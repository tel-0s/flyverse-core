# Synthetic radiance stimuli, the per-body RF localizer, and what the localizer found (round 2, `build:synthetic`)

**Status: built, tested (27 CPU tests in `tests/test_object_round2.py`, 19 of them in this file's class), run once on
the cluster (14 + 2 + 4 jobs, 0 failed, every run verified on CUDA); every number below is from a named file written
by `scripts/probe_synthetic_stimuli.py`. One run per stimulus family: magnitudes, no verdicts (`docs/INTERP.md` 2.4).
Three findings. (1) The localizer as specified (a 4.5-deg dark square, 200 ms per node) localizes Mi1 to within one
column (7 % coverage, 3.5 deg from the anatomical column, reproducible to 0.0 deg over three runs, false-fit rate
0 %) and **no LC11 or LC10a body**; with three passes and a 15-deg square (section 8) it localizes T2 / T3 / Tm5Y
(8-24 %, 4-5 deg from anatomy) and **13 LC10a bodies (4.7 %, half within 10 deg of their column) -- in ONE run of the
`gain_fb = 0` lobe, the only configuration the 15-deg localizer was ever run on**, while **LC11 stays at zero at
every size run (4.5 / 8.8 / 15 deg), its peak node at chance** -- the output of LC11 carries no retinotopic response
to a small dark square in this model although its inputs do. (2) The rate optic lobe is not stationary on a constant
scene (section 5): with the spiking feedback off it sustains a broadband population oscillation (peaks 4.4-4.8 Hz and
a sharp line at 32.1 Hz *modulo the 100-Hz frame sampling*; LC11 drive SD 1.25 mV in the population mean, 2.1 mV per
cell), the floor every within-run window comparison sits on and which two identical blank runs (`optic_measures.md`
6, 1e-9) could never show. (3) The specificity battery on the shipped lobe (one run each, section 6): the wide-field
families (grating, flicker, bar, 30-deg square) leave the one blank/blank draw behind at LC10a and every upstream
type; the small rectangles and 8.8-deg flashes do not, at LC11 nothing does -- but read that last clause with section
6.2: at elevation 0 the small rectangles and the flash sit in the sparsest patch of this retina, so it is partly a
statement about stimulus placement, not only about LC11.**

**Skeptic-checked (2026-09-13): `verify:synthetic`, verdict `mostly sound`, three refutations and five corrections,
plus one refutation from `verify:compare` that lands on the flash battery** (all verbatim in
`receptor_verification.md`, section "Object round 2 (2026-09-13) -- build skeptic verdicts"). Applied in place: the
4.4-deg rectangle's max coverage is **0.8853**, not 0.77 (1.1); the class runs **19** tests, not 16 (2); "the stored
radiance IS the presented input" is **false for `--null` runs** (3, 9); the cluster runs did not execute the
COMMITTED model (3, 9); the elevation-0 trajectory runs through a **retinal hole** (6.2); the 32.1-Hz line is
**aliased** by the 100-Hz frame sampling (5); the LC10a-at-15-deg and LC11 statements carry their `one run,
gain_fb = 0` qualifiers (4, 8); and, from `verify:compare`, the `flash` stimulus is a **periodic** square whose
whole-window statistic pools both transition polarities -- so the `flashon` / `flashoff` pair separates BRIGHT from
DARK, and the ON / OFF measurement is the new analysis-side split of **6.1**.

Neurome's intake (`docs/NEUROME_INTERFACE.md` 3b) asked for a matched visual assay -- constant elevation, angular
trajectory, angular speed, contrast and background; height and width ladders matching the published rectangles; a
fixed-centre square ladder; a per-body receptive-field localizer; blank and object radiance recorded in the loop;
per-body time courses before population maxima -- and, for the model comparison, bright / dark objects, isolated ON /
OFF transitions, stationary flicker, bars and gratings as specificity controls. This file documents the synthetic
half of that (the scene half is `scripts/probe_object_matched.py`, `build:sphere`): the stimuli, the localizer, the
RF-map format (`docs/INTERP.md` 2.7), the validation batch, and the localizer's result on LC11 / LC10a.

Files: `scripts/probe_synthetic_stimuli.py` (`preview` / `record` / `plan` / `verify` / `rfmap` / `analyse` /
`stationarity` / `divergence`), `tests/test_object_round2.py::SyntheticStimuliTests`, `docs/INTERP.md` 2.7 (the
RF-map file), `out/synth/` (batch 1, 12 runs + consoles + `batch.sh`), `out/synth_flash/` (batch 1b, the two flash
runs resubmitted), `out/synth2/` (batch 2), `out/synth_cluster.log` / `out/synthfl_cluster.log` /
`out/synth2_cluster.log`, `out/interp/synth/` (the Results: `rfmap_shipped.json`, `rfmap_fb0.json`,
`families.json`, `stationarity.json`, `divergence.json`, `stationarity_fb0_cpu.json`), `out/synth_ladder/batch.sh`
(the ladder batch, generated, NOT run).

## 1. What was built

### 1.1 The path

The fly is pinned and there is no body and no ray tracing: a `FlyBrain` (the room's LIF settings -- dt 0.5 ms,
prune_frozen, native CUDA kernels, `warp` CSR, `event_driven` on CUDA, as `probe_object_sweep.py`'s `Sim`) receives a
`(n_col, 4)` per-column radiance through `FlyBrain.vision(radiance)` every 10-ms frame, which `FlyBrain.step` hands to
`OpticLobe.step_frame` (`fly.py:233-235`). The array handed over is the stimulus' own stored array (`play`, one device
tensor filled by `copy_`), so the radiance written to `<run>_radiance.npz` IS the input the brain received:
`retina.mode = 'synthetic_direct'` in every provenance block, with `Retina.col_az_el` / `col_dir` / `col_side` /
`col_hex` beside it, and a per-frame checksum of what was presented, checked against the stored stimulus at the end of
every run (`checksum_per_arm`, both arms; `verify` refuses a run where it fails).

**Exception, named by `verify:synthetic`: in a `--null` run the stored `radiance` array is NOT what was presented.**
`--null` makes both arms blank (`blank_a` / `blank_b`) but the run still stores the family's own frames, so
`out/synth/null_blank_blank_radiance.npz` holds the 4.4 x 8.8 rect frames under key `radiance` (226 of its 300 frame
sums differ from the blank sum) although both arms presented only the constant background. The checksums themselves
are right (`checksum__blank_a` / `_blank_b` equal the blank sum exactly, which is what `verify` checks), but the
provenance sentence "the stored array IS the presented input" is written unconditionally into the retina record of
null runs too -- only `summary.arms` (`['blank_a', 'blank_b']`) and `prov.stimulus.null` disambiguate. Read those two
fields before reading a null run's `radiance`; the analysis never does (it takes the null's per-frame recordings, not
its radiance). The fix belongs in `cmd_record`'s retina record and is left to the probe's owner (section 9).

Units and background: the background is the per-channel [UV, B, G, R] mean of the room's blank arm at the object
sweep's pinned pose (`BACKGROUND` = 0.0364 / 0.0930 / 0.0949 / 0.0985, from
`out/apply_object/ladder/d114_null_s0_retina.npz`); a dark object is Weber contrast -0.995 (the black ball's radiance
is 0.0043-0.0052 of the wall's in `d300_stim_s0_retina.npz`), a bright one +0.995 (1.995 x background; the wall's own
per-column maximum is 2.0 x). Object radiance = background x (1 + contrast x coverage), where coverage is the fraction
of the column's acceptance kernel inside the object -- the SAME 7-ray, 4.5-deg Gaussian kernel the room's retina
samples with (`Retina.ray_directions`; the test proves the kernels equal to 1e-9). A 4.4-deg-wide rectangle therefore
never covers a whole column, as it does not in the scene: **max coverage 0.8853** over the sweep, measured on the
shipped retina from the stored radiance (`out/synth/rect_dark_h088_w044_radiance.npz`, coverage = (r/bg - 1) /
contrast, channel spread 1e-7). The **0.77** this file used to quote (`verify:synthetic`) is only the
exactly-centred value (0.3118 + 4 x 0.1147; the two rays at `daz` = +-2.25 fall outside a 4.4-deg window) -- a
0.05-deg offset admits a sixth ray and gives 0.8853. The qualitative claim (max < 1) stands, and 8.8-deg and larger
squares do reach exactly 1.0. The test at `tests/test_object_round2.py` asserts only `0.005 < rel.min() < 0.5`, so it
never pinned either number.

### 1.2 The stimuli (each returns `(n_frames, n_col, 4)` radiance plus the matched blank = the constant background)

| family | what | defaults | track |
|---|---|---|---|
| `rect` | a width x height rectangle translating in azimuth at constant angular speed at a fixed elevation, a triangle wave in ANGLE between -half_span and +half_span (the object sweep's lateral triangle wave in metres was 45.8 deg/s at the centre and 18.8 at the ends) | 4.4 x 8.8 deg, -0.995, 40 deg/s, elevation 0, +-30 deg, 3 s | az / el / width / height per frame |
| ladders (`plan --ladders`) | height ladder at width 4.4 (heights 2.2 / 4.4 / 8.8 / 15 / 22 / 30), width ladder at height 8.8 (same values), square ladder 4.5 / 8.8 / 11 / 15 / 20 / 30; dark and bright | `HEIGHT_LADDER`, `WIDTH_LADDER`, `SQUARE_LADDER` | |
| `bar` | a full-height vertical bar (every elevation) translating like the rectangles | 7 deg wide | |
| `grating` | `probe_motion.py`'s sine grating in the room's units: background x (1 + c sin(2 pi (x - s v t) / period)), x = abs(azimuth) for front->back / back->front, elevation for up / down | period 30 deg, c 0.5, 40 deg/s | phase shift |
| `flicker` | full-field square wave, bright half first, time-mean = background | 2 Hz, c 0.5 | level |
| `flash` | a stationary square at (az, el) shown for `on_s` every `period_s` from `first_s`. **PERIODIC, so every run contains BOTH transition polarities** (bright: ON at onset / OFF at offset; dark: the reverse -- the stimulus records which in `params['transition']`). The ON / OFF separation is an analysis-side split of the run, section 6.1 -- not two different stimuli | 8.8 deg at (0, 0), 0.5 s on, 1.5 s period, first at 0.5 s -> 4 transitions in 3 s | visible |
| `localizer` | a `size` square of contrast -0.995 flashed for 200 ms at every node of a 10-deg azimuth x elevation grid covering the eye (nodes that reach at least one column: **288** on the shipped retina, az -120..120, el -70..70), 300 ms of background between flashes, a fixed shuffled order per pass (`order_seed`), `passes` passes; stored as pattern + index | 4.5 deg, 1 pass -> 144.3 s, 14,430 frames | node / on per frame, node az / el, order |
| `blank` | the constant background (the stationarity control) | 3 s | |

### 1.3 Recording

`GatherRecorder` (a `common.Recorder` whose per-frame gather is done on the device) records `drive_mv`, `spike_count`
and `optic_dr` of `LC11, LC10a, T2, T3, Tm5Y, TmY21, Mi1` (7,031 cells; Mi1 because its hex-annotated cells are the
localizer's own ground truth) after every frame of both arms (stimulus, then the matched blank on a fresh brain of the
same seed, `probe_object_sweep`'s pattern; `--null` = blank / blank). Non-localizer runs keep the full `(T, n_cells)`
series (`<run>_stim.npz` / `_blank.npz`, the recording schema of `docs/INTERP.md` 2.3, provenance in the json
sidecar); the localizer keeps per-node reductions of every cell (`<run>_nodes.npz` for the stimulus arm and
`<run>_nodes_blank.npz` for the blank arm -- the blank arm's node windows are the fit's false-positive control -- with
`on` = the 20 flash frames, `off` = the 10 frames after the offset, `base` = the 10 frames before the onset, per
`drive_mv` / `optic_dr` / spikes-per-frame) plus per-type population means per frame. Every JSON carries the resolved
`LIFParams` / `OpticParams` (the stream-hook fields included), the realised device, the cache fingerprint and the
source fingerprint through `common.provenance`, and the `--optic` overrides pass through unchanged (`gain_fb=0` is the
deterministic lobe; the hook arms of `optic_stream_hooks.md` run on this battery through the same flag).

### 1.4 The RF map (`rfmap`; the file format is `docs/INTERP.md` 2.7)

Per body, from the per-node `on - base` response R (n_nodes,): the peak node is argmax |R|, the noise is 1.4826 x MAD
over nodes, a centre is FITTED when |peak| >= `z_min` (5) x noise, the centre is the (R - half-max)-weighted centroid of
the nodes at or above half-maximum of the peak's sign, the width the equivalent-disc FWHM of those nodes
(2 sqrt(n spacing^2 / pi), at least one spacing). Over several runs the map is the mean centre of the bodies fitted in
EVERY run, with the largest pairwise great-circle distance between the runs' centres per body (`centre_spread_deg`).
Beside the fit, criterion-free: the great-circle distance of each body's PEAK node from its anatomical column
(`trace.column_of_cells`) against the chance fraction of nodes within 15 deg of that column; the blank arm's per-node
SD as an independent noise (`z_blank`); the same fit rule applied to the blank arm (the false-fit rate); and a
sensitivity table of coverage / false-fit rate at `z_min` 3 / 4 / 5 / 8. CSV columns
`bodyId,type,az_deg,el_deg,width_deg,peak,n_nodes_above_threshold,...` (NaN centre when not fitted; a consumer
selects `fitted`), read by `probe_object_matched.py --rf-map` (its loader drops the unfitted rows; tested).

## 2. Validation on the CPU (`tests/test_object_round2.py::SyntheticStimuliTests`, 19 tests; 27 in the file, all pass)

```
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m unittest tests.test_object_round2
```

On a synthetic 27 x 18 column grid at the retina's 4.6-deg pitch, no connectome: the sampling kernel equals
`Retina.ray_directions` (weights and the 7 unit vectors, 1e-9); the rectangle's shape, its contrast (a fully covered
column reads background x (1 + contrast) to 1e-6; a 4.4-deg rectangle never covers a whole kernel), its constant
40 deg/s (every frame, 1e-9), its azimuth range and start, its constant elevation, and the position of the dimmed
columns against the track (centroid within 2.5 deg at three frames); the height ladder grows coverage monotonically
while keeping the azimuth extent at one column and the width ladder the converse; the bar dims every elevation row at
its azimuth; the grating reproduces `probe_motion`'s formula (1e-5) and averages to the background over one period; the
flicker's square wave, full field and mean; the flash's four transitions in 3 s, its name by contrast sign and
its position; **the ON / OFF transition split (two tests added with section 6.1): `luminance_state` maps a dark
square's visible frames to the DARKER level and a bright square's to the brighter one, `transition_frames` puts the
edges of a 0.5-s-on / 1.5-s-period flash at 0.5 / 1.0 / 2.0 / 2.5 s with the right polarity for each contrast sign
(dark: OFF at 0.5 / 2.0 s, ON at 1.0 / 2.5 s; bright: the reverse), 30-frame windows that are disjoint, exclude the
frames before the first edge, and truncate at the next edge when `on_s` is shorter than the window (20 frames at
`on_s` 0.2), and the same split on the flicker's level (`off` edges at 0.25 / 0.75 s, `on` at 0.5); then, on a
synthetic recording whose drive IS the frame index, `family_stats(frames=...)` equals the mean frame index of its own
windows for ON, OFF and pooled separately, the cumulative `spike_count` is re-accumulated over the kept frames only,
and `transition_stats` reproduces the same numbers with its frame and edge counts**; the localizer's grid reach, the
once-per-node 200-ms presentation, the 300-ms blanks, the node frames'
positions and the fixed order across runs, and `passes` (3 x 100 ms per node, a new order per pass, the reductions
averaging over the passes); the `NodeAccumulator`'s on / off / base windows and spikes per frame; `fit_rf` recovers a
Gaussian bump's centre within 3 deg and its sign, rejects a noise-only cell, and `merge_maps` reads 0 spread on
identical runs; **`play` hands the stored radiance to `vision` without aliasing it** (the previous attempt's
`torch.as_tensor(frame)` shared memory with the numpy view on the CPU, so the next `copy_` overwrote the stored
stimulus -- the smoke's `checksum_equal_stim_vs_presented: false`); the statistics definitions against
`probe_object_sweep.py` :277 / :291 / :292 (`diff_signed_best_cell` and `diff_abs_best_cell_mean` kept distinct) and
the time-course peak; the RF-map CSV columns and the sphere probe's loader; `verify_dir` on a CUDA / CPU / bad-checksum
triple; the CLI, `batch_jobs` (14), `ladder_jobs` (185 = 18 arms x 2 contrasts x 5 runs + 5 nulls) and `write_batch`;
**`make_stimulus` over every family with the batch's own argument strings** (the first batch's two flash jobs died on a
positional `background` landing in `first_s` -- the test that would have caught it, added with the fix);
`stationarity_stats` on a 4-Hz oscillation (peak 4.0 Hz) and a flat cell (0), pooled and per cell; `divergence_stats`
on identical-then-perturbed arms.

CPU smoke of the whole path (`--allow-cpu`; `out/synth_smoke/`): a 0.3-s rectangle and a coarse localizer (18 nodes,
3.7 s), both arm checksums true; `rfmap` and `analyse` run on them end to end (`rfmap_loc2.csv/json`,
`families_smoke.json`); a 3-s `blank` under `gain_fb=0` for section 5 (`blank_fb0_cpu_*`, `stationarity_fb0_cpu.json`).

## 3. The cluster batches (rented boxes; `out/synth*_cluster.log`)

| batch | jobs | log line | boxes | notes |
|---|---|---|---|---|
| `synth-4e31f2` (`out/synth/batch.sh`) | 14: localizer fb0 x1 + shipped x3 (seeds 0 / 1 / 2), rect dark and bright 4.4 x 8.8, squares 8.8 and 30, bar 7, grating, flicker 2 Hz, flash ON and OFF 8.8, one blank/blank null; 3 s + 2 s settle | `14 job(s), 0 failed (6.6 min)` | vast-a/b/c/d, H200 and B200 | the two **flash** jobs wrote nothing (`make_stimulus` defect, section 2); their job lines end in `tail -3` and reported `exit 0`, so the console's `0 failed` did not see it -- `verify` did, by the console without a summary (`out/synth/failed_first_batch/`) |
| `synthfl-153da8` (`out/synth_flash/batch.sh`) | 2: the flash jobs after the fix | `2 job(s), 0 failed (1.4 min)` | vast-a, vast-b | fetched into its own directory (one client per `--fetch` directory); `analyse --dir out/synth out/synth_flash` |
| `synth2-...` (`out/synth2/batch.sh`) | 4: localizer 8.8 deg x 3 passes (fb0, shipped), 15 deg x 3 passes (fb0), a `gain_fb=0` blank/blank with per-cell series | section 8 | | |

Every run: `verify --dir` = device `cuda` in the summary, the provenance and the console, both arm checksums true,
both arm recordings present (`out/synth/verify.json`: 12 / 12; `out/synth_flash`: 2 / 2). Localizer arms of 14,430
frames ran at 187-387 fps (37-77 s of wall per arm; H200 faster than B200 here), the 300-frame families at 46-400 fps.
Code identity on the cluster, corrected (`verify:synthetic`): **these batches did not run the COMMITTED model.**
`flyverse_commit` reads `{'commit': 'unknown', 'dirty': false}` -- and the `false` is an artefact of there being no
`.git` in the run dir, not evidence of a clean tree. The runs' `source_fingerprint` gives `flyverse/optic.py` =
`4e3dcca8ae86...`, which is the hooks task's UNCOMMITTED working-tree file (HEAD's blob is `1ea77ff68817...`), while
`flyverse/fly.py` (`75b93184...`) and `flyverse/senses.py` (`67590f95...`) match neither HEAD nor the current tree
(mid-edit proprioception versions). The skeptic tested the consequence: with the shipped defaults and seed 0, a clean
HEAD checkout (`git archive HEAD flyverse`) and the current tree give **bit-identical** `drive_mv` / `optic_dr` /
`spike_count` over 80 frames of the rect stimulus on CPU (max |diff| 0.0, both arms, all three quantities, repeat run
identical). So the numbers are almost certainly unaffected -- but every "shipped lobe" statement in this file inherits
the hooks task's own bit-identity claim on the CUDA path (`prov` shows `cuda_kernels` true, `hook_info.active` false),
and that claim is CPU / deterministic-backend only (`optic_stream_hooks.md` 3). Also on file: `source_fingerprint`
computed by content; compiled W md5 `ef23cc27bea13be7f6a96f3c04fd3737`, `sum_abs_W` 121,460,584, 167,106 neurons,
25,578,600 entries; torch 2.11.0+cu128; receptor model `sign` / `abs`; every `OpticParams` at its default
(`gain_fb` 0.5 / `gain_rr` 1 / `gain_in` 3 / `norm` l2 / `out_norm` l1 / `gain_out_mv` 100 / `drive_clip_mv` 35 /
`adapt_tau_ms` 400 / `adapt_gain` 1, stream hooks None) except `gain_fb 0` in the fb0 runs.

## 4. The RF localizer (4.5-deg dark square, 200 ms per node, 288 nodes, one pass)

### 4.1 Coverage and agreement (`out/interp/synth/rfmap_shipped.json`, three runs; `rfmap_fb0.json`, one run)

| type | bodies | shipped: fitted in all 3 runs (any run) | shipped: false fits on the blank arm | fb0: fitted (1 run) | fb0: false fits | Mi1-style agreement with the anatomical column |
|---|---|---|---|---|---|---|
| **LC11** | 143 | **0** (0) | 0.0 % | 1 (104 deg from its column) | 0.0 % | -- |
| **LC10a** | 275 | **0** (2) | 0.2 % | 1 (138 deg from its column) | 0.4 % | -- |
| T2 | 1,630 | 0 (11) | 0.1 % | 8 | 0.0 % | fb0: median 20.7 deg, 3/8 within 10 deg |
| T3 | 1,940 | 0 (3) | 0.0 % | 0 | 0.1 % | -- |
| Tm5Y | 898 | 0 (3) | 0.0 % | 0 | 0.0 % | -- |
| TmY21 | 372 | 0 (0) | 0.1 % | 0 | 0.3 % | -- |
| **Mi1** (the control) | 1,773 | **118** = 6.7 % (225 in any run) | 0.0 % | 172 = 9.7 % | 0.1 % | shipped: **median 3.5 deg from the anatomical column, 82 % within 10 deg, 85 % within 15 deg**, centre spread over the 3 runs median **0.0 deg**, width median 11.3 deg (one grid spacing); fb0: median 3.0 deg, 85.5 % within 10 deg |

Reading. The localizer works where it should: an Mi1 cell that is fitted sits at its hex-annotated column to within
one column, in every run, at a false-fit rate of 0-0.1 % on the blank arm. Its coverage is 7-10 % of Mi1 because a
4.5-deg square on a 10-deg grid samples about one column in five (a column between nodes gets a partial kernel
coverage at best), and at `z_min` 5 only the best-covered cells pass. **For LC11 and LC10a the map is empty**: no body
is fitted in all three shipped runs; on the deterministic lobe one body each is fitted and both sit > 100 deg from their
anatomical column, i.e. they are the false fits the blank arm's 0-0.4 % predicts.

### 4.2 Criterion-free: where the peak node falls, and how big the peak is

| type | peak node within 15 deg of the anatomical column: shipped / fb0 | chance | within 30 deg: shipped / fb0 | peak median, all bodies: shipped / fb0 | blank-arm node SD median | z_blank median: shipped / fb0 |
|---|---|---|---|---|---|---|
| LC11 | 0.000 / 0.014 | 0.022 | 0.056 / 0.147 | 1.98 / 4.78 mV | 1.93 / 1.96 mV | 0.93 / 2.41 |
| LC10a | 0.025 / 0.036 | 0.027 | 0.105 / 0.127 | 1.81 / 3.70 mV | 1.54 / 1.56 mV | 1.00 / 2.37 |
| T2 | 0.066 / 0.104 | 0.029 | 0.121 / 0.181 | 0.180 / 0.400 | 0.168 | 0.95 / 2.34 |
| T3 | 0.075 / 0.094 | 0.029 | 0.152 / 0.172 | 0.098 / 0.220 | 0.093 / 0.097 | 0.91 / 2.25 |
| Tm5Y | 0.085 / 0.095 | 0.027 | 0.153 / 0.175 | 0.181 / 0.398 | 0.170 | 0.95 / 2.25 |
| TmY21 | 0.046 / 0.011 | 0.024 | 0.086 / 0.083 | 0.217 / 0.538 | 0.221 | 0.91 / 2.36 |
| Mi1 | **0.290 / 0.298** | 0.029 | 0.334 / 0.368 | 0.058 / 0.109 | 0.038 / 0.039 | 1.18 / 2.41 |

Two facts. (1) At the small-field medulla / lobula types (T2, T3, Tm5Y) the peak node is at the anatomical column 2-3.5
times more often than chance (T3 0.075-0.094 vs 0.029; Mi1 10x): a retinotopic signal exists there that the `z_min 5`
fit does not pass. **At LC11 it is at chance or below** (0.000 / 0.014 vs 0.022), at LC10a at chance (0.025 / 0.036 vs
0.027). (2) The peaks are as large as the blank arm's own node-to-node scatter (shipped z_blank ~1 everywhere), and on
the deterministic lobe 2.4x it -- but not retinotopic. A flash anywhere moves LC11's drive by ~5 mV at the best node
and that node is not where the cell looks. Section 5 says why.

### 4.3 The criterion's trade-off (`rf_sensitivity`; coverage = fitted in every run, false-fit = fitted on any blank arm)

| type | z_min 3: coverage / false fit (shipped) | 4 | 5 | 8 | fb0 z_min 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| LC11 | 0.028 / 0.112 | 0.000 / 0.007 | 0.000 / 0.000 | 0 / 0 | 0.126 / 0.084 | 0.021 / 0.028 | 0.007 / 0.000 |
| LC10a | 0.055 / 0.244 | 0.004 / 0.065 | 0.000 / 0.007 | 0 / 0 | 0.135 / 0.127 | 0.018 / 0.025 | 0.004 / 0.004 |
| T3 | 0.021 / 0.107 | 0.000 / 0.009 | 0.000 / 0.001 | 0 / 0 | 0.053 / 0.056 | 0.006 / 0.006 | 0.000 / 0.001 |
| Mi1 | 0.166 / 0.101 | 0.094 / 0.010 | 0.067 / 0.000 | 0.026 / 0 | 0.230 / 0.059 | 0.147 / 0.010 | 0.097 / 0.001 |

Below `z_min` 5 the LC 'coverage' is the false-fit rate (LC10a at 3: 5.5 % fitted vs 24 % false on the blank arm);
Mi1 is the only type whose coverage exceeds its false-fit rate at every threshold. `z_min` 5 is kept as the default:
it is the lowest value at which the blank arm reads ~0 for every type.

## 5. The rate lobe is not stationary on a constant scene (`stationarity`, `divergence`)

The blank arm of every run is the constant background; on a settled deterministic lobe two windows 100 ms apart
should differ by ~0. They do not (`out/interp/synth/stationarity.json`, `--skip-s 1`):

| recording (arm) | type / quantity | per-cell SD over frames (median) | frame-to-frame abs diff (median) | 10-frame window means 200 ms apart, abs diff (median) | population-mean SD, first / last third | spectral peak of the population mean (share of power) |
|---|---|---|---|---|---|---|
| `null_blank_blank_blank_a` (shipped, per cell, 300 frames) | LC11 drive_mv | **2.53 mV** | 0.82 | 0.79 | 2.06: 1.53 / 2.27 | **4.5 Hz (0.54)** |
| | LC10a drive_mv | 1.82 | 0.95 | 0.73 | 0.83: 0.69 / 0.91 | 4.5 Hz (0.25) |
| | T3 optic_dr | 0.089 | 0.024 | 0.049 | 0.027: 0.022 / 0.030 | 4.5 Hz (0.53) |
| | T2 optic_dr | 0.156 | 0.040 | 0.089 | 0.063 | 4.5 Hz (0.43) |
| | Tm5Y optic_dr | 0.185 | 0.088 | 0.084 | 0.046 | 32 Hz (0.34) |
| | Mi1 optic_dr | 0.042 | 0.012 | 0.024 | 0.016 | 4.5 Hz (0.44) |
| `loc_fb0_s0_blank` (**gain_fb 0**, population means, 14,430 frames) | LC11 drive_mv | 1.25 (pooled) | 0.36 | 1.39 | 1.32 / 1.22 | 4.40 Hz (0.035) |
| | LC10a drive_mv | 0.67 | 0.62 | 0.31 | 0.68 / 0.66 | **32.09 Hz (0.084)** |
| | T3 optic_dr | 0.0171 | 0.0076 | 0.0129 | 0.0175 / 0.0168 | 4.40 Hz (0.022) |
| | T2 optic_dr | 0.0445 | 0.0127 | 0.064 | 0.046 / 0.044 | 4.40 Hz |
| | Tm5Y optic_dr | 0.038 | 0.046 | 0.012 | 0.039 / 0.038 | 32.09 Hz (0.146) |
| | TmY21 optic_dr | 0.067 | 0.040 | 0.004 | 0.067 / 0.066 | 32.09 Hz |
| | Mi1 optic_dr | 0.0089 | 0.0034 | 0.0019 | 0.0092 / 0.0089 | 4.74 Hz |
| `loc_shipped_s0_blank` (shipped, population means) | LC11 / LC10a / T3 / Mi1 | 1.35 / 0.71 / 0.018 / 0.010 | | | | 4.6 / 32.1 / 4.5 / 4.8 Hz |

* **The fluctuation is in the rate lobe itself, not the spiking feedback.** With `gain_fb = 0` the lobe's only input is
  the constant radiance and its population means still move: LC11's pooled drive SD 1.25 mV against 1.35 under the
  shipped feedback, T3 0.0171 vs 0.0179, Mi1 0.0089 vs 0.0102 -- the feedback adds ~10 %. The first and last thirds of
  a 144-s run have the same SD (1.32 / 1.22 mV), so it is sustained, not a settling transient. It is there on the CPU
  as well (`stationarity_fb0_cpu.json`, section 5.1).
* **It is a broadband population oscillation with two spectral lines**: 4.4-4.8 Hz at LC11, T2, T3, Mi1 (and on the
  shipped per-cell arm it carries 43-54 % of the population mean's power) and 32.1 Hz at LC10a, Tm5Y, TmY21. The
  timescales are the lobe's own (`tau_ms` 10 recurrent, `adapt_tau_ms` 400 adaptation); which loop makes which line is
  not determined here. **Aliasing caveat (`verify:synthetic`): the 32.09-32.11 Hz line is measured from a series
  sampled once per 10-ms frame (100 Hz) while the lobe integrates at `dt_ms` 1, so a true line at 67.9 or 132.1 Hz
  would alias onto 32.1 Hz.** Every statement of it reads "a sharp line at 32.1 Hz modulo the 100-Hz frame sampling";
  the 4.4-4.8 Hz line is well below Nyquist and is safe. Resolving it needs a per-substep record, which no tool here
  writes.
* **It is deterministic and perturbation-sensitive** (`out/interp/synth/divergence.json`): the stimulus and blank arms
  of `loc_fb0_s0` share the seed and the 2-s settle, and their population means agree to **1.5e-8 mV (LC11) / 0 (T3,
  Mi1)** over the 30 frames before the first flash; a single 4.5-deg, 200-ms flash then separates them (LC11 0.05 mV
  during the flash, 0.33 within 30 frames) and they never re-converge: over the last third of the run |stim - blank|
  averages 1.42 mV against a blank-arm SD of 1.22 and the two arms correlate at **-0.085** (T3 0.020 vs 0.017,
  corr -0.079). Under the shipped feedback the arms already differ before the first flash (0.24-0.56 mV: the LIF's
  nondeterminism enters through `W_rs` from frame 0). A stable fixed point would re-converge; this trajectory does
  not, which is why a within-run window comparison at LC11 / LC10a reads a ~1.5-2 mV floor on every node and why
  the deterministic lobe's 'responses' to the flash (section 4.2: 2.4 x the blank scatter, at no particular node) are
  the perturbed oscillation, not a receptive field.
* **Why round 1 did not see it.** `optic_measures.md` 6 established the `gain_fb = 0` null as "numerically zero": two
  blank runs agree to 1e-9 -- true, they are the same deterministic trajectory. The object sweep's statistics are
  12-s time-means, over which a stationary oscillation averages out identically in both arms. Neither tests
  stationarity within a run. Every per-frame or per-window quantity of this round (the localizer's on - base, the
  time-course peaks of section 6, any 'response latency') is read against this floor, and the size-ladder's
  `diff_max_over_cells_mean_mv` (a 12-s mean) is not.

What this licenses: nothing in the model changes. It is a **diagnosis for the optic-lobe owners** (`flyverse/optic.py`
is the hooks task's file this round): a recurrent rate lobe at `gain_rr` 1 with l2 fan-in normalisation and a 400-ms
adaptation loop sustains an autonomous oscillation on a constant scene. Whether the animal's optic lobe does is a
question for the literature; whether the round-1 figure-stage findings (`deficit_object.md`) depend on it is a
question for a rerun of `interp_apply_object.py perrun` with the window statistics of `stationarity` beside its
12-s means. What the localizer needs from it is stated in section 7.

### 5.1 The CPU check (`out/interp/synth/stationarity_fb0_cpu.json`, `record --stimulus blank --optic gain_fb=0 --allow-cpu`, 3 s from cold, `--skip-s 2`)

The same per-cell fluctuation on the CPU torch path (no CUDA kernels, `device cpu`), so it is not a kernel artefact:
per-cell SD over the last second LC11 drive **2.08 mV** (frame-to-frame 0.65, 10-frame windows 200 ms apart 1.34),
LC10a 1.76 (0.89 / 0.96), T3 optic_dr 0.092 (0.021 / 0.100), Mi1 0.038 (0.010 / 0.040); population-mean spectral peaks
4-5 Hz at LC11 / T3 / Mi1 and 32 Hz at LC10a (1-Hz resolution over 100 frames), shares 0.34-0.39.

## 6. The stimulus families on the shipped lobe (one run each, `out/interp/synth/families.json`)

`diff_max_over_cells_mean_mv` (LC types: max over cells of the per-cell time-mean stimulus - blank drive, the size
ladder's statistic) / `diff_signed_best_cell` (rate types: max over cells of |mean dr_stim - mean dr_blank|), with the
blank/blank draw in the last column. **One run per family and one null draw: magnitudes.** **Every column is a
WHOLE-window (3 s) mean, so the two flash columns are a dark flash and a bright flash -- each containing both an ON
and an OFF transition, pooled. The ON / OFF transition measurement is section 6.1.** The round's rule
(`INTERP.md` 2.4) needs >= 4-5 runs per arm in one submission before any of these is a difference; the generator for
that submission is `plan --ladders` (185 jobs at 5 runs -- split by `--family` and contrast to stay under ~20 jobs per
submission on the rented boxes).

| type | rect dark 4.4x8.8 | rect bright | square 8.8 | square 30 | bar 7 | grating | flicker 2 Hz | flash DARK 8.8 (`flash_off`) | flash BRIGHT 8.8 (`flash_on`) | **blank/blank** |
|---|---|---|---|---|---|---|---|---|---|---|
| LC11 (mV) | 0.083 | 0.182 | 0.110 | 0.185 | 0.085 | 0.741 | 0.239 | 0.119 | 0.176 | **0.135** |
| LC10a (mV) | 0.129 | 0.114 | 0.076 | 0.312 | 0.406 | 1.071 | 0.368 | 0.213 | 0.278 | **0.215** |
| T2 | 0.014 | 0.017 | 0.025 | 0.084 | 0.042 | 0.103 | 0.163 | 0.019 | 0.019 | **0.017** |
| T3 | 0.008 | 0.006 | 0.012 | 0.024 | 0.020 | 0.063 | 0.052 | 0.008 | 0.010 | **0.009** |
| Tm5Y | 0.014 | 0.012 | 0.011 | 0.055 | 0.030 | 0.144 | 0.112 | 0.016 | 0.020 | **0.018** |
| TmY21 | 0.017 | 0.010 | 0.017 | 0.042 | 0.036 | 0.161 | 0.155 | 0.020 | 0.029 | **0.022** |
| Mi1 | 0.011 | 0.008 | 0.008 | 0.040 | 0.041 | 0.036 | 0.032 | 0.008 | 0.006 | **0.008** |
| LC11 / LC10a best-cell rate diff (Hz) | 0 / 0.7 | 0 / 1.0 | 0.3 / 0 | 0.3 / 1.0 | 0.3 / 1.3 | 0.3 / 0.7 | 4.3 / 1.7 | 0.3 / 0.7 | 0 / 0.7 | 0 / 1.0 |

Read as magnitudes: the wide-field families (grating, flicker, bar, the 30-deg square) leave the one null draw behind
at every upstream type (T2 0.08-0.16 vs 0.017; Tm5Y 0.06-0.14 vs 0.018) and at LC10a (0.31-1.07 vs 0.215); the small
rectangles and the 8.8-deg flashes do not (LC11 0.08-0.18 around a null of 0.135; T3 0.006-0.012 around 0.009). This
is the size-ladder picture of round 1 on a matched stimulus with no retinal-position confound -- and still one draw,
and, for the small families, on a trajectory that barely reaches a column (6.2), with the flash rows pooled over both
transition polarities (6.1).
The per-frame population time courses (`tables.time_course`, every 50 ms) peak at +-4-6 mV at LC11 in every family
**and in the null**: that is the 4.5-Hz oscillation of two independently phased arms (section 5), not a response, and
the per-body rows (`tables.per_body`: every LC11 / LC10a body's time-mean difference, peak frame and rates in both
arms) carry the same floor. Read through the RF map (`--rf-map out/synth/rfmap_shipped.csv`), only Mi1 has fitted
bodies: 8 of its 118 bodies have the 30-deg square inside their RF (`diff_in_rf` best body +0.140), 23 the bar
(+0.072), 1 each the small rectangles; no LC11 / LC10a body has a window because none has a centre.

### 6.1 ON / OFF transitions: the pooled row is bright-vs-dark, the split rows are the measurement

`verify:compare` refuted the label "isolated ON and OFF flashes" on this battery, and it was right. `flash` is a
**periodic** square (`--on-s 0.5 --period-s 1.5` over 6 s = 4 cycles), so **every run contains both polarities**: a
bright square's onset is the ON transition and its offset the OFF transition, a dark square's the reverse (the
stimulus says so in `params['transition']`). The section-6 table above is `family_stats(..., window=None)`, the
WHOLE-window time mean, which pools the two edges inside each run -- so the `flash ON` / `flash OFF` pair of that
table (and the `flashon` / `flashoff` families of the model-comparison batch) separates a **BRIGHT flash from a DARK
one**, not an ON transition from an OFF one. Neurome asked for isolated ON / OFF transitions.

The stimulus is unchanged (it is on the cluster and its runs are on file). The fix is analysis-side, in the same
tool: `luminance_state` -> `transition_frames` -> `transition_stats` (`scripts/probe_synthetic_stimuli.py`) reduce a
run over **the frames after each rising luminance edge and after each falling edge separately** -- the same `on` /
`off` role split the localizer's `NodeAccumulator` already uses per node -- and `record` / `analyse` report them as
rows `transition = on | off` **beside** the pooled row (`transition = pooled`), never instead of it. Window:
`--transition-window-s`, default **0.3 s** (30 frames), truncated at the next edge so a frame belongs to exactly one
transition; the frames before the first edge belong to neither. The kept windows are concatenated, so a split row's
spike rate uses its own duration and its `tc_peak_t_s` is an offset within the split, not a time in the run. A blank arm has no edge, so the **blank/blank null
run is scored on each family's edge schedule** (column `edge_schedule`) -- that is the floor of the window statistic,
and it is the number the split rows have to be read against, because a 0.3-s window sits on the oscillation of
section 5 rather than on a 3-s time-mean that averages it out.

One run per family, 0.3-s windows (0.25 s in the flicker: truncated by its own 2-Hz half-period, 125 ON / 150 OFF
frames against the flashes' 60 / 60), two edges per polarity in the flash runs, 5 ON / 6 OFF in the flicker, shipped
lobe -- **magnitudes, no verdicts**; `diff_max_over_cells_mean_mv` (mV) for the LC types, `diff_signed_best_cell` for
the rate units; "floor" = the blank/blank run on the same windows:

| run | statistic | pooled (3 s) | ON | OFF | floor ON | floor OFF |
|---|---|---|---|---|---|---|
| dark flash 8.8 deg (`flash_off_088`) | LC11 | +0.119 | +0.602 | **+1.485** | +0.589 | +0.561 |
| | LC10a | +0.213 | +0.768 | **+1.164** | +0.837 | +0.664 |
| | T3 | +0.008 | +0.076 | +0.091 | +0.082 | +0.057 |
| bright flash 8.8 deg (`flash_on_088`) | LC11 | +0.176 | +0.617 | +0.477 | +0.561 | +0.589 |
| | LC10a | +0.278 | +0.925 | +0.565 | +0.664 | +0.837 |
| | T3 | +0.011 | +0.052 | +0.055 | +0.057 | +0.082 |
| flicker 2 Hz full field | LC11 | +0.239 | **+7.985** | **+4.470** | +0.258 | +0.509 |
| | LC10a | +0.368 | **+5.753** | **+6.071** | +0.600 | +0.855 |
| | T3 | +0.052 | **+0.503** | **+0.512** | +0.057 | +0.058 |

Reading (one run each, magnitudes): **of the twelve 8.8-deg flash split values in the table, only the three of the
DARK flash's OFF transition leave their own floor** -- that transition is the dark square's ONSET (the luminance step
down at 0.5 and 2.0 s; the dark square's offset is its ON transition): LC11 1.485 vs a floor of 0.561 (2.6x), LC10a
1.164 vs 0.664 (1.8x), T3 0.091 vs 0.057 (1.6x). Every other flash value is at or below its floor (0.6-1.4x),
including both ON transitions and every value of the bright flash. The full-field flicker leaves the floor at every
type by 7-31x, as
the pooled table already said of it. So on this battery an 8.8-deg flash at (0, 0) is a floor-level stimulus for
LC11 in both polarities -- consistent with section 6.2's placement finding (that flash ever touches 2 columns above
5 % coverage) -- and the one exception is a single run of one polarity, which a >= 5-run submission must confirm or
drop. Nothing here is a verdict.

Generator:
`PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python scripts/probe_synthetic_stimuli.py analyse --dir out/synth out/synth_flash --transition-window-s 0.3 --json out/interp/synth/families.json`
(the rows above were produced by that code path on a scratch copy holding just the four runs it needs -- both flash
runs, the flicker run and the blank/blank null -- with `--no-verify --allow-cpu`; `problems: none`). The
`out/interp/synth/families.json` on file predates the split and carries no `transition` rows; re-run `analyse` to
regenerate it. `summary.transition_split` records the window, the column names and this reading rule; `per_type`
gains the `transition` and `edge_schedule` columns and `families.<run>.transitions` / `.transition_floor` the
per-edge frame and edge counts.

### 6.2 The trajectory runs through the sparsest patch of this retina (`verify:synthetic`'s biggest substantive gap)

The chosen path (elevation 0, azimuth +-30 deg) is where the shipped retina has almost no columns. Of its 1,466
columns only **16** have |el| <= 6 within |az| <= 35, and exactly **ONE** column lies between az -15 and +15 (at
-4.3); the 10 x 10-deg cells along the equator hold 0-2 columns against 4-8 elsewhere. Measured from the stored
radiance: the 4.4 x 8.8-deg dark rectangle dims **no column at all** (max coverage < 1e-6) in **74 of 300 frames**
and dims a median of **0.23 column-equivalents**; the 8.8-deg square is invisible in **26 of 300** frames (median
0.77 column-equivalents); the 8.8-deg ON/OFF flash at (0, 0) ever touches **2 columns** above 5 % coverage, peaks at
**0.656** coverage, and has **one** column above half-max. So section 6's "the small rectangles and the 8.8-deg
flashes do not leave the null behind" is partly a statement about stimulus placement in a retinal hole, not only
about LC11 -- and the ladder batches inherit the same trajectory. The localizer's nodes are far better placed (median
1.0 / 4.0 / 11.4 column-equivalents at 4.5 / 8.8 / 15 deg), which is why it, and not this battery, is what
established the LC11 result of sections 4 and 8. Before a small-object family is read as a null, quote its
column-equivalents per frame; an elevation offset onto a populated patch (or the ray-traced assay's geometry,
`object_matched_assay.md`) is the fix, and neither is run here.

**Replicate discipline, stated plainly** (`verify:synthetic`): every family number in section 6 / 6.1 and every
batch-2 RF map is **n = 1 run per arm**, against the round's rule of >= 5 runs per arm in one submission
(`docs/INTERP.md` 2.4). This file uses no `compare()` verdict vocabulary anywhere, which is the right call; the only
multi-run measurement in the build is the three-run 4.5-deg shipped localizer (section 4). The LC10a-at-15-deg result
and the "the oscillation is sustained" statement each rest on one run (the latter at least a 143-s record with
internal thirds).

## 7. What follows for the localizer and the assay (diagnoses, not changes)

1. **A 4.5-deg, 200-ms single flash per node cannot localize an LC11 / LC10a body in this model**: the response at
   the pooled output (94 / 32 columns per cell, `out_norm l1`) is below the lobe's own 1.5-2 mV window-to-window
   fluctuation, and on the deterministic lobe the flash's main effect is to perturb that oscillation, at every node
   alike. The localizer's two knobs for this are shipped and tested: `--width` (the square) and `--passes` (repeat the
   grid; the per-node reductions average, the perturbed oscillation does not add up coherently across passes while a
   retinotopic response does) -- batch 2 (section 8) is the measurement with them: three passes of an 8.8-deg square
   localize T2 / T3 / Tm5Y and half of Mi1, a 15-deg square localizes 13 LC10a bodies, LC11 none. The fit rule should
   then read the peak against the blank arm's per-node scatter (`z_blank`, on file) rather than the stimulus arm's MAD.
2. **The sphere assay's RF windows** (`probe_object_matched.py --rf-map`) find no LC11 row in any map and LC10a rows
   only in `out/synth2/rfmap_150_fb0_p3.csv` (13 bodies, one run); its LC11 per-body statistics stay whole-window, and
   the anatomical column (`trace.column_of_cells`) remains LC11's only per-body region -- which section 4.2 / 8 say the
   localizer's peaks agree with at chance level.
3. **Within-run window statistics need the stationarity floor beside them.** Any latency, peak or on-off comparison
   on this lobe is read against `stationarity`'s window-mean difference (LC11 0.8-1.4 mV, T3 0.013-0.05), and the
   12-s time-means of the size ladder are the statistic that averages it out. Whether `gain_fb = 0` is "the exact null"
   depends on the statistic: exact for a time-mean, not for a window. The ON / OFF split of 6.1 is exactly such a
   window statistic, which is why `analyse` scores the blank/blank run on the same edge schedule and prints that floor
   beside it (LC11 0.26-0.59 mV over 0.3-s windows) -- a split row without its floor is not readable.
4. **Nothing here is a parameter proposal.** The oscillation is a property of the shipped rate lobe under the
   project's defaults; damping it (`adapt_gain`, `gain_rr`, `norm`) would be hand-tuning toward a behaviour. It goes
   to the optic-lobe owners as a measurement, with the generators to reproduce it.

## 8. Batch 2: larger squares, three passes, and a per-cell `gain_fb = 0` blank (`out/synth2/`, `synth2-5333f9`, `4 job(s), 0 failed (10.5 min)`, B200, all verified)

Three localizers with the two shipped knobs of section 7.1 -- an 8.8-deg square x 3 passes on the deterministic and
on the shipped lobe (309 nodes, 46,380 frames per arm, ~200 fps), a 15-deg square x 3 passes on the deterministic lobe
(327 nodes, 49,080 frames) -- and a `gain_fb = 0` blank/blank with full per-cell series (`null_fb0`). One run each.
`rfmap_088_fb0_p3.*`, `rfmap_088_shipped_p3.*`, `rfmap_150_fb0_p3.*` (CSV in `out/synth2/`, Results in
`out/interp/synth/`); `z_min` 5; false-fit rate on the blank arm 0-0.4 % for every type in every map.

| type | 4.5 deg x 1 pass, shipped x3 (section 4): fitted / peak within 15 deg of anatomy (chance) | 8.8 deg x 3, fb0 | 8.8 deg x 3, shipped | **15 deg x 3, fb0** |
|---|---|---|---|---|
| **LC11** (143) | 0 / 0.000 (0.022) | 0 / 0.014 (0.022) | 0 / 0.014 (0.022) | **0 / 0.021 (0.021)** |
| **LC10a** (275) | 0 / 0.025 (0.027) | 2 (both 77 deg from their column) / 0.047 (0.026) | 2 (both 6.8 deg from their column) / 0.055 (0.026) | **13 = 4.7 %, median 12.0 deg from the column, 46 % within 10 deg, 54 % within 15 / 0.105 (0.025)** |
| T2 (1,630) | 0 / 0.066 (0.029) | 97 = 6.0 %, median 3.0 deg, 91 % within 10 / 0.302 | 104 = 6.4 %, 3.6 deg, 85 % / 0.287 | 216 = 13.3 %, 3.8 deg, 86 % / 0.463 |
| T3 (1,940) | 0 / 0.075 (0.029) | 45 = 2.3 %, 7.0 deg, 73 % / 0.257 | 48 = 2.5 %, 6.1 deg, 79 % / 0.233 | 155 = 8.0 %, 4.7 deg, 77 % / 0.374 |
| Tm5Y (898) | 0 / 0.085 (0.027) | 78 = 8.7 %, 4.5 deg, 68 % / 0.305 | 78 = 8.7 %, 4.4 deg, 69 % / 0.317 | 217 = 24.2 %, 4.8 deg, 73 % / 0.478 |
| TmY21 (372) | 0 / 0.046 (0.024) | 0 / 0.027 | 2 (31 deg) / 0.046 | 11 = 3.0 %, 27.8 deg, 27 % / 0.075 |
| Mi1 (1,773) | 118 = 6.7 %, 3.5 deg, 82 % / 0.290 (0.029) | 855 = 48.2 %, 2.5 deg, 99.4 % / 0.644 | 867 = 48.9 %, 2.5 deg, 99.0 % / 0.644 | 1,104 = 62.3 %, 2.2 deg, 99.1 % / 0.734 |

Fitted widths (equivalent-disc FWHM): Mi1 11.3 deg (one spacing) at 4.5 and 8.8 deg, 19.5 at 15 deg; T2 / T3 / Tm5Y
16.0 at 8.8 deg and 21-23 at 15; LC10a 15.4 (8.8 deg, shipped) and 22.6 (15 deg). Coverage at `z_min` 3 / 4 / 5 / 8
for the 15-deg map: LC10a 0.41 / 0.12 / 0.05 / 0.004 against a blank-arm false-fit rate of 0.31 / 0.011 / 0 / 0; LC11
0.36 / 0.007 / 0 / 0 against 0.25 / 0.014 / 0 / 0 -- at 3 the LC 'coverage' is still mostly the false-fit rate, at 4
LC10a's 12 % stands over 1.1 %.

Reading (one run each, magnitudes):

* **Repetition and a larger square localize the small-field types.** Three passes of an 8.8-deg square take Mi1 from
  7 % to 48 % coverage at 2.5 deg from the anatomical column (99 % within 10 deg), and give T2 / T3 / Tm5Y a map for
  the first time (2-9 % of bodies, 3-7 deg from their column); 15 deg takes T3 to 8 % and Tm5Y to 24 %. The
  criterion-free peak-node fraction rises the same way (T3 0.075 -> 0.26 -> 0.37 against chance 0.03). The
  deterministic and the shipped lobe give the same map (8.8 deg: Mi1 855 vs 867 fitted, T3 45 vs 48, the same
  agreement): the shipped feedback is not what limits the localizer.
* **LC10a acquires a retinotopic signal at 15 deg, LC11 at no size.** Scope first (`verify:synthetic`): the sizes
  actually run are 4.5 deg (shipped x3 runs + fb0 x1), 8.8 deg x3 passes (fb0 and shipped, **one run each**) and
  15 deg x3 passes (**fb0 only, one run**) -- the 15-deg case, the one where LC10a first shows retinotopy, was
  **never run on the shipped lobe**, and the 13-body result is a single run of the non-shipped `gain_fb = 0`
  configuration. Every sentence below carries that; the tables label it. At 15 deg 13 LC10a bodies are fitted, half of
  them within 10 deg of their anatomical column, and the peak node sits within 15 deg of the column 4x more often than
  chance (0.105 vs 0.025) -- the first stimulus-driven per-body region for an LC type in this model, at the size
  Schretter et al. give as LC10a's preference (15-30 deg), on one run. **LC11 has no retinotopic response to a dark
  square of 4.5, 8.8 or 15 deg**: 0 bodies fitted in every map, and its peak node sits within 15 deg of its column
  exactly at chance (0.014-0.021 vs 0.021-0.022). Its per-node peaks (3.1-3.2 mV median) are 2.8-2.9 x the blank arm's
  node scatter and land anywhere -- the perturbed oscillation of section 5. This is the localizer's version of the
  size ladder's LC11 finding: whatever reaches LC11 from a small dark square is not retinotopic at the output, while
  its inputs T2 / T3 (8 % / 13 % localized at 15 deg, 4-5 deg from anatomy) are.
* **The `gain_fb = 0` per-cell blank** (`stationarity_fb0_gpu.json`, `null_fb0`): the two blank arms are the same
  trajectory to the last digit (identical per-cell statistics, as `optic_measures.md` 6 found) and each fluctuates:
  LC11 per-cell SD **2.12 mV** over the 2 s after settling (frame-to-frame 0.66, 200-ms window means 2.07 apart),
  LC10a 1.79 (0.88 / 1.24), T3 optic_dr 0.091 (0.022 / 0.098), T2 0.158, Tm5Y 0.181, TmY21 0.228, Mi1 0.039; spectral
  peaks 4.5 Hz (LC11 share 0.42, T3 0.30) and 32 Hz (LC10a 0.24, Tm5Y 0.39). The GPU and CPU (5.1) numbers agree.
  The divergence of the batch-2 pairs (`divergence2.json`) repeats section 5's: 1.5e-8 mV before the first flash,
  decorrelated afterwards (corr -0.01 to +0.08).
* For the sphere assay the usable per-body regions are therefore `out/synth2/rfmap_150_fb0_p3.csv` (LC10a: 13 bodies;
  T2 / T3 / Tm5Y / Mi1) or, on the shipped lobe, `rfmap_088_shipped_p3.csv` (T2 / T3 / Tm5Y / Mi1; LC10a 2); none has
  LC11, and every one of them is one run -- a map with a scatter needs >= 3 runs of the same localizer in one
  submission (`plan` takes `--loc-runs`; the 15-deg three-pass job costs ~8 min per run).

## 9. Files, provenance, defects

Generators (every quoted number): `scripts/probe_synthetic_stimuli.py` `record` (the runs), `verify`
(`out/synth/verify.json`), `rfmap` (`out/synth/rfmap_shipped.csv` + `out/interp/synth/rfmap_shipped.json`;
`rfmap_fb0.*`), `analyse` (`out/interp/synth/families.json` -- written before the 6.1 transition split existed, so it
carries no `transition` rows; re-run `analyse` with `--transition-window-s` to regenerate it), `stationarity` (`out/interp/synth/stationarity.json`,
`stationarity_fb0_cpu.json`), `divergence` (`out/interp/synth/divergence.json`), `plan` (`out/synth/batch.sh`,
`out/synth_flash/batch.sh`, `out/synth2/batch.sh`, `out/synth_ladder/batch.sh` + `batch_jobs.json`). Console logs:
`out/synth_cluster.log`, `out/synthfl_cluster.log`, `out/synth2_cluster.log`; per-run consoles `<run>.txt` beside the
data. Every Result passes `check()` (`problems: none`).

Defects found and fixed in this file's own code: the CPU aliasing of the stored radiance (section 2; GPU runs were
unaffected because `torch.as_tensor` copies to CUDA, but the check would have been false on any CPU run); the
`make_stimulus` flash mapping (section 3); `verify_dir` not seeing a console without a summary (a job line ending in
`tail -3` reports exit 0 whatever python did -- fixed; the same pattern is in every batch generator of the round).

Defects found by the skeptics and fixed in this revision (2026-09-13):

* **The whole-window statistic pooled the two transition polarities of the periodic flash** (`verify:compare`), so
  the `flashon` / `flashoff` pair measured bright-vs-dark and the predeclared "T2 and T3 keep both transitions" rule
  did not measure what its name said. Fixed analysis-side: `luminance_state` / `transition_frames` /
  `transition_stats`, the `transition = on | off` rows, the null's `edge_schedule` floor, `--transition-window-s`,
  two CPU tests (section 2, 6.1). No stimulus generator changed -- the runs on file are still valid inputs.
* Numbers corrected in place: max rectangle coverage **0.8853** not 0.77 (1.1); this class runs **19** tests, not 16
  (2); the 32.1-Hz line is quoted **modulo the 100-Hz frame sampling** (5).
* Statements qualified in place: the cluster runs executed an **uncommitted** `optic.py` / `fly.py` / `senses.py`
  (section 3, with the CPU bit-identity check that makes the numbers safe); the small-family nulls sit in a
  **retinal hole** (6.2); the LC10a / LC11 localizer sentences carry **one run, `gain_fb = 0`** (8); every family
  number is **n = 1** (6.2).

Known defect NOT fixed here, left to the probe's owner: in a `--null` run the stored `radiance` array is the family's
frames although both arms presented the blank, and the retina record's "the stored array IS the presented input"
sentence is written unconditionally (section 1.1). The checksums are correct and `verify` is unaffected; the fix is a
conditional provenance sentence (and, better, storing the blank as `radiance` for null runs) in `cmd_record`, which
this pass did not touch because it was scoped to the analysis side. Defects for other owners: none edited.
Observations for the optic-lobe owners: section 5.
