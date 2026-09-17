# The matched sphere assay (`scripts/probe_object_matched.py`) -- Neurome intake, experiment 1, the ray-traced arm

**Status (2026-09-13): BUILT and smoke-tested on the GPU; no size-tuning number is claimed here.** This file
documents the assay, its geometry checks and the one validation batch. The ladder itself (>= 5 runs per arm, one
submission) is the next task's; nothing below is a result about LC11 or LC10a.

**Skeptic-checked (2026-09-13): `verify:sphere`, verdict `mostly sound`, nothing refuted, seven corrections.** The
verdict is on file verbatim in `receptor_verification.md`, section "Object round 2 (2026-09-13) -- build skeptic
verdicts"; every correction is applied IN PLACE below -- the old-ladder generator (section 0), the unmatched
effective contrast and the bright/dark contrast mismatch (2.2, 3, 5), the wall-time range (2.2), the dimmed-band
interval and the centroid spread's run set (2.2, 5), the shipped tree's uncommitted files (2.2), and the skeptic's
own 5 v 5 replicate batch (2.3, exploratory).

Read first: `D:\Projects\neurome\reports\flyverse-size-tuning-intake.md` (the corrections), `docs/NEUROME_INTERFACE.md`
3b (what was agreed), `docs/audits/deficit_object.md` 5-6 (what the old ladder said and why it cannot separate size
from elevation), `docs/INTERP.md` 2.4 / 2.5 / 10.4 (replicates, provenance, process rules).

## 0. What Neurome named, and what this probe holds constant

The old ladder (`scripts/probe_object_sweep.py`, `docs/audits/object_sweep.md` 8.6, `deficit_object.md` 5) rested
the ball on the table 5 cm ahead of a walking fly's eye (1.2 mm above the table) and slid it on a lateral line at
0.04 m/s. Three things therefore changed with "size" (Neurome's figures in brackets). **Generator, corrected
(`verify:sphere`): `probe_object_matched.py geometry` computes only the MATCHED geometry -- it has no ball-on-table
model and no lateral-line sweep, so it cannot print this table.** The numbers below were recomputed by hand from
`probe_object_sweep.py`'s `BALL_R` / `AHEAD` / `HALF_SWEEP` / `SWEEP_S` and `body.FlyState.eye_height` (the skeptic
reproduced them and they are arithmetically right); the shipped generator for them is the ladder task's
`python scripts/object_round2_baseline.py oldladder`, which prints centre elevation 0.88 / 4.34 / 8.66 / 13.71 deg,
diameter 4.50 / 11.42 / 20.08 / 30.18 deg shrinking to 2.88 / 7.32 / 12.90 / 19.52 along the sweep and speed
45.84 -> 18.79 deg/s from those constants (`object_baseline_r2.md`). Cite that subcommand, not `geometry` -- and
prefer its numbers to the table's where they differ (+0.92 vs 0.88, +8.51 vs 8.66 deg, 4.6 vs 4.50 deg): the table
below was computed from the ROUNDED radii of its second column (2.0 / 5.0 / 8.7 / 13.4 mm), `oldladder` from
`probe_object_sweep`'s own `BALL_R`, which is why it reproduces Neurome's intake figures exactly:

| nominal | radius | centre elevation | angular diameter centre -> end of sweep | angular speed centre -> end |
|---|---|---|---|---|
| 4.5 deg | 2.0 mm | +0.92 deg [0.88] | 4.6 -> 2.9 deg | 45.8 -> 18.8 deg/s |
| 11.4 deg | 5.0 mm | +4.35 [4.34] | 11.4 -> 7.3 | same |
| 20 deg | 8.7 mm | +8.51 [8.66] | 19.7 -> 12.7 | same |
| 30 deg | 13.4 mm | +13.71 [13.71] | 30.0 -> 19.5 | same |

plus: the retina table Neurome received was a second-pass **geometry replay** at the pinned pose, and the blank
radiance was not shipped.

`probe_object_matched.py run` holds all of it constant, per Neurome's list:

* **(a) fly-relative centre, fixed elevation, fixed distance, radius-independent.** The centre is
  `eye + d * (cos el cos az, cos el sin az, sin el)` in the body frame (`centre_world`), `d` = `--distance-m` (0.05),
  `el` = `--elevation-deg` (0). The radius is *derived* from the requested angular diameter,
  `r = d sin(diam / 2)` (`radius_for`; 4.5 / 11 / 20 / 30 deg -> 1.963 / 4.792 / 8.682 / 12.941 mm at 5 cm), so the
  centre never moves with the size.
* **(b) an arc at constant angular speed.** Azimuth is a triangle wave between `+-az_max` (`--az-max` 50) at
  `--deg-per-s` (40): one way 2.5 s, left -> right first as the old sweep did (`azimuth_track`). Distance, elevation
  and angular diameter are constant along the arc by construction; the angular speed is 40 deg/s on every frame but
  the four reversal frames of a 10 s window (`make_track` flags them in `turn`).
* **(c) the ladder by `--diam-deg`, one value per run; dark by default, bright by option.** `--material black`
  (reflectance 0.01-0.02: the old ball) / `plate` (0.6-0.95, shaded like every other surface) / `lamp` (emissive
  0.6-1.0, unshaded). The tracer packs the material per object and `World.invalidate()` repacks it -- verified on the
  CPU by the unit test (`black` < 5 % of the wall's luminance, `lamp` > 1.5x, `plate` between) and on the GPU by the
  smoke's `d110_lamp` run (section 3).
* **(d) blank arm = the same timeline with the ball parked at (9, 9, 9)**, outside the room; `--null` makes arm A a
  second blank under the same seed (the blank/blank null of every `diff_*` statistic, as `probe_object_sweep --null`).
* **(e) in-loop capture.** After every `sim.step()` of BOTH arms the probe stores `sim.col_rad` -- the (1466, 4)
  radiance `fb.vision()` received on that very frame (`scripts/room_demo.py` `Sim.step`: `self.col_rad =
  self.column_radiance(); self.fb.vision(self.col_rad)`) -- as `rad_a` / `rad_b` in the run's npz, with the per-frame
  track (commanded `track_*` and, from the pinned pose the trace used, `seen_az_deg / seen_el_deg / seen_diam_deg /
  seen_dist_m`). The JSON says so: `provenance.retina.mode = "in_loop_capture"`, with the sampling sentence. This is
  not `interp_export.capture_retina`'s second pass.
* **(f) per-body per-frame recording** through `flyverse.interp.common.Recorder` at 10 ms frames over the scored
  window: LC11 / LC10a / LC10b / LPLC2 / LC4 (`drive_mv`, `spike_count`) and T2 / T3 / Tm5Y / TmY21 / TmY13 / TmY5a /
  Mi4 / Tm3 / Mi1 (`optic_dr`). The full (T, n_cells) arrays of both arms are in the npz (`a__lc_drive_mv`,
  `a__lc_spikes`, `a__optic_dr`, `b__...`), with `lc_body_ids` / `rate_body_ids`; `--save-recordings` also writes
  the Recording npz+json per arm.
* **(g) the statistics**, on the per-frame arrays (`summarize_arms`), with `probe_object_sweep.summarize`'s names and
  definitions (unit-tested against them): `diff_max_over_cells_mean_mv` (max over cells of the per-cell time-mean
  A - B drive, `probe_object_sweep.py:277`), `diff_mean_over_cells_mean_mv`, `diff_tuning_peak_mv` (now binned on
  the track *azimuth*, 20 bins), `diff_rate_hz_max_cell` / `_mean`, and for the rate units `diff_abs_best_cell_mean`
  (:291) and `diff_signed_best_cell` (:292) kept **distinct**, plus `diff_signed_mean`. PLUS the per-body table
  (`tables.per_body`: bodyId, model_index, type, quantity in drive_mv / rate_hz / optic_dr / optic_dr_abs, `a_mean`,
  `b_mean`, `diff`, the per-frame SDs, `n_frames`) -- the per-body time-mean before any population maximum -- and the
  sweep-locked per-body drive along the arc (`a__lc_drive_by_az_bin`, `b__...`, 20 azimuth bins). With `--rf-map`
  the table gains `rf_az_deg / rf_el_deg / rf_width_deg / rf_n_frames / rf_a_mean / rf_b_mean / rf_diff`: the mean
  over the frames on which the disc overlaps the body's RF box (`|d az| <= (width + diam) / 2`,
  `|d el| <= (height + diam) / 2`), the same frames in both arms, NaN (never 0) for a body without a localizer; and
  `summary.rf_windowed` the per-type max / mean over the windowed bodies.
* **(h) one `common.Result` JSON per run** (`tool trace`, `tool_version probe_object_matched/1`, run_id
  `objm-d<diam*10>-obj|null-r<seed>`) carrying the resolved LIFParams / OpticParams, the realised device per arm
  (`execution.device_by_arm`), the cache fingerprint, the git commit / source fingerprint, the stimulus block with
  the geometry record and the in-loop track check, the retina block, the populations, the tables `per_type` /
  `per_body` / `footprint_per_frame`; the npz beside it; a printed table.

The pinned pose, the empty fenced table, the wall behind (y = -2), wind 0, `fruit_set apple` with the apple parked:
all as `probe_object_sweep` (`POS (-0.20, 0.10, 0.75)`, heading -pi/2). One process = one run = one (A, B) pair under
one brain seed; runs are the replicate unit (`docs/INTERP.md` 2.4, 10.4 rule 1-2).

## 1. Two scene facts the matched geometry forces into the open (flags, defaults, both recorded)

**The table clips a ball at elevation 0 for a walking fly.** The eye is 1.2 mm above the table
(`body.FlyState.eye_height 0.0012`). The lower limb of a ball at elevation 0 is at `-diam / 2`; the table's horizon
at the ball's tangent distance is `-asin(eye_height / sqrt(d^2 - r^2))` = -1.38 to -1.42 deg across the ladder
(`substrate_clearance`; `geometry` subcommand):

```
python scripts/probe_object_matched.py geometry --diam-deg 4.5 11 20 30 --eye-height-m 0.0012
 diam_deg  radius_mm  lower_limb_deg  substrate_horizon_deg  clear
    4.500      1.963          -2.250                 -1.376  False
   11.000      4.792          -5.500                 -1.382  False
   20.000      8.682         -10.000                 -1.396  False
   30.000     12.941         -15.000                 -1.424  False
```

so every diameter would be clipped, by a size-dependent fraction -- a new confound in place of the old one. Two ways
out, both flags: raise the **eye** (`--eye-height-m`, default **0.15**: the body stays on the table, the eye is 15 cm
above it -- the tethered-fly analogue; the substrate horizon is then below -90 deg for every ball, i.e. the whole
ball is in the clear and the background behind the arc is the wall at y = -2), or raise the **elevation** for all
sizes (`--elevation-deg 20` at the walking eye height clears every diameter: lower limbs +17.75 / +14.5 / +10 / +5
deg). The probe **refuses** a clipped configuration unless `--allow-substrate-clip` is given; the check is in the JSON
(`stimulus.geometry_check.substrate`) either way. The eye height is an *assay* parameter (body geometry at a pinned
pose), not a model parameter: nothing under `flyverse/` changes.

**The ceiling lamp makes the ball cast a moving, size-dependent shadow.** `--light eye` (default) puts the scene's
point light at the eye: a headlamp, every shadow exactly behind its caster, the ball's near face lit; `--light
ceiling` keeps the room's lamp at (0, 0, 2.3), shadow included (then part of the captured stimulus and visible in
the footprint diagnostic as extra dimmed columns on the table). Both positions are recorded
(`stimulus.params.light_pos_m`, `room_lamp_m`). The wall behind the arc carries the room's vertical stripe pattern
(`world.MATERIALS['wall']`, period 0.5 m), so the background is not uniform; the footprint reports the blank
luminance under the object and its CV so that a background difference between diameters is visible, not assumed.

## 2. Validation

### 2.1 CPU geometry unit test (no connectome): `tests/test_object_round2.py::MatchedSphereGeometryTests`

`PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m unittest tests.test_object_round2 -v` -- 8 tests, all pass
(1.5 s):

* `test_arc_holds_elevation_diameter_distance_and_speed_for_three_diameters`: for (4.5 deg, el 0), (11, 0),
  (30, 12.5) the centres placed by `centre_world` and read back through `seen_from_eye` (a different code path: the
  world-frame vector projected on the fly's axes) give elevation drift < 1e-9 deg, angular-diameter drift < 1e-9,
  distance drift < 1e-12, azimuth = commanded to 1e-9, and |d az / dt| = 40 deg/s to 1e-6 on every non-turn frame
  of a 10 s track; 4 turn frames at 2.5 / 5 / 7.5 / 10 s; starts at +50 (left) moving right. The radius is
  distance-derived (20 deg at 0.08 m is another radius, the same angle).
* `test_substrate_clearance_names_the_table_clip_at_the_walking_eye_height`: the table above (clipped at 0.0012 m,
  horizon -1.4 deg; clear at 0.15 m; clear at elevation 20 for 30 deg).
* `test_tracer_supports_dark_bright_and_emissive_ball_materials`: a CPU `World` with a wall and one sphere:
  `black` < 5 % of the wall, `lamp` > 1.5x the wall, `plate` between; the light moved to the eye brightens the black
  ball's near face (the material / light switch goes through `invalidate()` and repacks).
* `test_footprint_recovers_a_dimmed_disc_and_reads_zero_on_blank_vs_blank`: a synthetic 4.6-deg column grid, a
  moving dark disc at el +5 -> centroid el 5 +- 2.5, moving azimuth, blank-vs-blank exactly 0 changed columns; a
  bright disc -> 0 dimmed, > 0 brightened, the same centroid.
* `test_summary_statistics_match_probe_object_sweep_definitions`: every statistic against its definition in
  `probe_object_sweep.py`; `diff_signed_best_cell != diff_abs_best_cell_mean` on the same data; per-body rows.
* `test_rf_map_loader_and_windowing`: the CSV interface, height defaulting to width, missing columns refused, the
  synthetic task's `fitted False` / NaN rows dropped, the window rule, NaN for a body without a localizer.
* `test_batch_generator_writes_the_cluster_rule_into_every_job_line`: `mkdir -p` + venv + CUDA assert + console
  redirect in every line; smoke = 7 jobs; ladder = diameters x seeds + seeds nulls.
* `test_cli_parses_the_documented_flags_without_touching_a_gpu`.

### 2.2 The GPU smoke batch (`out/objm/smk2/`, console `out/objm_smk2_cluster.log`)

Submitted through the shipped generator, `python scripts/probe_object_matched.py batch --kind smoke --dir
out/objm/smk2 --name objm-smk2 --minutes 30 --submit` (= `scripts/cluster_run.py` with 7 job lines, one
submission, fetched into the NAMED directory): the four diameters as object/blank pairs, one blank/blank null at
11 deg, and the two option variants at 11 deg (`--material lamp`, `--light ceiling`); 3 s window after 3 s
settle, seed 0, everything else at the defaults (elevation 0, 5 cm, +-50 deg at 40 deg/s, eye 0.15 m, dark ball,
headlamp). `out/objm_smk2_cluster.log`: `7 job(s), 0 failed (3.5 min)`, run dir `objm-smk2-1340f3` on vast-a /
vast-b / vast-c (NVIDIA B200 / H200), every console `device cuda`. Then
`python scripts/probe_object_matched.py verify out/objm/smk2 --json out/objm/smk2/verify.json` (CPU): **`problems:
none`** -- every run `retina_mode in_loop_capture`, realised device cuda in the JSON *and* the console, per-body
table and populations at **LC11 143 / LC10a 275** in all 7 runs (and every other recorded type at its population
count), JSON footprint == footprint recomputed from the npz radiance.

**Geometry as realised in the loop** (`seen_*` from the pinned pose the trace used, all 7 runs): elevation max
deviation 0.0 deg, angular diameter max deviation <= 1.4e-14 deg, distance <= 1.4e-17 m, azimuth vs commanded
<= 2.1e-14 deg, speed 40.00-40.00 deg/s off the single turn frame of a 3 s window. **Wall time is load-dependent,
not a property of the box (`verify:sphere`): roughly 6-19 s per arm for 600 frames** -- this smoke's own runs are
9.23-9.28 s/arm (H200) and 10.41-10.48 s/arm (B200), while the skeptic's 10-job batch of the same 3 s configuration
on the same boxes took 5.7-9.7 s/arm on H200 and 15.6-18.5 s/arm on B200 under concurrency. The 3 s smoke jobs took
66-86 s wall each end to end including the sim build; the "~3-4 min per 12 s ladder job" of section 3 is an
extrapolation from that and was never measured.

**The one thing the assay does NOT match: effective retinal contrast (`verify:sphere`'s main gap).** The ball's
material is identical at every rung, but the realised Weber contrast in the best column grows monotonically with
angular size, because a 4.5-deg ball only partially fills a column at 4.6-deg spacing. Measured from these same
captured radiance files (`out/objm/smk2/*.npz`): the median per-frame **extreme relative luminance change** is
**-0.475 / -0.889 / -0.935 / -0.954** at 4.5 / 11 / 20 / 30 deg, and the mean over the dimmed set
**-0.59 / -0.76 / -0.82 / -0.86**. This is the confound Neurome named in the intake (line 119, "increase in retinal
contrast footprint may contribute to the drive ordering"; line 193 lists contrast among the things to control), and
no claim of this file asserts matched contrast -- but "the matched assay" overstates coverage on that item, so:
**the small end of this ladder confounds size with effective contrast**, the per-rung contrast readout belongs in the
predeclared reading rules (section 3), and the contrast-matched families are the synthetic rectangle ladders
(`probe_synthetic_stimuli.py`, coverage-defined contrast). **The bright control is not contrast-matched to the dark
one either**: at the same 11 deg, `lamp` gives a median extreme relative change of **+0.511** and strongly changes
1.217 columns/frame, against `black`'s **-0.889** and 3.380 columns/frame (|rel| > 0.25: 2.353 vs 4.390). The tracer
supports the bright option (claim 6, verified), but a bright/dark specificity comparison at the shipped settings
compares two different effective contrasts and must say so.

**The footprint from the CAPTURED radiance** (`verify.json`; per frame, mean over the 300 window frames; Neurome's
diagnostic, its old-ladder values in brackets from the intake report):

| run | columns dimmed > 50 % | changed > 5 % | brightened > 50 % | centroid el mean +- SD (deg) | dimmed el band (deg) | centroid az range | blank luminance under the object (CV) |
|---|---|---|---|---|---|---|---|
| 4.5 deg dark | **0.45** [0.14] | 2.18 [1.51] | 0 | +0.04 +- 1.14 | -1.5 .. +1.6 | -49.4 .. +51.4 | 2.505 (0.21) |
| 11 deg dark | **3.38** [2.26 at 11.4] | 6.16 [4.78] | 0 | +0.15 +- 1.35 | -5.3 .. +5.4 | -49.4 .. +50.6 | 2.442 (0.15) |
| 20 deg dark | **11.39** [7.64] | 16.53 [12.41] | 0 | -0.05 +- 2.26 | -9.9 .. +8.5 | -49.8 .. +50.7 | 2.439 (0.14) |
| 30 deg dark | **25.95** [19.73] | 34.63 [26.78] | 0 | -0.64 +- 2.12 | -15.3 .. +14.6 | -49.8 .. +49.6 | 2.451 (0.09) |
| 11 deg dark, ceiling light | 3.26 | 5.87 | 0 | +0.20 +- 1.33 | -5.3 .. +5.4 | -49.4 .. +50.6 | 0.388 (0.23) |
| 11 deg `lamp` (bright) | 0.00 | 5.04 | 1.22 | +0.00 +- 1.05 | -6.1 .. +5.4 | -52.0 .. +50.4 | 1.958 (0.19) |
| 11 deg null (blank/blank) | 0 | **0** (exactly) | 0 | -- | -- | -- | -- |

**Centroid elevation across the four diameters: +0.04 / +0.15 / -0.05 / -0.64 deg, spread 0.83 deg -> the same band
(< one interommatidial angle, 4.6 deg)**, against 0.88 / 4.34 / 8.66 / 13.71 deg in the old ladder. Two numbers here
are corrected by `verify:sphere`: (i) the **0.83 deg spread is over all six non-null runs** (it includes the `lamp`
and `ceiling` variants) -- the four dark ladder rungs alone spread **0.787 deg**; (ii) the dimmed band is **not
symmetric about 0 at the large rungs**: measured it is [-1.49, +1.65], [-5.25, +5.41], [-9.85, +8.55],
[-15.29, +14.61] deg, i.e. band centres 0.00 / +0.08 / -0.65 / -0.34 deg, so the shorthand "+-9 / +-15" hides
[-9.85, +8.55] and [-15.29, +14.61]. The asymmetry is far inside one interommatidial angle (4.6 deg) and does not
change the matched-elevation reading; quote the interval, not `+-`. The count of columns dimmed > 50 %
grows 0.45 -> 3.38 -> 11.39 -> 25.95 per frame (the old 0.14 -> 19.73):
more than the old ladder at every rung because the whole disc is now above the substrate and the diameter no longer
shrinks along the sweep. The 4.5-deg rung dims a column by more than half on fewer than half the frames (0.45 /
frame; the ball is one column wide at 4.6-deg spacing and 4.5-deg acceptance): at that size the > 5 % count (2.18)
is the usable footprint. The blank arm is static to 0.0 frame-to-frame in every run, the null's A and B radiance are
identical to the float (0 columns changed on 300 frames), and the blank luminance under the object is the same wall
in all four dark runs (2.44-2.51; CV 0.09-0.21 = the wall's stripes). The two variants behave as documented: the
ceiling lamp lights the same wall 6x dimmer (0.388) and adds no visible shadow columns at this elevation (3.26 vs
3.38 dimmed), and the `lamp` ball brightens instead of dimming (1.22 columns > +50 %, 0 dimmed, the same centroid).

**The LC / upstream headline statistics of the smoke -- one draw per row, quoted as such, no comparison** (verify's
second table; `diff_max_over_cells_mean_mv` is a within-run maximum over cells of a 3 s time-mean, mV; rate = max
over cells of the A - B rate, Hz, on 3 s windows where one spike is 0.33 Hz):

| run | LC11 diff max (mV) | LC10a | LPLC2 | LC11 rate max (Hz) | LC10a | T3 signed best / abs best | Mi1 signed / abs |
|---|---|---|---|---|---|---|---|
| null 11 deg (blank/blank) | +0.172 | +0.229 | +0.507 | 0.00 | +0.33 | 0.013 / 0.043 | 0.006 / 0.017 |
| 4.5 deg | +0.120 | +0.180 | +0.369 | +0.67 | +0.67 | 0.012 / 0.052 | 0.008 / 0.040 |
| 11 deg | +0.187 | +0.199 | +0.611 | +0.67 | +0.33 | 0.013 / 0.021 | 0.011 / 0.062 |
| 20 deg | +0.200 | +0.317 | +1.552 | 0.00 | +0.67 | 0.016 / 0.038 | 0.028 / 0.091 |
| 30 deg | +0.296 | +0.392 | +2.886 | +0.33 | +1.33 | 0.024 / 0.079 | 0.046 / 0.099 |
| 11 deg ceiling | +0.207 | +0.217 | +0.554 | 0.00 | +0.33 | 0.013 / 0.033 | 0.013 / 0.063 |
| 11 deg lamp | +0.070 | +0.133 | +0.450 | +0.33 | +0.67 | 0.013 / 0.032 | 0.014 / 0.037 |

With one draw per arm and 3 s windows these are scatter-sized (the LC11 / LC10a null draw, +0.17 / +0.23, sits
inside the object rows), as the old ladder's 12-s nulls of +0.05-0.08 predicted for a window four times shorter;
the pipeline reproduces the known size-scaling of LPLC2's draw (+0.37 -> +2.89) and Mi1's (`diff_abs_best_cell`
0.040 -> 0.099) as a sanity check of the recording path, nothing more. `diff_signed_best_cell` and
`diff_abs_best_cell_mean` are two columns here and stay two columns. The LC cells fire in both arms (per-run
`cells_firing` 0-12 of 143 / 275 in 3 s): "no detected effect", never "silent".

**What the run JSON records** (`out/objm/smk2/d110_obj_s0.json`, checked): `provenance.compiled_connectome.md5
ef23cc27bea13be7f6a96f3c04fd3737` (the shipped cache; `sum_abs_W` 121,460,584; 167,106 neurons), dataset `v1.0
flat-connectome`, `flyverse_commit.commit unknown` on the rented box with `source_fingerprint.computed true`
(docs/INTERP.md 10.4 rule 6; the local checkout was `ae7c86f`, dirty), `execution.device cuda` / `device_name`
per box -- **and, named as `verify:sphere` asks: the tree this smoke shipped to the boxes was not HEAD.** It carried
two OTHER tasks' uncommitted changes to `flyverse/optic.py` (the hooks refactor, +215/-13 lines, touching the shipped
optic path) and to `flyverse/fly.py` (the proprioception task); the cluster log confirms both were shipped.
`provenance.flyverse_commit` on the git-less box reads `{commit: 'unknown', dirty: false, modified_files: []}`, which
UNDERSTATES that (the `false` is the absence of a `.git` in the run dir, not evidence of a clean tree) -- only the
per-file `source_fingerprint` sha256s make the tree recoverable. These numbers therefore rest on the hooks task's own
bit-identity claim, which this file did not verify and which `optic_stream_hooks.md` 3 now qualifies as CPU /
deterministic-backend only (`git diff -- flyverse/brain.py flyverse/body.py` was empty and the four hook fields are
`None` in every run JSON, so nothing here selects a hook). It is `common.provenance`'s behaviour on a git-less box,
not a defect of this probe, but the audit has to say which tree the smoke ran on -- this one. Also recorded:
`device_by_arm {a: cuda, b: cuda}`, resolved `model.lif` (`receptor_model sign`, `net_rule abs`) and
`model.optic` (`gain_fb 0.5`, `gain_out_mv 100`, `out_norm l1`, `drive_clip_mv 35`, ... and the hooks task's four
new opt-in fields `stream_rectify / stream_adapt / spatial_suppress / fb_hold`, all `None` = OFF in this batch),
`stimulus` (protocol `object_matched_sphere`, every parameter, the geometry record with the substrate check, the
in-loop `track_check`), `retina.mode in_loop_capture` with the npz path, 14 populations (LC11 143, LC10a 275,
LC10b 95, LPLC2 185, LC4 126, T2 1630, T3 1940, Tm5Y 898, TmY21 372, TmY13 432, TmY5a 1364, Mi4 1772, Tm3 2054,
Mi1 1773), `tables.per_body` 26,118 rows (every body x its quantities), `footprint_per_frame` 300 rows. The npz:
`rad_a` / `rad_b` (300, 1466, 4) float32, `a__lc_drive_mv` / `a__lc_spikes` (300, 824), `a__optic_dr`
(300, 12235), the `b__` twins, the track, the centres, the footprint per frame, the 20-bin azimuth tuning
(28.7 MB per run; the JSON 9.3 MB, most of it the per-body table).

### 2.3 The skeptic's 5 v 5 replicate batch (`out/objm/skept`, one diameter, 3 s windows) -- EXPLORATORY, not this build's

`verify:sphere` ran this probe's own CLI for 5 object + 5 blank/blank runs at one diameter in ONE submission and read
them through `common.compare`. Recorded here because it is the first evidence that the matched recording path detects
the object at all, and labelled as the skeptic labelled it: **exploratory, one diameter, no predeclared Holm family**.
Mi4 `diff_abs_best_cell_mean` +0.0362 +- 0.0033 vs null +0.0173 +- 0.0050 (z +3.78, U 25, p 0.008 -> `result`); Tm3
+0.0945 +- 0.0041 vs +0.0416 +- 0.0153 (z +3.46, p 0.008 -> `result`); LPLC2 `diff_max_over_cells_mean_mv` +0.6286 +-
0.0524 vs +0.3241 +- 0.1090 (z +2.79, U 25, p 0.008 = `p_floor` at 5 v 5 -> `null` only because z < 3, with all five
object runs above all five nulls); Mi1 z +2.05, Tm5Y +1.76, TmY21 +1.35, T3 abs +0.96, T2 +0.65 -> all `null`. So the
matched assay reproduces round 1's "carried upstream, attenuated at the LC10 / LC11 inputs" pattern WITHOUT the
elevation / speed confound. Power note for the ladder: `p_floor` is 0.008 at 5 v 5 (a 3-rung x 5-run family can be
Holm-corrected) and 0.029 at 4 v 4 (which cannot survive Holm over more than one test).

## 3. Reading rules for the ladder that follows (predeclared here, before any run)

* **Primary statistics, distinct per type.** LC11: `diff_max_over_cells_mean_mv` and `diff_rate_hz_max_cell` at the
  small end of the ladder (its biological target is a small dark object: preferred vertical extent 8.8 deg, width
  ~4.4 deg, Keles & Frye 2017 -- on a sphere ladder the 4.5-11 deg rungs); LC10a: the same statistics at 15-30 deg
  (Schretter et al. 2024, Fig 3a). The two must not share a target. Upstream: `diff_signed_best_cell` and
  `diff_abs_best_cell_mean` reported side by side, never collapsed.
* **Null = the blank/blank runs of the same submission** (`--null`), `common.compare` (z on the null SD, Welch, exact
  tie-aware U, `p_floor`), verdicts `result / null / underpowered / undetermined` only; Holm within the predeclared
  family; anything outside it exploratory. The 3-second smoke below is not that comparison.
* **Per-body first.** `tables.per_body` (and the RF-windowed variant once the synthetic task's localizer map exists)
  before `summary.diff_*`; the `a__lc_drive_by_az_bin` arrays are the sweep-locked per-body time course.
* **The footprint is the first thing to read in every run**: `n_dimmed_50pct` / `n_changed_5pct` per frame and the
  centroid elevation band. Equal centroid elevation across diameters is what "matched" means; if the ceiling light is
  used, the shadow's columns add to the count and the band widens downward (documented, not hidden).
* **A per-rung effective-contrast readout is part of the reading** (added after `verify:sphere`). The extreme
  relative luminance change per frame and its mean over the dimmed set (both computable from the npz radiance, and
  printed per rung by `object_round2_baseline.py analyse`) are reported beside every rung's statistic, because they
  are NOT matched across the ladder: -0.475 / -0.889 / -0.935 / -0.954 at 4.5 / 11 / 20 / 30 deg (section 2.2). A
  monotone size effect at the small end has a monotone contrast covariate; the contrast-matched arm of the round is
  the synthetic square/rectangle ladder, and a bright-vs-dark specificity call carries the +0.511 vs -0.889 mismatch
  with it.
* **The submission shape.** `batch --kind ladder --seeds 0 1 2 3 4 --diam-deg 4.5 11 20 --seconds 12` renders 3 x 5
  object jobs + 5 blank/blank nulls = 20 jobs (the cap of one submission on the rented boxes; ~1 min per job at
  3 s, expect ~3-4 min each at 12 s). Four diameters at 5 runs each is 25 jobs and would need two submissions, which
  breaks "all arms in one submission": either three rungs per submission (4.5 / 11 / 20 first -- LC11's rungs plus
  LC10a's lower target -- 30 deg in a second submission with its own nulls) or 4 runs per arm (the floor that can
  be called, `p_floor` 0.029 at 4 v 4). Decide before submitting, not after.
* **What the square-ladder cannot say** (Neurome's own caveat): a sphere changes width and height together; the
  rectangle experiments varied one at a time. Height / width ladders are the synthetic-stimuli probe's job
  (`scripts/probe_synthetic_stimuli.py rect`, radiance path, no ray tracing); this probe is the ray-traced,
  in-loop-captured baseline they are read against.

## 4. Files and interfaces

* `scripts/probe_object_matched.py` -- `run` (GPU; `--diam-deg --distance-m --elevation-deg --az-max --deg-per-s
  --eye-height-m --material --light --allow-substrate-clip --seconds --settle --null --rf-map --save-recordings
  --out` + `common.add_common_args`: `--optic KEY=VALUE`, `--lif`, `--receptor-model`, `--cache-dir`, `--device`);
  `verify` (CPU: footprint recomputed from the captured radiance per run, the elevation band across the object runs,
  body coverage of every recorded type against the JSON's own populations and the expected 143 / 275, the
  console-vs-meta device check; `--json`); `geometry` (CPU table); `batch` (`--kind smoke|ladder --dir --name
  --seeds --diam-deg --seconds --settle --minutes [--submit]`: prints the job lines, `--submit` runs
  `scripts/cluster_run.py` and tees `out/<name>_cluster.log`; refuses > 20 jobs).
* Per run: `<out>.json` (Result), `<out>.npz` (arrays listed in section 0 e-g), `<out>.txt` (console, written by the
  job line).
* `tests/test_object_round2.py::MatchedSphereGeometryTests` (CPU).
* The RF map (`--rf-map`): CSV `bodyId, az_deg, el_deg, width_deg[, height_deg]`; the synthetic task's
  `probe_synthetic_stimuli.py rfmap --csv` output (`bodyId, type, az_deg, el_deg, width_deg, peak,
  n_nodes_above_threshold, fitted, ...`) is read through the same columns with unfitted rows dropped. Written
  against that CSV; its `docs/INTERP.md` entry is the synthetic task's (not present at the time of writing), so the
  Result-JSON form (`tables.rf_map`) is a guess kept behind the same loader.
* Not touched: `scripts/probe_object_sweep.py` (imported for nothing -- its statistics are re-implemented on per-frame
  arrays and unit-tested against its definitions), `flyverse/*`. The optic-stream hooks of the concurrent hooks task
  reach this probe through `--optic KEY=VALUE` (`interp_trace.install_overrides` -> every `OpticParams` built by
  `room_demo.Sim`); no hook field is named here.

## 5. Defects and limits

* **Effective contrast is not matched across the ladder** (`verify:sphere`, section 2.2): median extreme relative
  change -0.475 / -0.889 / -0.935 / -0.954 at 4.5 / 11 / 20 / 30 deg, mean over the dimmed set -0.59 / -0.76 /
  -0.82 / -0.86; and the bright control is not contrast-matched to the dark one (+0.511 vs -0.889 at 11 deg). The
  geometry (elevation, distance, diameter, speed) IS matched; the realised contrast is not, and the small end of the
  ladder therefore confounds size with contrast. Section 3 predeclares the per-rung readout; the synthetic ladders
  are the contrast-matched families. The last bullet of this section names only the related 4.5-deg undersampling;
  this is the contrast statement it was missing.
* The dimmed band is quoted as an interval, not `+-`: [-1.49, +1.65], [-5.25, +5.41], [-9.85, +8.55],
  [-15.29, +14.61] deg (centres 0.00 / +0.08 / -0.65 / -0.34), and the 0.83-deg centroid spread printed by `verify`
  covers all six non-null runs (the four dark rungs alone: 0.787 deg).
* Wall time is a range, 6-19 s/arm for 600 frames depending on concurrency on the rented boxes; the 12-s ladder job
  estimate in section 3 is an untested extrapolation from 66-86 s wall per 3-s job.
* The old-ladder table of section 0 has no generator in THIS file; it is `object_round2_baseline.py oldladder`'s.
* `cmd_batch` streamed nothing to its log during the smoke because the child's stdout was block-buffered behind the
  pipe; fixed in this revision (`PYTHONUNBUFFERED=1`), after that batch had been submitted -- the log is complete,
  it just landed at the end.
* The first submission of this smoke (a prior pass of this task, `out/objm/smoke/batch_console.txt`) went to the
  house cluster and was cancelled after 3 min (`5 job(s), 5 failed`, fetch failed) -- the house cluster is reserved
  and disabled; nothing from it is quoted.
* One seed, one run per rung, 3 s windows: the smoke establishes that the pipeline produces the files, the footprint
  and the coverage it claims; it measures no effect. The LC statistics it prints are one draw each against a
  one-draw null and are quoted only as such in 2.2.
* `--eye-height-m 0.15` changes the scene relative to the old ladder (the background behind the arc, the distance to
  the wall, the shadow geometry). The comparison to the old ladder is therefore a comparison of assays, not of
  rungs; the old ladder stays the scene baseline it was declared to be.
* The footprint's centroid is unweighted over columns with |relative change| > 50 %; a 4.5-deg ball at 4.6-deg
  column spacing dims 0-2 columns per frame, so its centroid is quantised to the column grid (SD of order the
  spacing) -- read the band, not the SD, at the small end.
