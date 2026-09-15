# Demo media: what each file is, how it was made, what actually happened in it

Every file here is a recording of the shipped model at commit **`442c420`** (the tree the clips were rendered from
was a clean, detached `git worktree` of that commit; the per-run cluster copy has no `.git`, so the clip logs say
`commit unknown` -- the checkout is the one `scripts/cluster_run.py` shipped from the worktree). Nothing was staged,
tuned, retried or cherry-picked: each clip is the first and only run of its command line, and the captions report
what the run's state log says the fly did. Generator: `scripts/make_demo_media.py` (its `record` loop is
`room_demo.py`'s headless loop with a per-frame state log added; `CLIPS` there names the equivalent
`room_demo.py` line for each clip).

| file | size | what |
|---|---|---|
| `loom.gif` | 11.4 MB | clip 1, 25 s, 800x482, 10 fps, 128 colours (the README's animation) |
| `loom.mp4` | 3.6 MB | clip 1, 25 s, 1360x820 (the full console canvas), 12.5 fps, h264 crf 23 |
| `wind_apple.gif` | 10.3 MB | clip 2, 25 s, 800x482, 10 fps, 128 colours |
| `wind_apple.mp4` | 2.8 MB | clip 2, 25 s, 1360x820, 12.5 fps, h264 crf 23 |
| `toolkit_loom_gf.png` | 0.2 MB | the toolkit figure: `interp` `paths` stage map + `atlas` readout of the loom -> giant-fibre path |

## What is model output and what is UI

The console (`flyverse/room_ui.py`) draws, every 40 ms of brain time: the **body camera** (the ray-traced room from
the fly's eye -- this image is also the model's visual input: the two **retinal mosaics** are the 1,466 columns'
photoreceptor samples of it, UV/B/G false colour and R1-R6 contrast), a small **orbit view** of the table with the
fly's trail (UI only), the **LIF spikes per step** trace, the **antennal odour samples** per glomerulus (model input,
left / right), and the readout columns: **populations** (mean Hz per superclass), **command -> body** (the body
model's reading of the named motor groups: `v_cmd`, `yaw_cmd`, proboscis, pose), **walk / flight** (the named
descending / motor groups the body reads, in Hz). Everything in those columns is a model rate or the body model's
kinematics; the fly's motion in both views is the body model driven by those rates. The pop-up "Loom stimulus
presented" is a UI toast (the record loop calls `ui.notify` exactly as the `L` key does); the wind rose text
"wind from N deg" is the Air model's current direction.

## Clip 1 -- `loom.gif` / `loom.mp4`: walking, a loom at 8 s, the escape jump, the re-landing

Command (job 0 of the batch below; equivalent to
`python scripts/room_demo.py --headless --seconds 25 --loom-at 8 --seed 0 --gif out/media/loom_raw.gif`):

```
python scripts/make_demo_media.py record --clip loom --job fam_media
```

Device: `cuda`, NVIDIA B200, house cluster node1 (torch 2.11.0+cu128); 25 s of brain time in 74 s wall; 312 frames.

What happens (from `out/media/loom_log.json`, the run's own state log): the fly starts at its default spot on the
table top (x -0.50, y +0.05, heading +5 deg) and walks at ~0.9 cm/s, its heading drifting slowly from +5 to +22 deg
over the clip. At **8.00 s** a black ball (3 cm radius) approaches from the fly's left at 1 m/s from 0.5 m to 3.5 cm.
The giant fibre (`gf`, DNp01) rises from a walking baseline of 12-15 Hz to **33.2 Hz at 8.53 s**, the body's
`gf_threshold` (33 Hz), and the body model reads a **TAKEOFF** (TTMn 17.6 Hz, power 25.6 Hz); the giant fibre peaks
at 38.3 Hz at 8.72 s; the fly **LANDS at 8.86 s after 0.33 s in the air**, 12 cm further along +x, back on the table
top, and walks on. Nothing else happens: the nearest fruit (a lime) is 9-23 cm away throughout, never within the
1.5 cm taste range, so `taste/feed` stays 0 / 0 and MN9 stays at 0 Hz. The one thing the newcomer should watch is
the body camera at 8.0-8.7 s (the ball fills the left of the view), the `gf` bar crossing `gf_threshold`, and
`surface` flipping to `airborne` and back.

## Clip 2 -- `wind_apple.gif` / `wind_apple.mp4`: one apple, the default wind

Command (job 1; equivalent to
`python scripts/room_demo.py --headless --seconds 25 --fruit apple --seed 0 --gif out/media/wind_apple_raw.gif`):

```
python scripts/make_demo_media.py record --clip wind_apple --job fam_media
```

Device: `cuda`, NVIDIA B200, node1; 25 s in 72 s wall; 312 frames. Wind: the demo's default 0.3 m/s blowing from the
door at +x towards -x (`--wind-dir 180`), with the Air model's slow +-20 deg meander of direction; the apple is the
only fruit, so its plume is the only odour source.

What happens: **no wind orientation can be claimed from this clip, and no feeding.** The fly starts at heading
+5 deg, i.e. already within 5 deg of upwind, and walks at 0.85-0.9 cm/s from x -0.50 to -0.29 with its heading
drifting from +5 to +17.5 deg; per frame it stays within +-30 deg of upwind the whole time (mean |heading - upwind|
per 5 s window 8, 16, 17, 8, 25 deg, the variation being mostly the wind's own meander), which is what a fly that
simply walks straight from this start pose would also do. The distance to the apple falls from 71.7 to 50.1 cm --
it walks roughly towards it, but a 25 s clip at ~1 cm/s cannot reach a fruit 70 cm away; `tasting` and `feeding`
are 0 in every frame, MN9 0 Hz. The giant fibre's walking maximum is 21 Hz at 8.4 s, below the 33 Hz threshold: no
escape, no takeoff. The model's wind signal itself (DNp18 / DNp33 flipping with the wind side, +45 / -49 Hz) is
measured by `scripts/probe_wind.py`, not by this clip; the `wind_L` / `wind_R` readouts in the WALK column are what
the body reads from it here (both 10-30 Hz, the difference small with the wind near head-on).

## The toolkit figure -- `toolkit_loom_gf.png`

Left panel, `paths` (structural, CPU; `docs/INTERP.md` 4.3), run on this desktop at commit `442c420` with the
shipped cache (compiled W md5 `ef23cc27bea13be7f6a96f3c04fd3737`):

```
PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a "LC4|LPLC2" --b DNp01 --k 2 --top 12 --json out/media/paths_loom_gf.json
```

It is the type-level stage map of the k <= 2 signed walks from the two loom-sensitive visual projection types to the
giant fibre: the direct links carry +611 (LC4) and +485 (LPLC2) mV per post cell per presynaptic-type volley (these
include the x3 type-path gain of `brain.DEFAULT_TYPE_PATH_GAIN`); the eight strongest two-step relays each deliver a
few mV (PVLP024 and CB3513 inhibitory, the rest excitatory), i.e. under this model the loom reaches DNp01 almost
entirely monosynaptically. Numbers on the arrows are the `links` table's `mv_per_post_volley`.

Right panel, `atlas` (GPU; `docs/INTERP.md` 4.5): five independent runs (jobs 2 and 3 below, seeds 0-4), each a
`FlyBrain(batch=64)` in which row r pulses population r at 150 Hz for 400 ms after a 200 ms settle while four rows
get no pulse (the null), read out through `MotorRates`. Analysis:

```
PYTHONIOENCODING=utf-8 python scripts/interp_atlas.py analyse --runs "out/media/atlas_loom_r*" --no-connectome --json out/media/atlas_loom.json
```

Pulsing LC4 or LPLC2 on either side drives the `gf` readout (DNp01's mean rate over the pulse window) to 124-127 Hz
(run SD 0.3-0.7 Hz); DNp01 pulsed directly gives 55 Hz (its own pulse, a 1-cell population), LC6 4-6 Hz, LPLC1
0-0.4 Hz; the null rows are 0.00 Hz in every run. The verdict column is `common.compare`'s (`docs/INTERP.md` 2.4):
**`undetermined`** because the null arm is deterministic (the giant fibre never fires unstimulated, SD 0), so z is
undefined and the row is read by its `diff` and the exact Mann-Whitney p (1.9e-61 for every driven row);
`LPLC1_L` is `null` (0.0 Hz in all five runs). Device `cuda`, NVIDIA B200. The figure is drawn by
`scripts/make_demo_media.py figure` from the two JSONs; nothing in it is hand-entered.

## The cluster batch (four jobs, all on node1, every one `device cuda`)

Jobs 0-2 in one `cluster_run.py` call, job 3 (atlas seeds 3 and 4, to bring the atlas from three runs -- which
`compare` calls `underpowered` by rule -- to five) in a second; each command runs after
`mkdir -p out/media && source .venv/bin/activate &&` inside the per-run copy, with `SDL_VIDEODRIVER=dummy`:

```
python scripts/make_demo_media.py cluster --name media --minutes 30      # = python scripts/cluster_run.py --name media --minutes 30 --arm-block fam <job0> <job1> <job2> --fetch out/media/
job0: mkdir -p out/media && export FAM=fam_media && python -c 'import torch; assert torch.cuda.is_available()' && python scripts/make_demo_media.py record --clip loom --job $FAM > out/media/loom_record.txt 2>&1; st=$?; tail -8 out/media/loom_record.txt; exit $st
job1: mkdir -p out/media && export FAM=fam_media && python -c 'import torch; assert torch.cuda.is_available()' && python scripts/make_demo_media.py record --clip wind_apple --job $FAM > out/media/wind_apple_record.txt 2>&1; st=$?; tail -8 out/media/wind_apple_record.txt; exit $st
job2: mkdir -p out/media && export FAM=fam_media && python -c 'import torch; assert torch.cuda.is_available()' && for s in 0 1 2; do python scripts/interp_atlas.py run --populations 'LC4|LPLC2|LC6|LPLC1|DNp01' --pattern '^(DNp01|DNp11|PVLP151|PVLP122|PVLP024)$' --seed $s --out out/media/atlas_loom_r$s > out/media/atlas_loom_r$s.txt 2>&1 || exit 1; done && tail -3 out/media/atlas_loom_r2.txt
job3: the same as job2 with `for s in 3 4` (a second cluster_run.py call, --name media)
```

Run dirs `$CLUSTER_RUNS/media-d3317b` (jobs 0-2, `3 job(s), 0 failed`) and `media-0977ee` (job 3, `1 job(s), 0
failed`). The raw outputs (`out/media/*_raw.gif` at 1360x820, `*_last.png`, `*_log.json`, `atlas_loom_r*.npz/.json`,
`paths_loom_gf.json`) are git-ignored under `out/`.

## Encoding (this desktop, ffmpeg 8.0)

```
python scripts/make_demo_media.py encode        # raw GIF -> docs/media/<clip>.mp4 and .gif
```

MP4: every raw frame re-stamped at the true 80 ms cadence (`setpts=N/12.5/TB`; imageio's Pillow writer stores a
100 ms delay for room_demo's 0.08 s, so the raw GIF -- and `room_demo.py --gif` -- plays 25 s of brain time in 31 s),
libx264 crf 23, preset slow, yuv420p, 1360x820, 12.5 fps. GIF: the same frames at 10 fps, 800 px wide (lanczos),
128-colour palette (`palettegen stats_mode=diff`, bayer dither scale 3) -- the settings that keep each GIF under
15 MB. The MP4s are encoded from the 256-colour raw GIF frames (the record loop stores what `room_demo.py --gif`
stores), so they carry that quantisation; the console is flat-coloured and it is not visible at this size. Durations
checked with ffprobe: 25.0 s each (312 frames in the MP4s, 250 in the GIFs).


## Imposed compass memory figure (2026-09-15)

`compass_standin.png` shows actual EPG spike rates and decoded phase, raw versus the explicit
instrumented compass, seed 10 row 6 (+180 deg/s, stop, reverse). Sources are
`out/compass_standin/{raw,instrumented}_s10.npz`, submitted code `61d9415`, house CUDA; this is
an imposed angular-memory experiment, not a recovered biological compass. Recreate with
`python scripts/compass_driver_analyse.py`, then copy `out/compass_standin/analysis/turn_trace.png`
here. The plot uses unwrapped headings and masks phase when vector strength is below 0.6.
No trajectory was selected by outcome. See `docs/audits/compass_standin.md` for all 48 rows.
