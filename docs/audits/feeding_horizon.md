# A feeding-capable protocol: is the room's food-finding limited by starvation or by search?

Script: `scripts/probe_feeding_horizon.py` (new). Cluster batch `feed-horiz` (8 jobs, one batch); console log
`out/feeding_horizon_cluster.log`; per-job results `out/feedh_{default,off}_d{1.0,0.5,0.25}_{10,5}min_s{0,1}.{json,txt}`.
Aggregation and the CPU-side analyses, all in the same script and all reproducible without a GPU (saved in
`out/feeding_horizon_prior.log`):

    python scripts/probe_feeding_horizon.py --report out/feedh_*.json          # sections 5-6
    python scripts/probe_feeding_horizon.py --prior out/r5_*sustain*.json ...  # sections 2, 2.1
    python scripts/probe_feeding_horizon.py --geometry --fruit apple|all --minutes 5   # sections 1.2, 7(E)
    python scripts/probe_feeding_horizon.py --odour-profile --fruit apple      # section 2.2

This is the third handover item of the closed receptor plan (`docs/audits/receptor_verification.md` round 5,
"Handover" 3; `docs/NT_INTEGRATION.md` section 7). The inherited fact is that the room's feeding measures cannot rank
models: under the shipped `body.Metabolism` a 0.9 tank is empty at t = 170-180 s, 13-16 of 16 flies sit at energy 0 by
300 s, meals are 0.17-0.23 per fly per 5 min, and meals / final energy separate the shipped receptor default from
`off` at p 0.58 / 0.45.

Skeptic (dynamics round 1, 2026-09-12): verdict **mostly sound**. Its refutations and corrections are integrated in
place below -- the largest being that the closest-approach separation does not replicate (5.4) and that the negative
tail of `min_distance_cm` is airborne, not a walk-through (5.1 item 2). Its replicate batch is `sk-feed-048583` (4
jobs, 0 failed, 55.0 min, all `device=cuda`; `out/sk_feed_cluster.log`, `out/skfeed_*.json`) and its re-derivation
script is `scripts/sk_feeding_horizon_verify.py` (CPU, sections A-F). The verdict verbatim is in
docs/audits/receptor_verification.md, "Dynamics round 1 (2026-09-12) -- skeptic verdicts".

**`flyverse/body.py` is not touched here, and nothing in this audit is a proposed default.** The probe builds a normal
`BatchSim` and then writes `drain_per_s` / `walk_drain_per_m` on each fly's own `Metabolism` object before the loop;
`batch_body._metabolism` reads both off those scalar objects every frame (`flyverse/batch_body.py:104-121`, via
`attr()` at :13-14; docs/BATCH_SIM.md: "Changing a body's parameters affects the next batch update"), so
`--drain-scale 0.5` asks "what would half the drain give?" without an edit to the shared file. That this is equivalent
to editing the default is not an assumption: `tests/test_batch_sim.py` already compares every fly's whole `Metabolism`
dataclass -- drain parameters included -- between the batched and scalar models to 1e-12, over mixed states and across
the satiety / resume hysteresis boundary (:82-107, :248; re-run here, 12 tests, 8 pass and 4 skipped as opt-in
full-connectome integration). Under THE PROJECT RULE these are applied gains in an
experiment, stated as such. The drain default is the owner's decision; section 7 is a recommendation with measured
consequences, not a change.

## 1. What "starvation" actually is in this model (code, not measurement)

`body.Metabolism` (`flyverse/body.py:169-206`): `energy` in [0, 1], `drain_per_s` 1/300, `walk_drain_per_m` 0.15,
`feed_per_s` 1/15, `satiety` 0.95, `resume_below` 0.7, `hunger = clip(1 - energy, 0, 1)`.

A grep over the package for consumers of `energy` finds **five** (`flyverse/*.py`, `scripts/room_demo.py`):
`Metabolism.hunger` (:186), the satiety / resume hysteresis inside `update` (:191-201), the HUD string
`Metabolism.state` (:206), its use in `room_ui.py:445` -- and `flyverse/batch_body.py:110-120`, the vectorised
`_metabolism`, **which is the one that actually ran in every batch job in this document**: the scalar
`Metabolism.update` at `body.py:188-202` never executes under `BatchSim`. The conclusion is unchanged, because
`batch_body._metabolism` also has no death, no immobilisation and no episode termination at energy 0 (docs/BATCH_SIM.md
states this for the batch API too: "The API does not automatically reset starving flies or terminate episodes"); but
every code reading below should be taken against the batch path, not the scalar one.

`hunger` enters behaviour in one place, `programs.AnemotaxisProgram` -- whose decision logic `cx.CompassSteering`
reuses verbatim to build its PFL3 / DNp09 drive (`flyverse/cx.py:37-47`):

| term | code | value at energy 0.9 | at energy 0 |
|---|---|---|---|
| odour-gated upwind gain `hunger_scale = 0.1 + 0.9 * hunger` | `programs.py:47,104-105` | 0.19 | 1.00 |
| search program armed (`hunger > search_hunger = 0.3`) | `programs.py:57,108` | off | on |
| downwind drift `k_downwind * hunger` | `programs.py:112` | 0.1x | 1.0x |
| speed bonus / steering `* hunger_scale` | `programs.py:118-119` | 0.19 | 1.00 |

So **an empty tank is the model's state of MAXIMUM food-seeking drive, not of impairment.** Two consequences follow
without any simulation:

1. **Energy has no dynamic range as a score.** It is a floor at 0 for almost every fly: 44 of 48 flies (default) and
   46 of 48 (off) end a 300 s rollout at exactly 0 (`out/r5_adopt_sustain_live_{1,2,3}.json`,
   `out/r5_sustain_off_live_{1,2,3}.json`).
2. **Hunger has no dynamic range after ~175 s.** Every fly's motivational gain is clamped at 1, identical across flies
   and models, so the one state variable the programs read carries no information for the rest of the run.

And the meal gate never blocks a hungry fly: `feeding = tasting and not sated`, and `sated` is cleared as soon as
`energy < resume_below`, so at energy 0 any fruit contact starts a meal immediately.

### 1.1 The drain arithmetic, and what a "meal" counts

At the room's measured walking speed (path 3.67-3.75 m per 300 s over 16 flies, `out/r5_adopt_sustain_live_1.json`
mean 3.67 m = 1.223 cm/s) the total drain is 1/300 + 0.15 x 0.01223 = 0.005168 per s, of which the resting term is
64 % and walking 36 %. A scalar `Metabolism` integrated at that speed (`body.Metabolism.update`, dt 10 ms) gives:

| drain-scale | drain per s | 0.9 tank empties | energy crosses 0.7 (search arms) | re-feed interval 0.95 -> 0.70 |
|---|---|---|---|---|
| 1.0 (shipped) | 0.005168 | **174.2 s** | 38.7 s | 48.4 s |
| 0.5 | 0.002584 | 348.3 s | 77.4 s | 96.8 s |
| 0.25 | 0.001292 | 696.6 s | 154.8 s | 193.5 s |

174.2 s reproduces the observed 170-180 s exactly (`out/r5_adopt_sustain_live_1.txt`: at t = 170 s the fourteen
non-feeding rows hold 0.026-0.051; at t = 180 s all fourteen are 0.0). Because every fly starts at the same energy and
walks at nearly the same speed, **the tank empties within a ~5 s window of every fly at once** (the 0.026-0.051 spread
is 5.0-9.9 s of drain), so "time at energy 0" is a property of the protocol, not of the fly.

A meal is `meals += 1` on the rising edge of feeding, so the per-fly meal count is

> meals = (fruit encounters) + (re-feeds), re-feed interval = (satiety - resume_below) / drain = 0.25 / drain

and **lowering the drain lowers the re-feed term** (48 s -> 97 s -> 194 s). The measured composition of the 0.17-0.23
meals per fly is visible in the round-5 logs. In `out/r5_adopt_sustain_live_1.txt` two of sixteen flies ever fed:
row 6 fed once (between 90 and 120 s: 0.459 -> 0.922) and drained to 0 without returning; row 12 fed at ~50 s
(0.901 at t = 60), drained to 0.462 by 150 s and fed again by 180 s (0.942). Row 12's second meal is **not** a re-feed:
a fly that had stayed in taste range would have resumed the moment energy crossed 0.7, near t = 105 s. So the three
meals are three separate arrivals by two flies, and in the measured behaviour the metabolic re-feed term contributes
nothing -- a sated fly leaves the fruit (by design: `satiety` ends the meal) and takes longer to come back than the
48 s re-feed window. **Meals, as measured, is an arrival counter carried by 1-2 flies out of 16** -- which is why its
between-replicate scatter is a factor 5 (section 2).

### 1.2 The assay's geometric ceiling (why the horizon may not be the lever either)

The fenced single-apple table is a 1.2 x 0.8 m walkable surface (`world.make_room`, `table_extent`, fence =
table extent) with one target: a sphere of radius 4 cm at (0.25, 0.15, 0.79) resting on the table top,
`fruit_set='apple'` keeping only `fruit[:1]` (`flyverse/world.py:338-351`). `BatchSim.step` tastes at surface distance
< 1.5 cm (`flyverse/batch_sim.py:174`), which for a fly walking in the plane z = 0.75 is an **in-plane capture radius
of 3.77 cm** (sqrt(0.055^2 - 0.04^2)), i.e. a capture disc of 0.0045 m^2 = 0.47 % of the table.

For an ideal searcher that never revisits, expected encounters = path x swath / area = L x 0.0755 / 0.96:

| horizon | path at 1.223 cm/s | ideal expected encounters per fly |
|---|---|---|
| 5 min | 3.7 m | 0.29 |
| 10 min | 7.4 m | 0.58 |
| 17.3 min | 12.7 m | 1.00 |

**No horizon below ~17 minutes can give one encounter per fly for a BLIND random sweep.** An ideal blind searcher's
chance of finding the apple in 5 minutes is 1 - exp(-0.289) = 0.251; the measured model feeds 6 of 48 and 8 of 48 flies
(0.125 / 0.167, section 2), i.e. **50-67 % of that rate**.

**That 17.3 minutes is a blind-search baseline, not a ceiling, and the searcher is the binding constraint -- not the
assay.** The protocol starts every fly at (-0.15, 0.15), exactly on the apple's own y axis (0.25, 0.15), 40 cm
straight downwind of it, and the capture disc subtends 10.78 deg from there (3.00 % of headings). A straight upwind
walker holding heading within +-5.39 deg reaches the disc in 29.6 s = **0.49 min** at 1.223 cm/s. An odour-guided fly
whose own rule is "steer upwind" therefore has a directed route two orders of magnitude cheaper than the sweep, and
running at 43-53 % of a blind sweep indicts the searcher -- which is exactly what section 6 concludes. The batch in
section 4 tests the other half of it: that the limit is not the tank.

(`python scripts/probe_feeding_horizon.py --geometry --fruit apple --minutes 5` prints this table; the union capture
area is integrated on a 1201 x 801 grid and overlapping discs are merged into target groups, which matters only for
`--fruit all` in section 7.)

## 2. What the present measure can and cannot resolve (prior, re-derived from files)

Re-derived here from the round-5 room batches (energy 0.9, 300 s, 16 flies, `--program cx --fruit apple --fence`;
shipped-gains default vs `off`, the `off` arm under the damped gains -- `docs/audits/receptor_integration.md` caveat):

| arm | files | flies | meals | meals/fly | flies that fed | max/fly | final energy | flies at 0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default (sign/abs) | `out/r5_adopt_sustain_live_{1,2,3}.json` | 48 | 8 | 0.167 | 6 | 2 | 0.0278 | 44 |
| off (presynaptic) | `out/r5_sustain_off_live_{1,2,3}.json` | 48 | 10 | 0.208 | 8 | 2 | 0.0380 | 46 |

Two-sided Mann-Whitney over the 48 + 48 flies: meals U 1106, **p 0.586**; final energy U 1196, p 0.447. (This
reproduces the handover's "p 0.58" from the files.)

Between-batch scatter of the *same* condition, meals per fly: default 0.188 / 0.125 / 0.188 (round 5) and
0.125 / 0.062 / 0.312 / 0.250 (round 4, `out/r4_sustain_default_{1,1b,2,3}.json`); off 0.062 / 0.312 / 0.250 (round 5)
and 0.062 / 0.000 / 0.250 / 0.188 (round 4).

Widening to **every 300 s, energy-0.9 room batch on file** (mixing the damped and shipped gains and the `--gf-hz 1e9`
arms, so this pools conditions and is a scatter statement, not a comparison):

| | batches of 16 x 300 s | fly-rollouts | meals | meals/fly | flies that fed | ended at energy 0 | per-batch meals/fly |
|---|---:|---:|---:|---:|---:|---:|---|
| default | 15 | 240 | 36 | 0.150 | 31 (12.9 %) | 220 (91.7 %) | 0.000 - 0.375 |
| off | 11 | 176 | 22 | 0.125 | 20 (11.4 %) | 170 (96.6 %) | 0.000 - 0.312 |

**Replicates of one condition span 0.000 to 0.375 meals per fly**, and the two conditions' ranges coincide, while the
difference to be detected is 0.150 vs 0.125. That is the whole reason the measure cannot rank anything.

Power, stated as a requirement rather than a complaint. At 0.19 meals per fly per 300 s a two-sided Poisson rate test
at 80 % power needs ~95 meals per arm to resolve a 1.5x difference and ~33 for a 2x one -- **151,000 and 52,000
fly-seconds per arm**; the round-5 take-off instrument, which does rank the models, spends 14,400. Turned round, the
smallest rate ratio meals can resolve with the samples this project actually runs:

(at the pooled 0.139 meals per fly and the pooled final-distance spread 20.81 +- 9.51 cm; `--prior` prints this table)

| flies per arm | fly-s per arm | expected meals | smallest resolvable meal-rate ratio | smallest resolvable shift in final distance to the fruit |
|---:|---:|---:|---:|---:|
| 16 (one batch) | 4,800 | 2.2 | 14.2x | 9.4 cm |
| 48 (three batches) | 14,400 | 6.7 | 4.6x | 5.4 cm |
| 96 | 28,800 | 13.4 | 3.0x | 3.8 cm |
| 240 (every default batch on file) | 72,000 | 33.5 | 2.0x | 2.4 cm |

The distance column uses the measured between-fly spread of the final surface distance to the apple: default
20.54 +- 9.58 cm, off 19.62 +- 9.60 cm over 48 flies each (`out/r5_adopt_sustain_live_*`,
`out/r5_sustain_off_live_*`; MWU U 1218, p 0.631, t 0.47, p 0.639 -- so the two models do not differ on it either, but
the *measure* resolves 5.4 cm at the sample size where meals resolves only a factor 4.6). **Every fly contributes to a
distance; one fly in eight contributes to a meal.** That is the argument for changing the score rather than the
metabolism.

### 2.1 Where the flies actually end up (416 final positions already on file)

The same 26 batches carry one measure that was never read: `rows[].position`. Pooling all 416 fly-rollouts (mixed
conditions, so this is a description of the protocol, not a model comparison) and comparing with a uniform point on the
fenced table -- the apple sits at (0.25, 0.15), the fly starts at (-0.15, 0.15), and `WindParams.direction_deg = 180`
means the wind blows towards -x (`flyverse/air.py:33`), so **the fly starts 36 cm downwind of the source and upwind is
+x**:

| shell (surface distance to the apple) | flies | % of 416 | uniform % | enrichment |
|---|---:|---:|---:|---:|
| 0-4 cm (the capture disc is 3.8 cm) | 3 | 0.7 | 1.6 | 0.46 (n = 3) |
| 4-8 cm | 15 | 3.6 | 2.6 | 1.38 |
| 8-12 cm | 29 | 7.0 | 3.7 | 1.90 |
| 12-16 cm | 85 | 20.4 | 4.7 | **4.32** |
| 16-20 cm | 82 | 19.7 | 5.7 | 3.45 |
| 20-25 cm | 105 | 25.2 | 8.0 | 3.16 |
| 25-30 cm | 47 | 11.3 | 8.2 | 1.39 |
| 30-40 cm | 33 | 7.9 | 15.2 | 0.52 |
| 40-70 cm | 15 | 3.6 | 36.4 | 0.10 |

Against a uniform point the flies are enriched 2.81x within 20 cm of the apple and 3-4x in the 12-25 cm shells, and
**not at all at the target itself** (3 of 416 inside 4 cm against a uniform expectation of 6.5). 81.2 % of them end at
x > 0.25, i.e. past the source going upwind (median final x 0.415 = 16.5 cm beyond it). It is not fence-sticking:
10.1 % end within 2 cm of a fence edge against a uniform 8.1 % (per arm 4.2-16.7 %, i.e. 2-8 flies of 48, so read the
pooled figure), and 45.0 % within 10 cm against 37.4 %.

**Most of that enrichment is the upwind march, not the plume.** Against an x-matched null (the observed final-x
marginal, y uniform), which isolates the plume-specific lateral signature, the enrichments fall to 12-16 cm **2.21x**,
16-20 cm **1.65x**, 20-25 cm **1.53x**, within 20 cm **1.62x** -- and inside 12 cm there is none at all (**1.07x**),
inside 8 cm a depletion (**0.86x**) (`scripts/sk_feeding_horizon_verify.py` section E). So the honest statement is
narrower than "flies reach the source region": they end up at the right *distance downwind-to-upwind* because they walk
upwind, and the plume adds only a modest lateral concentration around the axis -- which strengthens, not weakens, the
terminal-approach conclusion. Two further caveats: this is a single end-of-run snapshot, not an occupancy measure; and
the lateral signature itself is modest (37.5 % end within 10 cm of the plume axis y = 0.15 against a uniform 25 %, and
the fly also *starts* on that axis). The `min_distance_cm` rows of the new probe settle it, because
they measure the closest approach rather than the last frame -- and they show the approach going much further in than
the end-of-run snapshot suggests: median closest approach 2.3-10.8 cm, 58 of 64 flies inside 10 cm at the 10-minute
horizon (section 5.3). So the flies reach the source region, come within a few centimetres, and then leave it again;
the 12-25 cm shell is where they happen to be when the clock stops, not how close they got.

The mode fractions over the same 416 rollouts say the same thing from the other side: **29.14 % of every rollout is
spent "surging"** (the lateral-horn odour gate above 0.6), 8.64 % casting after plume loss, 10.93 % labelled
`searching` -- which is the mode name for the frames where the search program is NOT armed -- 50.89 % `exploring`
(the armed fraction; the two labels are the wrong way round, `programs.py:120`) and 0.40 % feeding (1.2 s per fly per
300 s). The fly is in a strong odour signal for roughly a third of the run and still touches the fruit in 12.9 % of
rollouts. Odour detection is not the limit, and
starvation cannot be the limit (section 1); the limit is between the surge and the last 12 cm.

### 2.2 Why the last 12 cm is hard (the odour field, from `air.py`)

The plume-height artefact of session 8 is fixed: `Air` gives a source a plume "as wide as its source", so a fly walking
under the apple's rim is on the plume axis (`flyverse/air.py:102`, `sig = max(sigma0, src_rad) + spread * max(x, 0)`;
the design is stated in the `Air.__init__` docstring at :51-53). But the
Gaussian term exists **only downwind** (`np.where(x > 0, ...)`, :104); upwind of a source the only odour is the
isotropic near-field `near_gain / (1 + (r / 0.03)^2)`. Evaluating the shipped field on the plume axis at the fly's
walking height (apple `strength = radius / 0.02 = 2.0`, `puff_depth` set to 0 for the mean; summed over the apple's
glomeruli):

| position on the axis | surface distance | total concentration | of which the near field |
|---|---:|---:|---:|
| 40 cm downwind (the start) | 36.2 cm | 0.482 | 0.029 |
| 15 cm downwind | 11.5 cm | 1.231 | 0.188 |
| 4.5 cm downwind | 1.9 cm | 2.759 | 1.056 |
| 5 cm **upwind** | 2.3 cm | 0.954 | 0.954 |
| 10 cm upwind | 6.7 cm | 0.377 | 0.377 |
| 15 cm upwind | 11.5 cm | 0.188 | 0.188 |
| 20 cm upwind | 16.4 cm | 0.110 | 0.110 |

The approach is well signposted -- concentration rises 5.7x from the start to 4.5 cm downwind -- but **past the source
the fly is still in odour** (0.19 at 15 cm upwind, against 0.48 where it started), and the model's response to odour is
"steer upwind", which from there points away from the fruit. Nothing in `AnemotaxisProgram` / `CompassSteering` uses a
concentration *change* to stop or reverse; the gate only switches between surge, cast and search. That is consistent
with the 12-25 cm shell and the 81 % upwind excess of section 2.1, and it names the missing term: the antennal-contrast
(klinotaxis) stage, which session 8 measured closing the last 1.4-1.5 cm in 2 of 3 seeds where `cx` alone stalled at
8-17 cm. **This is a dynamics question for the search thread, not a metabolism question.**

## 3. What the probe records

`scripts/probe_feeding_horizon.py` writes one JSON per run: a `summary`, a `trajectory` (energy and distance-to-fruit
every `--trajectory-s`), cumulative `checkpoints` every `--checkpoint-s` (so a 10-minute run can be read at the
5-minute horizon), and a `rows` entry per fly with

* `meals`, `meal_times_s`, `first_meal_s`, `feeding_s`;
* `energy`, `min_energy`, `mean_energy`, `t_zero_s` (when the tank first empties), `frac_at_zero`, `s_at_zero`;
* `contacts` (rising edges of `BatchSim.tasting`), `first_contact_s`, `contact_s` -- the *arrival* measure a meal needs;
* `distance_cm` (final), `min_distance_cm`, `t_min_distance_s`, `near_frac` at 2 / 5 / 10 cm -- the *approach* measures,
  to which every fly contributes whether or not it ever feeds. `min_distance_cm` is the minimum over all frames,
  airborne ones included, while a taste needs the fly on the ground -- see the confound in 5.4;
* `hops` / `hops_escape` / `hops_voluntary` / `gf_max_walk_hz` (the round-5 take-off instrument, read from
  `BatchSim.hops_escape` / `hops_voluntary` exactly as `scripts/batch_sustain.py` does, so the drain-1.0 arms are also
  extra draws of handover item 5 at a new horizon), `path_m`, `airborne_frac`, `modes`, `position`;
* `before_zero` / `after_zero`: seconds, path, meals and contacts on each side of the moment the tank empties. Since
  section 1 shows an empty tank has no motor consequence, the contact rate before vs after is the direct test of
  whether starvation limits food-finding at all.

The metabolism block records the shipped parameters, the applied ones and the scale, so every run says what gain it
applied.

`--geometry` (CPU, no brain) prints the section-1.2 ceiling for any fruit set, and `--report` aggregates finished
JSONs. Both are analysis paths added after the batch below was submitted; the rollout path the jobs ran is unchanged.

## 4. The batch

One cluster batch, `feed-horiz`, 8 concurrent jobs, `--batch 16 --energy 0.9 --program cx --fruit apple --fence
--cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch --trajectory-s 5 --checkpoint-s 60 --log-every 30`,
brain seed 0 with environment seeds 0-15 and brain seed 1 with environment seeds 16-31 (the round-5 convention):

| job | receptor model | drain-scale | horizon | brain seed | file stem |
|---|---|---|---|---|---|
| 1 | shipped default (sign/abs) | 1.0 | 10 min | 0 | `out/feedh_default_d1.0_10min_s0` |
| 2 | `--receptor-model off` | 1.0 | 10 min | 0 | `out/feedh_off_d1.0_10min_s0` |
| 3 | shipped default | 1.0 | 10 min | 1 | `out/feedh_default_d1.0_10min_s1` |
| 4 | `off` | 1.0 | 10 min | 1 | `out/feedh_off_d1.0_10min_s1` |
| 5 | shipped default | 0.5 | 5 min | 0 | `out/feedh_default_d0.5_5min_s0` |
| 6 | `off` | 0.5 | 5 min | 0 | `out/feedh_off_d0.5_5min_s0` |
| 7 | shipped default | 0.25 | 5 min | 0 | `out/feedh_default_d0.25_5min_s0` |
| 8 | `off` | 0.25 | 5 min | 0 | `out/feedh_off_d0.25_5min_s0` |

The seed-1 pair was spent on the drain-1.0 / 10-minute arm because it is the only arm that applies no metabolic gain
(so its numbers describe the shipped model as it stands) and it carries twice the fly-seconds per job. The
`--checkpoint-s 60` series lets the 10-minute arms be read at 300 s and compared with the round-5 batches of section 2.

Predictions entered before the results (section 1.1-1.2), so that the batch can falsify them:

* **P1** the drain-1.0 / 10-min arms empty the tank at ~174 s and spend ~71 % of the run at energy 0;
* **P2** meals per fly stay below 1 in every arm -- the 10-minute ideal-searcher ceiling is 0.58 encounters per fly;
* **P3** *lowering* the drain *lowers* the meal count (the re-feed interval goes 48 -> 97 -> 194 s) and weakens
  food-seeking (the hunger gain `0.1 + 0.9 * hunger` never reaches 1: at drain-scale 0.25 a 5-minute run ends at
  energy 0.512, gain 0.54, against gain 1.00 from t = 174 s at drain-scale 1.0, and the search program does not arm
  until 155 s instead of 39 s). Because `cx` turns that gain into its DNp09 forward drive
  (`cx.py:41-47`: `fwd = intent.speed / (wind_speed_bonus + search_speed)`, both 0.006 m/s, `programs.py:46,65`), the
  low-drain arms should also **walk less** than the 3.67-3.75 m per 300 s of the drain-1.0 batches;
* **P4** the fruit-contact rate is the same before and after the tank empties, because an empty tank has no motor
  consequence -- if anything it is higher afterwards, since the hunger gain is then maximal.

## 5. Results

Console log `out/feeding_horizon_cluster.log`; aggregated in `out/feeding_horizon_report.log`
(`--report out/feedh_*.json`). All eight jobs ran on one B200 share each under heavy cluster contention
(load average ~50, ~20 other jobs from other threads): 1.26 aggregate fly-s per wall second and 63 wall minutes for
a 16 x 300 s job, 1.39 and 115 minutes for a 16 x 600 s job, against the 6.29 of docs/BATCH_SIM.md for this
configuration. 8 jobs, 0 failed, 116.6 min for the batch; every job reports `device=cuda`.

### 5.1 Taking starvation away changes nothing about meals (drain-scale 0.5 and 0.25, 5 min)

At drain-scale 0.5 the 0.9 tank lasts 348 s and at 0.25 it lasts 697 s, so in all four arms **no fly reaches energy 0
at any point** (minimum energy over the run 0.127-0.133 at 0.5 and 0.523-0.525 at 0.25; `frac_at_zero` 0.0 in 64 of 64
rollouts). The comparison arm is every drain-1.0 300 s batch on file (section 2).

| arm | flies | tank empties | meals / fly | flies fed | contacts / fly | min distance median | min distance mean | path m | mean energy | search program armed |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drain 1.0 (26 batches, section 2) | 416 | **at 174 s** | 0.139 | 12.3 % | not recorded | not recorded | -- | 3.67-3.75 | -- | 50.9 % |
| default, drain 0.5 | 16 | never | 0.188 | 3/16 | 0.188 | 6.0 cm | 7.8 | 3.46 | 0.564 | 39 % |
| off, drain 0.5 | 16 | never | 0.250 | 4/16 | 0.250 | 10.8 cm | 11.1 | 3.32 | 0.579 | 27 % |
| default, drain 0.25 | 16 | never | 0.188 | 3/16 | 0.188 | 5.9 cm | 6.7 | 3.26 | 0.728 | 29 % |
| off, drain 0.25 | 16 | never | 0.125 | 2/16 | 0.125 | 8.4 cm | 10.6 | 3.19 | 0.730 | 25 % |

("search program armed" is the `exploring` mode fraction: `programs.py:120` labels the frame `exploring` when the
search program is on and `searching` when it is not -- the names are the wrong way round.)

Six readings, all of which need the one-replicate caveat except where a p-value is quoted:

1. **Starvation was not the limit.** With the tank never empty, meals per fly are 0.125-0.250 in these four arms
   against 0.139 at the shipped drain. Removing starvation entirely moves the feeding measure by less than its own
   replicate scatter. The scatter is also wider than the 0.000-0.375 of section 2: two further drain-0.5 runs at
   brain seed 1 reach **0.500 meals per fly** (8 meals over 16 flies; 7 of 16 flies fed in the default arm, 8 of 16 in
   off -- `out/skfeed_default_d0.5_5min_s1.json`, `out/skfeed_off_d0.5_5min_s1.json`), so the quoted range is
   0.000-0.500 and the highest single run is 0.500, not 0.375. The central negative -- never 1 meal per fly --
   survives at 0.500.
2. **`meals` equals `contacts` in all four arms** (3/3, 4/4, 3/3, 2/2), and in these four arms `contacts` equals the
   number of flies whose closest approach `min_distance_cm` fell below the 1.5 cm taste threshold, fly for fly (12 of
   64 flies, 12 of 64 fed). The metabolic re-feed channel contributes exactly zero meals at any drain, so **`meals` is
   a binary threshold at 1.5 cm on the closest-approach distribution, AND on the ground** (`batch_sim.py:174` ANDs the
   1.5 cm test with `~airborne`) -- a distribution that runs from -2.1 cm to +34.1 cm with a per-arm median of
   5.9-10.8 cm. **43 of the 64 flies come within 10 cm of a target whose capture radius is 3.8 cm; 12 cross it and
   score, and the other 31 score zero.** (In the 10-minute arms the identity `contacts` = flies inside 1.5 cm breaks
   by four flies that got there in flight -- see the confound in 5.4.)

   **The negative tail is airborne, not a walk-through.** The explanation given here and in 5.3 item 4 -- "the fly
   walks through the apple's footprint: there is no sphere collision for a walking fly, so `|p - centre| - radius`
   goes negative" -- is geometrically impossible. Under `--fence`, `flyverse/batch_body.py:145-157` updates only x, y
   and heading, so a walking fly sits at exactly z = table_top = 0.75 (all 416 re-read final positions carry z = 0.75
   and no other value); the apple is `Sphere((0.25, 0.15, 0.79), r = 0.04)` (`flyverse/world.py:338`) and
   `flyverse/batch_sim.py:151-155` returns `|p - centre| - radius`, whose ground floor is therefore exactly
   sqrt(0.04^2) - 0.04 = **0.0000 cm**. Empirically: all 20 flies with a negative minimum took off (20/20), and all 65
   flies that never took off bottom out at +0.597 cm. The premise is true in x, y and irrelevant: negative values
   require z > 0.75, i.e. flight.
3. **The drain is not behaviourally neutral, but its direction on food-finding is UNRESOLVED.** Path length per 300 s
   falls monotonically with the drain: 3.67-3.75 m at drain 1.0, 3.46 / 3.32 at 0.5, 3.26 / 3.19 at 0.25; the fraction
   of the run with the search program armed falls 50.9 % -> 39 / 27 % -> 29 / 25 %. Less drain means less hunger, and
   hunger is the only motivational gain the programs have (section 1), so the slower-drain fly walks less and searches
   less. **What does not follow is that it finds less food.** The comparison above sets drain 0.5 / 0.25 at brain seed
   0 against a 26-batch pool of other seeds; the **seed-matched** comparison, available in these same files (the 300 s
   checkpoints of the 10-minute runs), goes the other way at brain seed 1 / env 16-31: halving the drain takes meals
   per fly **0.312 -> 0.500** (default) and **0.250 -> 0.500** (off), flies fed 5 -> 7 and 4 -> 8 of 16, and the
   closest-approach median 6.20 -> 2.52 cm and 7.12 -> 2.31 cm. At brain seed 0 / env 0-15 it does not (default
   0.250 -> 0.188, off 0.000 -> 0.250). **Three of the four seed-matched drain-1.0 -> drain-0.5 transitions on file
   raise meals**, so the drain's effect on food-finding is unresolved at this sample size, not established as adverse.
   (The path and search-arming effects above are seed-matched within each run and stand.)
4. **`min_distance_cm` separates the models where `meals` cannot -- but it does not replicate.** Default is closer
   than off in both cells here -- 6.71 vs 10.60 cm at drain 0.25 (U 88, p 0.137) and 7.82 vs 11.07 at 0.5 (U 111,
   p 0.534) -- while `meals` on the very same rollouts gives p 0.653 and 0.693. Two independent replicate cells run
   afterwards break the streak: see 5.4.
5. **The round-5 take-off excess replicates at both drains**: default 13 hops (6 escape + 7 voluntary) and 18
   (5 + 13) against off 3 (3 + 0) and 4 (4 + 0) over 16 x 300 s each; U 175.5, p 0.035 at drain 0.25 and U 204,
   p 0.0019 at 0.5. Off makes no voluntary take-off in either arm, as in all six round-5 batches.
6. **Instrument check.** For the flies that never fed, never took off and never hit the floor, the drain is an exact
   identity: `energy = 0.9 - drain_per_s * T - walk_drain_per_m * path_m`. Measured residuals (median over 3-13
   qualifying flies) -2.9e-04 to -9.8e-04, **almost always negative** -- range -2.7e-03 to +4.5e-06, the one positive
   being a fly in `out/feedh_default_d0.25_5min_s0.json` -- the expected sign and size of fence clamping (a fly held
   against the fence is charged for a commanded speed it does not travel). Scope: **4 of the 8 arms have zero
   qualifying flies** (every fly in the four 10-minute arms fed, hopped or hit the floor), so this check covers the
   5-minute arms only. The parameter write does what editing `body.py` would.

### 5.2 The 10-minute horizon at the shipped drain: meals reach 0.31 per fly, not 1

Two replicates per arm (brain seed 0 with environment seeds 0-15, brain seed 1 with 16-31), 32 flies and 19,200 fly-s
per arm.

| | default s0 | default s1 | off s0 | off s1 |
|---|---:|---:|---:|---:|
| meals (= contacts) | 4 | 6 | 1 | 6 |
| meals per fly | 0.250 | 0.375 | 0.062 | 0.375 |
| flies that fed | 3/16 | 6/16 | 1/16 | 5/16 |
| first meal, median | 116 s | 171 s | 457 s | 100 s |
| tank empties, median (first fly) | 177.8 s (174.3) | 176.4 (173.5) | 178.4 (174.1) | 179.8 (175.5) |
| fraction of the run at energy 0 | 64.9 % | 65.1 % | 68.8 % | 63.4 % |
| closest approach, median | 4.1 cm | 2.3 cm | 7.2 cm | 3.8 cm |
| path | 7.63 m | 7.64 m | 7.44 m | 7.52 m |
| take-offs (escape + voluntary) | 36 = 15 + 21 | 41 = 16 + 25 | 2 = 2 + 0 | 3 = 3 + 0 |

Cumulative meals (= contacts) per fly against time, averaged over the two replicates of each arm -- the direct answer
to "at what horizon do meals become countable":

| t | 60 s | 120 | 180 | 240 | 300 | 360 | 420 | 480 | 540 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 0.06 | 0.16 | 0.16 | 0.22 | 0.28 | 0.28 | 0.28 | 0.31 | 0.31 | 0.31 |
| off | 0.03 | 0.09 | 0.12 | 0.12 | 0.12 | 0.16 | 0.16 | 0.19 | 0.19 | 0.22 |

**Meals never become countable, and the curve is saturating, not linear.** Half of the default arm's arrivals happen in
the first 120 s -- the initial upwind surge from the start position 36 cm downwind of the apple -- and the second five
minutes add 0.03. Extrapolating the measured efficiency rather than the curve: the ideal-searcher ceiling is 0.58
encounters per fly in 10 minutes (section 1.2) and the model delivers 0.31, i.e. 53 %, so **one meal per fly needs
about 33 minutes per fly** (17.3 ideal minutes / 0.53) = 31,700 fly-s per 16-fly replicate -- 84 wall minutes per
batch at the 6.29 aggregate fly-s per wall second of docs/BATCH_SIM.md, 6.3 hours at the 1.39 these jobs measured
under contention, and 8.4 GPU-hours for three replicates of two arms. The saturating shape says even that is
optimistic.

The 300 s checkpoints of these runs reproduce the round-5 batches they should: 0.250 / 0.312 meals per fly (default)
and 0.000 / 0.250 (off), inside the 0.000-0.375 replicate range of section 2, with 14-16 of 16 rows at energy 0.

### 5.3 Starvation is not the limit -- four independent ways of saying so

1. **Remove starvation and nothing changes.** In the drain-0.5 and drain-0.25 arms no fly ever reaches energy 0, and
   arrivals per fly over 300 s are 0.125-0.250 against 0.139 pooled over 26 drain-1.0 batches and 0.12-0.16 at the
   300 s checkpoint of the drain-1.0 10-minute arms.
2. **An empty tank makes the fly walk faster, not slower.** Mean walking speed before and after the tank empties, from
   the probe's split: default 1.161 -> 1.325 and 1.176 -> 1.319 cm/s; off 1.144 -> 1.298 and 1.131 -> 1.302. A
   12-15 % *increase*, which is what section 1 predicts -- hunger, and with it the `cx` program's forward drive,
   saturates at its maximum.
3. **Flies keep finding fruit on an empty tank.** 4 of the default arm's 10 arrivals and 2 of off's 7 occurred after
   that fly's energy had reached 0.
4. **The failure is geometric, in the last few centimetres.** The closest approach over 10 minutes has a median of
   2.3-7.2 cm against a capture radius of 3.8 cm, and it is a continuous distribution: in the four 10-minute runs the
   per-fly closest approaches run from -2.5 cm to 12.1 cm, with 58 of 64 flies inside 10 cm and 41 inside 5 cm. The
   negative values are **airborne only** -- a walking fly is pinned at z = 0.75 and cannot go below 0.0000 cm, and the
   65 flies in this batch that never took off bottom out at +0.597 cm (5.1 item 2). Surge-level odour fills 21.1-23.0 % of these
   10-minute rollouts (29.14 % is the 5-minute, 416-rollout figure of section 2.1). The flies are not starving out of
   the assay; they are missing a 3.8 cm target they walk past.

The one number that *looks* like a starvation effect is the raw contact rate before vs after the tank empties
(default 0.95 / 1.00 falling to 0.16 / 0.46 per 1,000 fly-s; off 0.00 / 1.44 falling to 0.15 / 0.16). It is not one:
the arrival flux is front-loaded by the protocol -- every fly starts 36 cm straight downwind of the source and its
best chance is the first upwind surge -- and the no-starvation arms show the same accumulation shape without any empty
tank (their arrivals are, if anything, *later*: all 0.19 of the drain-0.25 default arm's arrivals fall in the last
120 s). The before/after split cannot separate starvation from time-since-start; items 1-3 can, and they say no
effect.

### 5.4 The measure, not the metabolism, decides whether models can be ranked

The same rollouts, scored two ways (two-sided Mann-Whitney over per-fly values, default vs off within each cell):

| cell | meals per fly | p | closest approach (mean cm) | p |
|---|---|---:|---|---:|
| drain 1.0, 10 min (n 32 each) | 0.312 vs 0.219 | 0.402 | **2.80 vs 5.20** | **0.042** |
| drain 0.5, 5 min (n 16 each) | 0.188 vs 0.250 | 0.693 | 7.82 vs 11.07 | 0.534 |
| drain 0.25, 5 min (n 16 each) | 0.188 vs 0.125 | 0.653 | 6.71 vs 10.60 | 0.137 |
| combined over the three cells | -- | Fisher 0.756 | default closer in 3 of 3 | Fisher 0.072, stratified z -2.42, p 0.016 |

`meals` cannot rank the models at any drain or horizon tested, exactly as inherited.

**The closest-approach separation FAILED replication and must be read as a hypothesis, not a finding.** Two
independent replicate cells run afterwards on the cluster (batch `sk-feed-048583`, 4 jobs, 0 failed, 55.0 min, all
`device=cuda`; `out/sk_feed_cluster.log`) break the 3-of-3 streak:

| new cell | default | off | result |
|---|---|---|---|
| drain 1.0, 5 min, brain seed 2 (env 32-47) | 8.401 cm mean / 8.325 median | **6.557 / 7.317** | **reversed**; U 145, p 0.534 |
| drain 0.5, 5 min, brain seed 1 (env 16-31) | 3.148 / 2.517 | 3.552 / **2.308** | null; U 130, p 0.955 |

(`out/skfeed_{default,off}_d{1.0,0.5}_5min_s{2,1}.json`.) At the replicate level over all six run pairs the default is
closer in **5 of 6**, exact sign test **p 0.2188**, Wilcoxon signed-rank p 0.1562, paired t p 0.1068. The original "3
of 3 cells" was a four-run streak that breaks at six runs. The per-fly statistics themselves reproduce exactly (an
independent re-derivation gives U 360 / p 0.0419, U 111 / 0.534, U 88 / 0.137, Fisher 0.0720, stratified z -2.506
p 0.0122), so this is a replication failure, not an arithmetic error.

Three further caveats on that statistic. (i) It is a **32-flies-per-arm** p: at the replicate level the nonparametric
floor is p 0.125 with four pairs and p 0.2188 with six, and the 10-minute cell alone permutes to exactly 2/6 = 0.333.
(ii) The stratified z and the Fisher combination -- the headline p 0.016 -- are **in no shipped generator**
(`probe_feeding_horizon.py` calls only `scipy.stats.mannwhitneyu`, :477); `scripts/sk_feeding_horizon_verify.py`
reproduces them. (iii) The main table quotes asymptotic p and the never-took-off control quotes exact p (0.212 /
0.238 are scipy's exact values; the asymptotic ones are 0.203 / 0.232) -- both defensible, but the mixture should be
stated. The exact meals p-values are more conservative than the quoted ones (0.544 / 0.780 / 0.780 vs 0.402 / 0.693 /
0.653), which only strengthens the negative on meals. The project rule asks for three replicates before a difference
is called a result; a batch that wants to settle it should run three replicates of **one** cell and score a
**ground-only** closest approach (below).

**The approach measure is entangled with the take-off measure -- more deeply than "it includes airborne frames".**
`min_distance_cm` is the minimum over *all* frames, airborne included, while `BatchSim.tasting` requires the fly to be
on the ground: 19 of the 64 flies in the 10-minute runs came within the 1.5 cm taste radius but only 15 tasted, the
four-fly difference (3 default, 1 off) being approaches made in flight. But the coupling is not only those four flies.
**Within each arm, hopping predicts a closer approach**: Spearman(hops, `min_distance_cm`) rho -0.517 p 0.0024 (off,
10 min), -0.751 p 0.0008 (off, drain 0.5), -0.346 p 0.0521 (default, 10 min), and pooled cell-z-scored -0.248 p 0.049
(default) and -0.384 p 0.0017 (off); across all 128 rollouts **27 of 63 hoppers came inside 1.5 cm and 23 fed, against
4 of 65 never-hoppers**. Causality is unresolvable from these files -- approaching a 4 cm sphere should loom and
trigger a GF escape, which would make proximity cause hops rather than the reverse -- but it means recommendation (A)
would install as the primary score a quantity co-determined by the one arm difference that is already the most robust
on file (hops, p 2.7e-11). Two partial controls, neither sufficient. Clipping the impossible negative tail at the
ground floor leaves the direction intact (gap 2.060 vs 2.402 cm at 10 minutes, p 0.0472 vs 0.0419; stratified z -2.452
vs -2.506), so the airborne tail alone is not what breaks the result -- replication is. Restricting to flies that
never took off is possible only in the 5-minute cells (4 and 8 default flies against 12 and 13 off), where the sign
survives and weakens with the sample: 8.17 vs 15.22 cm (p 0.212) and 9.11 vs 11.05 (p 0.238), stratified z -1.81,
p 0.070; in the 10-minute cell exactly 1 of 32 default flies never took off. A next version of the probe should record
a **ground-only** closest approach alongside this one; it is a three-line change to the accumulator, and it is the
measure the next round should score (7.2).

Two supporting notes. `near_frac` at 5 cm and 2 cm does *not* separate the arms (1.55 / 0.64 % default vs
1.59 / 0.64 % off at 10 minutes): occupancy near the fruit is dominated by the few flies that arrive and sit there, so
the closest approach is the better of the two approach measures. And `path_m` separates the models in these runs
(7.636 vs 7.478 m, p 0.0011 at 10 minutes; 3.463 vs 3.323, p 0.0056 at drain 0.5), but **that it is a locomotor
difference is not established**: `path` accumulates ||d(x,y)|| over every frame including airborne ones
(`probe_feeding_horizon.py:188-189`) and the arms differ 15x in take-offs. At 10 minutes the entire arm-level gap
(0.1582 m per fly, 5.063 m over 32 flies) sits inside the ballistic budget of the 72 extra hops: it needs 0.378 m/s
averaged over the 0.419 s per fly of extra airborne time, against `Flight.jump_speed` 0.6 m/s at 45 deg pitch with
drag 2.5/s (~6 cm per hop, ~4.6 m over the arm). Neither replicate cell reproduces the significance (p 0.376 and
p 0.158). The drain-0.5 gap (~35 %) is too large for that accounting, so this is a quantified caveat rather than a
refutation -- but the 2-4 % figure should not be quoted as a locomotor result. (The before/after-zero speed rise of
5.3 item 2 is unaffected: the off arm shows the same 12-15 % rise with 5 take-offs in 32 flies.)

### 5.5 A by-product: the round-5 take-off cost at a 10-minute horizon

These runs carry the round-5 instrument unchanged, and at 19,200 fly-s per arm this is the largest single-horizon
measurement of it on file:

| arm | take-offs | escape / 1,000 fly-s | voluntary / 1,000 fly-s | walking-GF median |
|---|---|---:|---:|---:|
| default (2 x 16 x 600 s) | 77 = 31 escape + 46 voluntary | 1.56, 1.67 | 2.19, 2.60 | 32.03, 32.28 Hz |
| off (2 x 16 x 600 s) | 5 = 5 + 0 | 0.21, 0.31 | **0.00, 0.00** | 27.52, 28.10 Hz |

Per-fly counts 2.406 vs 0.156, U 983, p 2.7e-11. Off makes no voluntary take-off in 19,200 fly-s, as in all six
round-5 batches (0 in 28,800 fly-s there), and the walking-GF medians reproduce round 5's 31.9 vs 27.2 Hz. The
drain-0.5 and drain-0.25 arms replicate it as well (default 18 and 13 take-offs vs off 4 and 3; p 0.0019 and 0.035),
so the take-off cost is not a starvation artefact either -- it is there in the arms where no fly starves. Note that
this `off` is `receptor_model=None` under the **shipped** gains, whereas round 5's `off` arm still carried the retired
GF damping, so these are not the same reference numbers.

### 5.6 The predictions of section 4

| | prediction | outcome |
|---|---|---|
| P1 | the tank empties at ~174 s and ~71 % of a 10-minute run is spent at 0 | **held**: median 176.4-179.8 s (first fly 173.5-175.5), 63.4-68.8 % at 0 |
| P2 | meals per fly stay below 1 in every arm | **held**: 0.062-0.375, highest at the 10-minute horizon |
| P3 | lowering the drain lowers meals and weakens food-seeking | **half held**: it does not lower meals (there is nothing to lower -- re-feeds are zero at every drain; and seed-matched, 3 of 4 drain-1.0 -> drain-0.5 transitions on file RAISE meals, 5.1 item 3), but it does lower the locomotor proxies: path 3.7 -> 3.46 / 3.32 -> 3.26 / 3.19 m per 300 s, search program armed 50.9 % -> 39 / 27 % -> 29 / 25 % |
| P4 | the contact rate is the same or higher after the tank empties | **not as stated**: the raw rate falls 3-6x, but the fall is the protocol's arrival transient, not starvation (5.3). The mechanistic half held exactly: walking speed *rises* 12-15 % once the tank is empty |

## 6. The answer

**Neither the horizon nor the drain makes meals countable, and the `cx` program's food-finding is limited by search --
specifically by the last few centimetres of the approach -- not by starvation.**

* **Countability.** No tested point reaches 1 meal per fly: 0.139 pooled at 5 minutes and the shipped drain,
  0.125-0.500 at 5 minutes with the drain halved or quartered (no fly starving at all), 0.219-0.312 at 10 minutes.
  `meals` is identically equal to `contacts` in all runs, so it is an arrival counter thresholded at 1.5 cm and at
  being on the ground, and arrivals accumulate sub-linearly. At the measured 53 % of the blind-sweep rate, one meal per
  fly needs about 33 minutes per fly -- but the 17.3 minutes is a blind-search baseline, not a geometric ceiling: a
  straight upwind walker covers the 40 cm to the capture disc in 0.49 min (section 1.2).
* **Whether the models then differ.** Not on meals, anywhere (p 0.40, 0.65, 0.69; Fisher 0.76) -- and meals cannot be
  made to answer, since at 0.3 meals per fly a 32-fly arm resolves only a factor ~3 in rate. On the closest approach,
  measured on the same rollouts, the default was nearer the fruit in 3 of 3 cells (2.80 vs 5.20 cm at 10 minutes,
  p 0.042; stratified p 0.016) -- but **two independent replicate cells did not reproduce it** (one reversed, one
  null; run-level 5 of 6, sign test p 0.219, Wilcoxon 0.156), and the measure is entangled with the take-off rate
  (Spearman(hops, min_distance) -0.25 to -0.75 within arm). It is a hypothesis for a three-replicate, ground-only
  batch, not a result (5.4).
* **Starvation or search.** Search. Removing starvation entirely leaves arrivals unchanged; an empty tank *raises*
  walking speed 12-15 % because hunger is the model's only motivational gain and it saturates; 4 of the default arm's
  10 arrivals happen on an empty tank; and the closest-approach median is 2.3-7.2 cm against a 3.8 cm capture radius,
  with 58 of 64 flies inside 10 cm, 41 inside 5 cm, while 21-23 % of a 10-minute rollout (29.14 % of a 5-minute one)
  is spent at surge-level odour. The open question is the
  terminal approach -- upwind anemotaxis past a source that emits no directional plume upwind of itself (section 2.2),
  with no concentration-change rule to stop or reverse -- which belongs to the search / small-object dynamics thread,
  not to the metabolism.

Limits of this round. One brain seed in the two 5-minute cells and two in the 10-minute cell (the skeptic's batch adds
a third and a second, 5.4); 16 flies per run; one fruit geometry (the fenced single apple) and one wind direction
throughout; the `off` arm is `receptor_model=None` under the shipped gains, so its take-off numbers are not round-5's
`off` numbers; the drain's direction on food-finding is unresolved rather than measured (5.1 item 3); and the whole
batch ran under heavy cluster contention (1.26-1.39 aggregate fly-s per wall second against the 6.29 of
docs/BATCH_SIM.md), which affects the wall-clock estimates and nothing else.

## 7. Design options and their measured consequences (recommendation only -- `body.py` is the owner's)

The structure of the problem is a trilemma, not a bad parameter value. The shipped drain does roughly what its own
docstring says: `drain_per_s = 1/300` makes a *full* tank last 300 s at rest and, at the room's measured 1.223 cm/s,
193.5 s = 3.2 minutes of walking against the documented "~4 min" (the docstring's number corresponds to 0.56 cm/s, and
the `cx` fly walks 2.2x that). The 5-minute assay is simply **1.7 tanks long and starts at 0.9**, and a 10-minute one
is 3.4 tanks. For the tank to outlast an assay of T seconds from 0.9 the drain must be <= 0.9 / T: drain-scale <= 0.58
for 5 minutes, <= 0.29 for 10. So exactly one of three things has to give -- the horizon, the drain, or the use of
energy and meals as scores. **The measurements of section 5 say it should be the scores** (option A below, with E when
meals must be the score): the drain arms bought a clean energy axis and no clear change in meals (seed-matched they
went up in 3 of 4 transitions, 5.1 item 3) while costing path length and search time, and the 10-minute horizon
bought 0.03 extra meals per fly in its second five minutes.

**(A) Keep the drain; change the score. This is the recommendation, and it is a hypothesis to be tested, not a
settled improvement.** Primary measures that every fly contributes to and that do not saturate: `min_distance_cm`
(the closest approach -- the measure that separated the models in this round's three cells at stratified p 0.016, and
then **failed to replicate** in two further cells, run-level 5 of 6, sign test p 0.219; 5.4), per-fly contact as a
binomial, and `first_contact_s` as a latency. Reference them against the blind-searcher null of section 1.2 (0.29
expected encounters per fly per 5 min, 0.58 per 10; P(find) 0.25 / 0.44), remembering that this null is a floor, not a
ceiling. Keep `meals` and `energy` in the JSON as descriptions of the episode, not as scores -- section 5.4 shows that
on identical rollouts they say p 0.76 where the closest approach says 0.016. Cost: no change to `body.py` at all, and
`scripts/probe_feeding_horizon.py` already records every column bar one. Three caveats measured here: `near_frac` at
2 / 5 cm is *not* a good approach measure (it is dominated by the few flies that arrive and sit); `path_m` includes
airborne frames, so its 2-4 % gap is not established as locomotor (5.4); and `min_distance_cm` as shipped is
co-determined by the take-off rate, so **the score to adopt is a ground-only closest approach** -- the one column the
probe does not yet record, a three-line change to the accumulator.

**(B) Slow the drain (a `body.py` default change). Measured: it buys a clean energy axis and costs path length and
search time; its effect on food-finding is unresolved.** What drain-scale 0.5 and 0.25 actually gave, at 5 minutes
(section 5.1):

| | drain 1.0 | drain 0.5 | drain 0.25 |
|---|---|---|---|
| 0.9 tank empties | 174 s (inside every assay) | 348 s | 697 s |
| flies at energy 0 by 300 s | 91.7-96.6 % | **none** | **none** |
| meals per fly at 300 s | 0.139 (26 batches) | 0.188 / 0.250, and 0.500 / 0.500 at brain seed 1 | 0.188 / 0.125 |
| path per 300 s | 3.67-3.75 m (other scripts' batches; this probe's own 300 s checkpoints give 3.557-3.699) | 3.46 / 3.32 | 3.26 / 3.19 |
| search program armed | 50.9 % of the run | 39 / 27 % | 29 / 25 % |

So it does remove the floor on energy, and it reduces the distance the fly walks by 5.3 % and then 11.0 % on this
probe's own instrument (6 % and 13 % against the other scripts' reference) and the time it spends searching by a fifth
and then two fifths. What it does to food-finding is **unresolved**. Three cautions the data support:

* The drain is **not behaviourally neutral.** `hunger` is the model's only motivational gain (section 1), so scaling
  the drain rescales the `cx` program's steering gain, its forward drive and the arming of its search program over the
  whole run -- a behaviour change expressed as a time constant. But the locomotor proxies and the outcome disagree:
  seed-matched, 3 of the 4 drain-1.0 -> drain-0.5 transitions on file **raise** meals (5.1 item 3), so "the measured
  direction is against food-finding" is not supported -- only "against path length and search arming" is.
* It changes the conditions of `benchmark.py`'s opt-in `hops` section, which runs this same protocol at energy 0.9 for
  **2.5 simulated minutes** (`scripts/benchmark.py:33-36, 720-760`). That window ends at 150 s, just short of the
  174 s at which the tank empties, so the section is the one existing protocol that is (barely) starvation-free -- and
  it sits on the edge: its flies run from energy 0.9 down to 0.125, a `hunger_scale` sweeping 0.19 -> 0.89 with a
  window mean of 0.54. At drain-scale 0.5 they would end at 0.513, gain 0.539 (= 0.1 + 0.9 x (1 - 0.5124)), window
  mean 0.36 -- a 1.5x change in the
  `cx` program's steering and forward gain across the whole scored window, so the round-5 voluntary / escape /
  walking-GF references would need re-deriving. (Those references themselves survive a drain change in *sign*: the
  take-off excess is present in the drain-0.5 and drain-0.25 arms too, section 5.5.)
* If a value is wanted rather than a scale: `walk_drain_per_m = 0.068` is what the docstring's own design ("a full tank
  lasts ~4 min of walking") implies **at the measured walking speed** (1/240 = 1/300 + w x 0.01223). That is a
  derivation from the stated design rather than a fit to a behaviour, but it only moves the 0.9-tank lifetime from
  174 s to 216 s -- still inside a 5-minute assay.

**(C) A longer horizon, no code change. Measured: 10 minutes is not enough and the curve is saturating.** Meals per
fly went 0.139 (5 min) to 0.219-0.312 (10 min), with half the arrivals in the first two minutes and 0.03 added by the
second five (section 5.2). The ceiling is geometric -- 12.7 m of path = 17.3 minutes of walking per expected encounter
for an ideal searcher -- and the model runs at 53 % of it, so one expected meal per fly needs ~33 minutes per fly =
31,700 fly-s per 16-fly replicate: 84 wall minutes per batch at the 6.29 aggregate fly-s per wall second of
docs/BATCH_SIM.md (6.3 hours at the 1.39 these jobs saw under contention), or 8.4 GPU-hours for three replicates of
two arms. Affordable, but it buys a measure whose own scatter is still a factor 5 and which is a 1.5 cm threshold on a
continuous quantity (section 5.1 item 2).

**(D) A starvation-free assay (a hunger clamp), no `body.py` change.** `--drain-scale 0 --energy E` holds every fly at
a chosen hunger for the whole run. Three properties follow from section 1's code reading:

* energy stops being a floor and hunger stops drifting, so the motivational gain is stationary and identical across
  flies, models and replicates -- the confound is removed rather than rescaled;
* **meals degenerates into an exact per-fly encounter counter**: with no drain and `E < resume_below`, a fly that
  reaches fruit feeds to satiety 0.95 and never falls back below 0.7, so `meals` is 1 if it found the food and 0 if
  not -- a clean binary outcome instead of a compound of arrival and a metabolic clock. (The price is that after its
  one meal the fly is sated at hunger 0.05 for the rest of the run and stops searching, so the measure is
  *first* encounter and post-meal behaviour is not comparable; `first_contact_s` is the matching latency.)
* hunger becomes a *swept independent variable* (`--energy 0.0 / 0.3 / 0.6 / 0.9` at drain 0) -- a dose-response on the
  lateral-horn odour gate, which is a better experiment than the present one.

Cost: the foraging economy is open-loop. This is an assay for ranking models, not a model of foraging, and it should be
labelled as one. (Verified to work as described: `--drain-scale 0 --energy 0.3`, 2 flies x 0.6 s on CPU, holds every
row at energy 0.300 with no row at zero and the before/after split reporting `n/a` for the empty half.)

**(E) Change the target, not the body.** The encounter rate, not the metabolism, is what makes meals rare, and the room
already contains a much richer table. `python scripts/probe_feeding_horizon.py --geometry --fruit all --minutes 5`:

| fruit set | sources | target groups | union capture area | summed capture width | ideal encounters / fly / 5 min |
|---|---:|---:|---:|---:|---:|
| `apple` (the current assay) | 1 | 1 | 44.8 cm^2 = 0.47 % of the table | 7.5 cm | 0.29 |
| `all` | 19 | 7 | 552.5 cm^2 = 5.76 % | 59.7 cm | **2.28** |

A 7.9x higher encounter rate -- enough for meals >= 1 per fly at the present drain and the present 5-minute horizon,
with no change to `body.py`. NOTES session 8's full-table rows (plain model 1 / 0 / 1 and, earlier, 3 meals per 5 min)
are consistent with that scale. The cost is scientific, not computational: the single apple was chosen *because* it
"separates strategies" (`flyverse/world.py:350`), and a table with seven target groups rewards wandering. The
recommendation is therefore to split the two jobs -- keep the fenced single apple for strategy comparisons, scored by
approach (A), and use the fenced full fruit set when meals must be the score. This was not measured this round; the
cheapest confirmation is one 16-fly 5-minute batch with `--fruit all`.

### 7.1 What the measurements say to do

Ranked by what this round measured, not by preference:

1. **(A) now, and as a hypothesis to be tested.** Score the fenced single-apple assay by a **ground-only** closest
   approach. It costs nothing, it is the only measure in this round that separated the two models at all, and it makes
   the existing 26 batches of round-4/5 data re-analysable (their `position` rows already gave section 2.1) -- but the
   separation itself did not replicate (5.4), and the shipped `min_distance_cm` is entangled with the take-off rate,
   so the first job of the round that adopts it is to test it, not to rely on it.
2. **(E) when meals must be the score.** One 16-fly 5-minute batch with `--fruit all` to confirm the 2.28 expected
   encounters per fly, then use that room for meal-scored work and keep the single apple for strategy comparisons.
   Commands to confirm it: `--geometry --fruit all --minutes 5` for the prediction, then
   `probe_feeding_horizon.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit all --fence ...`.
3. **(D) for the next model comparison.** `--drain-scale 0 --energy 0.0` gives a stationary, maximally hungry fly, a
   binary per-fly "found it", and a `first_contact_s` latency, with the drain removed as a confound rather than
   rescaled. Sweeping `--energy` turns hunger into an independent variable.
4. **(B) only if the owner wants the energy axis back for its own sake**, knowing it costs 5-13 % of the path length
   and a fifth to two fifths of the search time, and that `benchmark.py`'s `hops` references must be re-derived.
   Nothing in this round supports slowing the drain *in order to* make feeding measurable -- and nothing in it
   supports the converse either: seed-matched, the drain's effect on meals is unresolved (5.1 item 3).
5. **(C) as a supplement, not a fix.** ~33 minutes per fly x 3 replicates x 2 arms if a meal-scored answer at the
   present geometry is wanted anyway.

The one thing this round says should *not* happen is tuning the drain against a feeding difference between models:
there is no meals difference at any drain, so there is nothing to tune against.

### 7.2 What the next thread should pick up

The open mechanism is the terminal approach, and it is a search question: the plume is Gaussian **downwind only**
(section 2.2), upwind of the source only the isotropic 1/(1 + (r/3 cm)^2) near field remains, the odour-gated rule is
"steer upwind", and nothing uses a concentration change to stop or reverse -- so a fly that overshoots keeps being
driven away while still in odour. The measured signature is a closest-approach distribution whose median sits at
2.3-7.2 cm around a 3.8 cm capture radius, with 81 % of flies ending upwind of the source (section 2.1). Session 8
already measured the candidate fix as a module: klinotaxis (the antennal left-right contrast) closed the last
1.4-1.5 cm in 2 of 3 seeds where `cx` alone stalled at 8-17 cm.

**The round-2 design this leaves (R2-5), no `body.py` change anywhere in it.**

1. **A ground-only closest approach plus `first_contact_s`, three replicates of one cell.** Add a ground-only
   accumulator to `probe_feeding_horizon.py` (three lines) and score that, not `min_distance_cm`: it removes the
   take-off entanglement (5.4) and the airborne negative tail (5.1 item 2) in one change, and `first_contact_s` is
   already recorded. One cell, three brain seeds per arm, default vs off -- the replication the 5-of-6 run-level
   result needs.
2. **`--fruit all` against the 2.28 ideal-encounter prediction.** One 16-fly 5-minute batch confirms whether the
   7.9x richer table (section 7 option E) delivers >= 1 meal per fly; if it does, meal-scored comparisons move there
   and the single apple keeps the strategy comparisons.
3. **The hunger clamp as an exact binary.** `--drain-scale 0 --energy 0.0` holds every fly at maximum hunger, so
   `meals` becomes exactly "found it / did not" per fly with no metabolic clock in it (option D), the drain is removed
   as a confound rather than rescaled -- which is what the unresolved drain direction (5.1 item 3) calls for -- and
   sweeping `--energy` turns hunger into an independent variable.
4. **`cx+klinotaxis` against `cx`, scored by the same ground-only approach.** One 16-fly batch per arm; this is the
   mechanism test the section-2.2 reading points at, and it is the only arm here that changes behaviour rather than
   scoring.

## Files

New: `scripts/probe_feeding_horizon.py`, `docs/audits/feeding_horizon.md`.
Batch `feed-horiz-a9dfeb` (8 jobs, 0 failed, 116.6 min, all `device=cuda`): console log `out/feeding_horizon_cluster.log`;
`out/feedh_{default,off}_d1.0_10min_s{0,1}.{json,txt}`, `out/feedh_{default,off}_d{0.5,0.25}_5min_s0.{json,txt}`.
Aggregations: `out/feeding_horizon_report.log` (`--report`), `out/feeding_horizon_prior.log` (`--prior`, `--geometry`
for `apple` at 5 and 10 minutes and for `all`, `--odour-profile`).
Re-read, not re-run: `out/r4_sustain_{default,off}_*.json`, `out/r5_adopt_sustain_live_*.json`,
`out/r5_sustain_{default,off}_{live,nogf}_*.json`, `out/r5_skeptic_sustain_live_4.json`,
`out/sk5_sustain_{default,off}_live_1r.json` (26 batches, 416 fly-rollouts) and `out/r5_adopt_sustain_live_1.txt`.
Skeptic replication (5.1, 5.4, 2.1): `scripts/sk_feeding_horizon_verify.py` (CPU, sections A-F; run with
`--glob 'feedh_*.json,skfeed_*.json'`), batch `sk-feed-048583` console log `out/sk_feed_cluster.log`, and
`out/skfeed_{default,off}_d1.0_5min_s2.{json,txt}`, `out/skfeed_{default,off}_d0.5_5min_s1.{json,txt}`.
Nothing in `flyverse/` or in any other task's file was modified.
