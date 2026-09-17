# Plume steering and source approach (Astra, 2026-09-16)

## 1. Diagnostic declaration

The owner reports nearly straight walking and fruit missed while starving with
`compass plume hunger flight`. The previous flight-priority correction prevents perpetual
powered flight but explicitly does not establish food finding. Two possible gaps need
separating: the goal is pure upwind during sustained odor, and its synthetic PFL3 input may
not produce an adequate turn through the native descending circuit.

Before changing the controller, one house job runs `scripts/plume_steering_probe.py` with
`compass`, then `compass_ring`. Each is a full MaleCNS B=6 room, neural seed 31, environment
seeds 0-5, all fruit, the default start, no fence, initial energy 0.1. Initial headings are
5, 90, -90, 185, 5, 5 degrees, to include both strong imposed heading errors and the user's
near-upwind default. Sixty simulated seconds, samples every 0.1 s: antenna concentrations,
LH odor gate, heading strength and phase, goal, signed demand, PFL3 L/R, DNa02 L/R, actual
yaw/heading, energy, distance (measurement only) and feeding. No gain sweep, statistical
success claim or admission gate; these traces diagnose where the signal stops. Parameters
and this source are frozen before submission. Raw and body constants are unchanged.

`world.make_room` places the apple, orange, banana and lime at seed-independent coordinates and the start is
identical in every row, so the six seeds differ only in grape/blueberry jitter, plume puff phase, brain RNG stream
and the imposed initial heading (three rows share 5 deg). The lime is the nearest fruit at 22.6 cm and four of the
six compass rows fed there.

## 2. Findings

Diagnostic source `f687f96`, one house submission, both heading providers. Every sampled
heading is above the existing 0.6 confidence threshold: losing the compass is not the
immediate problem here. For `compass`, mean absolute DNa02 L-R is only 0.281-0.687 Hz,
despite goal-error p95 as large as 167.70 degrees. The default-heading rows (seeds 0,4,5)
span 38.40, 34.94 and 21.66 degrees over 60 s. Mean absolute synthetic demand is
0.057-0.074, so the one-way 80 Hz input scale often drives PFL3 too weakly to recruit DNa02.
The native route exists: PFL3 soma-L -> DNa02-R sums to 380 compiled weight units,
PFL3 soma-R -> DNa02-L to 356, with both same-side blocks zero (unscaled `W[post,pre]`).

Three of six compass rows feed, including one for only 0.76 s; three ring rows feed. These
are diagnostic traces, not a rate-half comparison. The source goal also stays upwind whenever
odor remains high, so approaching an off-axis fruit is not supplied by that policy.

## 3. Declared correction (unverified engineering)

Keep the published PFL3 comparator, neural heading/LH inputs, hunger gain and parent graph.
Make two explicit changes inside the existing optional `plume` instrument:

1. Close the output bridge using DNa02 L-R feedback. Its target is `40*turn` Hz, which cancels
   the original comparator's division by 40: the desired descending difference is the ideal
   comparator difference times odor/hunger/confidence gating. Signed PFL3 input is
   `clip(80*turn + integral, -80, 80)` Hz; integral adds `5*dt*(target - measured DNa02 L-R)`,
   bounded at +/-80 Hz, with conditional anti-windup. Positive input still goes to PFL3-R,
   negative to PFL3-L. No DNa02 rate, body command or weight is overwritten. Disabled
   navigation/feeding/satiety clears the integral and both inputs. The gain 5/s and target
   bridge are declared engineering, not a DNa02 physiological law or a fitted constant.
2. For walking, use bilateral physical smell samples when available. Sum the supplied
   glomerular concentrations independently at each antenna; smooth `(L-R)/(L+R)` for 0.25 s.
   Infer a lateral log-concentration gradient as `2*atanh(contrast)/0.001 m`, clamping contrast
   to +/-0.999 before `atanh`. Set the local goal to `heading + atan(0.1 m * gradient)`.
   The 1 mm baseline is the room's current antenna spacing; the 0.1 m response length and
   filter are unverified choices. This chooses a turn toward stronger odor, not a world
   coordinate or a remembered fruit position. The neural LH/heading gate still applies.
   Airborne or absent/zero physical smell retains the original wind/entry-memory policy.
   0.1 m / 0.001 m gives a contrast gain of 200; at the 0.26-0.57 % bilateral contrast observed in the rooms this
   yields 25-41 deg mean goal offsets (p95 54-66 deg), so the law behaves as a near-sign-of-contrast turn. The
   contrast is read from the physical concentration field, so it is noise-free and would not survive the model's
   own ORN transduction (1 + 150c/(c+0.5) plus Poisson spiking).

The qualitative bilateral turn sign is motivated by [Gaudry et al. 2013, Figure 1](https://doi.org/10.1038/nature11747),
which demonstrates odor lateralization and turning toward stronger antennal input in walking
adult flies. It does **not** validate this concentration sum, sensitivity, gradient estimator,
response length, filter or controller. In particular, its ORN asymmetry measurements are not
concentration thresholds for this simulation. Mixed chemical concentrations are treated as
one attractive cue only in this stand-in; there is no claimed general valence model.

`FlyBrain.smell()` forwards its existing physical inputs to an explicitly named receiver,
analogously to the existing held wind/yaw inputs. Empty instrumented and raw receive no new
module. Concentrations are validated before mutation; new per-row state checkpoints and resets
through the extension scheduler. The full computation stays on the module device without
host scalar reads. No motor/body defaults, raw weights or sensory transduction laws change.
Removing the instrument removes the correction. Older plume checkpoints reject the changed
state schema; start a fresh episode.

## 4. Correction validation frozen before submission

One sequential house job, `navigation_batch.py --plume-validation`, with all constants above
fixed before results. No sweep, data-driven gain fit or preset adoption. Save all failures.

- CPU: signed bilateral responses against opposing wind, concentration-scale invariance,
  airborne fallback, hunger/feeding, thresholded-plant rate tracking, reversed attachment
  order, invalid inputs, checkpoint/row reset, full CPU and original MaleCNS golden.
- Existing 26 exact CUDA lifecycle checks, extended with changing held smell samples for
  both heading providers. Captured and checked eager states must agree, including after
  resets/checkpoints and detach.
- Full MaleCNS native CUDA B=8, optic off, seed 31, LH forced at 40 Hz for 30 s; read the last
  10 s. Rows: left/right walking gradients opposed by wind; equal antennas; feeding; sated;
  airborne wind right/left despite opposed gradients; no smell with left wind. Concentrations
  1.02 versus 1.0, energy 0.1 except flags. Gates: actual DNa02 L-R >2 / <-2 Hz for left/right;
  mean tracking error <2 Hz in the five active directional rows; balanced |difference| <2 Hz;
  zero added PFL3 input while feeding/sated; airborne and absent-smell rows follow wind.
- Repeat the diagnostic's 60 s full rooms, same seeds, initial headings/energy and defaults,
  now also recording target DNa02 difference, PFL3 input and bilateral contrast. The user's
  `compass` configuration must have at least four of six rows feed for >=1 s as a functional
  source-approach check. Apply the same criterion descriptively to `compass_ring`. This is
  not a broad success-rate estimate, three-draw admission study, or proof of native navigation.

Results are recorded in section 6. The independent skeptic pass (Opus, 2026-09-17) is complete and quoted at
the end of this audit. No new physiological claim or adoption.

## 5. Scalar demo integration declaration

After the compass B=6 room passes with six feeding rows, separately check the user's actual
scalar simulator and backend. No controller parameter/code change: `plume_steering_probe.py
--mode scalar` uses `room_demo.Sim`, seed 0, all fruit/default start, initial energy 0.1,
`compass plume hunger flight`, native CUDA events, warp sparse and captured frames, for 60 s.
Gate: >=1 s feeding; record position, heading, yaw, energy, distance and contact. One additional
house submission then performs a 1 s headless UI launch with brain map and screenshot. This
checks scalar integration and rendering availability, not interactive FPS or generalization.
Freeze both commands before submission; all controller laws remain exactly those of section 3.

## 6. Results and reproduction

All declared correction gates pass. Correction source/protocol `32acee2`; scalar-only harness
`6863f39`. The three submissions (diagnostic, correction validation, scalar/UI integration)
each succeeded on their first attempt. The controller's constants stayed fixed throughout
validation and the scalar follow-up. All GPU work was on the house B200.

- Twenty-six exact CUDA lifecycle checks pass, including both heading providers, changing
  held smell/wind/internal inputs, graph replay, pulses, reset, checkpoint and detach.
- The eight-row neural assay passes all five gates. Left/right walking cues give DNa02 L-R
  **+9.378 / -8.945 Hz**, against targets +8.944 / -8.944 Hz. The balanced row is +0.061 Hz;
  feeding/sated added PFL3 input is zero. Airborne wind rows give -6.971 / +7.061 Hz; the
  absent-smell wind row gives +6.836 Hz. The largest absolute *mean* target error in the
  five directional rows is 0.434 Hz. This is not a bound on instantaneous fluctuations.
- All **six compass rows feed for 15 s** in the 60 s room run. 15 s is the shipped meal length
  (`Metabolism.feed_per_s` = 1/15 up to the 0.95 satiety threshold), so it is a ceiling reached by any row that
  makes contact, not a graded measure. First contacts occur at
  32.0, 37.3, 27.4, 32.9, 20.1, 21.9 s (environment seeds 0-5, 0.1 s sampling).
  The original diagnostic has two rows above the predeclared >=1 s feeding threshold,
  plus one 0.76 s contact. These seeds/configurations are observations, not a general
  success-rate estimate or an independent causal separation of the two corrections.
- The `compass_ring` comparison has **five of six** rows feed for >=1 s. Seed 3 remains a
  failure despite passing close to fruit; its complete trajectory and zero feeding are
  retained. The alternative ring has not acquired a calibrated biological angular gain.
- The actual single-fly `room_demo.Sim`, seed 0, native warp sparse, reaches first feeding
  at **21.6 s**, feeds **15 s**, and ends at energy **0.841211**. The separate headless UI
  launch with brain map succeeds and its screenshot was inspected. No interactive FPS
  claim follows from this check.

The starting energy is only 0.1 and every row reaches zero energy at 18.0-18.2 s, before any first contact
(20.1-37.3 s); the hunger gain is therefore pinned at 1.0 for most of each episode and never leaves [0.90, 1.0],
so these rooms do not test metabolic modulation.
**Withdrawn:** "The starting energy is only 0.1 and some trajectories reach zero before finding food." -- it is
every row, not some, and every one of them reaches zero before its own first contact.
The shipped metabolism does not model death at zero; these results establish contact and
refeeding in that model, not survival or a general starvation-recovery guarantee. Goal
selection remains a simplified attractive-gradient policy. Mixed odor valence, noisy or
adversarial plumes, alternate starts, repeated neural draws, wall/ceiling walking and long
episodes remain validation gaps. Neither the admission suite nor the 300 s room rate-half
is replaced by these functional checks. No default, native mechanism or physiological gain
is adopted, and no full-brain eager/captured equality or speedup is claimed.

`world.make_room` places the apple, orange, banana and lime at seed-independent coordinates and the start is
identical in every row, so the six seeds differ only in grape/blueberry jitter, plume puff phase, brain RNG stream
and the imposed initial heading (three rows share 5 deg). The lime is the nearest fruit at 22.6 cm and four of the
six compass rows fed there.

The native event-driven CUDA path is not bit-reproducible (the exact-workload gate in
[navigation_instruments.md](navigation_instruments.md) fails with per-cell rate differences up to 44.9 Hz under
provably identical inputs), so these first-contact times and the 0.841211 end energy are one draw and do not pin a
number on re-running (INTERP 10.4 item 2). The analyser re-derives the tables from the stored records; it does not
re-run the rooms.

In the 60 s rooms the integral supplies more than half the PFL3 input (mean |integral| 3.2-6.9 Hz vs mean |base|
2.9-6.6 Hz), and the balanced control row needs -5.94 Hz to hold DNa02 L-R at +0.061 Hz, so the loop inverts the
parent PFL3 -> DNa02 gain rather than relying on it. The comparator's normalised demand is bounded by 0.281, so
the [-1,1] clip and the 80 Hz one-way scale are unreachable (max 22.5 Hz).

No run in these audits shows artificially powered flight and feeding in the same episode: the only powered-flight
rooms are the superseded navigation flight arm (59.67 s airborne, zero feeding in all six rows), and `powered_s`
is 0.00 in all 12 flight-priority room rows and all 12 plume room rows.

Every per-environment row is script-emitted in [rooms.md](data/plume_steering/rooms.md).
Exact numbers, all neural controls, gates and hashes are in
[summary.json](data/plume_steering/summary.json). The [trajectory figure](data/plume_steering/trajectories.svg)
shows the full 60 s paths with first feeding marked; it is a top-down projection, including
walks that leave the table. Declaration snapshots are beside those files.

```sh
python scripts/plume_steering_analyse.py
```

The analyzer verifies each of three 79-file freezes against its committed snapshot and the
exact remote archive, then checks 52 shared source hashes in each of ten runtime provenance
records. It also asserts identical controller bytes in validation and scalar submissions.
Raw results, logs, screenshot and `source.tar.gz` archives (extracted under `source/`) stay
ignored in `out/plume_diagnostic_v1`, `out/plume_validation_v1`, `out/plume_scalar_v1`.
Public derived artifacts omit host/path-bearing provenance. Keep those raw directories.

## 7. Author self-review (the independent skeptic pass is the section after this one)

CPU suite: **503 passed / 19 skipped / 220 subtests**. Targeted navigation/golden:
27 passed / 10 subtests. New/changed instrument and assay files pass Ruff; diff whitespace
check passes. Original MaleCNS golden and both worktrees' three cache MD5s are unchanged:
`c50c598a708b5b373cbaffca7d6a9d82`, `ac131529cebf98decde58d0c227b7954`,
`bf01d724acf2a1fec8fdb60ef8a9e066`. Compiled CSR: `ef23cc27bea13be7f6a96f3c04fd3737`.

The new receiver sees only existing physical smell arguments, never fruit/world coordinates.
All neural reads are frame-boundary samples; the only sibling state read is `hunger.level`, which is written
solely by `observe_internal` and never during a module step, so execution order cannot change the result.
**Withdrawn:** "All neural reads are frame-boundary samples, so no mutable sibling state is read during a module
step." -- `PlumeNavigation.step` and `FlightDrive.step` both read `self.hunger.level`.
Only existing PFL3/DNp09 Poisson targets are written; feedback does not overwrite
DNa02 or body commands. Integrator limits, anti-windup and disabled-state clearing are explicit.
All added state participates in checkpoint and row reset; old incompatible plume checkpoints
fail explicitly. Capture uses device tensor arithmetic and persistent held input buffers.
Finite/nonnegative/shape validation occurs before either antenna state or neural input changes.
The raw smell path retains the same sensory calculation; raw/golden checks pass.

This is the author's review, not the independent skeptic pass. That pass ran on 2026-09-17 (Opus, verdict
mostly sound for this audit); its verdict line and its eight claim lines are quoted verbatim in the section below,
and the corrections it required are applied above.


## Skeptic pass (independent, Opus, 2026-09-17)

An independent skeptic pass ran on 2026-09-17 (Opus, CPU only, no cluster job, nothing adopted). Its verdict line
and its claim lines are quoted verbatim below. The CORRECTIONS REQUIRED list is applied in place in the sections
above; where a correction replaced a sentence that stated a finding, the original sentence stays in the record
marked **Withdrawn:** (INTERP 10.4 rule 29 iii). The pass's NOT CHECKED list is recorded verbatim with this
round's entry in [receptor_verification.md](receptor_verification.md).

One pass covered the three navigation audits and carried one verdict line per audit; the line for
this audit is quoted below, and all eight claims are quoted in each of the three.

### Verdict

```text
VERDICT (one per audit):
- plume_steering: mostly sound -- every number reproduces exactly; one sentence is factually wrong ("some trajectories reach zero" -> all of them), and four material characterisations a reader needs are missing.
```

### Claims

```text
CLAIMS 1-8
1. Plume headline: confirmed; no geometric shortcut. plume_steering_analyse.py reproduces feeding 15.00 s x6 and first contacts 32.0/37.3/27.4/32.9/20.1/21.9 s; ring 5/6 (seed 3 = 0.00 s, nearest fruit -0.0012 m); scalar 21.6 s / 15 s / 0.8412110863728103. Initial body_heading at t=0.1 s = 5.03/90.00/-89.98/185.04/5.00/5.00 deg; start (-0.4996, 0.0500); energy 0.0996; fence: false. The walking goal uses only the antennal samples: over all 3600 logged samples, |wrap(goal-heading) - atan(200*atanh(contrast))| < 1e-5 for 99.64 % (the other 13 are brief airborne fallback samples). The instrument's only body-derived inputs are observe_wind(dL,dR), observe_smell(cL,cR) and observe_internal(airborne,feeding); sim.nearest_fruit() is probe logging and never reaches it; Air.antennae really samples the field 1 mm apart. But: 0.1 m / 0.001 m x 2 = a contrast gain of 200, and the observed mean |contrast| is only 0.26-0.57 % (p95 0.7-1.1 %), giving 24.5-41.0 deg mean goal offsets. It is effectively a sign-of-contrast turn read from the physical, pre-transduction field -- the model's own ORN law (1 + 150c/(c+0.5)) would render a 0.3 % difference ~0.1 Hz/ORN, well under the Poisson noise of a 0.25 s window (~0.8 Hz for ~500 ORNs/side).
2. Neural controls: all confirmed. From boundary.json: left/right actual DNa02 L-R = +9.3777 / -8.9450 Hz against targets +-8.9436; balanced +0.0610; feeding and sated added PFL3 input exactly 0; airborne -6.9708 / +7.0610; no-smell +6.8361; max |mean target error| 0.4341 Hz; all five gates True. Diagnostic mean |DNa02 L-R| = 0.281-0.687 Hz, heading_valid_fraction 1.0, mean |demand| 0.0573-0.0737, goal-error p95 up to 167.70 deg. Two undisclosed facts: the integral supplies more than half the PFL3 drive in every room row (mean |integral| 3.2-6.9 Hz vs mean |base| 2.9-6.6 Hz), and the balanced row needs -5.94 Hz of input to hold DNa02 at +0.06 Hz, i.e. the loop is cancelling the parent circuit's own L/R bias. On a 72-heading grid the comparator's normalised demand is bounded by 0.281, so the documented clip(...,-1,1) and the "up to 80 Hz" one-way scale are structurally unreachable (max 22.5 Hz) -- that, not just a small demand, is why the one-way bridge failed.
3. Navigation: all confirmed; the identity failure is under-reported. compass 0 fed x6; plume seed 4 = 8.08 s; flight 59.67 s airborne x6 at 99.84-100.52 Hz with zero feeding; recurrent +125.264/-124.959 deg/s, drift 0/+0.563 deg, strengths 0.922/0.894, EB=2.7 ~ 3e-10 deg/s -- "NOT calibrated" is the right reading. workload_identity = false. But the audit reports only "count differences up to 21 and 13", whereas profile.json -> identity shows at B=8/32 per-cell rate differences up to 39.0 / 44.9 Hz, voltages to 77.6 / 42.2 mV, conductances to 193 / 247 -- while poisson_p and drive are bit-identical, proving the inputs were equalised and the kernels are nondeterministic. Consequence: the 26 exact lifecycle checks (rtol=atol=0, torch-sparse 3-row subgraph) do not transfer, and no 60 s room in any of the three audits is reproducible -- re-running the same command with the same seeds gives a different trajectory and could give different feeding outcomes (INTERP 10.4 item 2). navigation_instruments.md hedges this; plume_steering.md does not mention it while quoting contacts to 0.1 s and an end energy to six decimals.
4. Flight priority: all confirmed; the policy is provably frozen. room_table.md regenerates byte-identically: depleted first landings 0.47/0.47/0.48/0.52/0.48/0.52 s, airborne 0.46-0.88 s, seeds 2 and 5 feed 15.00 s, seeds 0/1/3/4 end at 0.0000; normal-energy powered_s = 0.00 x6, one 0.38 s native hop (seed 0), seed 0 feeds 8.06 s. Freeze: predeclared.json stamped_utc 2026-09-16T02:00:28Z == the commit time of be1d893, artifacts fetched 19:05; sections 1-2 with the 20 s / 8 s / 0.35 / 0.50 / 0.60 / 0.25 policy unchanged since the freeze, and FlightDrive.parameters still carries exactly those values. Same pattern verified for the other two audits.
5. Vocabulary clean; sample sizes disclosed except one thing. No audit or handoff uses result / null / underpowered / undetermined in the verdict sense, and there is no p, z or significance anywhere. The 29-check suite and 300 s rate-half are explicitly disclaimed in navigation_instruments.md and plume_steering.md; flight_foraging_priority.md disclaims "adoption suite" and "statistical causal comparison" but never names them. The rooms are one draw each (one job, one B=6 batch, one brain seed -- 0, 0, 31), disclosed in all three. Undisclosed: the six "environment seeds" are near-repeats. world.make_room places apple, orange, banana and lime at seed-independent coordinates; only 9 grapes and 6 blueberries jitter; the start is identical and three rows share the same 5 deg initial heading. The lime is the nearest fruit at 22.6 cm and four of the six compass rows fed at it. Also: the plume rooms are a different task from the navigation/flight rooms (19 sources vs 1 apple, fence off vs on, start (-0.5,0.05) vs (-0.15,0.15), energy 0.1 vs 0.6, brain seed 31 vs 0) -- the only matched baseline for "all six feed" is the diagnostic (2/6 >= 1 s), which the audit does correctly use.
6. Labelling sound. NeuralInstrument.describe() hardcodes law="unverified", kind="stop-gap" for all four; compass_ring carries "calibrated": false. Gaudry 2013 (Nature 493:424-428) verified externally: flies turn toward the more strongly stimulated antenna via ~40 % greater ipsilateral release per ORN spike -- the audit uses it for the turn sign only. Mussells Pires 2024: the model form f(cos(H-H_pref) + d*cos(G-G_pref)) confirmed in the PMC and bioRxiv copies; the Methods with the numeric constants were not retrievable, so 29.23 / 2.17 / 0.63 / -0.7 and the two 12-element arrays are not independently verified. No sentence in any audit claims a recovered mechanism. Two minor labelling gaps: navigation_v1/recurrent.json carries the same bare-graph preset=raw, instruments=[] root label the audit discloses only for lifecycle.json; and navigation_instruments.md section 2's plume paragraph still states the pre-correction law and omits smell() concentrations from its input list.
7. Provenance sound. Every stamped_utc equals its declaration commit's timestamp and precedes its own results; navigation_v1's recorded commit (c1fa446a) is the pre-freeze parent exactly as the audit explains, and its 77 hashes verify against bbb0e2f plus the shipped archive. verify_frozen passed for all five freezes. Runtime records: 10 (plume) / 7 (flight) / 22 (navigation), each asserting preset == "instrumented" with 4 instruments and 52 shared source files; compiled CSR ef23cc27bea13be7f6a96f3c04fd3737 throughout. The three cache MD5s verified locally. FlyBrain.smell still runs the original ORN encoding and forwards concentrations in parallel, so the raw path is untouched. The lifecycle artifacts store only check labels; "26 exact checks pass" rests on the job's exit status plus the in-probe assert_close(rtol=0, atol=0), not on stored residuals.
8. Things the audits do not say that a reader needs. (a) The instrument/connectome split: the goal, the 200x gradient and the integral are the instrument's, and because the integral dominates, the biological PFL3 -> DNa02 stage is being servo-inverted to a setpoint the instrument computes rather than consulted for its gain; the clearly connectome-supplied stage is DNa02 -> motor readout -> yaw. The instrument also writes up to 60 Hz to DNp09, so the forward run is instrument-commanded too (path 0.53 m compass-only -> 0.70 m with plume in the matched rooms). (b) There is no compass-with-goal-but-without-feedback arm; mean |demand| did not rise between diagnostic and validation (0.057-0.074 -> 0.036-0.083) while mean |yaw| tripled (0.038-0.067 -> 0.106-0.245 rad/s), so the feedback bridge is the load-bearing change and a goal-only arm would probably still show 0.28-0.69 Hz DNa02 differences. Untested. (c) "Feeds" = nearest_fruit_distance < 0.015 m while not airborne, for any of the 19 fruit; every fruit is an odour source. The 15 s is the shipped meal length (feed_per_s = 1/15 to the 0.95 satiety threshold) -- a ceiling, not a graded measure. (d) All six validation rows (and all six ring rows, all six diagnostic rows, and the scalar demo) hit energy exactly 0.0000 at t = 18.0-18.2 s, i.e. before every first contact. (e) Consequently the hunger gain is pinned at 1.0 from 18 s on and never leaves [0.90, 1.0], so these rooms do not test hunger modulation at all.
```
