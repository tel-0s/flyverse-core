# Flight foraging priority (engineering correction)

## 1. Problem and scope

The original optional `flight` instrument requested 100 Hz wing-power activity whenever energy
exceeded 0.05, except while feeding or sated. `hunger` scaled navigation but never interrupted
flight. In the original six 60 s room observations every fly spent 59.67 s airborne and none
fed (see `navigation_instruments.md` section 4). The owner independently observed flight
continuing during starvation. This is an instrument policy defect, not evidence for a missing
biological transmitter, receptor or energy cost.

The correction stays inside `FlightDrive`, through its existing neural extension boundary.
No parent synapse, body constant, raw path, preset default or metabolism law changes.
The new policy is **unverified engineering**, declared before the assays below:

- Search on the ground for 20 s before requesting voluntary flight. Feeding, satiety, low
  reserves and strong odor interrupt this interval. Actual touchdown starts a new interval.
- Power each bout for at most 8 s, including a failed takeoff attempt. A spontaneous native
  takeoff can be assisted for one bounded bout if no interruption applies.
- At energy <= 0.35 withdraw artificial wing drive. Re-arm only at energy >= 0.50.
  These are policy thresholds, not measurements of Drosophila hunger or endurance.
- Read the same biological LH berry/apple populations and normalization as `plume`, through
  this module's own frame-boundary inputs. Low-pass for 1 s, interrupt at normalized odor >=
  0.60 and clear at <= 0.25. Missing LH groups are explicitly absent in provenance; reduced
  graphs retain reserve and duration limits without a fabricated odor signal.
- Once interruption begins in the air, hold the artificial drive at zero until actual
  touchdown, even if the initiating cue disappears. Feeding also zeros drive immediately.

This withdraws the extra power and steering Poisson input, allowing the shipped physics to
bring the fly down. It is not a precision landing controller or a choice of landing site.
Native activity, including GF escape hops, remains possible. Odor is a neural cue, not food
distance; this policy neither locates fruit nor establishes successful plume navigation.
The existing 100 Hz servo target and gains are unchanged while a bout is active. The 100 Hz
equilibrium is from the body equations, not physiology. Namiki et al. 2022 (the original
flight source) does not supply the new bout, reserve or landing rules. No new literature
claim or physiological fit is made. Retire this policy when validated neural flight-state,
feeding-priority and descending/VNC mechanisms supply the same behavior.

All new per-row state stays on the module device, advances without host scalar reads and is
included in checkpoint, full reset and row reset. No mutable `plume` state is read, so
attachment order cannot alter the decisions. Direct users must continue supplying
`interoception`; the room and BatchSim already do so.

## 2. Protocol frozen before submission

One sequential house submission, generated with
`python scripts/navigation_batch.py --flight-priority --out out/flight_priority_v1`.
The generator hashes the source and this declaration before submission. GPU work is house
only; CPU regression tests are local. No parameter sweep, food-finding fit or preset admission.

1. Existing `navigation_probe.lifecycle`: 13 exact scheduler/checkpoint/reset/detach checks
   for each of `compass` and `compass_ring`, with plume/hunger/flight attached.
2. `flight_priority_probe.transitions`: 31 s, biological selected graph, B=3, seed 31,
   torch sparse, no optic or receptor model. Captured and checked eager paths compared
   exactly at 17 declared frames, all brain and instrument state. Body observations and
   LH frame-boundary rates are prescribed (0 or 40 Hz), not behavioral results. Rows are
   a healthy ground wait and timed bout, an odor-interrupted airborne bout, and depleted
   airborne reserves. Gates: capture actually used; no launch before 20 s; healthy bout
   begins and expires by 28.03 s; odor stops artificial power by 2 s and remains off until
   touchdown at 10 s; depleted row never requests power.
3. Two 60 s full MaleCNS B=6 rooms, neural seed 0, environment seeds 0-5, apple fruit,
   fenced table, no program, `compass plume hunger flight`, native CUDA kernels/events,
   automatic optic and captured sensory/brain frames. First arm starts on the table at
   (-0.15, 0.15, 0.75), energy 0.6. Second starts airborne at z=1 m, zero velocity,
   energy 0.1; otherwise identical settings. Gates: no artificial power at energy <=
   0.35, each commanded bout <= 8.02 s (float32 accumulation/frame tolerance), and all
   initially depleted airborne rows touch down within 5 s. Record airtime, powered time,
   feeding, first landing, native/voluntary hops, energy, trajectory and policy states.
   Food finding is descriptive, with no pass threshold. These are not independent
   physiological replicates, an adoption suite or a statistical causal comparison with
   the prior room runs. Endogenous escape hops do not fail the controller gates.

Neither the three-draw 29-check admission suite nor the 300 s room rate-half is replaced by these checks.

No performance speedup is predicted from timings on this shared machine. The controller
adds only per-row tensor arithmetic and small LH reductions; capture validity is tested.
No end-to-end UI performance claim follows from headless rooms.

## 3. Results

All frozen functional gates pass on B200. Implementation and frozen protocol:
`266d2741671fc79013cea30ee861f04f5c345791`. One house job, four sequential assays, no
retry or parameter change. The exact remote source archive matches all 79 frozen LF-normalized
hashes and that commit; each of the seven provenance records agrees on 52 shared runtime
source files. Whole MaleCNS compiled CSR MD5: `ef23cc27bea13be7f6a96f3c04fd3737`.

- All 26 existing exact multi-module lifecycle checks pass (13 per heading provider).
- All 17 additional captured/eager transition checkpoints agree exactly, with six policy
  gates passing. The prescribed healthy row waits, powers a bout and times out; the odor
  row interrupts and stays off until touchdown; the depleted row never powers flight.
- All six initially depleted airborne room rows land in 0.47-0.52 s. A ballistic fall of the 0.25 m drop at the
  body's 3.0 m/s^2 gravity takes 0.41 s, so the declared 5 s gate tests only that no power was applied.
  No artificial power is
  delivered at low reserves. Two later native escape hops occur, one in each of seeds 1
  and 5; there are no voluntary launches. Two rows feed for 15 s each; four exhaust their
  energy. This is not a successful starvation-recovery policy for every fly.
- In the normal-energy room arm, all six rows request zero artificial wing power; the neural
  odor interruption prevents the ground interval completing. With the shipped LH normalisation the odour gate
  reads 0.66-0.71 mean (max 0.95-0.99) even with a single apple 40 cm away and the fence on, so the 0.60/0.25
  hysteresis latches for essentially the whole episode: under this policy artificial flight is unreachable in any
  room containing fruit, not merely bounded. Powered flight is demonstrated only in the transition assay, where
  LH is prescribed at 0 Hz. One native escape hop lasts
  0.38 s. Seed 0 feeds for 8.06 s; the other five do not feed. This differs from the original
  sustained-flight observation, but it is not a statistical food-finding improvement claim.
  The bounded-bout room gate is vacuous here because there are no powered bouts; the
  non-vacuous timeout test is the controlled transition assay and CPU regression.

No run in these audits shows artificially powered flight and feeding in the same episode: the only
powered-flight rooms are the superseded navigation flight arm (59.67 s airborne, zero feeding in all six rows),
and `powered_s` is 0.00 in all 12 flight-priority room rows and all 12 plume room rows.

All per-environment rows, script-emitted without selection, are in
[`data/flight_priority/room_table.md`](data/flight_priority/room_table.md). Exact values, gates,
raw artifact hashes and source hashes are in [`summary.json`](data/flight_priority/summary.json);
the original declaration is [`predeclared.json`](data/flight_priority/predeclared.json).
The original navigation results remain unchanged in their audit.

Reproduce the derived files with:

```sh
python scripts/flight_priority_analyse.py --source out/flight_priority_v1 --out docs/audits/data/flight_priority
```

The ignored raw directory retains all four JSONs, the original declaration, console/launcher
logs and exact remote `source.tar.gz` (extracted under `source/`). Public derived files exclude
host/path-bearing provenance. No gain, timing, threshold or protocol was changed after results.

## 4. Review status

Author self-review: all state transitions use held internal observations and frame-boundary
neural rates; there are no world-position/fruit-distance inputs, body commands or synapse writes.
The odor population read avoids module-order dependence; the CPU regression crosses the odor
threshold with reversed attachment order. Capture keeps persistent state buffers, and the house
fixture exercises both decision boundaries and the existing checkpoint/row-reset/detach paths.
Failed takeoffs time out and touchdown resets the ground interval. The old flight checkpoint
schema is deliberately rejected rather than silently omitting the new state.

CPU: **500 passed / 19 skipped / 220 subtests**. Targeted navigation plus original golden:
24 passed / 10 subtests. Ruff and diff whitespace checks pass. Both working copies retain
cache MD5s `c50c598a708b5b373cbaffca7d6a9d82`, `ac131529cebf98decde58d0c227b7954`, and
`bf01d724acf2a1fec8fdb60ef8a9e066`. No full-brain eager/captured equality, runtime improvement,
precision landing or reliable food-finding claim follows from these checks.

The independent skeptic pass ran on 2026-09-17 (Opus, verdict mostly sound for this audit); its verdict line
and its eight claim lines are quoted verbatim in the Skeptic pass section below, and the corrections it required
are applied above. Nothing adopted into raw or made a preset default. The energy/bout/odor policy remains an
explicit unverified stand-in.


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
- flight_foraging_priority: mostly sound -- every number reproduces, the policy is provably frozen before the run, but the results omit that the policy makes artificial flight unreachable in any fruited room and that the 5 s landing gate is near-vacuous.
```

### Claims

```text
CLAIMS 1-8
1. Plume headline: confirmed; no geometric shortcut. plume_steering_analyse.py reproduces feeding 15.00 s x6 and first contacts 32.0/37.3/27.4/32.9/20.1/21.9 s; ring 5/6 (seed 3 = 0.00 s, nearest fruit -0.0012 m); scalar 21.6 s / 15 s / 0.8412110863728103. Initial body_heading at t=0.1 s = 5.03/90.00/-89.98/185.04/5.00/5.00 deg; start (-0.4996, 0.0500); energy 0.0996; fence: false. The walking goal uses only the antennal samples: over all 3600 logged samples, |wrap(goal-heading) - atan(200*atanh(contrast))| < 1e-5 for 99.64 % (the other 13 are brief airborne fallback samples). The instrument's only body-derived inputs are observe_wind(dL,dR), observe_smell(cL,cR) and observe_internal(airborne,feeding); sim.nearest_fruit() is probe logging and never reaches it; Air.antennae really samples the field 1 mm apart. But: 0.1 m / 0.001 m x 2 = a contrast gain of 200, and the observed mean |contrast| is only 0.26-0.57 % (p95 0.7-1.1 %), giving 24.5-41.0 deg mean goal offsets. It is effectively a sign-of-contrast turn read from the physical, pre-transduction field -- the model's own ORN law (1 + 150c/(c+0.5)) would render a 0.3 % difference ~0.1 Hz/ORN, well under the Poisson noise of a 0.25 s window (~0.8 Hz for ~500 ORNs/side).
2. Neural controls: all confirmed. From boundary.json: left/right actual DNa02 L-R = +9.3777 / -8.9450 Hz against targets +-8.9436; balanced +0.0610; feeding and sated added PFL3 input exactly 0; airborne -6.9708 / +7.0610; no-smell +6.8361; max |mean target error| 0.4341 Hz; all five gates True. Diagnostic mean |DNa02 L-R| = 0.281-0.687 Hz, heading_valid_fraction 1.0, mean |demand| 0.0573-0.0737, goal-error p95 up to 167.70 deg. Two undisclosed facts: the integral supplies more than half the PFL3 drive in every room row (mean |integral| 3.2-6.9 Hz vs mean |base| 2.9-6.6 Hz), and the balanced row needs -5.94 Hz of input to hold DNa02 at +0.06 Hz, i.e. the loop is cancelling the parent circuit's own L/R bias. On a 72-heading grid the comparator's normalised demand is bounded by 0.281, so the documented clip(...,-1,1) and the "up to 80 Hz" one-way scale are structurally unreachable (max 22.5 Hz) -- that, not just a small demand, is why the one-way bridge failed.
3. Navigation: all confirmed; the identity failure is under-reported. compass 0 fed x6; plume seed 4 = 8.08 s; flight 59.67 s airborne x6 at 99.84-100.52 Hz with zero feeding; recurrent +125.264/-124.959 deg/s, drift 0/+0.563 deg, strengths 0.922/0.894, EB=2.7 ~ 3e-10 deg/s -- "NOT calibrated" is the right reading. workload_identity = false. But the audit reports only "count differences up to 21 and 13", whereas profile.json -> identity shows at B=8/32 per-cell rate differences up to 39.0 / 44.9 Hz, voltages to 77.6 / 42.2 mV, conductances to 193 / 247 -- while poisson_p and drive are bit-identical, proving the inputs were equalised and the kernels are nondeterministic. Consequence: the 26 exact lifecycle checks (rtol=atol=0, torch-sparse 3-row subgraph) do not transfer, and no 60 s room in any of the three audits is reproducible -- re-running the same command with the same seeds gives a different trajectory and could give different feeding outcomes (INTERP 10.4 item 2). navigation_instruments.md hedges this; plume_steering.md does not mention it while quoting contacts to 0.1 s and an end energy to six decimals.
4. Flight priority: all confirmed; the policy is provably frozen. room_table.md regenerates byte-identically: depleted first landings 0.47/0.47/0.48/0.52/0.48/0.52 s, airborne 0.46-0.88 s, seeds 2 and 5 feed 15.00 s, seeds 0/1/3/4 end at 0.0000; normal-energy powered_s = 0.00 x6, one 0.38 s native hop (seed 0), seed 0 feeds 8.06 s. Freeze: predeclared.json stamped_utc 2026-09-16T02:00:28Z == the commit time of 266d274, artifacts fetched 19:05; sections 1-2 with the 20 s / 8 s / 0.35 / 0.50 / 0.60 / 0.25 policy unchanged since the freeze, and FlightDrive.parameters still carries exactly those values. Same pattern verified for the other two audits.
5. Vocabulary clean; sample sizes disclosed except one thing. No audit or handoff uses result / null / underpowered / undetermined in the verdict sense, and there is no p, z or significance anywhere. The 29-check suite and 300 s rate-half are explicitly disclaimed in navigation_instruments.md and plume_steering.md; flight_foraging_priority.md disclaims "adoption suite" and "statistical causal comparison" but never names them. The rooms are one draw each (one job, one B=6 batch, one brain seed -- 0, 0, 31), disclosed in all three. Undisclosed: the six "environment seeds" are near-repeats. world.make_room places apple, orange, banana and lime at seed-independent coordinates; only 9 grapes and 6 blueberries jitter; the start is identical and three rows share the same 5 deg initial heading. The lime is the nearest fruit at 22.6 cm and four of the six compass rows fed at it. Also: the plume rooms are a different task from the navigation/flight rooms (19 sources vs 1 apple, fence off vs on, start (-0.5,0.05) vs (-0.15,0.15), energy 0.1 vs 0.6, brain seed 31 vs 0) -- the only matched baseline for "all six feed" is the diagnostic (2/6 >= 1 s), which the audit does correctly use.
6. Labelling sound. NeuralInstrument.describe() hardcodes law="unverified", kind="stop-gap" for all four; compass_ring carries "calibrated": false. Gaudry 2013 (Nature 493:424-428) verified externally: flies turn toward the more strongly stimulated antenna via ~40 % greater ipsilateral release per ORN spike -- the audit uses it for the turn sign only. Mussells Pires 2024: the model form f(cos(H-H_pref) + d*cos(G-G_pref)) confirmed in the PMC and bioRxiv copies; the Methods with the numeric constants were not retrievable, so 29.23 / 2.17 / 0.63 / -0.7 and the two 12-element arrays are not independently verified. No sentence in any audit claims a recovered mechanism. Two minor labelling gaps: navigation_v1/recurrent.json carries the same bare-graph preset=raw, instruments=[] root label the audit discloses only for lifecycle.json; and navigation_instruments.md section 2's plume paragraph still states the pre-correction law and omits smell() concentrations from its input list.
7. Provenance sound. Every stamped_utc equals its declaration commit's timestamp and precedes its own results; navigation_v1's recorded commit (048a2fab) is the pre-freeze parent exactly as the audit explains, and its 77 hashes verify against 3cac3cc plus the shipped archive. verify_frozen passed for all five freezes. Runtime records: 10 (plume) / 7 (flight) / 22 (navigation), each asserting preset == "instrumented" with 4 instruments and 52 shared source files; compiled CSR ef23cc27bea13be7f6a96f3c04fd3737 throughout. The three cache MD5s verified locally. FlyBrain.smell still runs the original ORN encoding and forwards concentrations in parallel, so the raw path is untouched. The lifecycle artifacts store only check labels; "26 exact checks pass" rests on the job's exit status plus the in-probe assert_close(rtol=0, atol=0), not on stored residuals.
8. Things the audits do not say that a reader needs. (a) The instrument/connectome split: the goal, the 200x gradient and the integral are the instrument's, and because the integral dominates, the biological PFL3 -> DNa02 stage is being servo-inverted to a setpoint the instrument computes rather than consulted for its gain; the clearly connectome-supplied stage is DNa02 -> motor readout -> yaw. The instrument also writes up to 60 Hz to DNp09, so the forward run is instrument-commanded too (path 0.53 m compass-only -> 0.70 m with plume in the matched rooms). (b) There is no compass-with-goal-but-without-feedback arm; mean |demand| did not rise between diagnostic and validation (0.057-0.074 -> 0.036-0.083) while mean |yaw| tripled (0.038-0.067 -> 0.106-0.245 rad/s), so the feedback bridge is the load-bearing change and a goal-only arm would probably still show 0.28-0.69 Hz DNa02 differences. Untested. (c) "Feeds" = nearest_fruit_distance < 0.015 m while not airborne, for any of the 19 fruit; every fruit is an odour source. The 15 s is the shipped meal length (feed_per_s = 1/15 to the 0.95 satiety threshold) -- a ceiling, not a graded measure. (d) All six validation rows (and all six ring rows, all six diagnostic rows, and the scalar demo) hit energy exactly 0.0000 at t = 18.0-18.2 s, i.e. before every first contact. (e) Consequently the hunger gain is pinned at 1.0 from 18 s on and never leaves [0.90, 1.0], so these rooms do not test hunger modulation at all.
```
