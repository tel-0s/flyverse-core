# Results: what the model was asked, and what came back

One place to read what every round found. Each behaviour gets a row in the table and an entry below it: the
question, the arms that were run, the verdict with its numbers, what the verdict means, the audit that carries
the numbers, and the independent skeptic's verdict line quoted verbatim.

**Every number here is a number from an audit.** Nothing is computed, combined or rephrased into a new
quantity in this file: a sentence that quantifies something names the audit and section it comes from, and a
quoted skeptic is quoted as written in
[audits/receptor_verification.md](audits/receptor_verification.md), this project's record of the independent
passes.

**Reading the verdicts.** Comparisons between arms use four words and only four: `result` (the difference is
called under the declared rule), `null` (it is not), `underpowered` (the design could not have called it) and
`undetermined` (the reference arm has no scatter, so the statistic is undefined). The 29-check regression
suite has its own three statuses, `PASS` / `FAIL` / `KNOWN GAP`, which are statements about a check against a
bound and not about a comparison between arms.

**What `raw` is.** `FlyBrain()` is `FlyBrain(preset="raw")`: the unchanged connectome plus the documented
physiological assumptions of [REPRODUCIBILITY.md](REPRODUCIBILITY.md) section 1. `raw` is byte-identical on
every path (`tests/test_bit_identity.py`) and is the default. **Nothing in any round below was adopted into
`raw`.** The `instrumented` preset adds labelled stand-ins; every one of them declares a `kind`, the named
gap it fills, its law (`unverified`) or its source, a removal condition, and a `replaces` field
(`"input"` / `"computation"` / `"configuration"`) -- [PRESETS_SPEC.md](PRESETS_SPEC.md) sections 3 and 5,
inventory in [INSTRUMENTS.md](INSTRUMENTS.md). An instrument result is never attributed to the wiring.

**The milestone, in the two sentences that are licensed.** Under the `instrumented` preset the fly turns and
finds food on the **physical** antennal contrast -- the shipped `plume` instrument fed in 6 of 6 rooms --
while substituting the model's own ORN population rates at the same gain gives 1 of 6 on a cue uncorrelated
with the true lateral contrast; and **no episode shows artificially powered flight and feeding together**
([NOTES.md](NOTES.md) "Session 14, continued: the transduced-plume rooms";
[audits/plume_transduced.md](audits/plume_transduced.md) section 7;
[audits/navigation_instruments.md](audits/navigation_instruments.md) section 2, correction 13).

---

## A. Behaviours and status

`status (raw)` is what the plain model does; "n/a (instrument)" means the behaviour exists only under
`instrumented` and is not a property of the wiring.

| behaviour | status (raw) | audit(s) | independent skeptic verdict |
|---|---|---|---|
| Looming escape (giant fibre) | suite `PASS`, 9 of 9 runs | [guard_suites_r3.md](audits/guard_suites_r3.md) 1-4, [anti_runaway.md](audits/anti_runaway.md) round 7 | `verify:guards  --  verdict: **mostly sound**` |
| Direction selectivity (T4 / T5) | suite `PASS`, 9 of 9 runs | [guard_suites_r3.md](audits/guard_suites_r3.md) 1, [optic_measures.md](audits/optic_measures.md) | `verify:optic-audit  --  verdict: **mostly sound**` |
| Wind direction (DNp18 / DNp33) | suite `PASS`, 9 of 9 runs | [guard_suites_r3.md](audits/guard_suites_r3.md) 1 | `verify:guards  --  verdict: **mostly sound**` |
| Taste, and bitter suppression | suite `PASS`, 9 of 9 runs; knife-edge readout | [guard_suites_r3.md](audits/guard_suites_r3.md) 1, [monoamine_slow_term.md](audits/monoamine_slow_term.md) 0, [instrumented_suite.md](audits/instrumented_suite.md) 2 | `verify:monoamines  --  verdict: **mostly sound**` |
| Walking straightness | measured, localized deficit | [deficit_turning.md](audits/deficit_turning.md) 0, [connectome_backends.md](audits/connectome_backends.md) | no verdict line in the record (see entry) |
| Small-object pathway (LC11 / LC10a) | suite `KNOWN GAP`; `null` on the matched assay | [deficit_object.md](audits/deficit_object.md) 0, [object_baseline_r2.md](audits/object_baseline_r2.md), [object_compare_r2.md](audits/object_compare_r2.md), [object_localizer_r3.md](audits/object_localizer_r3.md), [object_rectangles_r3.md](audits/object_rectangles_r3.md), [object_samedevice_r3.md](audits/object_samedevice_r3.md) | mostly sound (object round 3: samedevice, rectangles, localizer) |
| Turning: the DNa02 choke | measured, localized deficit | [deficit_turning.md](audits/deficit_turning.md) 0, 4-6 | no verdict line in the record (see entry) |
| Turning: the body model (leg cycle, rounds 4-4d) | `result` on the relay and DNa02-left; `null` on behaviour | [body_sided_state.md](audits/body_sided_state.md), [level_matched_control.md](audits/level_matched_control.md), [level_controls.md](audits/level_controls.md), [level_controls_r2.md](audits/level_controls_r2.md), [level_fixed_point.md](audits/level_fixed_point.md) | `VERDICT: mostly sound` (round 4d) |
| Compass, rounds 5A-6B (cx5-cx7) | `null` at the shipped gains | [compass_ring_mechanism.md](audits/compass_ring_mechanism.md), [glno_relabel.md](audits/glno_relabel.md), [compass_dc_balance.md](audits/compass_dc_balance.md), [compass_local_recurrence.md](audits/compass_local_recurrence.md) | `**Mostly sound.**` (5A, 5B, 6A); 6B has no final verdict block (see entry) |
| Compass, rounds 7-8 (cx8r, cx8t, cx9) | `result` on the GLNO side report; `null` on rotation | [compass_velocity_route.md](audits/compass_velocity_route.md), [compass_sign_control.md](audits/compass_sign_control.md) | `VERDICT: mostly sound`; `- compass_sign_control: mostly sound` |
| Flight: the octopamine state | not adoptable; `null` on behaviour, runaway above | [monoamine_slow_term.md](audits/monoamine_slow_term.md) 0 | `verify:monoamines  --  verdict: **mostly sound**` |
| Flight: the priority policy | n/a (instrument) | [flight_foraging_priority.md](audits/flight_foraging_priority.md) | `- flight_foraging_priority: mostly sound` |
| Fruit finding: the plume instrument | n/a (instrument) | [plume_steering.md](audits/plume_steering.md), [navigation_instruments.md](audits/navigation_instruments.md) | `- plume_steering: mostly sound` |
| Fruit finding: the goal-only arm | n/a (instrument; goal-only feeds 4.67 +/- 0.52 of six) | [plume_goal_only.md](audits/plume_goal_only.md) | `- plume_goal_only: mostly sound` |
| Fruit finding: the transduced cue | n/a (instrument; 1 of 6 rooms) | [plume_transduced.md](audits/plume_transduced.md) 3, 7 | `**VERDICT: mostly sound; merge with fixes.**` |
| Monoamines (round 3) | not adoptable | [monoamine_slow_term.md](audits/monoamine_slow_term.md) | `verify:monoamines  --  verdict: **mostly sound**` |
| Unitary strength (round 3) | not adoptable | [unitary_strength.md](audits/unitary_strength.md) | `verify:unitary  --  verdict: **mostly sound**` |
| Guards and retirement candidates (round 3) | both candidates refused | [guard_suites_r3.md](audits/guard_suites_r3.md) 4, [anti_runaway.md](audits/anti_runaway.md) round 7 | `verify:guards  --  verdict: **mostly sound**` |
| Cross-connectome (FAFB / BANC) | descriptive | [connectome_backends.md](audits/connectome_backends.md), [connectome_backends_review.md](audits/connectome_backends_review.md) | `**Verdict: merge with fixes** (B1-B4)` (review, not `receptor_verification.md`) |
| Determinism | one protocol repeats exactly; no room repeats | [determinism_gate.md](audits/determinism_gate.md) 3-4 | `- determinism_gate: mostly sound` |

---

## B. The entries

### 1. Looming escape (giant fibre)

**The question.** Does an approaching object drive LC4 / LPLC2 into the giant fibre hard enough to jump, and
does self-motion through a textured room stay below that threshold?

**The arms.** The 29-check suite at three independent draws per arm over three arms -- the shipped default and
the two retirement candidates `no_drive_clip` and `pair_gain_lpi_x1` -- in one submission, 27 jobs, 0 failed
([guard_suites_r3.md](audits/guard_suites_r3.md) section 1); plus the opt-in `hops` section (section 2) and the
room take-off protocol at three brain seeds x 16 flies (section 3).

**The verdict.** `PASS` in all nine runs. `loom.GF_peak_hz` (bound `>= 20`) reads 47.22 in all three default
draws; `loom_escape.GF_peak_hz` (bound `>= 33`) reads 46.72 / 48.68 / 49.50; `loom_escape.escapes` (bound
`>= 1`) reads 3 in all nine runs; and the walking giant-fibre tail `walk_gf.p99_hz` (bound `< 38`) reads
17.22 / 25.38 / 18.32 (section 1).

**What it means.** The escape is carried by the wiring under the shipped assumptions, and the margin between
the walking tail and the threshold is what stops self-motion from faking it -- exactly the margin the two
retirement candidates move: `pair_gain_lpi_x1` puts the walking-GF median at 38.1 Hz with 48 of 48 flies above
the threshold, and `no_drive_clip` raises room take-offs to 5.069 against 3.125 per 1,000 fly-s. Both were
refused ([guard_suites_r3.md](audits/guard_suites_r3.md) section 4).

**Skeptic.** `verify:guards  --  verdict: **mostly sound**`

### 2. Direction selectivity (T4 / T5)

**The question.** Do the eight T4 / T5 subtypes prefer the directions the animal's do, from the wiring alone?

**The arms.** `probe_motion.py` inside the 29-check suite: 60 deg/s, 30 deg grating, four directions
([guard_suites_r3.md](audits/guard_suites_r3.md) section 1; protocol table in
[benchmark_suite.md](audits/benchmark_suite.md)).

**The verdict.** `PASS`. `motion.min_dsi` (bound `>= 0.1`) reads 0.24109642 / 0.24109674 / 0.24109651 in the
three default draws, and `motion.correct_directions` (bound `== 8`) reads 8 in all nine runs
([guard_suites_r3.md](audits/guard_suites_r3.md) section 1). The earlier full-suite run records the eight
preferred directions "correct and identical in every rerun including RTX 4090 -> B200 -- the optic lobe is
deterministic" ([benchmark_suite.md](audits/benchmark_suite.md), section detail).

**What it means.** The cleanest wiring-only behaviour in the model, and the most reproducible -- the optic
lobe is a graded rate stage. It is also where the one openly mislabelled gain lives: the "T5 delayed
inhibition" x5 pair gain multiplies 101,619 edges of which 78 % are cholinergic Tm9 / Tm4, so it is mostly a
drive gain ([optic_measures.md](audits/optic_measures.md)).

**Skeptic.** `verify:optic-audit  --  verdict: **mostly sound**`

### 3. Wind direction (DNp18 / DNp33)

**The question.** Does the model carry a lateralised wind signal to the descending neurons that steer?

**The arms.** `screen_steering.py`'s clean site inside the 29-check suite: wind 0.3 m/s, heading -90 against
+90, pinned, 10 s each ([benchmark_suite.md](audits/benchmark_suite.md)).

**The verdict.** `PASS` in all nine runs. `wind.DNp18_flip_hz` (bound `>= 15`) reads +45.84 / +44.70 / +45.25
and `wind.DNp33_flip_hz` (bound `<= -15`) reads -49.67 / -49.42 / -50.01 across the three default draws
([guard_suites_r3.md](audits/guard_suites_r3.md) section 1).

**What it means.** The strongest lateralised signal the wiring produces, and the one sensory route whose
left-right difference reaches a descending neuron at a size the body can steer with. It is kept in every suite
run as a regression guard rather than as a new finding.

**Skeptic.** `verify:guards  --  verdict: **mostly sound**`

### 4. Taste, and bitter suppression

**The question.** Do the labellar sugar GRNs drive the proboscis motor neuron MN9 through the known
second-order cells, and do the bitter GRNs shut it off?

**The arms.** `probe_taste.py` and `probe_bitter.py` in the suite, under this project's calibration and under
Shiu et al.'s uniform-synapse rules ([guard_suites_r3.md](audits/guard_suites_r3.md) section 1). Separately,
the monoamine slow class at `add_low` on CPU and on CUDA
([monoamine_slow_term.md](audits/monoamine_slow_term.md) section 0), and the three-instrument list beside raw
([instrumented_suite.md](audits/instrumented_suite.md) section 2).

**The verdict.** `PASS`. `taste.MN9_hz` (bound `> 2`) reads 10.93 in all nine round-3 suite runs, and the four
bitter checks read 5.518 / 0 / 139.9 / 0.818 in all nine -- calibrated sugar, calibrated sugar + bitter, Shiu
sugar, Shiu sugar + bitter ([guard_suites_r3.md](audits/guard_suites_r3.md) section 1).

**What it means.** The sugar -> proboscis path is the one behaviour Shiu et al.'s point model already
reproduced, and it survives every change this project made. But MN9's rate is a residual of two nearly equal
inputs -- +1,786 against -1,677 mV/s in `off` -- a knife-edge readout, and where things break first: the
shipped `slow_gain` 0.02 under `full` puts it at 1.9669914245605469 against the `> 2` bound on the CPU while
the same configuration reads 5.0909 with the class off
([monoamine_slow_term.md](audits/monoamine_slow_term.md) section 0). Under round 8's three-instrument list it
is the one non-uniform row of 29, and the instability is raw's own
([instrumented_suite.md](audits/instrumented_suite.md) section 2).

**Skeptic.** `verify:monoamines  --  verdict: **mostly sound**`

### 5. Walking straightness

**The question.** How straight does the plain fly walk, and is that a measurement or an artefact of the
protocol?

**The arms.** The plain-fly room, three runs x 16 flies x 60 s, clean-frame statistics with the fence frames
removed ([deficit_turning.md](audits/deficit_turning.md) sections 2-3); and the same protocol in the
cross-connectome walking replicate at five seeds x 16 flies x 60 s
([connectome_backends.md](audits/connectome_backends.md)).

**The verdict.** The realised yaw-rate SD is 2.56 deg/s once the fence's 1.2 % of frames is removed, against
8.3 deg/s with them, and the yaw command's own SD is 3.5 deg/s (runs 3.48 / 3.58 / 3.56)
([deficit_turning.md](audits/deficit_turning.md) section 0). The MaleCNS walking replicate reads clean yaw SD
2.641 +/- 0.146 deg/s and straightness 0.9943 ([connectome_backends.md](audits/connectome_backends.md)).

**What it means.** The animal turns in discrete events of 200-450 deg/s about every 250 ms
([round3_integration.md](audits/round3_integration.md) section 4); the model does not, and the reason is entry
7, not a parameter. A reporter-only defect in `probe_walk_straightness.py`'s table-exit counter (a bounds
tuple read as half-extents) was fixed with `metrics_version` 2, which changes the exit counts and not the yaw
statistics ([connectome_backends.md](audits/connectome_backends.md)).

**Skeptic.** [receptor_verification.md](audits/receptor_verification.md) carries no verdict line for
`deficit_turning.md` itself; the nearest pass that re-measured these quantities with its own clean-frame rule
is round 3's `verify:body-state  --  verdict: **mostly sound**`.

### 6. Small-object pathway (LC11 / LC10a)

**The question.** Where does a small moving object stop being carried between the medulla and the lobula, and
what kind of fact is that loss?

**The arms.** A depth trace from the photoreceptors to the LC cells plus eight lesion arms on fixed anatomy
([deficit_object.md](audits/deficit_object.md) 1, 4); the matched sphere ladder at six runs per arm
([object_baseline_r2.md](audits/object_baseline_r2.md)); the fixed-anatomy comparison of rectification,
adaptation and spatial suppression ([object_compare_r2.md](audits/object_compare_r2.md)); and round 3's three
closers -- same-device, contrast-matched rectangles, the receptive-field localizer -- each replicated at fresh
seeds on a second GPU model ([object_samedevice_r3.md](audits/object_samedevice_r3.md),
[object_rectangles_r3.md](audits/object_rectangles_r3.md),
[object_localizer_r3.md](audits/object_localizer_r3.md)).

**The verdict.** The figure is carried at depth 1-4 (L1 +13.3, Mi1 +17.2, T4c +14.5, T4d +15.6 z above the
none-vs-none null) and every input class of LC11 and LC10a sits at the null; LC11 reads +0.1 and LC10a +0.0
([deficit_object.md](audits/deficit_object.md) 0). On the matched ladder no size preference is called: LC11's
twelve-member family is `null` throughout with a smallest exact-U p of 0.180, and LC10a's smallest is 0.0087
against the 0.00417 Holm needs ([object_baseline_r2.md](audits/object_baseline_r2.md) 0c). The round-3
closers change no answer: 40 of 40 rectangle verdicts `null` in both batches
([object_rectangles_r3.md](audits/object_rectangles_r3.md)), zero LC11 fits at the fixed z = 5 on both lobes
in both maps ([object_localizer_r3.md](audits/object_localizer_r3.md)), and a same-device re-run that
reproduces on one GPU model and is `PARTIAL` on the other
([object_samedevice_r3.md](audits/object_samedevice_r3.md)). The suite row `object.LC10a_flip_hz` is
`KNOWN GAP` at 0.0086 / 0.0023 / 0.0037 against `abs >= 1`
([guard_suites_r3.md](audits/guard_suites_r3.md) 1).

**What it means.** Three scoped statements, not one. "At the null at the small-field stage" is a threshold
call: the same quantity reads Tm5Y z +0.86, +2.19, +2.65 and +3.65 in four independent batches
([deficit_object.md](audits/deficit_object.md) section 0). The localizer negative is scoped to a static
4.5-deg probe and is about received drive, not firing -- 137 of 143 LC11 bodies emit exactly zero spikes over
the 1,466.5 s recording ([object_localizer_r3.md](audits/object_localizer_r3.md)). And no mechanism arm
passes: per-stream rectification is the only one that carries the upstream figure and it releases bar, grating
and full-field flicker at LC11, the opposite of the animal
([object_compare_r2.md](audits/object_compare_r2.md)).

**Skeptic.** Three verdict lines, one per audit, quoted verbatim:

```text
### object:samedevice (`docs/audits/object_samedevice_r3.md`)  --  verdict: **mostly sound**
### object:rectangles (`docs/audits/object_rectangles_r3.md`)  --  verdict: **mostly sound**
### object:localizer (`docs/audits/object_localizer_r3.md`)  --  verdict: **mostly sound**
```

Round 2's build passes: `verify:baseline  --  verdict: **mostly sound**`,
`verify:compare  --  verdict: **mostly sound**`.

### 7. Turning: the DNa02 choke

**The question.** Why does the plain fly not turn, and is the block a tuning problem or a named one?

**The arms.** `decompose` on DNa02 and the leg motor neurons, `atlas` over 963 stimulation rows, and `paths`
from the movers back to the central complex and the senses, all on the shipped weights
([deficit_turning.md](audits/deficit_turning.md) sections 4-6).

**The verdict.** DNa02's rate-weighted synaptic input in the room is -326 / -392 mV/s per cell, which in the
model's own units is -1.63 / -1.96 mV against a 7.0 mV threshold gap -- about 25 % of the barrier -- although
the static wiring is net excitatory per volley (+466.1 / +447.6 mV). The inhibition is sign-correct by
transmitter: IN12B014 -89 / -124 mV/s, GNG562 -94 / -80, PS059 -95 / -49, LT51 -57 / -70. DNa02 fires 0.011 /
0.075 Hz and its L-R is -0.064 Hz. PFL3 reads 0.000 Hz in every cell of every fly (24 cells, max 0.000);
AOTU015 and AOTU001 read 0.000 Hz as means (AOTU015 max 10 Hz in 2 of 8 cells)
([deficit_turning.md](audits/deficit_turning.md) section 0).

**What it means.** The deficit is localized to a named population and a named link rather than to a gain. The
one qualification the audit makes about itself is that "PFL3 is the only CX gate" is an overstatement: 8 of
the 199 non-PFL3 CX populations move `turn_LR` with verdict `result` at 0.6-1.6 Hz, which at this file's
`k_turn` is the order of the whole yaw-command SD
([deficit_turning.md](audits/deficit_turning.md) section 0 item 2). The ledger row `rotation.DNa02.rate_hz`
records the gap against Schnell et al. 2010.

**Skeptic.** [receptor_verification.md](audits/receptor_verification.md) carries no verdict line for
`deficit_turning.md`. The structural claims were re-derived by later rounds that do carry one -- round 3's
`verify:body-state  --  verdict: **mostly sound**` and rounds 4-4d below.

### 8. Turning: the body model (the leg cycle, rounds 4-4d)

**The question.** If the never-driven leg afferents are made to fire, does DNa02 fire -- and if it does, is
that the afferents' *level* or their per-phase *structure*?

**The arms.** Five batches of one protocol: a tripod leg cycle plus a side-split haltere readout
([body_sided_state.md](audits/body_sided_state.md)); a level-matched control L at `mn_ref_hz` 8.84
([level_matched_control.md](audits/level_matched_control.md)); the unsided, channel-matched and
modulation-only controls at six arms x five seeds ([level_controls.md](audits/level_controls.md)); the
modulation-only arm re-run at the cycle's own realised amplitude 0.948, four arms x six seeds
([level_controls_r2.md](audits/level_controls_r2.md)); and a control matched on all three leg channels at once
([level_fixed_point.md](audits/level_fixed_point.md)).

**The verdict.** With the cycle on, the commanded chordotonal rate goes 23.4 -> 88.1 Hz, DNa02_L / _R go
0.031 / 0.104 -> 0.540 / 0.383 Hz and the clean-frame yaw SD goes 3.35 -> 7.87 +/- 0.14 deg/s
([body_sided_state.md](audits/body_sided_state.md); [NOTES.md](NOTES.md) Session 11). Against the
level-matched control, round 4d's three-channel-matched arm gives C minus L3 of +2.6687 / +3.8782 Hz at the
ascending relay AN04B003 and a `result` on DNa02-left, with DNa02-right, the clean yaw SD, straightness and
DNa02 L-R all `null`; the modulation-only arm M2 is `null` against C on all seven primaries
([level_fixed_point.md](audits/level_fixed_point.md); [NOTES.md](NOTES.md) Session 13). Round 4c had already
closed that half: the level-corrected structure term is +4.60 +/- 0.31 Hz without the amplitude law against the
cycle's +4.72 +/- 0.32 ([level_controls_r2.md](audits/level_controls_r2.md)).

**What it means.** Three things that are not turning. No clean frame in any arm of any batch exceeds
100 deg/s against an animal's 200-450 ([round3_integration.md](audits/round3_integration.md) section 4); the
extra yaw is a fixed left drift of +1.23 +/- 0.16 deg/s in 78 of 80 flies, which is the connectome's own
asymmetry; and the sided afferent that reaches DNa02 correlates with the realised yaw with the **kinematic**
sign -- the turn shapes the afferent ([body_sided_state.md](audits/body_sided_state.md);
[NOTES.md](NOTES.md) Session 11). The module ships opt-in and off.

**Skeptic.** Round 4d: `VERDICT: mostly sound`. Round 4b: `**Mostly sound.**` Round 4c: `**MOSTLY SOUND.**`
Round 4's pass refuted the *attribution* rather than the numbers -- "every one of the fifteen family
comparisons reproducing to the printed precision with my own exact Mann-Whitney, my own z and my own Holm --
not one verdict flips", then "What is unsound is the **attribution**"
([receptor_verification.md](audits/receptor_verification.md), "Level-matched control").

### 9. Compass, rounds 5A-6B (cx5, cx5b, cx6, cx7)

**The question.** At the shipped gains, is there a type-level change the data imply that makes the ring hold a
bump -- and if the answer is no, what is the binding constraint?

**The arms.** 5A: twelve arms over four seeds -- the shipped path, the GLNO relabel, the receptor `sign+gain`
tier, their pair, a labelled global instrument and two labelled references
([compass_ring_mechanism.md](audits/compass_ring_mechanism.md)). 5B: the GLNO = glutamate candidate through
the 29-check suite at 3 + 3 draws and the efferent compass at 4 seeds
([glno_relabel.md](audits/glno_relabel.md)). 6A: an `edges`-kind counterfactual holding 2 ExR6 + 4 ER6 + 11
ER4m at 0 onto PEN and EPG, eight arms x five seeds
([compass_dc_balance.md](audits/compass_dc_balance.md)); 6B: the same hold plus EPG-only and global recurrence
([compass_local_recurrence.md](audits/compass_local_recurrence.md)).

**The verdict.** `bump_survival_s` is 0.00 s in 4 of 4 seeds in every arm at the shipped gains that is not a
labelled instrument or reference (S, G, C, CG), and no arm meets the predeclared working-compass rule; every
surviving bump -- F, CF, CFG and the references R / RG -- fails the rate row by about 3x
([compass_ring_mechanism.md](audits/compass_ring_mechanism.md)). The hold lifts the driven PEN population from
0.29-0.66 Hz to 40.2-48.3 Hz (`result`, z 294.0, Holm 0.0317) and buys `bump_survival_s` 0.00 in 5 of 5
seeds, because the same term is what holds the unstimulated ring at rest
([compass_dc_balance.md](audits/compass_dc_balance.md)). 6B adds 0 of 5 joint successes in every arm and the
first measured per-type ring rates: post-pulse ExR6 41.3-42.8 Hz, ER6 23.1-23.9
([compass_local_recurrence.md](audits/compass_local_recurrence.md)). The GLNO relabel is `null` on all 15
predeclared members at 4 v 4 ([glno_relabel.md](audits/glno_relabel.md)).

**What it means.** The shipped-gain question is closed as a `null`, not as a `result`, and the constraint is a
DC balance on the relays rather than a missing ring mode. Everything that holds a bump is a labelled
instrument or a labelled reference. GLNO's transmitter is a rate parameter of the ring, not a signal
parameter: glutamate and GABA produce a bit-identical W under the shipped receptor model, so the decision on
the table is "sign GLNO -1", and it was not taken.

**Skeptic.** 5A, 5B and 6A each return `**Mostly sound.**` 6B is the exception: the handoff wording was
"mostly sound", the independent reviewer hit its usage limit before writing a final verdict block, its
completed notes are appended verbatim, and **no final verdict is reconstructed or attributed to it**
([receptor_verification.md](audits/receptor_verification.md), "Compass round 6B").

### 10. Compass, rounds 7-8 (cx8r, cx8t, cx9)

**The question.** Does a signed report of the fly's own turn reach the compass, does the ring rotate when it
does, and is the effect sign-specific?

**The arms.** Round 7: eight predeclared arms x six seeds on the velocity route, including the sign-flipped
afferent inside the held ring; then cx8t, a direct 90 Hz GLNO challenge at four predeclared contrasts, 36 runs
([compass_velocity_route.md](audits/compass_velocity_route.md) 5, 6.4-6.7). Round 8 item 2: the sign-flipped
afferent **alone**, no hold and no relabel, beside re-runs of S and V, six seeds each
([compass_sign_control.md](audits/compass_sign_control.md)).

**The verdict.** V minus S raises GLNO L-R by +2.2183 Hz (`result`, Holm p 0.0130)
([compass_velocity_route.md](audits/compass_velocity_route.md) section 5.2). Rotation does not follow: a hump
forms in HG / HGV / HGV- at vector strength ~0.72, ~160 Hz and ~3.9 wedges, and in the three gate-passing runs
the confined-frame centre slope is +0.137, -0.086 and -0.069 wedges/s against an ideal +4.0. cx8t puts a
correctly signed +2.9939 Hz into PEN (z 3.1904) while the side-balanced PEN mean falls from 24.0426 Hz to
5.0211 / 3.6345. Round 8's V- minus S is -2.0665 +/- 0.6461 Hz (`result`, Holm p 0.0065, all six seeds
separated), so primary 3 is sign-specific **in the unheld ring**
([compass_sign_control.md](audits/compass_sign_control.md) section 3).

**What it means.** A well-localized `null` on rotation with a `result` on the side report. Two verdicts in
this pair are orientation-dependent, because `common.compare` divides by the reference arm's SD: primary 3
reversed gives |z| 2.963 and a null, and cx8t's transfer reversed gives z -2.7036 and a null, while the
symmetric Welch statistics are +7.26 and +5.05 and U is unchanged
([receptor_verification.md](audits/receptor_verification.md), round 7 corrections 9-10). The held arms are a
separate, unresolved question: HGV against HGV- on GLNO L-R recomputes as a `null` (+3.1092 Hz, z +1.625), so
their non-reversal is not a demonstrated absence
([compass_sign_control.md](audits/compass_sign_control.md) section 4).

**Skeptic.** Round 7 and cx8t: `VERDICT: mostly sound`. Round 8 item 2: `- compass_sign_control: mostly sound`.

### 11. Flight: the octopamine state

**The question.** Can the 1.88 M monoamine synapses that carry sign 0
([monoamine_slow_term.md](audits/monoamine_slow_term.md) section 1a) be routed through a slow receptor class,
so that an octopaminergic flight state exists in the model at all?

**The arms.** One class scalar at four settings -- `add_low` 0.02 additive (the shipped value), `add_mid` 0.2,
`add_high` 1.0 and `gain_mid` -- at 5 runs x 8 flies x 30 s per arm, plus a CPU replay of the taste section
([monoamine_slow_term.md](audits/monoamine_slow_term.md) section 0).

**The verdict.** `add_low` is `null` on every behaviour key at 5 v 5 and again at 4 v 4; baseline activity
moves +3.0 % (`result` but tiny) and `taste.MN9_hz` fails on the CPU path (entry 4). `add_mid` and `add_high`
run away in 5 of 5 runs each, aborted at t = 0.51 s at 604-680 and 1,273-1,277 spikes/step against the off
arm's 52.1. `gain_mid` gives x2.55 spikes, OA-VUMa2 at 329 Hz, 1.5 voluntary take-offs per fly and DNa02 at
0.1 Hz -- a baseline change, not a behavioural gain
([monoamine_slow_term.md](audits/monoamine_slow_term.md) section 0; [NOTES.md](NOTES.md) Session 11).

**What it means.** The two literature anchors are 50-100x apart on one scalar -- Cohn 2015 gives ~0 mV on
Kenyon cells, Longden 2010 / Maimon 2010 give x1.5-2 octopaminergic gain on HS / VS -- so **one class scalar
cannot carry dopamine, octopamine and serotonin at once**. The octopamine anchor row records that its upper
edge 2.3 "has NO number in any cited source and is HEADROOM, not a literature value"
([monoamine_slow_term.md](audits/monoamine_slow_term.md) 2, ledger row `lit.HS.oa_response_gain`, all five
anchors `op report`). A per-transmitter split, a KC -> MBON plasticity module and VNC receptor rows are what
an adoption would need.

**Skeptic.** `verify:monoamines  --  verdict: **mostly sound**`

### 12. Flight: the priority policy

**The question.** Under `instrumented`, can a wing-drive instrument be given a policy that does not let the
fly fly while it starves?

**The arms.** The corrected policy -- ground search, bounded bouts, an energy reserve and odour interruptions
-- frozen before the run, with 26 lifecycle checks, 17 exact captured/eager transition checkpoints and the
room interruption gates ([flight_foraging_priority.md](audits/flight_foraging_priority.md) 2-3).

**The verdict.** All six initially depleted airborne room rows land in 0.47-0.52 s. The policy is provably
frozen before the run (`predeclared.json` stamped at the commit time of be1d893). Two things the skeptic added
in place: with the shipped lateral-horn normalisation the odour gate reads 0.66-0.71 mean (max 0.95-0.99) even
with a single apple 40 cm away, so the hysteresis latches for essentially the whole episode and **artificial
flight is unreachable in any room containing fruit, not merely bounded**; and a ballistic fall of the 0.25 m
drop takes 0.41 s, so the declared 5 s landing gate tests only that no power was applied
([flight_foraging_priority.md](audits/flight_foraging_priority.md) section 3).

**What it means.** Engineering policy, not physiology: no receptor, transmitter, synapse or default changed.
It is also why the second licensed milestone sentence holds -- `powered_s` is 0.00 in all 12 flight-priority
room rows and all 12 plume room rows, and the only powered-flight rooms are the superseded navigation flight
arm, which fed in none of six ([navigation_instruments.md](audits/navigation_instruments.md) 2, correction
13).

**Skeptic.** `- flight_foraging_priority: mostly sound -- every number reproduces, the policy is provably frozen before the run, but the results omit that the policy makes artificial flight unreachable in any fruited room and that the 5 s landing gate is near-vacuous.`

### 13. Fruit finding: the plume instrument

**The question.** Under `instrumented`, can a labelled goal-and-steering stand-in take the fly to fruit in a
60 s tabletop room, with no world position, wind angle, fruit position or distance reaching it?

**The arms.** The corrected bilateral walking goal plus DNa02 L-R feedback onto the existing PFL3 inputs,
over six environment seeds, with the compass and ring variants
([plume_steering.md](audits/plume_steering.md) 3, 6; [navigation_instruments.md](audits/navigation_instruments.md)).

**The verdict.** Six of six compass rows feed for at least one second; the instrument's only body-derived
inputs are antennal deflections, bilateral concentrations and interoception flags, and the walking goal is
`atan(200*atanh(contrast))` on the antennal samples alone in 99.64 % of 3,600 logged samples
([plume_steering.md](audits/plume_steering.md) section 3).

**What it means.** Four characterisations the skeptic required, all now in the audit. The 0.1 m / 0.001 m
contrast gain of **200** acts on a noise-free pre-transduction field, giving 25-41 deg mean goal offsets at the
0.26-0.57 % contrast observed, so the law behaves as a near-sign-of-contrast turn. The integral supplies more
than half the PFL3 input and the balanced control row needs -5.94 Hz to hold DNa02 L-R at +0.061 Hz, so the
loop inverts the parent PFL3 -> DNa02 gain rather than relying on it. Every row reaches zero energy at
18.0-18.2 s, before any first contact at 20.1-37.3 s, so **these rooms do not test metabolic modulation**. And
the six environment seeds are near-repeats: the lime is the nearest fruit at 22.6 cm and four of the six rows
fed there ([plume_steering.md](audits/plume_steering.md) section 6, corrections 1, 3, 5, 6).

**Skeptic.** `- plume_steering: mostly sound -- every number reproduces exactly; one sentence is factually wrong ("some trajectories reach zero" -> all of them), and four material characterisations a reader needs are missing.`

### 14. Fruit finding: the goal-only arm

**The question.** Which half of the plume instrument is load-bearing -- the walking goal, or the DNa02
feedback?

**The arms.** Three arms at six runs each on the shipped six rooms and on six drawn starts: `full` as shipped,
`goal-only` (the bilateral walking goal with the DNa02 integral gain at 0) and `feedback-only` (the feedback
with the pre-correction upwind / entry-memory goal) -- 36 runs, 0 problems
([plume_goal_only.md](audits/plume_goal_only.md) sections 1-3).

**The verdict.** The walking goal is the load-bearing part. `goal-only` feeds 4.67 +/- 0.52 rows of six at a
DNa02 |L-R| of 0.62 Hz, while `feedback-only` feeds 0.17 +/- 0.41 with the **largest** DNa02 |L-R| of the
three arms (1.97 Hz): the loop turns the fly hard toward a goal that does not point at food. `full` is 6/6,
and the row `goal-only` never feeds is the 185 deg start
([plume_goal_only.md](audits/plume_goal_only.md) sections 3-4).

**What it means.** The earlier skeptic's inference -- that the feedback was the load-bearing change, and that
a goal-only arm would show 0.28-0.69 Hz DNa02 differences -- is refuted on its own prediction. The claim is a
**simple effect**, not a main effect: the three arms are three cells of a 2 x 2 whose fourth cell, the
pre-correction goal at gain 0, is not in this batch. And the drawn set is weaker than designed -- two of its
six rows start 1.6 and 3.2 cm from the nearest fruit surface against a 1.5 cm feeding shell and are fed in
every arm and run ([plume_goal_only.md](audits/plume_goal_only.md) section 4).

**Skeptic.** `- plume_goal_only: mostly sound`

### 15. Fruit finding: the transduced cue

**The question.** If the bilateral cue is read from the model's own ORN population rates instead of the
physical concentration field -- letting the Poisson noise in -- does the fly still find food?

**The arms.** On the CPU first: the shipped ORN law `1 + 150c/(c+0.5)`, the 100 ms rate filter and the 250 ms
contrast filter, over the contrast range the rooms saw ([plume_transduced.md](audits/plume_transduced.md) 3).
Then 18 rooms: `full`, `transduced` (`plume:bilateral=orn`) and `goal_only` (`plume:feedback=off`) over six
seed-drawn 60 s starts **shared across the arms**, one fly per room (same audit, section 7).

**The verdict.** Fed for at least one second: **full 6/6, transduced 1/6, goal_only 3/6**. In the transduced
arm the cue is uncorrelated with the physical lateral contrast the antennae actually saw -- per-run Pearson r
-0.07 to +0.24, the cue's sign right in 0.52 +/- 0.10 of samples against 0.87-0.98 in the full arm -- and
`|200*atanh(contrast)| > 1` in 66-87 % of samples, so the commanded turn saturates on the sign of noise.
Excluding the two starts that begin 3.9 and 5.7 cm from a fruit surface leaves 4/4 full, 1/4 transduced, 3/4
goal_only ([plume_transduced.md](audits/plume_transduced.md) section 7). On the CPU the same contrast arrives
at cascade SNR 0.16-0.34 with the correct sign in 0.667 of 250 ms windows (analytic 0.635)
([plume_transduced.md](audits/plume_transduced.md) section 3).

**What it means.** The shipped `plume` instrument finds food on a physical antennal contrast, and substituting
the model's own ORN rates at the matched gain drops it to 1 of 6 on a cue uncorrelated with the truth. **The
rooms do not confirm the CPU SNR finding** -- the in-room ORN L-R and its temporal SD are different quantities
from the CPU's fixed-contrast signal and its 250 ms counting-window noise, and are not an SNR. The licensed
statement is the one in the audit: the rooms add that in-room cue noise is 1.3-2.6x the analytical estimate and
that the cue's sign was right in only 0.52 +/- 0.10 of samples
([plume_transduced.md](audits/plume_transduced.md), skeptic correction C7). Two design caveats travel with it:
the six runs of an arm differ in start *and* seed, so the across-run SD is start heterogeneity and the
paired-by-start reading is the stronger statement the tables do not make; and `goal_only`'s three successes
are exactly the three starts already pointing near fruit (bearing error 19-88 deg), so that arm is closer to
"hold the start heading" than to a plume-competence floor. The gain of 200 stays underived.

**Skeptic.** `**VERDICT: mostly sound; merge with fixes.**`

### 16. Monoamines, unitary strength and the guards (round 3)

**The question.** Three adoption candidates at once: a monoamine slow class, a per-transmitter unitary synapse
strength, and the retirement of two anti-runaway rules.

**The arms.** Five threads in one round, five independent skeptics with their own fresh-seed replications, 165
cluster jobs in six completed submissions plus 58 skeptic jobs, 0 failed
([monoamine_slow_term.md](audits/monoamine_slow_term.md), [unitary_strength.md](audits/unitary_strength.md),
[guard_suites_r3.md](audits/guard_suites_r3.md), [anti_runaway.md](audits/anti_runaway.md) round 7,
[round3_integration.md](audits/round3_integration.md); [NOTES.md](NOTES.md) Session 11).

**The verdict.** Monoamines: entry 11. Unitary strength: the one anchor with both the EPSP and the count
measured (ORN -> PN, ~5 mV over ~23 synapses) is x0.79 of the shipped 0.275 mV, no per-synapse fast IPSP in a
Drosophila central neuron is on record, and every data-anchored inhibitory bracket breaks the suite --
`walk.power_sustained_hz` reads 89-194 Hz against a bound of `< 50` in every bracket and every draw
([unitary_strength.md](audits/unitary_strength.md); [NOTES.md](NOTES.md) Session 11). Guards: both retirement
candidates are refused on the room -- `no_drive_clip` at 5.069 against 3.125 take-offs per 1,000 fly-s, and
`pair_gain_lpi_x1` at 34.6 per 1,000 fly-s with 48 of 48 flies above the escape threshold
([guard_suites_r3.md](audits/guard_suites_r3.md) sections 3-4).

**What it means.** The refusals are the finding: one class scalar cannot carry three transmitters, one
unitary strength is a cholinergic calibration rather than a constant, and the two rules that look like
hand-tuning are what keep the closed loop off the escape threshold. **No model default changed**: every new
field defaults off, the shipped path is bit-identical on CPU with the cycle attached and the sense off, and
the 13 new ledger rows are all `op report` ([NOTES.md](NOTES.md) Session 11).

**Skeptic.** `verify:unitary  --  verdict: **mostly sound**`, `verify:guards  --  verdict: **mostly sound**`,
`verify:integrate  --  verdict: **mostly sound**`.

### 17. Cross-connectome (FAFB v783 and BANC v888)

**The question.** Does the same model, unchanged, run on two female FlyWire reconstructions -- and does the
walking result transfer?

**The arms.** A full MaleCNS recompile with the branch code against the shipped cache, the FAFB column-map
validations, and a walking replicate on BANC at five seeds x 16 flies x 60 s at the shipped parameters, 30 of
30 simulations exit 0 ([connectome_backends.md](audits/connectome_backends.md)).

**The verdict.** MaleCNS identity is preserved: the recompile reproduces the shipped cache byte for byte, file
for file, with the fingerprint unchanged. On the walking replicate the female CNS walks straighter than the
male at the shipped defaults -- clean yaw SD 0.275 +/- 0.004 deg/s (BANC) against 2.641 +/- 0.146 (MaleCNS),
straightness 0.9984 against 0.9943. The leg-cycle yaw increase replicates qualitatively (BANC 0.381 -> 2.651
deg/s, `result` within-dataset at 5 v 5) but the neural pattern does not: BANC's DNa02_L does not fire under
the cycle arm while DNa02_R reaches 0.0985 Hz, and its PS059 sits at 0.00087 / 0 Hz where MaleCNS runs
20.06 / 16.77 ([connectome_backends.md](audits/connectome_backends.md); [NOTES.md](NOTES.md) Session 11
addendum).

**What it means.** **This is a descriptive cross-connectome comparison, not a sex test.** One male
reconstruction against one female reconstruction, with different animal, lab, synapse threshold and yield,
different optic capability, different naming and no per-individual replication; DNa02's total raw input runs
about 1 : 0.59 : 0.28 across the three releases -- a descriptive anchor scale, not a universal correction
([connectome_backends.md](audits/connectome_backends.md)) -- and fractions are normalised within each release,
while BANC's optic lobes are under-proofread (T2 853 against FAFB 1,466 and MaleCNS 1,630,
[flywire_banc_survey.md](audits/flywire_banc_survey.md)). A missing type match is not established sex
specificity. What the survey does establish is that DNa02's excitation : inhibition ratio holds across all
three releases (2.0 : 1 MaleCNS, 1.96 : 1 FAFB, 1.7 : 1 BANC), so the net-inhibited resting state the model
finds is a rate statement, not a reconstruction artefact of the male graph
([flywire_banc_survey.md](audits/flywire_banc_survey.md)).

**Skeptic.** The one row whose independent pass is **not** in
[receptor_verification.md](audits/receptor_verification.md): the review is its own file, and its verdict line
is `**Verdict: merge with fixes** (B1-B4)`
([connectome_backends_review.md](audits/connectome_backends_review.md)); the follow-up review carries
`**Verdict: merge with fixes.**`
([connectome_backends_followups_review.md](audits/connectome_backends_followups_review.md)).

### 18. Determinism

**The question.** Does anything in this model repeat run to run on a GPU, and what may therefore be quoted as a
number?

**The arms.** Each of five protocols run twice, sequentially, in one job on one pinned GPU, with every saved
array and numeric metric compared exactly under a rule frozen before submission: the `cx_wedge` compass
protocol, and the raw and plume rooms at B=6 on both the native and the torch paths
([determinism_gate.md](audits/determinism_gate.md) sections 1-3).

**The verdict.** The `cx_wedge` protocol repeats exactly -- 31 of 31 arrays and 546 of 546 metrics, the third
exact repeat of that protocol. **No B=6 room repeats on either path**: the first differing frame is 4 (raw
native), 30 (plume native), 86 (raw torch) and 60 (plume torch), after which the two runs differ as two seeds
do ([determinism_gate.md](audits/determinism_gate.md) section 3).

**What it means.** The skeptic localized the divergence further than the audit had: the first difference in
every room pair is exactly +-1 spike in exactly one of the six rows, that row's body follows 1-51 frames later
and the other five rows stay bit-identical for a further 215-329 frames, so neither the body integrator nor the
air / odour field is the source; the same B=6 raw room on the torch path repeats bit-exactly across two
separate CPU processes for 120 frames with hash randomisation on, which excludes the Python layer; and on the
native path the event-scatter kernel accumulates with a float `atomicAdd` (`flyverse/kernels/neural.cu:122`),
order-nondeterministic by construction and a sufficient mechanism for that path
([determinism_gate.md](audits/determinism_gate.md) section 4.2). The frozen decision governs everything after
it: rooms are at least 6 draws with the run as the replicate unit, no room number is quoted beyond its
across-draw SD, and **every room number quoted before the gate stays one draw**.

**Skeptic.** `- determinism_gate: mostly sound`

---

## C. What was withdrawn

A withdrawn claim is one that was published in an audit, refuted by an independent pass, and replaced in place
with a `**Withdrawn:**` note under [INTERP.md](INTERP.md) 10.4 rule 29 iii. The pattern they share -- a right
conclusion supported by a wrong argument -- is the most common failure in this record.

1. **"The hemibrain name GLNO encodes glutamate."** Cited to the owner as one of three sources for the GLNO
   relabel. No accessible text defines the "G" or attaches a transmitter prediction to GLNO. *Correction:* the
   expansion is not a source and is not to be cited again without a page reference; what remains is two
   low-confidence EM classifiers calling glutamate and a third calling GABA, all three agreeing only that GLNO
   is inhibitory ([glno_relabel.md](audits/glno_relabel.md); [NOTES.md](NOTES.md) Session 12 round 5B).

2. **"What is still missing is not excitation at the tile but inhibition everywhere else"** (compass 6B).
   *Correction:* H3E's far half is already at 8.7-12.3 Hz, near the 10 Hz background, with only 0-2 cells above
   22 Hz; the excess lies adjacent to the driven block, and recurrence makes the far half quieter than H3 while
   recruiting Delta7. The failure is a seven-wedge, high-rate hump with a systematic +1.58-1.73 wedge offset
   ([compass_local_recurrence.md](audits/compass_local_recurrence.md)).

3. **Three claims in the same audit.** "Every firing ring cell reads 2.1-2.4 mV whatever its rate"; **4.6-11.8
   mV as an established bracket** for the smoothed f-I sigma (the 11.8 mV reading supplies a quasi-static
   sensitivity scenario, not an upper bound; the tool was evaluated at 2.0, 4.6 and 11.8 without changing its
   default); and **a general determinism claim from the cx7 repetition** -- the recorded values reproduce
   exactly on that backend, which does not establish GPU determinism for other configurations or machines
   ([compass_local_recurrence.md](audits/compass_local_recurrence.md)).

4. **"The mirror of V (GLNO_L 2.59 / GLNO_R 0.37)"** as an exact mirror, and **"76 of 76 (arm, key) rows are
   exactly equal"** without the word *metric*. *Correction:* V's GLNO_R equals S's in only three of six seeds;
   `replication.csv` compares the run JSON's 38 metric keys and no array -- though the 384 ledger arrays were
   separately verified bit-equal ([compass_sign_control.md](audits/compass_sign_control.md)).

5. **"The pinned id in `CUDA_VISIBLE_DEVICES`" as a statement about every run**, and **"every loaded source
   file at the predeclared hash"** without its qualification. *Correction:* `cx_wedge` records no
   `CUDA_VISIBLE_DEVICES` at all, and three loaded paths fall outside `source_hashes()` and are silently
   skipped ([determinism_gate.md](audits/determinism_gate.md)).

6. **Two claims about the instrumented suite.** "This list, unlike the compass stand-in, moves no suite row"
   -- values move on 20 of the 29 rows; what does not move is any row's **status**. And "the meals descriptive
   was not declared" / "undeclared descriptives from the same runs" -- meals, path and the walking-GF median
   are all named in `predeclared.json`'s `rule.descriptive`, and only rows at the giant-fibre threshold are
   undeclared ([instrumented_suite.md](audits/instrumented_suite.md)).

7. **"The starting energy is only 0.1 and some trajectories reach zero before finding food."** *Correction:* it
   is every row, at 18.0-18.2 s, and every one of them reaches zero before its own first contact
   ([plume_steering.md](audits/plume_steering.md)).

8. **"All neural reads are frame-boundary samples, so no mutable sibling state is read during a module
   step."** *Correction:* `PlumeNavigation.step` and `FlightDrive.step` both read `self.hunger.level`, which
   is written solely by `observe_internal` and never during a module step
   ([plume_steering.md](audits/plume_steering.md)).

9. **"Energy reaches zero before every first contact as before"** and **"two of the six drawn rows begin
   within a fruit's reach."** *Correction:* the first holds for the shipped-start `full` rooms only -- 74 of
   151 first contacts in the goal-only batch precede energy zero; and neither drawn row starts inside the
   feeding shell (0.0163 m and 0.0318 m against a 0.015 m radius), so both are counted in every drawn figure
   rather than excluded ([plume_goal_only.md](audits/plume_goal_only.md)).

10. **"Both antennas receive the same glomerulus mass"** as a claim about the pre-fix attached instrument.
    *Correction:* before the fix the scheduler sorted only the indices, producing +3.57679 Hz under symmetric
    stimulation; the claim holds for the fixed weights
    ([plume_transduced.md](audits/plume_transduced.md) section 1).

11. **The 4-run / 5-run agreement claim of round 3's body-state audit** ("the five-run and four-run tables
    differ in NO verdict"). *Correction:* 15 of 765 pairwise and 7 of 548 room-table verdict cells flip; no
    headline row is among them ([NOTES.md](NOTES.md) Session 11;
    [receptor_verification.md](audits/receptor_verification.md) `verify:body-state`).

12. **Round 2's transducer numbers as published.** The one `unsound` verdict in the record:
    `verify:vnc-drive  --  verdict: **unsound**`. 125 of the 128 room numbers quoted in
    [vnc_drive.md](audits/vnc_drive.md)'s sections 0, 4 and 5 came from a partial, superseded staging
    directory while the audit asserted they came from the final one. *Correction:* the qualitative conclusions
    survive and the generator is sound -- an independent recompute reproduces the final table with 0 mismatches
    on all 51 rows x 5 arms -- but two of 204 verdicts flip stage to final, and the document's own provenance
    sentences were false ([receptor_verification.md](audits/receptor_verification.md), "Dynamics round 2").

13. **Two further single-claim withdrawals.** "The arms' ranges do not overlap on any `result` row"
    ([vnc_drive.md](audits/vnc_drive.md)); `spearman_voluntary` -0.80, a tie-breaking artefact
    ([anti_runaway.md](audits/anti_runaway.md)).

14. **A per-seed scatter that was in no file.** Round 4b's audit printed DNa02_L per-run lists for four of six
    arms that occur nowhere in the recordings -- constructed to carry the published means rather than
    transcribed. The decision table itself was and is correct; the skeptic caught the prose while reproducing
    all 49 rows exactly. *Correction:* the four lists were replaced from `analysis/room_table.csv`, and
    [INTERP.md](INTERP.md) 10.4 item 28 now requires any per-run scatter quoted in prose to be emitted into a
    named file and pasted from it ([level_controls.md](audits/level_controls.md); [NOTES.md](NOTES.md)
    Session 12 addendum).

---

## D. Nothing adopted

The defaults that did **not** change, and why. The standard is the round-2 rule: a candidate is adoptable only
if no check has a status in any candidate draw that is not the status of every shipped draw (the suite half, at
three draws or more), **and** the room take-off rate-half passes at six runs per arm or more, **and** a data
change meets the source standard of the entries already in the table
([glno_relabel.md](audits/glno_relabel.md) 4.3; [guard_suites_r3.md](audits/guard_suites_r3.md) 4).

| candidate | why it was not adopted |
|---|---|
| `LIFParams.w_syn_by_nt` (per-transmitter unitary strength) | Kept as an **opt-in instrument, not a mechanism**: default `None`, shaped weights byte-identical under `None` / `{}` / all-ones, with a CPU bit-identity test. No bracket of the field is adoptable -- every data-anchored inhibitory bracket puts `walk.power_sustained_hz` at 89-194 Hz against `< 50` ([unitary_strength.md](audits/unitary_strength.md) 5). |
| The monoamine slow class | One class scalar cannot carry dopamine, octopamine and serotonin: the two literature anchors are 50-100x apart. At the shipped 0.02 it fails `taste.MN9_hz` on the CPU; at 0.2 and 1.0 the mushroom body runs away in 5 of 5 runs ([monoamine_slow_term.md](audits/monoamine_slow_term.md) 0). |
| `drive_clip_mv` retirement (`no_drive_clip`) | Refused on the room: 5.069 against 3.125 take-offs per 1,000 fly-s, the excess entirely voluntary, and round 6's walking-GF prediction not confirmed ([guard_suites_r3.md](audits/guard_suites_r3.md) 3-4; [anti_runaway.md](audits/anti_runaway.md) round 7). |
| LPi -> LPLC2 pair gain x1 (`pair_gain_lpi_x1`) | Refused on the room: 34.6 take-offs per 1,000 fly-s, 451 escapes against 17, and 48 of 48 flies above the escape threshold ([guard_suites_r3.md](audits/guard_suites_r3.md) 3-4). |
| `body.LegCycle` and the side-split haltere | Ship **opt-in and off**. The attribution replicates across the rounds 4-4d batches, but it moves yaw without turning: no clean frame exceeds 100 deg/s, the extra yaw is a fixed drift, and the sided afferent follows the turn rather than leading it ([body_sided_state.md](audits/body_sided_state.md) 8; [level_fixed_point.md](audits/level_fixed_point.md)). |
| `unsided`, `leg_cycle_flat`, `flat_amplitude_value` | Labelled **control constructions**, never candidates for a default; the shipped CPU path is bit-identical with all three absent ([level_controls.md](audits/level_controls.md), [level_controls_r2.md](audits/level_controls_r2.md)). |
| GLNO -> glutamate (`TYPE_NT_OVERRIDE`) | The suite half is met and nothing else: the room rate-half was not run, and the source standard of the three existing entries -- each naming a transcriptome or an EASI-FISH source -- is not met. Because the candidate's W is bit-identical to the GABA cache under the shipped receptor model, the decision actually on the table is "sign GLNO -1", which the data do not settle ([glno_relabel.md](audits/glno_relabel.md) 4.3). |
| The ExR6 / ER6 / ER4m hold (`--hold-edges`) | A hold is a counterfactual and can never be a default: it defaults to `None` and the shipped path is bit-identical with it absent. It also removes the ring's resting state ([compass_dc_balance.md](audits/compass_dc_balance.md)). |
| `same_type_gain` x0.1 | Kept. In 5A, removing it is the only configuration that carries a bump at the shipped gains, and it is a **labelled global instrument**, not a mechanism -- and it fails `compass.EPG.bump_rate_hz` by about 3x ([compass_ring_mechanism.md](audits/compass_ring_mechanism.md)). |
| Per-stream rectification / adaptation / suppression at the small-field stage | No mechanism passes. Rectification carries the upstream figure and releases bar, grating and full-field flicker at LC11, the opposite of the animal ([object_compare_r2.md](audits/object_compare_r2.md)); `rectify` and `suppress` remain hand-set opt-in control arms ([object_samedevice_r3.md](audits/object_samedevice_r3.md)). |
| The five navigation stand-ins (`compass`, `compass_ring`, `plume`, `hunger`, `flight`) | Admitted under `instrumented` only, never into `raw`. Round 8 found the three-instrument list **admissible** under [PRESETS_SPEC.md](PRESETS_SPEC.md) section 2 item 5 -- no row's status changes and the room rate-half passes at 110 take-offs against 101, one-sided exact Poisson p 0.291 -- "and nothing more: nothing is adopted, `raw` stays the default, the afferent's law is still `unverified` and the relabel still has no transmitter source" ([instrumented_suite.md](audits/instrumented_suite.md) 2-4; [NOTES.md](NOTES.md) Session 14 round 8). |
| The compass stand-in as a preset default | Its own admission run rejected it: `taste.MN9_hz` moved 1.690 -> 4.296 Hz at seed 1, FAIL -> PASS outside the declared gap ([compass_standin.md](audits/compass_standin.md); [NOTES.md](NOTES.md) Session 13). |

**The code state that goes with all of it.** `raw` is byte-identical: the bit-identity golden predates the
instrument branch, the three cache MD5s are unchanged (`c50c598a...`, `ac131529...`, `bf01d724...`) with
compiled CSR `ef23cc27bea13be7f6a96f3c04fd3737`, and no default moved anywhere in `flyverse/` or `scripts/` --
every change is a new trailing keyword with a neutral default. The same MD5s and fingerprint are recorded
unchanged through rounds 4d, 7, 8 and the plume rooms ([NOTES.md](NOTES.md) Session 13).

---

**Where to go next.** The numbers batch by batch: [audits/](audits/). The independent passes verbatim:
[audits/receptor_verification.md](audits/receptor_verification.md). What was learned in what order:
[NOTES.md](NOTES.md). Which number was measured at which commit: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
section 8. The assays: [BENCHMARK_BATTERY.md](BENCHMARK_BATTERY.md). The procedure a failing ledger row
triggers: [INTERP.md](INTERP.md) section 10.
