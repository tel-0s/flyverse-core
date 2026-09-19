# Model presets: `raw` and `instrumented`

Spec, 2026-09-15 (owner decision of the same day: labelled stand-ins are acceptable so long as they are as physiologically
realistic as the literature allows, optional, and a `raw` preset returns the pure brain with none of them). Nothing in
this document changes a default; it fixes the vocabulary and the acceptance rules so the round that ships the first
instrument has something to be measured against. Two owner extensions (sections 5 and 6) widen what may be *named* as
an instrument; neither relaxes section 2 and neither touches `raw`. What ships, and how to turn one on, is
[INSTRUMENTS.md](INSTRUMENTS.md); the gate results are section 7.

Section numbering is stable: code and audits cite these sections by number.

## 1. Two presets, one brain

| preset | what it is | bit-identity |
|---|---|---|
| `raw` | The connectome, the LIF, the receptor table, the senses and the motor readout exactly as shipped. No module attached, no held edge, no relabel beyond the sourced `TYPE_NT_OVERRIDE` rows. This is the plain model every audit refers to. | Byte-identical to today's `FlyBrain()` on every path (`tests/test_bit_identity.py` extends to assert it). |
| `instrumented` | `raw` plus a **named list of instruments**, installed through the existing surface for their kind: neural computations use ordinary `attach()` modules; body-derived inputs use the proprioception transducer; the held-edge and relabel candidates of section 3 verify an explicit caller configuration. No runtime module rewrites a parent synapse. | Provenance records `preset` and each instrument's `describe()`. An empty list on the original connectome and LIF parameters reproduces `raw`. |

The preset is one keyword on the constructors and the CLIs (`FlyBrain(preset="raw")`, `--preset raw`), threaded into
`provenance()` beside `connectome_fingerprint`. **The default is `raw`** (`flyverse/fly.py`; round-7 review
clarification, 2026-09-15), `instrumented` is one word away, and every audit states which preset it ran.
`preset="raw"` with an instrument raises; the constructor never builds an instrument on its own.

Legacy explicit `--hold-edges` / `--nt-override` diagnostics keep their separate provenance under `raw`, as the
round-7 handoff requires; `raw` without those flags is the plain comparator.

## 2. What an instrument must be

An instrument stands in for a piece of physiology the connectome names but the model cannot yet supply. It is admitted
only with all of:

1. **A named gap.** The docstring says which cells / synapses / physiology it replaces and points at the audit that
   established the gap (e.g. "PS196_b receives no signed turn input in the shipped body: `vnc_drive.md` section 6,
   NOTES compass round 2; Wang 2026 finding 3 names the same cell from the GLNO side").
2. **A source for its law.** The transfer it implements is taken from a measurement (a rate, a time constant, a sign,
   a tuning), quoted at page / figure level per `INTERP.md` 10.4 item 29, or the law is marked `unverified` in the
   docstring and in `describe()`.
3. **The neural boundary only**, in exactly one of three shapes, and never the body's yaw or speed (that is a
   *program*, `flyverse.programs` / `flyverse.cx`, which receives the body object and stays labelled as such):
   - a **module** reads `rate_hz` / `drive_mv` of selected cells and writes `poisson_hz` / `drive_mv` to selected
     cells;
   - a **body-derived transducer** reads the existing sensory arguments and writes the same neural input channels;
   - a **configuration record** (the `edges` and `relabel` kinds) verifies a caller configuration before stepping and
     performs no runtime mutation. Removing such a record alone does not undo that configuration; returning to `raw`
     also means restoring the original connectome and parameters.

   No instrument of any shape touches weights, receptors or `NT_SIGN` at runtime.
4. **A held edge is an instrument too.** `--hold-edges` (6A) is an `edges`-kind hold; under `instrumented` it is
   attached and described like any module, so a preset with the ExR6 / ER6 / ER4m hold *says so* in every JSON.
5. **Its own suite run.** The 29-check suite under `instrumented` is a second column beside `raw`; an instrument that
   moves a row the gap does not cover is rejected (the round-2 rule: >= 3 draws, no status change outside the rows the
   instrument is declared to touch; the room rate-half at >= 6 runs per arm). Results to date: section 7.
6. **A removal condition.** The docstring names what result would retire it (a receptor row, a physiology paper, a
   data-driven default that reproduces its effect).

`flyverse.instruments._check_instrument` enforces what it can: every `describe()` must carry `name`, `kind`, `law`,
`gap`, `removal`, `audits` and `replaces`, and a `law` other than the word `unverified` must come with a nonempty
`source` or `sources` (item 2).

## 3. The candidates on the table (2026-09-15), in the order they would be tried

| instrument | kind | gap it fills | law and source | status |
|---|---|---|---|---|
| `sided_turn_afferent` | stop-gap | PS196_b's signed turn input (AN07B037_a/_b, CB0675 / GNG580 / PS047_b): the shipped body's report arrives unsigned (`vnc_drive.md`, NOTES compass round 2); Wang 2026 finding 3 makes PS196_b GLNO's largest non-ring input (1,801 syn, reproduced) | Poisson drive on the ascending afferents proportional to signed yaw velocity from the haltere / leg readout; gain **unverified** (no PS196_b recording exists -- searched 2026-09-15: Wang's audit, Hulse 2021, the two Rockefeller theses) | built and run: round 7 arms V / HGV / HGV- / HGVp and the round-8 V- sign control; **admissible** at `k` 0.5 as part of the three-instrument list (section 7) |
| `ring_dc_hold` | edges | The ExR6 / ER6 / ER4m DC term on PEN / EPG that removes the ring's resting state (`compass_dc_balance.md`) | the 6A `--hold-edges` hold; removal condition: receptor rows at the EB / GA contacts (Eddy 2026 preprint is the lead) | built and run: round 6A, round 7 component; **admissible** as part of the three-instrument list (section 7) |
| `glno_sign` | relabel | GLNO sign 0 (two disagreeing EM predictions, `glno_relabel.md`) | glutamate per the stronger prediction; safe on the suite, fixes nothing alone (5B) | built and run: round 2 / 5B relabel, round 7 component; **admissible** as part of the three-instrument list (section 7) |
| `flight_state` | mechanism | The OA-dependent flight state (the octopaminergic drive that gates wing MNs and the haltere loop); nothing in the model supplies it | law from the flight-initiation literature; **not yet specified** | **not built.** The shipped `flight` instrument (section 6) is a different object: an engineering wing-power servo on the wing MNs, not this flight state |
| `CompassSteering` (`--program cx`) | program | the whole compass -> PFL3 -> DNa02 stage | already ships, labelled, writes PFL3 / DNp09 | stays a `--program`, not an instrument, because it is a body-side program object. The criterion originally given for that exclusion -- "it replaces a computation, not a missing input" -- was superseded on 2026-09-17; see the note below |

**Why `CompassSteering` is still a program (reconciliation, 2026-09-17).** The original reason given in the table --
that it "replaces a computation, not a missing input" -- no longer excludes anything, because the owner's section-5
extension admits program-shaped stand-ins as instruments. The surviving reason is structural and checkable in the
code: `flyverse.cx.CompassSteering.apply(fb, motor, fly, metabolism, dt_s)` receives the **body object and the
metabolism** and drives the brain with `fb.stimulate`, outside the module scheduler. That is what section 2 item 3
excludes, and it is unchanged by section 5. `CompassDriver`, `plume` and `flight` receive no body object, run inside
the extension scheduler, and declare `replaces: "computation"`.

**The order was forced, and round 7 settled what it could.** No instrument downstream of a heading bump (hDelta
integration, goal comparison) was worth building until `sided_turn_afferent` + `ring_dc_hold` + `glno_sign` either
produced a bump that follows the fly's own turn or were shown not to. Round 7 ran
(`docs/audits/compass_velocity_route.md` section 5.1): the signed afferent reaches GLNO -- V minus S raises GLNO L-R
by +2.2183 Hz, a `result` at Holm p 0.0130 -- but the round **demonstrates no turn-following compass**. Primaries 1
and 2 are `undetermined`: no HGV run passes the predeclared 50 % turn-window confinement gate, so those are
unavailable comparisons, not zero velocity and not statistical nulls. PEN L-R and DNa02 L-R are `null`. Nothing
downstream of a heading bump has been built since.

## 4. What this is not

Not a licence to tune. An instrument's gain is a declared parameter with a source or the word `unverified`, swept only
as a labelled arm; it never becomes a fitted constant hidden in a default. Not a change to `raw`. Not a way to move a
suite row: the `raw` column is the one the README reports as the model's own behaviour, and the `instrumented` column
is reported beside it with the instrument list in the caption. Passing the section-2 item-5 gate makes a list
*admissible* and nothing more; it is not adoption.

## 5. Owner extension: imposed compass memory (2026-09-15)

After round 7 and its direct-GLNO follow-up failed to recover heading integration, the owner explicitly
authorized a simplified program/control arm under `instrumented`. `CompassDriver` is this larger stand-in:
it supplies angular memory and a continuous EPG Poisson representation, rather than claiming to repair
a discovered physiological mechanism. Its exact law is **unverified**. The source graph and raw default stay
unchanged. It is selectable by `instruments=["compass"]`; `instrumented` alone remains empty.

The existing proprioception entry point may forward its held `yaw_rate` to an explicitly named instrument's
`observe_turn` receiver. The compass receives no body object, absolute heading, visual landmark, goal or
motor command. Its neural output, stepping, checkpoint and reset use the ordinary module interface. This
extends the earlier round-7 clarification; `SidedTurnAfferent` still lives in its original sense transducer.

Admission checks and the distinction between an experimental option and default adoption are recorded in
[audits/compass_standin.md](audits/compass_standin.md). A functional phase-tracking pass does not establish
food finding, circuit recovery, or permission to change the raw model's constants.

**Program-shaped stand-ins (owner decision, 2026-09-17).** Section 3's CompassSteering row made "it replaces a
computation, not a missing input" the reason a program is not an instrument, while `CompassDriver`, `plume` (PFL3 +
DNp09) and `flight` (wing MNs) are exactly that shape and ship as instruments. The reconciliation is explicit, not
implicit: program-shaped stand-ins -- `CompassDriver`, `plume`, `flight`: the ones that replace a *computation*
rather than supply a missing *input* -- are admitted as instruments **ONLY under the owner's section-5 extension**;
they are distinguished in every `describe()` by a `replaces` field (`"input"` | `"computation"`); and the README and
any suite column that uses them name them as computation stand-ins. The section-3 configuration records
(`edges`, `relabel`) supply neither and carry `replaces: "configuration"`. `flyverse.instruments._check_instrument`
requires the field on every instrument, so an instrument that does not declare what it replaces cannot attach.
Nothing here relaxes section 2: the law is still `unverified` or sourced, the neural boundary is unchanged, and the
`raw` default is unchanged.

## 6. Owner extension: navigation experiments (2026-09-15)

The owner subsequently requested a more direct Wang/paper implementation, plume/flight controls,
and metabolic modulation under explicit instruments. `compass_ring` implements a reduced recurrent
rate model as an alternative to `compass`. `plume` supplies an ideal goal memory and a published
PFL3 comparator with an engineering neural-output bridge. `hunger` optionally scales navigation
from normalized energy; `flight` is an explicit wing-MN control experiment. All remain labelled
`stop-gap`, `law=unverified`: measured parts do not validate the complete composite law.

Named receivers may consume held antennal deflections through `wind()`, bilateral physical
concentrations through `smell()`, and normalized energy,
sated/airborne/feeding flags through `interoception()`, as the existing compass consumes held yaw.
This is explicit body-derived sensory input; no module receives the body object or world geometry.
These receivers are constructed only when named. General module read/write boundaries and the raw
default are unchanged. See [the navigation audit](audits/navigation_instruments.md) for exact laws,
source limits, incompatible combinations, and functional versus admission gates.

The 2026-09-16 plume correction reads its existing smell arguments for a local walking goal
and closes its PFL3 output using measured DNa02 L-R. Both are explicitly unverified stand-ins;
no source position, body command or synaptic edit is introduced. Details and admission limits:
[audits/plume_steering.md](audits/plume_steering.md).

The round-8 lever set is the same instrument under keywords, recorded in its own `describe()`:
`plume:bilateral=orn` ([audits/plume_transduced.md](audits/plume_transduced.md)),
`plume:feedback=off` (spelled `plume:feedback=0` as well) and `plume:walking_goal=0`
([audits/plume_goal_only.md](audits/plume_goal_only.md)). Bare `plume` is unchanged, parameter for parameter.

## 7. Admission results against the section-2 item-5 gate

The gate is section 2 item 5. Passing it makes a list *admissible*; nothing is adopted, and `raw` stays the default.

| list | suite half (3 + 3 draws) | room rate-half (>= 6 runs per arm) | verdict | record |
|---|---|---|---|---|
| `sided_turn_afferent:k=0.5` + `ring_dc_hold` + `glno_sign` | no row changes status: 27/0/2, 26/1/2, 27/0/2 under both presets; the one non-uniform row, `taste.MN9_hz`, is raw's own instability and is identical under the instruments | passes: 110 take-offs against 101 over 28,800 fly-s each (3.82 vs 3.51 per 1,000 fly-s; one-sided exact Poisson p 0.291; run-level `null`) | **admissible** | [audits/instrumented_suite.md](audits/instrumented_suite.md) |
| `compass` | one status change outside the declared gap: `taste.MN9_hz` seed 1, 1.690456 -> 4.295961 Hz, FAIL to PASS (seed 0 falls 10.934177 -> 2.478648 Hz while retaining PASS) | not run (the 300 s rate-half is owed) | **rejected** | [audits/compass_standin.md](audits/compass_standin.md) |
| everything else that ships (`ring_dc_hold_pen`, `glno_pen_hold`, `edge_gain_<i>`, `compass_ring`, `plume` and its variants, `hunger`, `flight`) | not run | not run | **untested by the gate** | their own audits; see [INSTRUMENTS.md](INSTRUMENTS.md) |

Two readings the admissible result does **not** license. It is not vacuous -- under the three instruments the ring's
operating state moves substantially during the suite's wedge drive (PEN 2.01 / 1.78 / 1.08 Hz under `raw` against
7.72 / 7.66 / 7.22 Hz, rest-EPG 0.000 x 3 against 1.36 / 2.12 / 3.36 Hz, three draws against three with no overlap)
while every check keeps its status; values move on 20 of the 29 rows and 8 rows are bit-identical to `raw`. And it is
not an endorsement: the afferent's law is still `unverified`, the relabel still has no transmitter source, a
predeclared fewer-meals descriptive (1-4 per run instrumented against 0-7 raw, `null` at 6 v 6) is on the record, and
so is `gf_max_hz`, the one measure in that set with a rank-test p below 0.05 (exact U p 0.0152, undeclared, `null`
under the z >= 3 rule). The suite half tests the hold and the relabel everywhere and the afferent only where a body
feeds it yaw, and no run record carries the afferent's rate, so both halves establish that it was attached, not that
it fired ([audits/instrumented_suite.md](audits/instrumented_suite.md) section 6).
