# Model presets: `raw` and `instrumented`

Spec, 2026-09-15 (owner decision of the same day: labelled stand-ins are acceptable so long as they are as physiologically
realistic as the literature allows, optional, and a `raw` preset returns the pure brain with none of them). Nothing in
this document changes a default; it fixes the vocabulary and the acceptance rules so the round that ships the first
instrument has something to be measured against.

## 1. Two presets, one brain

| preset | what it is | bit-identity |
|---|---|---|
| `raw` | The connectome, the LIF, the receptor table, the senses and the motor readout exactly as shipped. No module attached, no held edge, no relabel beyond the sourced `TYPE_NT_OVERRIDE` rows. This is the plain model every audit refers to. | Byte-identical to today's `FlyBrain()` on every path (`tests/test_bit_identity.py` extends to assert it). |
| `instrumented` | `raw` plus a **named list of instruments**, installed through the existing surface for their kind: neural computations use ordinary `attach()` modules; body-derived inputs use the proprioception transducer; the held-edge and relabel candidates below verify an explicit caller configuration. No runtime module rewrites a parent synapse. | Provenance records `preset` and each instrument's `describe()`. An empty list on the original connectome and LIF parameters reproduces `raw`. |

The preset is one keyword on the constructors and the CLIs (`FlyBrain(preset="raw")`, `--preset raw`), threaded into
`provenance()` beside `connectome_fingerprint`. Which preset is the default is the owner's call and is **not** decided
here; either way `raw` stays one word away, and every audit states which preset it ran.

Round-7 review clarification (2026-09-15, before submission): the handoff explicitly places
`SidedTurnAfferent` in `senses.Proprioception`, reading the existing `yaw_rate` channel. This is the
body-derived input case above, not a new body-to-neural-module API. The section-3 `edges` and `relabel`
candidates are configuration records: they verify holds already installed through `LIFParams` and
labels already compiled in a scratch connectome. Removing such a record alone does not undo that
configuration; returning to raw also means restoring the original connectome and parameters.
The default remains `raw`. Legacy explicit `--hold-edges` / `--nt-override` diagnostics retain their
separate provenance under raw, as required by the handoff; raw without those flags is the plain comparator.

## 2. What an instrument must be

An instrument stands in for a piece of physiology the connectome names but the model cannot yet supply. It is admitted
only with all of:

1. **A named gap.** The docstring says which cells / synapses / physiology it replaces and points at the audit that
   established the gap (e.g. "PS196_b receives no signed turn input in the shipped body: `vnc_drive.md` section 6,
   NOTES compass round 2; Wang 2026 finding 3 names the same cell from the GLNO side").
2. **A source for its law.** The transfer it implements is taken from a measurement (a rate, a time constant, a sign,
   a tuning), quoted at page / figure level per `INTERP.md` 10.4 item 29, or the law is marked `unverified` in the
   docstring and in `describe()`.
3. **The neural boundary only.** It reads `rate_hz` / `drive_mv` of selected cells and writes `poisson_hz` / `drive_mv`
   to selected cells. It never writes the body's yaw or speed (that is a *program*, `flyverse.programs`, and stays
   labelled as such), never touches weights, receptors or `NT_SIGN`.
   A body-derived transducer reads the existing sensory arguments and writes the same neural input channels.
   Held-edge/relabel records verify caller configuration before stepping; they do not perform runtime mutation.
4. **A held edge is an instrument too.** `--hold-edges` (6A) is an `edges`-kind hold; under `instrumented` it is
   attached and described like any module, so a preset with the ExR6 / ER6 / ER4m hold *says so* in every JSON.
5. **Its own suite run.** The 29-check suite under `instrumented` is a second column beside `raw`; an instrument that
   moves a row the gap does not cover is rejected (the round-2 rule: >= 3 draws, no status change outside the rows the
   instrument is declared to touch; the room rate-half at >= 6 runs per arm).
6. **A removal condition.** The docstring names what result would retire it (a receptor row, a physiology paper, a
   data-driven default that reproduces its effect).

## 3. The candidates on the table (2026-09-15), in the order they would be tried

| instrument | kind | gap it fills | law and source | status |
|---|---|---|---|---|
| `sided_turn_afferent` | stop-gap | PS196_b's signed turn input (AN07B037_a/_b, CB0675 / GNG580 / PS047_b): the shipped body's report arrives unsigned (`vnc_drive.md`, NOTES compass round 2); Wang 2026 finding 3 makes PS196_b GLNO's largest non-ring input (1,801 syn, reproduced) | Poisson drive on the ascending afferents proportional to signed yaw velocity from the haltere / leg readout; gain **unverified** (no PS196_b recording exists -- searched 2026-09-15: Wang's audit, Hulse 2021, the two Rockefeller theses) | round 7 arm |
| `ring_dc_hold` | edges | The ExR6 / ER6 / ER4m DC term on PEN / EPG that removes the ring's resting state (`compass_dc_balance.md`) | the 6A `--hold-edges` hold; removal condition: receptor rows at the EB / GA contacts (Eddy 2026 preprint is the lead) | round 7 arm, as a component |
| `glno_sign` | relabel | GLNO sign 0 (two disagreeing EM predictions, `glno_relabel.md`) | glutamate per the stronger prediction; safe on the suite, fixes nothing alone (5B) | round 7 arm, as a component |
| `flight_state` | mechanism | The OA-dependent flight state (the octopaminergic drive that gates wing MNs and the haltere loop); nothing in the model supplies it | law from the flight-initiation literature; **not yet specified** | after turning |
| `CompassSteering` (`--program cx`) | program | the whole compass -> PFL3 -> DNa02 stage | already ships, labelled, writes PFL3 / DNp09 | stays a program, not an instrument: it replaces a computation, not a missing input |

The order is forced: no instrument downstream of a heading bump (hDelta integration, goal comparison) is worth
building until `sided_turn_afferent` + `ring_dc_hold` + `glno_sign` either produce a bump that follows the fly's own
turn or are shown not to (round 7, `docs/audits/compass_velocity_route.md`).

## 4. What this is not

Not a licence to tune. An instrument's gain is a declared parameter with a source or the word `unverified`, swept only
as a labelled arm; it never becomes a fitted constant hidden in a default. Not a change to `raw`. Not a way to move a
suite row: the `raw` column is the one the README reports as the model's own behaviour, and the `instrumented` column
is reported beside it with the instrument list in the caption.

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
