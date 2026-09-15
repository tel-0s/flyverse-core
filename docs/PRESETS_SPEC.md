# Model presets: `raw` and `instrumented`

Spec, 2026-09-15 (owner decision of the same day: labelled stand-ins are acceptable so long as they are as physiologically
realistic as the literature allows, optional, and a `raw` preset returns the pure brain with none of them). Nothing in
this document changes a default; it fixes the vocabulary and the acceptance rules so the round that ships the first
instrument has something to be measured against.

## 1. Two presets, one brain

| preset | what it is | bit-identity |
|---|---|---|
| `raw` | The connectome, the LIF, the receptor table, the senses and the motor readout exactly as shipped. No module attached, no held edge, no relabel beyond the sourced `TYPE_NT_OVERRIDE` rows. This is the plain model every audit refers to. | Byte-identical to today's `FlyBrain()` on every path (`tests/test_bit_identity.py` extends to assert it). |
| `instrumented` | `raw` plus a **named list of instruments**, each a `flyverse.modules` object of kind `stop-gap` or `mechanism`, attached through the ordinary `attach()` surface -- an external input at the neural boundary, never a rewritten synapse (CONTROL_SURFACE "Hooks, modules, graph extension" already forbids that). | Provenance records `preset` and the instrument list with each instrument's `describe()`; removing every instrument reproduces `raw`. |

The preset is one keyword on the constructors and the CLIs (`FlyBrain(preset="raw")`, `--preset raw`), threaded into
`provenance()` beside `connectome_fingerprint`. Which preset is the default is the owner's call and is **not** decided
here; either way `raw` stays one word away, and every audit states which preset it ran.

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
