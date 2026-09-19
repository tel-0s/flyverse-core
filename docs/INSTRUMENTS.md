# Instruments: what ships, how to turn one on, what it stands in for, what retires it

The contract is [PRESETS_SPEC.md](PRESETS_SPEC.md) (owner decision of 2026-09-15: labelled physiological stand-ins
are acceptable if sourced, optional, and `raw` returns the pure brain). This page is the inventory, one uniform block
per shipped instrument.

**Nothing here is on by default.** `FlyBrain()` is `FlyBrain(preset="raw")` -- `preset="raw"` is the same model
as `FlyBrain()`, and bit-identity is a CPU claim (`tests/test_bit_identity.py`, `RawPresetTests`;
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) 4.4); `preset="raw"` with an instrument raises; `preset="instrumented"`
with an empty list reproduces `raw` on the original connectome and LIF parameters (PRESETS_SPEC section 1); and every
audit states which preset it ran. Nothing on this page is adopted into `raw`, and no instrument changes a suite
row's *status* outside the rows its gap is declared to touch -- an instrument that does is rejected, as `compass`
was (the policy is PRESETS_SPEC section 4, the results section 7). Values do move: 20 of the 29 rows under the
admissible list, with 8 rows bit-identical to `raw` ([audits/instrumented_suite.md](audits/instrumented_suite.md)
section 4).

## The two licensed milestone sentences

These are the only behavioural milestone claims the instruments license, and this page is where they are placed for
the release; neither is restated in looser words anywhere in this file. Both are quoted, and their audit homes are
named with each quote below.

**On finding food under the instrumented preset** -- the sentence
[docs/NOTES.md](NOTES.md) "Session 14, continued" records as the skeptic's proposed wording, quoted for the owner to
place (the room evidence is [audits/plume_transduced.md](audits/plume_transduced.md) section 7):

> In six 60 s tabletop rooms per arm, the shipped `plume` instrument -- which reads the physical odour-concentration
> difference between the antennae -- fed in 6/6 rooms, while the opt-in `plume:bilateral=orn` variant, which
> substitutes the model's own ORN population rates at the same gain, fed in 1/6 and steered on a cue uncorrelated
> with the true lateral contrast; a descriptive room observation, not a significance test, an SNR measurement, or a
> claim about flies.

**On powered flight and feeding** -- the sentence carried verbatim by
[audits/navigation_instruments.md](audits/navigation_instruments.md) section 4,
[audits/flight_foraging_priority.md](audits/flight_foraging_priority.md) section 3 and
[audits/plume_steering.md](audits/plume_steering.md) section 6, required of all three by
[audits/receptor_verification.md](audits/receptor_verification.md) (the merge record; "Corrections required", item 13):

> No run in these audits shows artificially powered flight and feeding in the same episode: the only powered-flight
> rooms are the superseded navigation flight arm (59.67 s airborne, zero feeding in all six rows), and `powered_s`
> is 0.00 in all 12 flight-priority room rows and all 12 plume room rows.

What the rooms do *not* do is confirm the CPU SNR characterisation of
[audits/plume_transduced.md](audits/plume_transduced.md) section 3. The licensed wording is that audit's own:

> The in-room ORN L-R and its temporal SD are different quantities from the CPU's fixed-contrast signal and 250 ms
> counting-window noise, and are not an SNR; what the rooms add is that the measured in-room cue noise is 1.3-2.6x
> the analytical estimate and that the cue's sign was right in only 0.52 +/- 0.10 of samples. The rooms do not
> confirm the section-3 SNR result and are not offered as confirming it.

## Turning one on

```python
from flyverse import instruments as fi
from flyverse.fly import FlyBrain
from flyverse.interp.common import provenance

inst = fi.SidedTurnAfferent(c, k_hz_per_deg_s=0.5, sign=+1, cells="AN07B037")
fb = FlyBrain(c, preset="instrumented", instruments=[inst])       # preset="raw" with instruments raises
fb.proprioception(leg_L, leg_R, haltere, airborne, yaw_rate=...)  # the body channel that already exists feeds it
provenance(c, fb=fb)["preset"], provenance(c, fb=fb)["instruments"]   # 'instrumented', [inst.describe()]
```

`instruments=` takes objects or the names below (a bare string is one name). Construct an instrument on the effective
connectome: a different cell ordering is refused (`validate_parent`). `BatchSim(..., preset="instrumented",
instruments=[...], proprioception="all")` forwards both and re-installs a sense-side stand-in onto its own sense
(the spec grows `+turn_afferent`); naming `all+turn_afferent` under the instrumented preset also registers the
token's default stand-in, and `raw` refuses that token. Checkpoints require the same preset and the same
`describe()` records, including sign and gain, before loading; full and partial resets carry every instrument.

On the command line, `--preset raw|instrumented` and `--instruments NAME [NAME ...]` (with the repeatable
`--instrument NAME` alias) are accepted by `scripts/room_demo.py`, `scripts/benchmark.py` and `scripts/cx_wedge.py`:

```sh
python scripts/room_demo.py --preset instrumented --instruments compass plume hunger flight --brain-map
python scripts/benchmark.py --sections all --preset instrumented --instruments sided_turn_afferent:k=0.5 \
    --hold-edges '^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)' --nt-override GLNO=glutamate
```

| grammar | meaning |
|---|---|
| `--preset raw` | the default; byte-identical to omitting the flag |
| `--preset instrumented` | required before any instrument; `--instrument` implies it in `cx_wedge.py` |
| `NAME[:key=value]...` | the instrument spec grammar (`flyverse.instruments.SPEC_HELP`); the canonical spelling of a configuration is that instrument's own `describe()['parameters']['variant_spec']` |
| `--hold-edges PRE_RE:POST_RE` | `cx_wedge.py`, `benchmark.py`; refused under `raw` in `benchmark.py`. Under `instrumented` it is also recorded as an `edges` instrument |
| `--edge-gain PRE_RE:POST_RE:FACTOR` | `cx_wedge.py`; recorded as `edge_gain_<i>` under `instrumented` |
| `--nt-override TYPE=nt` | `cx_wedge.py`, `benchmark.py`; recorded as a `relabel` instrument under `instrumented` |
| `--turn DEG_S --turn-window START:END` | `cx_wedge.py` ledger runs only: a prescribed signed yaw (positive = a left turn) after the pulse |

Without `--preset instrumented` those three flags behave and record exactly as they did in rounds 6A / 6B / 2, in the
row rather than in the instrument list (`cx_wedge.build_instruments`).

## Composition

Duplicate names, incompatible heading providers, unmet dependencies and overlapping neural writes raise errors, at
CLI parse time (`validate_cli`) and again before installation (`validate_composition`). Detaching a required provider
also raises; remove dependent instruments first.

| rule | instruments |
|---|---|
| incompatible | `compass` with `compass_ring` (two heading providers) |
| requires one of | `plume` needs `compass` or `compass_ring`; `hunger` needs `plume` or `flight` |
| no two writers on one cell and channel | enforced over every resolved `writes` selection |

Direct `FlyBrain` callers supply the body-derived inputs themselves; the room demo and `BatchSim` do it
automatically. Energy is in `[0,1]`, flags are Boolean, each accepts a scalar or one value per batch row, and inputs
persist until updated:

```python
fb.wind(dL, dR)            # signed antennal deflections, not a world wind angle
fb.smell(cL, cR)           # per-glomerulus concentration dictionaries, one per antenna
fb.interoception(energy, sated=sated, airborne=airborne, feeding=feeding)
fb.proprioception(0, 0, 0, airborne, yaw_rate=yaw_rad_per_s)
fb.step(10)
```

A named receiver may consume these on a selected graph that lacks the corresponding sensory cells -- `plume` can
receive wind without JO cells, the compass can receive yaw without a proprioceptive afferent module -- exactly as
PRESETS_SPEC section 6 authorizes. That receiver does not enable any other sensory pathway.

## How to read a block

Every block below carries the same nine fields, in the same order -- the six PRESETS_SPEC section 2 requires of an
instrument, plus its name and grammar, its `replaces` class (section 5) and the round(s) it was used in. Each field
is checked against that instrument's `describe()` in `flyverse/instruments.py`, `flyverse/navigation.py` or
`flyverse/compass.py`.

1. **Grammar** -- the spec string and the class.
2. **Kind** -- `stop-gap` / `mechanism` / `edges` / `relabel`.
3. **Replaces** -- `input` / `computation` / `configuration` (PRESETS_SPEC section 5).
4. **Law** -- the word `unverified`, or the law with its source (section 2 item 2).
5. **Gap** -- the physiology it stands in for, with the audit that established it (item 1).
6. **Reads / writes** -- the neural boundary, with every body-derived input named (item 3).
7. **Removal** -- what result retires it (item 6).
8. **Admission** -- its standing against the section-2 item-5 gate.
9. **Rounds** -- where it has actually been run.

"Admissible" means only the PRESETS_SPEC section 2 item 5 gate. It is not adoption, not a claim that a gain is right,
and not permission to change a default.

## `sided_turn_afferent`

- **Grammar** -- `sided_turn_afferent[:k=0.25|0.5|1.0][:sign=+1|-1][:cells=AN07B037|CB0675|GNG580|PS047_b|all][:max_hz=HZ]`;
  `flyverse.instruments:SidedTurnAfferent`. Defaults `k=0.5`, `sign=+1`, `cells=AN07B037`, `max_hz=250`.
- **Kind** `stop-gap`. **Replaces** `input` -- a missing body-derived input, not a computation.
- **Law** `unverified`. `poisson_hz = k * max(0, sign * yaw_deg_s)` on the left afferents and
  `k * max(0, -sign * yaw_deg_s)` on the right, clamped to `max_hz`; positive yaw is a left turn
  (`body.Locomotion`). `k` is a declared level in Hz per deg/s over `K_LEVELS = (0.25, 0.5, 1.0)`, 0.5 the primary
  ([audits/compass_velocity_route.md](audits/compass_velocity_route.md) section 2); `sign = -1` is the control arm.
  No PS196_b / AN07B037 recording exists -- searched 2026-09-15 in Wang's fly-circuit-exploration audit, Hulse et al.
  2021 (eLife 66039) and the two Rockefeller theses (PRESETS_SPEC section 3).
- **Gap** -- PS196_b receives no *signed* turn input in the shipped body. PS196_b is GLNO's largest non-ring input
  (1,801 synapses, 19.2 % of GLNO's input in [audits/cx_shift.md](audits/cx_shift.md) section 1, quoted as 19-21 %
  in [audits/compass_velocity_route.md](audits/compass_velocity_route.md) section 1; Wang 2026 finding 3,
  reproduced to the synapse), and GLNO is PEN's one nodulus
  input of size (16,371 synapses, 19.4 % of PEN's raw input, fully contralateral --
  [audits/cx_shift.md](audits/cx_shift.md) section 1). The ascending report that reaches PS196_b arrives unsigned:
  under the labelled Coriolis stop-gap its L-R moves the same way in both turn directions
  ([audits/vnc_drive.md](audits/vnc_drive.md) section 6, [docs/NOTES.md](NOTES.md) compass round 2).
- **Reads / writes** -- reads the body's yaw rate (rad/s) through `FlyBrain.proprioception(..., yaw_rate=)`, the
  channel the `haltere_coriolis` stop-gap already reads; writes `poisson_hz` on the afferent cells through
  `FlyBrain._input` (`'proprioception_turn_afferent'`). Never a weight, a receptor, `NT_SIGN` or the body. Zero yaw
  is zero everywhere (`tests/test_instruments.py`); `max_hz` is a safety ceiling, not a law.
- **Removal** -- a recording of PS196_b / AN07B037 during turning (then a `mechanism` with a source, or dropped if
  the tuning is flat); a sided ascending report that reaches PS196_b signed on its own (the round-3 sided tokens did
  not: [audits/body_sided_state.md](audits/body_sided_state.md)); or a round-7 `null` on measure 3 at every declared
  level.
- **Admission** -- part of the admissible three-instrument list at `k=0.5`
  ([audits/instrumented_suite.md](audits/instrumented_suite.md); see **Admission status** below). Both halves
  establish that it was attached, not that it fired: the afferent receives yaw only where a body feeds one (the room
  sections), and no suite run record carries its rate (that audit's section 6).
- **Rounds** -- round 7 arms V / HGV / HGV- / HGVp and the k sweep
  ([audits/compass_velocity_route.md](audits/compass_velocity_route.md) sections 2, 5); round 8 item 2, the V- sign
  control ([audits/compass_sign_control.md](audits/compass_sign_control.md)); round 8 item 3, the instrumented suite
  ([audits/instrumented_suite.md](audits/instrumented_suite.md)).

### Where it lives, and why

It is a **body-derived** signal (the realised yaw rate), so it lives where the `haltere_coriolis` stop-gap lives: the
`senses.Proprioception` transducer, as the extra channel `'turn_afferent'` (never part of `'all'`), injected through
`FlyBrain._input` like every other sense. A `flyverse.modules` module sees neural quantities only, and no neural
quantity in this model carries the fly's own turn with a sign, so this instrument stays in the transducer. The later
compass candidate has an explicit held-yaw receiver authorized by PRESETS_SPEC section 5; that does not change the
afferent's implementation (PRESETS_SPEC section 1, round-7 review clarification, and section 5's own sentence that
`SidedTurnAfferent` still lives in its original sense transducer).

### The side, from the graph

`describe()['routing']` is read from `W[post, pre]` with `somaSide` at construction, with raw counts through
`interp.common.raw_counts` so the sign-0 GLNO -> PEN block counts too. On the shipped cache every hop of the route is
contralateral ([audits/instruments_review.md](audits/instruments_review.md) section 3):

| hop | L -> L | L -> R | R -> L | R -> R |
|---|---:|---:|---:|---:|
| AN07B037_a -> PS196_b | 0 | 202 | 214 | 3 |
| AN07B037_b -> PS196_b | 0 | 28 | 24 | 0 |
| PS196_b -> GLNO | 0 | 832 | 966 | 3 |
| GLNO -> PEN_a(PEN1) | 0 | 5,024 | 4,782 | 0 |
| GLNO -> PEN_b(PEN2) | 0 | 3,212 | 3,353 | 0 |

So with `sign=+1` a **left** turn (positive yaw, the body's convention) drives the **left** AN07B037 and the chain
lands on `PS196_b_R -> GLNO_L -> PEN_R` (`describe()['chain_for_positive_yaw']`). The `cells` variants are
ipsilateral onto PS196_b -- CB0675 51 / 51, GNG580 10 / 13, PS047_b 185 / 217 same-side synapses, same source table --
so their chain lands one side over, and the record says so per variant. Which PEN side *should* carry a left turn is
exactly what no recording settles; that is why the sign arm exists and why `sign` is a parameter, not a constant.

### In `cx_wedge` there is no body

The turn is a prescribed protocol parameter (`--turn 90 --turn-window 0.5:3.5`), the analogue of the 90 deg/s
imposed visual rotation of [audits/deficit_rotation.md](audits/deficit_rotation.md); the efferent protocol's "fly
turning itself at 86-113 deg/s" is the odour-room body, not this script. The ledger records per-side PEN / GLNO /
DNa02 / PS196_b / afferent rates per frame, and `metrics` carries the round-7 measures `<G>_LR_hz` and
`bump_follow_wedges_per_s` -- read beside `bump_follow_confined_frac`, because on a dead bump the centre is noise.
The JSON carries `preset`, `instruments` and every `describe()` in `provenance`.

## `ring_dc_hold`

- **Grammar** -- `--hold-edges '^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)'` under `--preset instrumented`;
  `flyverse.instruments:EdgeHold` at factor 0, auto-named `ring_dc_hold` by `cx_wedge.build_instruments`.
- **Kind** `edges`. **Replaces** `configuration` -- a record of a caller configuration, neither an input nor a
  computation. The object does not install the hold; `LIFParams.type_path_gain` does, and `install()` refuses a
  FlyBrain whose gain list does not carry it, so the record cannot lie.
- **Law** `counterfactual`: `W[post ~ ^(PEN_|EPG$), pre ~ ^(ExR6|ER6|ER4m)$] x 0`. **No transfer is claimed.**
- **Gap** -- the ExR6 / ER6 / ER4m DC term on PEN / EPG removes the ring's resting state
  ([audits/compass_dc_balance.md](audits/compass_dc_balance.md)); receptor placement and kinetics at those contacts
  are open. On the shipped cache the hold selects 17 presynaptic cells onto 88 postsynaptic cells (46 EPG + 42 PEN),
  1,149 weight entries, 37,256 synapses ([audits/instruments_review.md](audits/instruments_review.md) section 3;
  [audits/compass_dc_balance.md](audits/compass_dc_balance.md) section 2).
- **Reads / writes** -- neither. It reads and writes no neural channel; it verifies a `LIFParams` configuration before
  stepping, and records the resolved counts in `describe()['resolved']`.
- **Removal** -- sourced receptor placement and kinetics at the EB / GA contacts of ExR6 / ER6 / ER4m that reproduce
  the physiological operating state. ExR6 glutamate and ER6 GABA already have type-level support
  ([audits/exr6_evidence.md](audits/exr6_evidence.md)); neither label is changed here.
- **Admission** -- part of the admissible three-instrument list
  ([audits/instrumented_suite.md](audits/instrumented_suite.md)). The hold is shown to act: during the suite's 2 s
  wedge drive PEN goes 2.01 / 1.78 / 1.08 Hz under `raw` to 7.72 / 7.66 / 7.22 Hz under the instruments and rest-EPG
  0.000 x 3 to 1.36 / 2.12 / 3.36 Hz, three draws against three with no overlap, while no check changes status.
- **Rounds** -- round 6A ([audits/compass_dc_balance.md](audits/compass_dc_balance.md)); round 6B as the common hold
  ([audits/compass_local_recurrence.md](audits/compass_local_recurrence.md)); round 7 arms HG / HGV / HGV- and the
  GLNO-to-PEN follow-up ([audits/compass_velocity_route.md](audits/compass_velocity_route.md)); round 8 item 3
  ([audits/instrumented_suite.md](audits/instrumented_suite.md)).

## `ring_dc_hold_pen`

- **Grammar** -- `--hold-edges '^(ExR6|ER6|ER4m)$:^PEN_'` under `--preset instrumented`; the same `EdgeHold` class,
  auto-named `ring_dc_hold_pen`.
- **Kind** `edges`. **Replaces** `configuration`.
- **Law** `counterfactual`: the same factor-0 hold restricted to PEN. **No transfer is claimed.**
- **Gap** -- the same DC-term gap as `ring_dc_hold`, narrowed: the full hold removes ring inhibition from every EPG,
  driven and off-block alike, so round 6B cannot say which part of its wide hump matters; this variant keeps EPG's
  DC input to test whether that narrows it ([audits/compass_velocity_route.md](audits/compass_velocity_route.md)
  section 2, arm HGVp). On the shipped cache it selects the same 17 presynaptic cells onto 42 PEN, 402 weight
  entries, 7,893 synapses ([audits/instruments_review.md](audits/instruments_review.md) section 3).
- **Reads / writes** -- neither; a configuration record, as above.
- **Removal** -- the same receptor / operating-state evidence as the full hold.
- **Admission** -- **untested by the gate.** It was not in the list [audits/instrumented_suite.md](audits/instrumented_suite.md)
  submitted, and no suite or room rate-half has been run with it.
- **Rounds** -- round 7 arm HGVp only ([audits/compass_velocity_route.md](audits/compass_velocity_route.md) sections
  2, 5). Its contrast with HGV is `null` under the full rule: Holm p 0.0130 but |z| = 2.3613, below the declared
  threshold of 3, and the PEN-only hold gives no confined post-pulse frames in any seed (that audit, section 5.1).

## `glno_pen_hold`

- **Grammar** -- `--hold-edges '^GLNO$:^PEN_'` under `--preset instrumented`; the same `EdgeHold` class, auto-named
  `glno_pen_hold`, with its own `gap` / `source` / `removal` / `audits` text.
- **Kind** `edges`. **Replaces** `configuration`.
- **Law** `counterfactual`: a pathway-removal control, not a receptor model.
- **Gap** -- the transfer of a GLNO side signal into PEN in round 7. On the GLNO-glutamate scratch graph it selects
  84 weight entries / 16,371 synapses from four GLNO onto 42 PEN
  ([audits/compass_velocity_route.md](audits/compass_velocity_route.md) section 6.1).
- **Reads / writes** -- neither; a configuration record.
- **Removal** -- retire after the diagnostic; no adoption is proposed.
- **Admission** -- **untested by the gate**, and not an adoption candidate.
- **Rounds** -- the round-7 GLNO-to-PEN follow-up, arms C0 / CL / CR
  ([audits/compass_velocity_route.md](audits/compass_velocity_route.md) section 6). Read its results with the
  audit's own caveat: the unforced cut control C0 already carries PEN L-R -11.0738 Hz, so removing the path shifts
  the operating state even without the challenge, and the held-path contrasts do not identify a unitary transfer at
  matched presynaptic rates (section 6.4).

## `glno_sign`

- **Grammar** -- `--nt-override GLNO=glutamate` under `--preset instrumented`;
  `flyverse.instruments:TypeRelabel`, auto-named `glno_sign`. The relabel must already be compiled into the scratch
  cache: `install()` refuses a connectome whose GLNO cells do not carry the transmitter.
- **Kind** `relabel`. **Replaces** `configuration`.
- **Law** `unverified`: type GLNO compiled as glutamate (its `NT_SIGN` sign) in a scratch cache -- the transmitter
  two of the three low-confidence EM classifiers call.
- **Gap** -- GLNO's transmitter is unknown: three EM classifiers agree only that GLNO is inhibitory and disagree on
  the transmitter (MaleCNS `unclear` at 0.48, glutamate 51 %; BANC glutamate 4/4 at ~0.5; FlyWire GABA 3/4 at ~0.3),
  none of the three a confident call: [audits/glno_relabel.md](audits/glno_relabel.md) 1.2,
  [audits/cx_glno.md](audits/cx_glno.md). Glutamate is safe on the suite and fixes nothing alone (round 5B).
- **Reads / writes** -- neither; it verifies a compiled label before stepping.
- **Removal** -- a transmitter source for GLNO that is not one EM classifier, at which point the row enters
  `TYPE_NT_OVERRIDE` itself.
- **Admission** -- part of the admissible three-instrument list
  ([audits/instrumented_suite.md](audits/instrumented_suite.md)). Admissible is all it is: the relabel still has no
  transmitter source.
- **Rounds** -- round 2 of the receptor integration and round 5B ([audits/cx_glno.md](audits/cx_glno.md),
  [audits/glno_relabel.md](audits/glno_relabel.md)); round 7 arms HG / HGV / HGV- / HGVp and the follow-up
  ([audits/compass_velocity_route.md](audits/compass_velocity_route.md)); round 8 item 3
  ([audits/instrumented_suite.md](audits/instrumented_suite.md)).

## `edge_gain_<i>`

- **Grammar** -- `--edge-gain PRE_REGEX:POST_REGEX:FACTOR` (repeatable) under `--preset instrumented`;
  `flyverse.instruments:EdgeGain`, named `edge_gain_0`, `edge_gain_1`, ... in flag order.
- **Kind** `edges`. **Replaces** `configuration`.
- **Law** `instrument (a per-type gain, no transfer claimed)`: `W[post ~ POST, pre ~ PRE] x FACTOR` through
  `LIFParams.type_path_gain`, appended after the holds. Its `source` field says so in words: no measurement -- a gain
  factor is not one.
- **Gap** -- a per-type undamping that rounds 5A / 6A asked for
  ([audits/compass_local_recurrence.md](audits/compass_local_recurrence.md)).
- **Reads / writes** -- neither; a configuration record with the resolved counts in `describe()['resolved']`.
- **Removal** -- a measured recurrence gain, or the round that retires the question.
- **Admission** -- **untested by the gate.**
- **Rounds** -- round 6B ([audits/compass_local_recurrence.md](audits/compass_local_recurrence.md)), where the row
  records `edge_gains` and `edge_gains_resolved` beside the instrument.

## `compass`

- **Grammar** -- `--instruments compass`, or `instruments=["compass"]`; `flyverse.compass:CompassDriver`. Parameters
  `peak_hz=50`, `width_deg=35`, `initial_phase_deg=0`, `velocity_gain=1` (a negative gain is an explicit sign-control
  arm). Incompatible with `compass_ring`. Requires `preset="instrumented"`; `raw` rejects it.
- **Kind** `stop-gap`. **Replaces** `computation` -- a program-shaped stand-in, admitted as an instrument **only**
  under the owner's PRESETS_SPEC section 5 extension.
- **Law** `unverified`: `phase += velocity_gain * yaw_rad_s * dt_s`; EPG Poisson Hz =
  `peak * exp(-0.5 * (wrapped angle error / width)^2)`, on the 16 interleaved EB wedges. The kinematic law, the
  35 degree Gaussian width and the 50 Hz peak are engineering choices. The wedge mapping follows the instance
  annotations and the interleaved map checked by Wang (pinned commit 80b94e6); the functional motivation is
  Turner-Evans et al. 2017, eLife 23496. Neither source validates this law.
- **Gap** -- the plain EPG ring does not maintain and update a confined heading bump (compass rounds 6-7):
  [audits/compass_standin.md](audits/compass_standin.md),
  [audits/compass_velocity_route.md](audits/compass_velocity_route.md).
- **Reads / writes** -- reads nothing neural (`reads = {}`); receives held body `yaw_rate` in rad/s through
  `FlyBrain.proprioception(0, 0, 0, False, yaw_rate=...)` -> `observe_turn`, held until updated, zero it on stopping.
  Writes a continuous Poisson drive on the biological EPG cells and nothing else. No body object, absolute heading,
  visual landmark, goal or motor command; initial phase is arbitrary; manual room pose teleports are not sensed
  turns. Parent spikes and edges are untouched. Full and partial resets and checkpoints include the driver.
- **Removal** -- a native circuit that maintains and integrates a heading bump under the same sustained-turn,
  reversal and stationary/dark-memory checks.
- **Admission** -- **rejected.** [audits/compass_standin.md](audits/compass_standin.md) ran the three-draw 29-check
  suite: no passing row regresses, but `taste.MN9_hz` at seed 1 moves 1.690456 -> 4.295961 Hz, FAIL to PASS, outside
  the declared heading gap, and that fails strict admission regardless of its favourable direction; seed 0 taste
  falls 10.934177 -> 2.478648 Hz while retaining PASS. The 300 s room rate-half was not run. That audit's Report
  verdict line, quoted: `experimental option only; admission rejected; no physiological default adopted`.
- **Rounds** -- its own admission study ([audits/compass_standin.md](audits/compass_standin.md)); the heading
  provider in every navigation experiment since
  ([audits/navigation_instruments.md](audits/navigation_instruments.md),
  [audits/plume_steering.md](audits/plume_steering.md),
  [audits/flight_foraging_priority.md](audits/flight_foraging_priority.md),
  [audits/plume_goal_only.md](audits/plume_goal_only.md),
  [audits/plume_transduced.md](audits/plume_transduced.md) section 7).

A functional phase-tracking pass does not establish food finding, circuit recovery, or permission to change the raw
model's constants. The UI names the preset and the instrument; provenance carries the complete law, the target
bodyIds and the wedge assignment.

## `compass_ring`

- **Grammar** -- `--instruments compass_ring`; `flyverse.navigation:RecurrentCompass`. Parameters `eb_ratio=0.0`,
  `velocity_gain=0.3/(pi/2)`, `tau_ms=50`, `substep_ms=2`, output `0.0001 + 10 * e[wedge]` Hz. Incompatible with
  `compass`; naming both is an error.
- **Kind** `stop-gap`. **Replaces** `computation` (PRESETS_SPEC section 5).
- **Law** `unverified`. Wang's reduced recurrent 16 EPG / 16 PEN rate loop, independently expressed as batched
  matrix dynamics (no external code is imported or executed; noise is 0 instead of Wang's 0.02). Memory lives in
  recurrent rates, not an integrated phase scalar. The default EB/PB ratio 0 is a **declared textbook
  counterfactual**, never a parent edge deletion; the diagnostic ratio 2.7 is a hemibrain count ratio, not a
  measured efficacy in MaleCNS. `v = clip(yaw_rad_s * 0.3/(pi/2), -0.8, 0.8)` is **uncalibrated**: 90 deg/s in does
  not imply 90 deg/s of bump velocity ([audits/navigation_instruments.md](audits/navigation_instruments.md) section
  2). `parameters['calibrated']` is `False` in the record.
- **Gap** -- a recurrent angular-memory experiment beside the kinematic compass stand-in
  ([audits/navigation_instruments.md](audits/navigation_instruments.md)).
- **Reads / writes** -- reads nothing neural; receives held body `yaw_rate` through
  `FlyBrain.proprioception` -> `observe_turn` (no absolute heading, goal or world geometry). Writes Poisson Hz on
  the biological EPG cells, with an explicit positive 0.0001 Hz engineering floor. No visual anchoring, tilt
  compensation, or inference of the native operating state.
- **Removal** -- a validated native heading circuit, or a quantitatively validated connectome-fitted model.
- **Admission** -- **untested by the gate**: no three-draw suite and no room rate-half has been run with it
  ([audits/navigation_instruments.md](audits/navigation_instruments.md) section 3 closing paragraph).
- **Rounds** -- [audits/navigation_instruments.md](audits/navigation_instruments.md) (engineering checks and room
  observations); the second arm of the [audits/plume_steering.md](audits/plume_steering.md) diagnostic.

## `plume`

- **Grammar** -- `--instruments compass plume`, or `PlumeNavigation(c)`;
  `plume[:bilateral=concentration|orn][:feedback=on|off|GAIN][:walking_goal=0|1]`. `feedback=off` and `feedback=0`
  are the same arm; a number sets the DNa02 integral gain in Hz per unit error per second.
  `flyverse.navigation:PlumeNavigation`. Requires `compass` or `compass_ring`, and EPG / LH / PFL3 / DNa02 / DNp09
  populations -- a missing population is an error, not a no-op.
- **Kind** `stop-gap`. **Replaces** `computation` (PRESETS_SPEC section 5).
- **Law** `unverified` as a composite. Its measured parts do not validate the whole: the PFL3 comparator
  `r = 29.23 * softplus(2.17 * (cos(H-Hpref) + 0.63 * cos(G-Gpref) - 0.7))` Hz with both published preferred-angle
  arrays is implemented directly from Mussells Pires et al. 2024 Methods "Full PFL3 model"; the odour-gated
  allocentric wind goal is motivated by Matheson et al. 2022 and its 2024 addendum; the entry-bearing return memory
  by Siliciano et al. 2026; the 0.45 s casting delay by van Breugel and Dickinson 2014; the bilateral turn sign by
  Gaudry et al. 2013 Figure 1. The goal policy, the output bridge to biological PFL3 soma sides and DNp09, the
  1 mm antenna baseline, the 0.1 m response length, the 80 / 60 / 40 Hz injection scales and the 5.0 /s DNa02
  integral gain are **declared engineering assumptions**
  ([audits/navigation_instruments.md](audits/navigation_instruments.md) sections 1-2,
  [audits/plume_steering.md](audits/plume_steering.md) section 3). The 0.1 m / 0.001 m ratio is a contrast gain of
  200; at the 0.26-0.57 % bilateral contrast observed in the rooms it yields 25-41 degree mean goal offsets
  (p95 54-66 degrees), so the law behaves as a near-sign-of-contrast turn (plume_steering.md section 3). There is no
  food-location oracle and no general odour-valence model.
- **Gap** -- heading, odour and wind signals lack a functional goal-memory / steering bridge
  ([audits/navigation_instruments.md](audits/navigation_instruments.md),
  [audits/plume_steering.md](audits/plume_steering.md)).
- **Reads / writes** -- reads the biological EPG rates (occupancy-balanced over 16 wedges), the LH berry / apple
  odour channels and DNa02 soma-L / soma-R rates. Body-derived inputs: held antennal deflections through
  `FlyBrain.wind` -> `observe_wind`, held bilateral glomerular concentrations through `FlyBrain.smell` ->
  `observe_smell`, airborne / feeding flags through `FlyBrain.interoception` -> `observe_internal`, plus
  `hunger.level` when `hunger` is named. **No world heading, wind angle, source position or distance.** Writes
  Poisson Hz on biological PFL3 soma-L / soma-R and on DNp09.
- **Removal** -- native odour / wind goal memory and PFL3 steering that pass the same sensory controls.
- **Admission** -- **untested by the gate.** The room results below are descriptive observations, not the three-draw
  suite or the 300 s rate-half ([audits/navigation_instruments.md](audits/navigation_instruments.md) section 3
  closing paragraph; [audits/plume_steering.md](audits/plume_steering.md) section 4).
- **Rounds** -- the pre-correction law in [audits/navigation_instruments.md](audits/navigation_instruments.md); the
  walking-goal and DNa02-feedback correction in [audits/plume_steering.md](audits/plume_steering.md); round 8 item 4
  as the `full` arm ([audits/plume_goal_only.md](audits/plume_goal_only.md)); the v5 rooms as the `full` arm
  ([audits/plume_transduced.md](audits/plume_transduced.md) section 7).

Older `plume` checkpoints are rejected: the sensory and feedback state schema changed with the correction. Start a
fresh episode. Every configuration records its own canonical spec in
`describe()['parameters']['variant_spec']` beside `variant`, `walking_goal_enabled`, `steering_integral_gain_per_s`
and (in `orn` mode) `bilateral_source`.

### `plume:bilateral=orn` -- the transduced-contrast variant

- **Grammar** -- `--instruments compass plume:bilateral=orn hunger flight`, or `PlumeNavigation(c, bilateral="orn")`.
  `variant` `transduced full`; `variant_spec` `plume:bilateral=orn`. Requires matched left/right antennal ORN groups.
- **Kind** `stop-gap`. **Replaces** `computation`, as `plume`.
- **Law** `unverified`. It replaces the physical walking concentration cue with matched left/right ORN population
  rates read from `brain.rate` at the frame boundary, filtered over 0.25 s:
  `goal = heading + atan(200 * atanh(EMA_0.25s((rL - rR)/(rL + rR))))`. Glomerulus weights are
  `nL*nR/(nL+nR)`, normalized -- a provably minimum-variance pooling rule for equal independent cell rates, an
  engineering rule, not a fitted decoder. `rate_contrast_gain` 200.0 carries
  `rate_contrast_gain_status: "unverified and underived"` in the record, retained for comparability with the default
  law, not as a concentration inversion.
- **Gap** -- the `plume` gap above, narrowed to one question: whether the model's own ORNs can deliver the lateral
  cue that the shipped law reads from the physical field. The shipped law reads a noise-free physical contrast that
  "would not survive the model's own ORN transduction"
  ([audits/plume_steering.md](audits/plume_steering.md) section 3); this variant is that substitution
  ([audits/plume_transduced.md](audits/plume_transduced.md)).
- **Reads / writes** -- adds `antenna_L` / `antenna_R` neural reads; **installs no physical concentration receiver**
  (`observe_smell` is `None` in this mode), so it receives no `smell()` samples at all. Writes are unchanged.
- **Removal** -- as `plume`; additionally, the 200x gain is retired by a derivation or a measurement of the
  rate-contrast law, which `rate_contrast_gain_status` says does not exist.
- **Admission** -- **untested by the gate.**
- **Rounds** -- the CPU characterisation and the v5 rooms of
  [audits/plume_transduced.md](audits/plume_transduced.md). The CPU half is unfavourable for reliable instantaneous
  lateralization: at the historical median total concentration and 0.57 % physical contrast, four explicit
  odour-mixture scenarios give 0.022-0.056 Hz of weighted L-R signal against 0.224-0.273 Hz of SD in a 250 ms
  counting window; with the 100 ms rate filter and the 250 ms contrast filter the approximate signal/noise ratio is
  0.16-0.34, still below one (that audit, section 0). The room half is the licensed sentence quoted at the top of
  this page, with the cue-fidelity figures recorded there: per-run Pearson r of the cue against the physical lateral
  contrast -0.07 to +0.24 in the transduced arm against 0.56-0.95 in the full arm, sign right in 0.52 +/- 0.10 of
  samples against 0.87-0.98, and `|200*atanh(contrast)| > 1` in 66-87 % of transduced samples. Excluding the two
  starts that begin 3.9 and 5.7 cm from a fruit surface leaves 4/4 full, 1/4 transduced, 3/4 goal_only. The six runs
  of an arm differ in start *and* seed, so the across-run SD is start heterogeneity and bounds nothing about
  run-to-run nondeterminism; the arms do share their six starts, so the paired-by-start reading is the stronger one.

### `plume:feedback=off` -- the physical-goal-only control

- **Grammar** -- `plume:feedback=off`, spelled `plume:feedback=0` as well (the same arm), or
  `PlumeNavigation(c, feedback_gain_per_s=0.0)`. `variant` `goal-only`; canonical `variant_spec`
  `plume:feedback=off`.
- **Kind** `stop-gap`. **Replaces** `computation`, as `plume`.
- **Law** `unverified`. The DNa02 L-R integral is held at zero, so the PFL3 input is the one-way
  `clip(80 * turn, -80, 80)` bridge of the original diagnostic; the bilateral walking goal is unchanged.
- **Gap** -- the `plume` correction bundled a bilateral walking goal with a DNa02 integral feedback, and no arm
  separated the two; an earlier skeptic inferred that the feedback was the load-bearing part
  ([audits/plume_goal_only.md](audits/plume_goal_only.md) opening).
- **Reads / writes** -- unchanged from `plume` (DNa02 is still read; only the integral correction is disabled).
- **Removal** -- as `plume`; this lever itself is a control arm, retired when the separation question is closed.
- **Admission** -- **untested by the gate.**
- **Rounds** -- round 8 item 4 as the `goal-only` arm ([audits/plume_goal_only.md](audits/plume_goal_only.md)) and
  the v5 rooms as the `goal_only` arm ([audits/plume_transduced.md](audits/plume_transduced.md) section 7). Round 8
  reading: **the walking goal is the load-bearing part, not the feedback** -- goal-only feeds 4.67 +/- 0.52 rows of
  six on the shipped starts at a mean |DNa02 L-R| of 0.62 Hz, while `full` feeds 6/6 and `feedback-only` feeds
  0.17 +/- 0.41; what the feedback adds is the last rows, and the row goal-only never feeds is the 185 degree start
  (1.00 of the +1.33 rows, the 90 degree row the other 0.33). Tests 1 and 4 are `null` under the declared rule
  (`compare` divides by the reference arm's SD and needs |z| >= 3) although the samples separate completely; the
  reading is the magnitude, +1.3 to +1.7 fed rows out of six. In the v5 rooms this arm fed in 3/6, and those three
  are exactly the starts whose initial heading already pointed near fruit (bearing error 19-88 degrees) while its
  three failures faced away (102-157 degrees), so that arm is closer to "hold the start heading" than to a
  plume-competence floor ([audits/plume_transduced.md](audits/plume_transduced.md) section 7). A fourth arm with no
  plume cue at all -- the zero-information floor -- has not been run, so "transduced 1/6 < goal_only 3/6" is not a
  comparison against zero.

### `plume:walking_goal=0` -- the feedback-only complement

- **Grammar** -- `plume:walking_goal=0`, or `PlumeNavigation(c, walking_goal=False)`. `variant` `feedback-only`;
  `variant_spec` `plume:walking_goal=0`.
- **Kind** `stop-gap`. **Replaces** `computation`, as `plume`.
- **Law** `unverified`. The DNa02 feedback at the shipped 5.0 /s gain, closed on the **pre-correction** upwind /
  entry-memory goal law; the bilateral state is still tracked and checkpointed but does not set the goal.
- **Gap** -- the complement of the arm above, and the other half of the same separation question
  ([audits/plume_goal_only.md](audits/plume_goal_only.md) opening).
- **Reads / writes** -- unchanged from `plume`.
- **Removal** -- as `plume`; a control arm, retired when the separation question is closed.
- **Admission** -- **untested by the gate.**
- **Rounds** -- round 8 item 4 as the `feedback-only` arm ([audits/plume_goal_only.md](audits/plume_goal_only.md)).
  It feeds 0.17 +/- 0.41 rows of six on the shipped starts while carrying the *largest* mean |DNa02 L-R| (1.97 Hz),
  |yaw| (0.176 rad/s) and integral (8.1 Hz) of the three arms: the loop turns the fly hard toward a goal that does
  not point at food. The +5.83-row contrast with `full` is a **simple effect** -- the three arms are three cells of a
  2 x 2 whose fourth cell, the pre-correction goal at gain 0, is not in that batch.

## `hunger`

- **Grammar** -- `--instruments ... hunger`; `flyverse.navigation:HungerGain`. Requires `plume` or `flight`.
- **Kind** `stop-gap`. **Replaces** `computation` (PRESETS_SPEC section 5).
- **Law** `unverified`: `gain = (1 - energy) * (not sated)`, an instantaneous engineering control. Root et al. 2011
  supports state-dependent food search through insulin / sNPF modulation of specific ORNs; this global navigation
  gain is an explicit engineering alternative, **not** Or42b / sNPFR1 kinetics, a hormone model, a receptor row or a
  fabricated hunger-neuron mapping ([audits/navigation_instruments.md](audits/navigation_instruments.md) section 2).
- **Gap** -- the environment's metabolism is not reported to navigation circuits.
- **Reads / writes** -- **neither**: `reads` and `writes` are both empty. It receives normalized energy and the sated
  flag through `FlyBrain.interoception` -> `observe_internal`, and the sibling instruments that bind it read
  `hunger.level` -- it scales `plume`'s signed turn demand into the PFL3 Poisson input and `flight`'s lift request.
  Without this instrument those consumers use gain 1. The value is written only by `observe_internal`, never during
  a step, so attachment order cannot change a frame's result.
- **Removal** -- a sourced metabolic transducer with verified target cells and dynamics.
- **Admission** -- **untested by the gate.**
- **Rounds** -- [audits/navigation_instruments.md](audits/navigation_instruments.md),
  [audits/plume_steering.md](audits/plume_steering.md),
  [audits/flight_foraging_priority.md](audits/flight_foraging_priority.md), and every round-8 plume arm
  ([audits/plume_goal_only.md](audits/plume_goal_only.md)) and v5 room arm
  ([audits/plume_transduced.md](audits/plume_transduced.md) section 7).

## `flight`

- **Grammar** -- `--instruments ... flight`; `flyverse.navigation:FlightDrive`. Requires nonempty wing power and
  steering MN groups and PFL3 on both sides. Available LH odour groups add the landing cue; a reduced graph keeps
  the reserve and duration limits without a fabricated odour signal.
- **Kind** `stop-gap`. **Replaces** `computation` (PRESETS_SPEC section 5).
- **Law** `unverified`. A PI power servo, `clip(100 + 0.25*error + integral, 0, 250)` Hz with integral gain 0.5 /s
  bounded to [-100, 150], and a steering bridge `10 +/- clip(0.1*(PFL3_R - PFL3_L), -10, 10)` Hz. The 100 Hz target
  is the **shipped body's lift equilibrium** (`20 + 3/(1.5/40) = 100`), not a measured motor-neuron rate. The
  foraging-priority policy -- 20 s of ground search before each voluntary flight, each powered bout capped at 8 s,
  withdraw at energy <= 0.35 and re-arm only at >= 0.50, odour interruption at normalized odour >= 0.60 clearing at
  <= 0.25 over a 1 s low-pass, and hold the drive at zero until actual touchdown once interruption begins in the air
  -- is **unverified engineering**, declared before its assays
  ([audits/flight_foraging_priority.md](audits/flight_foraging_priority.md) section 1). Namiki et al. 2022 supports
  investigating descending wing-power control; it does not establish these gains or takeoff sufficiency.
- **Gap** -- wing power cannot sustain level flight and the walking steering readout does not control wings. The
  controller deliberately bypasses the unvalidated DNg02 / VNC route by driving the biological wing MNs directly.
- **Reads / writes** -- reads biological wing-power MN rates, PFL3 soma-L / soma-R and the LH berry / apple channels
  (its own frame-boundary reads, not `plume`'s mutable state, so attachment order does not matter). Receives
  normalized energy and the sated / airborne / feeding flags through `FlyBrain.interoception` -> `observe_internal`,
  plus `hunger.level` when `hunger` is named; **no altitude, world position or fruit distance**. Writes Poisson Hz
  on the biological wing power and steering MNs. `altitude_control` and `flight_energy_cost` are `False` in the
  record: no altitude feedback, no collision avoidance, no landing-site choice, no added flight metabolic cost.
  Native escape hops still occur. Its timers, reserve / odour hysteresis and landing latch are in checkpoints and
  row resets; older checkpoints carrying the original `flight` state are rejected.
- **Removal** -- validated descending / VNC flight-state and steering mechanisms that supply the same control.
- **Admission** -- **untested by the gate.**
- **Rounds** -- the original sustained-power servo in
  [audits/navigation_instruments.md](audits/navigation_instruments.md) (preserved there as a historical record: in
  those six 60 s rooms every fly spent 59.67 s airborne and none fed); the corrected policy in
  [audits/flight_foraging_priority.md](audits/flight_foraging_priority.md); the round-8 `full` arm
  (`compass plume hunger flight`) of [audits/plume_goal_only.md](audits/plume_goal_only.md) and every v5 room arm
  of [audits/plume_transduced.md](audits/plume_transduced.md) section 7. Read the corrected policy with its own
  finding: with the shipped LH normalisation the odour gate reads 0.66-0.71 mean (max 0.95-0.99) even with a single
  apple 40 cm away and the fence on, so the 0.60 / 0.25 hysteresis latches for essentially the whole episode -- under
  this policy artificial flight is unreachable in any room containing fruit, not merely bounded, and powered flight
  is demonstrated only in the transition assay, where LH is prescribed at 0 Hz
  ([audits/flight_foraging_priority.md](audits/flight_foraging_priority.md) section 3).

## The inventory at a glance

| instrument | kind | replaces | law | admission | rounds |
|---|---|---|---|---|---|
| `sided_turn_afferent` | stop-gap | input | unverified | admissible at `k=0.5`, in the three-instrument list | 7, 8 (items 2, 3) |
| `ring_dc_hold` | edges | configuration | counterfactual | admissible, in the three-instrument list | 6A, 6B, 7, 8 (item 3) |
| `ring_dc_hold_pen` | edges | configuration | counterfactual | untested by the gate | 7 (arm HGVp) |
| `glno_pen_hold` | edges | configuration | counterfactual | untested by the gate | 7 (follow-up) |
| `glno_sign` | relabel | configuration | unverified | admissible, in the three-instrument list | 2 / 5B, 7, 8 (item 3) |
| `edge_gain_<i>` | edges | configuration | a per-type gain, no transfer claimed | untested by the gate | 6B |
| `compass` | stop-gap | computation | unverified | **rejected** (compass_standin.md) | its own suite; every navigation round since |
| `compass_ring` | stop-gap | computation | unverified | untested by the gate | navigation_instruments, plume_steering |
| `plume` | stop-gap | computation | unverified | untested by the gate | navigation_instruments, plume_steering, 8 (item 4), v5 rooms |
| `plume:bilateral=orn` | stop-gap | computation | unverified | untested by the gate | plume_transduced (CPU + v5 rooms) |
| `plume:feedback=off` | stop-gap | computation | unverified | untested by the gate | 8 (item 4), v5 rooms |
| `plume:walking_goal=0` | stop-gap | computation | unverified | untested by the gate | 8 (item 4) |
| `hunger` | stop-gap | computation | unverified | untested by the gate | navigation_instruments and every later navigation round |
| `flight` | stop-gap | computation | unverified | untested by the gate | navigation_instruments, flight_foraging_priority, 8 (item 4), v5 rooms |

`flyverse.instruments._check_instrument` requires `name`, `kind`, `law`, `gap`, `removal`, `audits` and `replaces`
on every `describe()`, and requires a nonempty `source` or `sources` whenever `law` is not the word `unverified`, so
an instrument that does not declare what it replaces, or claims a law without naming where it came from, cannot
attach.

## Admission status

The gate is PRESETS_SPEC section 2 item 5: the 29-check suite under `instrumented` beside `raw` at >= 3 draws with
no status change outside the rows the instrument is declared to touch, and the room rate-half at >= 6 runs per arm.

**Passed.** The three-instrument list `sided_turn_afferent:k=0.5` + `ring_dc_hold` + `glno_sign` is **admissible**
([audits/instrumented_suite.md](audits/instrumented_suite.md) section 4). The suite changes **no row's status** in
three instrumented draws beside three raw draws -- 27/0/2, 26/1/2, 27/0/2 under both presets; the one non-uniform
row, `taste.MN9_hz`, is raw's own instability and is identical under the instruments; the compass row stays KNOWN
GAP. Values move on 20 of the 29 rows and 8 rows are bit-identical to raw. The room rate-half passes at six
seed-matched runs per arm: 110 take-offs against 101 over 28,800 fly-s each (3.82 against 3.51 per 1,000 fly-s;
one-sided exact Poisson p 0.291, run-level `null`). A fewer-meals descriptive -- 1-4 per run instrumented against
0-7 raw, a predeclared descriptive, `null` at 6 v 6 -- is on the record for the next reader, as is `gf_max_hz`, the
one measure in the set with a rank-test p below 0.05 (exact U p 0.0152, undeclared, `null` under the z >= 3 rule).
Admissible is all it is: nothing is adopted, `raw` stays the default, the afferent's law is still `unverified` and
the relabel still has no transmitter source.

**Rejected.** `compass` ([audits/compass_standin.md](audits/compass_standin.md)), on the `taste.MN9_hz` status
change described in its block above.

**Untested by the gate.** `ring_dc_hold_pen`, `glno_pen_hold`, `edge_gain_<i>`, `compass_ring`, `plume` and its
three variants, `hunger`, `flight`. Their audits carry engineering and functional checks, not preset admission:
"Neither the three-draw 29-check admission suite nor the full room rate-half is replaced by these checks"
([audits/navigation_instruments.md](audits/navigation_instruments.md) section 3).

## Experimental availability versus adoption

An instrument is not a default, not a tuned constant, and not a way to move a suite row (PRESETS_SPEC section 4).
The `raw` column stays the model's own behaviour; an `instrumented` column is reported beside it with the instrument
list in the caption; an instrument that changes a row's status outside its declared gap fails admission
(PRESETS_SPEC section 2 item 5). Passing the gate makes a list *admissible* and nothing else. No new native-circuit
or food-finding result follows from an instrument's availability beyond the two licensed sentences at the top of
this page.

See [PRESETS_SPEC.md](PRESETS_SPEC.md) for the contract, [CONTROL_SURFACE.md](CONTROL_SURFACE.md) for the caller
surface, [EXTENSIBILITY.md](EXTENSIBILITY.md) for the module lifecycle an instrument rides on, and
[audits/navigation_instruments.md](audits/navigation_instruments.md) for the navigation instruments' papers, exact
versus approximate computations, CUDA checks and outcomes.
