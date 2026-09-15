# Instruments: what ships, how to turn one on, what it stands in for, what retires it

The contract is [PRESETS_SPEC.md](PRESETS_SPEC.md) (owner decision of 2026-09-15: labelled physiological stand-ins are
acceptable if sourced, optional, and `raw` returns the pure brain). This page is the inventory. **Nothing here is on by
default**: `FlyBrain()` is `FlyBrain(preset="raw")`, byte-identical on every path (`tests/test_bit_identity.py`,
`RawPresetTests`), and every audit states which preset it ran.

## Turning one on

```python
from flyverse import instruments as fi
from flyverse.fly import FlyBrain
inst = fi.SidedTurnAfferent(c, k_hz_per_deg_s=0.5, sign=+1, cells="AN07B037")
fb = FlyBrain(c, preset="instrumented", instruments=[inst])      # preset="raw" with instruments raises
fb.proprioception(leg_L, leg_R, haltere, airborne, yaw_rate=...)  # the body channel that already exists feeds it
provenance(c, fb=fb)["preset"], provenance(c, fb=fb)["instruments"]   # 'instrumented', [inst.describe()]
```

`BatchSim(..., preset="instrumented", instruments=[inst], proprioception="all")` forwards both and re-installs the
stand-in onto its own sense (the spec grows `+turn_afferent`). Naming `all+turn_afferent` with the instrumented
preset also registers the token's default stand-in; raw refuses that token. Construct the instrument on the
effective connectome: a different cell ordering is refused. Checkpoints require the same preset and descriptions,
including sign and gain, before loading. On the CLI (`scripts/cx_wedge.py`):

```
--preset raw|instrumented            default raw (byte-identical to no flag); --instrument implies instrumented
--instrument NAME[:k=..][:sign=..][:cells=..]   repeatable; k in {0.25, 0.5, 1.0} Hz per deg/s, sign -1 = the control
--turn DEG_S --turn-window START:END  ledger runs only: a prescribed signed yaw (positive = a left turn) after the pulse
```

Under `--preset instrumented` the 6A hold (`--hold-edges`), the GLNO relabel (`--nt-override GLNO=glutamate`) and a 6B
edge gain are also recorded as instruments (`ring_dc_hold`, `glno_sign`, `edge_gain_<i>`; PRESETS_SPEC 2.4) beside
the flags that install them; without the preset flag those flags behave and record exactly as in 6A / 6B. The eight
predeclared arms (six primary arms plus two descriptive gain levels) of [audits/compass_velocity_route.md](audits/compass_velocity_route.md) are each one command line
(`scripts/cx_velocity_route.py --plan-batch out/cx8` writes them). After review and merge, run
`python scripts/cx_velocity_route.py --predeclare out/cx8` to freeze the protocol, family, gate, source hashes and
resolved model parameters before submission. Neither the plan nor declaration can be overwritten after freezing.
The wrapper explicitly selects house in both calls and stops on a failed client. Submission and results are recorded
in the round-7 audit; the generated wrapper's draft comment describes its status at generation.

## The inventory

The explicit `compass` candidate adds a continuous heading-memory input through an ordinary module:

```sh
python scripts/room_demo.py --preset instrumented --instrument compass --cuda-graphs --brain-map
```

In Python, use `FlyBrain(..., preset="instrumented", instruments=["compass"])` or the same arguments on
`BatchSim`. Both room implementations feed realized yaw velocity; callers stepping FlyBrain directly must
call `fb.proprioception(0, 0, 0, False, yaw_rate=rad_per_s)` as motion changes. No other proprioceptive sense
is required. The input is held until updated; call it with zero on stopping. Full and partial resets and
checkpoints include the driver. Manual room pose teleports do not constitute sensed turns; phase is arbitrary.

The program holds phase and writes a 50 Hz, 35 degree Gaussian Poisson profile on the 46 biological EPGs.
These constants and the angular-velocity law are **unverified engineering choices**. It neither changes
synapses nor sets motor commands; it lacks visual anchoring, tilt compensation and steering/goal memory.
It is retired when a native circuit passes the same sustained-turn, reversal and stationary-memory checks.
See [the audit](audits/compass_standin.md) for source reading, validation and candidate/admission status.
The UI names the preset and instrument; provenance includes its complete law and target IDs. `raw` rejects it.

| instrument | kind | class | stands in for | law | retired by |
|---|---|---|---|---|---|
| `sided_turn_afferent` | stop-gap | `instruments.SidedTurnAfferent` | PS196_b's signed turn input: Poisson spikes on its named ascending afferents (AN07B037_a / _b by default; `cells=` CB0675 / GNG580 / PS047_b / all) at `k * max(0, +-yaw_deg_s)` on the side the graph implies. In the shipped body the report reaching PS196_b is unsigned (`audits/vnc_drive.md` 6, NOTES compass round 2); Wang 2026 finding 3 makes PS196_b GLNO's largest non-ring input. | **unverified**: no PS196_b / AN07B037 recording exists (searched 2026-09-15: Wang's audit, Hulse 2021, the two Rockefeller theses). `k` is a declared level, `sign` the HGV- control. | a recording of PS196_b / AN07B037 during turning (then a `mechanism` with a source, or dropped); a sided ascending report that reaches PS196_b on its own; a round-7 `null` on measure 3 at every declared `k` |
| `ring_dc_hold` | edges | `instruments.EdgeHold` | the ExR6 / ER6 / ER4m DC term on PEN / EPG held at 0 (6A's `--hold-edges '^(ExR6\|ER6\|ER4m)$:^(PEN_\|EPG$)'`: 17 pre cells onto 88, 1,149 entries, 37,256 synapses) | a counterfactual: no transfer claimed | sourced receptor placement / kinetics at the EB / GA contacts and a physiological operating state. ExR6 glutamate and ER6 GABA already have type-level support (`audits/exr6_evidence.md`); neither label is changed here |
| `ring_dc_hold_pen` | edges | `instruments.EdgeHold` | the same 17 pre cells held only onto 42 PEN cells: `^(ExR6\|ER6\|ER4m)$:^PEN_`, 402 entries, 7,893 synapses; EPG keeps its ring input | a counterfactual: no transfer claimed | the same receptor / operating-state evidence as the full hold; round 7 tests whether retaining EPG's DC input changes confinement |
| `glno_pen_hold` | edges | `instruments.EdgeHold` | the round-7 transfer control: `^GLNO$:^PEN_` held at zero on the GLNO-glutamate scratch graph, 84 entries / 16,371 synapses onto 42 PEN | pathway-removal counterfactual, not a receptor model | retired after the diagnostic; no adoption proposed |
| `glno_sign` | relabel | `instruments.TypeRelabel` | GLNO compiled as glutamate in a scratch cache (`--nt-override GLNO=glutamate`) | **unverified**: the stronger of two disagreeing EM predictions (`audits/glno_relabel.md`) | a transmitter source for GLNO that is not one EM classifier (then a `TYPE_NT_OVERRIDE` row) |

### `sided_turn_afferent`: where it lives and why

It is a **body-derived** signal (the realised yaw rate), so it lives where the `haltere_coriolis` stop-gap lives: the
`senses.Proprioception` transducer, as the extra channel `'turn_afferent'` (never part of `'all'`), fed through
`FlyBrain.proprioception(..., yaw_rate=)` and injected through `FlyBrain._input` like every other sense. A
`flyverse.modules` module sees neural quantities only, and no neural quantity in this model carries the fly's own turn
with a sign, so this instrument stays in the transducer. The later compass candidate has an explicit held-yaw
receiver authorized by PRESETS_SPEC section 5; that does not change the afferent's implementation.

**The side, from the graph** (`describe()['routing']`, read from `W[post, pre]` with `somaSide`; raw counts through
`common.raw_counts` so the sign-0 GLNO -> PEN block counts too). On the shipped cache every hop of the route is
contralateral:

| hop | L -> L | L -> R | R -> L | R -> R |
|---|---|---|---|---|
| AN07B037_a (2 L, 2 R) -> PS196_b (1 L, 1 R) | 0 | 202 | 214 | 3 |
| AN07B037_b (1 L, 1 R) -> PS196_b | 0 | 28 | 24 | 0 |
| PS196_b -> GLNO (2 L, 2 R) | 0 | 832 | 966 | 3 |
| GLNO -> PEN_a(PEN1) (10 L, 10 R) | 0 | 5,024 | 4,782 | 0 |
| GLNO -> PEN_b(PEN2) (11 L, 11 R) | 0 | 3,212 | 3,353 | 0 |

So with `sign=+1` a **left** turn (positive yaw, the body's convention) drives the **left** AN07B037 and the chain
lands on `PS196_b_R -> GLNO_L -> PEN_R` (`describe()['chain_for_positive_yaw']`). The variants are ipsilateral onto
PS196_b (CB0675 51 / 51, GNG580 10 / 13, PS047_b 185 / 217 same-side synapses), so their chain lands one side over
and the record says so per variant. Which PEN side *should* carry a left turn is exactly what no recording settles;
that is why the sign arm exists and why `sign` is a parameter, not a constant.

**The boundary.** Reads the yaw through the existing proprioception call, writes `poisson_hz` on the afferent cells,
never a weight, a receptor, `NT_SIGN` or the body. Every rate is clamped to `max_hz` (250 Hz, a safety ceiling, not a
law). Zero yaw is zero everywhere (`tests/test_instruments.py`).

**In cx_wedge there is no body**, so the turn is a prescribed protocol parameter (`--turn 90 --turn-window 0.5:3.5`:
the analogue of the 90 deg/s imposed visual rotation of `audits/deficit_rotation.md`; the efferent protocol's
"fly turning itself at 86-113 deg/s" is the odour-room body, not this script). The ledger records per-side PEN / GLNO
/ DNa02 / PS196_b / afferent rates per frame and the round-7 measures `<G>_LR_hz`, `bump_follow_wedges_per_s`
(read beside `bump_follow_confined_frac`: on a dead bump the centre is noise) in `metrics`, and the JSON carries
`preset`, `instruments` and every `describe()` in `provenance`.

## What an instrument is not

Not a default, not a tuned constant, not a way to move a suite row (PRESETS_SPEC 4). The `raw` column stays the
model's own behaviour; an `instrumented` column is reported beside it with the instrument list in the caption, and an
instrument that moves a row its gap does not cover is rejected (PRESETS_SPEC 2.5, not yet run for any instrument:
round 7 has to return first).
