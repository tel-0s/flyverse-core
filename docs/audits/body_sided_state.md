# Sided body state -> sided afferents: a stance / swing leg cycle and a side-split haltere readout (thread body-state, round 3)

Generator: `scripts/probe_vnc_drive.py --family body` (`room` / `compass` GPU, `plan` / `analyse` / `pairs` CPU). Data: `out/vncd3/`
(two submissions of ONE plan -- `out/vncd3/batch.sh` -> `vncd3-f3bb50`, `batch_b.sh` -> `vncd3b-c2eeaf`; consoles
`out/vncd3_cluster.log` / `out/vncd3b_cluster.log`; section 3), analysis `out/vncd3/analysis/` (and `analysis_4runs/`). CPU
smokes `out/vncd3_smoke/` (no number below comes from them). Tests: `tests/test_body_cycle.py` (new, 10),
`tests/test_proprioception.py` (+2). Code authored by this task: `flyverse/body.py` (`LegCycle`, `FlyState` leg fields,
`Locomotion.cycle` / `Locomotion.proprio_state`), `flyverse/batch_body.py` (`BatchBody.leg_cycle`, `_cycle`,
`proprio_state(haltere_sides=)`), `flyverse/motor.py` (`haltere_side_groups`, `read_haltere_sides`), `flyverse/senses.py`
(`Proprioception` tokens `leg_cycle` / `haltere_sided`, per-leg per-phase laws, `take_body`, `haltere_sides`),
`flyverse/batch_sim.py` (the one per-frame proprioception call), `scripts/probe_vnc_drive.py` (the `body` family),
`scripts/probe_walk_straightness.py` (`--leg-cycle`, the new spec tokens). Nothing in `brain.py`, `optic.py`, `fly.py`,
`connectome.py`, `room_demo.py`, `room_ui.py`, the receptor table or the ledger CSV was touched. **Every new field
defaults OFF and the shipped path is bit-identical on CPU** (section 5; `tests/test_bit_identity.py`'s recorded golden
passes unchanged -- which is also why the side split is a function beside `MotorRates` and not a field on it, section 1.2).

## 0. Results

Every number below is a run mean +- SD over the five runs of an arm (16 flies x 55 s window each; runs are the
replicate unit) unless a per-seed list is given; verdicts are `flyverse.interp.common.compare` on runs (5 v 5, floor
p 0.0079). The four runs of the single submission `vncd3b` (seeds 0 1 3 4; 4 v 4, floor 0.029;
`out/vncd3/analysis_4runs/pairs_console.txt`) give the SAME call on every row quoted in this section (the two tables
do differ in 22 verdict cells overall -- 15 of 765 pairwise, 7 of 548 room-table -- none of them a row quoted here;
section 4 lists them). Adjacent-arm
verdicts (`out/vncd3/analysis/pairwise.csv`, `probe_vnc_drive.py pairs`) are the ones the questions turn on: C vs B is
the leg cycle, D vs C the side-split haltere, E vs D the labelled stop-gap.

**(1) DNa02 fires under the leg cycle, and it is the cycle -- not the sidedness of anything -- that makes it fire.**
DNa02_L 0.031 +- 0.004 Hz (B) -> **0.540 +- 0.018** (C), DNa02_R 0.104 +- 0.007 -> **0.383 +- 0.020**, the fraction of
walking frames with DNa02 above 5 Hz 0.010 -> **0.066 +- 0.001**: C vs B `result` on all three (z +141 / +43 / +100,
p 0.008). D vs C: -0.028 / +0.010 / -0.001, `null` (p 0.056 / 0.42 / 0.42); E vs D `null` (+0.023 / +0.001 / +0.002).
The side-split haltere readout and the Coriolis control change nothing DNa02 does in the room. Why it fires
(section 4.4): the leg-afferent term on DNa02 outgrows the haltere-afferent inhibition -- AN04B003/L +57.9 (B) ->
**+360.4 mV/s** (C) against PS059/L -114.1 -> -207.5, so DNa02_L's net rate-weighted input crosses from -233 to
**+110 mV/s** (DNa02_R: -345 -> -155). The PS059 cancellation of round 2 does NOT break when the haltere channel is
sided (PS059/L -207.5 (C) -> -205.5 (D) -> -209.6 (E); /R -169.9 -> -171.0 -> -177.6; run SD 2-4); it is simply
outgrown, by a factor the round-2 law never had.

**(2) The C-vs-B difference is a LEVEL effect of the leg afferents, and nothing in this round separates it from the
phase structure.** The commanded chordotonal mean goes 23.4 +- 0.1 Hz (B) -> **88.1 +- 1.8** (C), hair plate 14.1 ->
47.4, leg campaniform 50.0 -> 24.9 -- every one inside its ledger bracket (10-150, 5-100, 0-100 Hz) before and after.
The round-2 law scaled the drive by the side's leg-MN rate / 30 Hz (~0.1 at the 2.5-3 Hz the leg MNs run at); the
cycle's law averages ~0.6 of the range over a step (swing 0.24 of the cycle at drive 1, stance 0.76 at mean |2p - 1|
= 0.5, amplitude ~0.95 at 9 mm/s). No number was set to a behaviour and the brackets pin neither mean, so the honest
statement is: the transducer at literature-typical FeCO rates drives DNa02 through AN04B003; the round-2 transducer at
MN-scaled rates did not. Attributing anything to the per-leg / per-phase structure needs the level-matched control arm
of section 8 (the round-2 law with `mn_ref_hz` set so its mean is 88 Hz -- a labelled control, not a default).

**(3) The fly turns more, but not because it senses a turn.** Yaw-rate SD on clean walking frames 2.64 +- 0.13 (A) ->
3.35 +- 0.09 (B) -> **7.87 +- 0.14 deg/s** (C; per seed 7.84 / 7.96 / 8.01 / 7.66 / 7.88), D 7.73 +- 0.18, E 7.93 +-
0.18 (C vs B `result` z +51; D vs C, E vs D `null`); median |yaw| 0.74 -> 1.13 -> 1.57 deg/s; 99th percentile 15.2 ->
18.0 -> 25.8; straightness 0.995 -> 0.979 -> 0.826 +- 0.032; net heading change 0.03 -> 0.09 -> 0.22 turns per fly.
But: the leg L-R is still a fixed positive bias -- **0 of 80 fly-runs negative in every arm** (0.141 +- 0.006 Hz under
C, 0.151 under A: the cycle does not touch it, it reads the same leg MNs) -- and DNa02's L-R has a fixed sign too:
DNa02_L > DNa02_R in **68 / 67 / 68 of 80** fly-runs (C / D / E) against 5 / 80 under A and 8 / 80 under B; the mean
signed yaw is **+1.23 +- 0.16 deg/s (a left drift) in 78 / 80 flies** under C (D +0.99, E +1.08; A +0.21 in 66 / 80).
The turn is the connectome's own left-right asymmetry amplified (DNa02_L's net input +110 vs DNa02_R's -155 mV/s; the
inhibitory rows differ by side, section 4.4), not a report of the body's yaw. Consequences: 7.4 +- 1.8 of 16 flies walk
off the table top within 60 s (A 0.6, B 0.2), hops 0.46 +- 0.24 per fly (A 0.08), and the plain `yaw_sd_deg_s` of
round 2 (93.8 +- 17.7 deg/s under C) is an edge-crossing artefact -- 0.96 frames per fly with |yaw| > 720 deg/s -- which
is why every yaw number here is the clean-frame one (section 4.1).

**(4) A sided AN04B003 term survives per frame, tripod-locked, and reaches DNa02 -- and does not steer.** With the
cycle on, AN04B003's L-R swings **-3.39 +- 0.06 Hz** between the two tripod half-cycles (E[L-R | commanded chordotonal
L-R > 0] - E[. | < 0]; B +0.67), DNa02's L-R command follows by **-0.35 +- 0.02 Hz** (B +0.02), corr(AN04B003 L-R,
chordotonal L-R) -0.30 +- 0.01 (B +0.13), corr(DNa02 L-R, chordotonal L-R) -0.14 (B +0.01). The sign is the wiring:
AN04B003_L reads the L1 and L2 femoral chordotonal afferents (93 + 171 synapses; L3 34), AN04B003_R reads R2 and R3
(218 + 163; R1 48) -- ipsilateral per leg, but L2 swings with R1 R3 and R2 with L1 L3, so the side-summed L-R and the
AN's L-R alternate in antiphase. The alternation is at the step frequency (7.9 +- 0.2 Hz), the walk smooths yaw at
80 ms, and the turn asymmetry the yaw adds is |amp L-R| 0.016 (0.02 of the amplitude) against a per-frame |chordotonal
L-R| of 12.5 Hz from the alternation. corr(realised yaw, chordotonal L-R) is -0.15 +- 0.00 in C / D / E -- the sign the
kinematics impose (a left turn shortens the left steps) and 0.29 (the opposite sign) under B, where the L-R came from
the leg-MN bias; corr(AN04B003 L-R, yaw) -0.07. A sided term, real and body-locked, a 30th of the alternation it rides
on.

**(5) The side-split haltere is sided by construction and inert in practice.** Haltere MN L 15.33 +- 0.22 / R 18.06 +-
0.18 Hz under D (the readout exists in every arm: A 7.63 / 9.39, the right side always higher); the haltere afferent
command becomes L 15.3 / R 18.1 (C: 16.75 both), L-R -2.73 +- 0.13 Hz, and corr(DNa02 L-R, haltere command L-R) rises
from 0.01 (C) to 0.14 (D), corr(PS196_b L-R, haltere L-R) 0.00 -> 0.15, corr(PS059 L-R, haltere L-R) -0.04 -> +0.08:
the sided haltere reaches PS059 / PS196_b / DNa02 as a signal. But the haltere MN L-R is another fixed bias (R > L in
every arm, corr with the realised yaw 0.08), so what it carries is not a turn, and D vs C is `null` on every behavioural
and DNa02 row.

**(6) The compass does not see the self-turn** (`scripts/probe_vnc_drive.py compass --family body`, gE 2 / gD 15,
DNa02_L / _R at 20 Hz for 10 s, 3 seeds per arm, 3 v 6 per phase = `underpowered` by the rule; section 6):
* PS196_b's L-R does NOT flip with the turn direction. D: rest +3.43, ccw +2.33, rest2 +3.90, cw +3.48 Hz; flip
  (ccw - cw) -1.14 +- 1.31 (seeds 0.00 / -2.57 / -0.86) against the rest2 - rest null +0.48 +- 1.15. E: +3.86 / +4.52 /
  +3.52 / **+7.24**; flip -2.71 +- 1.31 (-3.86 / -1.29 / -3.00) against -0.33 +- 1.30 -- under the Coriolis control the
  L-R RISES in both directions (+0.7 ccw, +3.7 cw against its rests) while PS196_b itself goes from 7.6 / 3.7 Hz (L / R,
  rest) to 20.9 / 16.3 (ccw) and 20.8 / 13.5 (cw): driven 2.7x, unsigned, the round-2 pattern again. A: 0.00 / -0.05 /
  -0.05 / -0.10 (silent).
* GLNO's L-R stays +27.2 / +27.4 / +27.1 / +26.9 Hz (D; flip +0.52 +- 0.32 against -0.10 +- 0.48), +27.0 / +27.0 /
  +26.9 / +26.5 (E), +28.0 at every phase (A).
* The bump does not drift: ccw +0.0015 +- 0.0034 w/s (D; seeds -0.0019 / +0.0016 / +0.0049), cw +0.0041 +- 0.0009;
  E +0.0029 / +0.0032; A +0.0010 / +0.0028 -- against +-4.0 ideal at the realised 99 / -87 deg/s; vector strength 0.76,
  peak 252 Hz, pinned as in `cx_shift.md`.
* Where the signed report IS: at depth 1. AN04B003's L-R -1.00 rest, **-5.79 ccw**, -0.60 rest2, **+4.02 cw** (D; flip
  -9.81 +- 1.60, every seed -8.2 to -11.4, null +0.40 +- 0.53); E -10.70 +- 0.18. AN06A026 -2.21 (D) / -4.81 (E),
  AN07B035 -1.88 / -1.64. The haltere MN L-R itself: -1.56 rest, **-4.39 ccw**, -1.26 rest2, **+0.95 cw** (D; the
  haltere afferent command L-R -4.39 / +0.96; E -14.1 / +0.39 with the Coriolis multiplier). At depth 2, PS047_b -1.71
  (D) / -3.33 (E) and PS196_b as above; at depth 3, GLNO +0.5; at depth 4, PEN_a -0.10, PEN_b -0.19, EPG +0.58 (A
  +0.63). The report is sided at the ascending neurons (a quarter of their 22-27 Hz), 1-3 Hz two synapses on, and
  gone at GLNO. PS059 (+4.9) and IN19A003 (+5.7) flip in D as they do in A (+4.1 / +6.7): they follow the DNa02
  stimulus, not the body.

**Under the project rule** (section 2): the leg cycle and the side-split readout are body-model mechanisms and stay
opt-in; nothing is adopted. What a default adoption would require is in section 8: a level-matched control that
separates the FeCO rate level from the phase structure, the pinned-body compass under the cycle at the realised
walking speed, the full suite with `all+leg_cycle+haltere_sided`, four ledger rows, and an owner decision on
`MotorRates.haltere_L / _R` (a golden regeneration).

## 1. The two mechanisms, and what they are under the project rule

Round 2 (`vnc_drive.md` 0(2), 0(3), 8(4)) left two body-model gaps that no transducer parameter could close: the leg
channels read the side's leg-MN rate, which differs by 0.3 Hz between the sides in every one of 400 fly-runs and never
changes sign (`k_leg_turn` turns that into a fixed +0.5 -> +1.0 deg/s bias), so the "sided" excitatory term on DNa02
(AN04B003/L +57, /R +56 mV/s) was symmetric by construction; and `MotorRates.haltere` was one bilateral number, so the
haltere afferents -- the only afferent class with a two-step route into PS196_b -- were unsided by construction and the
compass report of a self-turn arrived unsigned (PS196_b's L-R moved the same way for both turn directions at 11 Hz under
the stop-gap). Both are BODY-MODEL facts: the body had no leg cycle and no per-side haltere. This round builds the two
body mechanisms the audits named, opt-in, and re-runs the round-2 protocols under them.

### 1.1 The stance / swing leg cycle (`body.LegCycle`)

A kinematic model of the six legs driven by the body's REALISED walking speed and yaw rate (`FlyState.speed`,
`FlyState.yaw_rate` after `Locomotion.step`'s 80 ms smoothing). It feeds nothing back into the walk: the trajectory of a
fly with the cycle attached is bit-identical to one without (section 5(a), `test_shipped_path_is_bit_identical_with_the_cycle_attached_and_the_sense_off`),
so it is a readout of the body's own kinematics, exposed on `FlyState` (`leg_phase` (6,), `leg_stance` (6,), `leg_amp` (6,),
`stance_frac`, `stance_load_L`, `stance_load_R`; order L1 R1 L2 R2 L3 R3) and read by the transducer.

| element | law | source (the numbers taken) |
|---|---|---|
| gait | tripod: L1 R2 L3 share one phase, R1 L2 R3 the phase + 1/2; `phase += f dt` (mod 1); stance = phase < beta | Strauss & Heisenberg 1990, J Comp Physiol A 167:403, verbatim: "For fastest walking alternating tripod coordination is observed which slightly deviates towards tetrapody as a function of step period" -- **tripod at the FASTEST walking, not at every speed** (an earlier version of this row glossed it as "at every walking speed"; skeptic pass); Mendes et al. 2013, eLife 2:e00231 Fig 4 (tripod index rises with speed); DeAngelis et al. 2019, eLife 8:e46409 Fig 2 (a continuum around the tripod, 5-foot stance the most frequent category at 7 mm/s, 3-foot stance peaking at 24 mm/s). The paper that does support a tripod across all speeds is *Drosophila uses a tripod gait across all walking speeds*, eLife 2021;10:e65878 -- cited here for the assumption, not used for a number |
| stance duration | `tau_st(v) = 0.9328 s * (v / 1 mm/s) ^ -1.025` | DeAngelis et al. 2019 Fig 1E / Materials: "tau_stance ~ v^-1.025, R^2 = 0.59", the fit `932.8 ms * (v / 1 mm/s)^-1.025` |
| swing duration | `tau_sw = 30 ms`, constant with speed | Mendes et al. 2013 Fig 2B: "swing phase duration remains mostly constant while stance phase duration is inversely proportional to speed"; step period plateaus at ~60 ms (16 Hz) above ~30 mm/s with swing ~ stance at the fastest speeds -> 30 ms; DeAngelis et al. 2019 Fig 1E "roughly constant". Mendes' Fig 2B reads ~50 ms at slower speeds, so the bracket is **30-50 ms: UNCERTAIN**, and 30 ms is the plateau reading |
| step frequency / stance fraction | `f = 1 / (tau_st + tau_sw)`, `beta = tau_st / (tau_st + tau_sw)` floored at 1/2 (three legs always down) | derived; 7.1 Hz / 0.79 at 8 mm/s (the body's baseline speed), 13.7 Hz / 0.59 at 20 mm/s, 16.5 Hz / 0.51 at 28 mm/s (Mendes 2013's representative speed: "16 cycles per s") -- `tests/test_body_cycle.py::TimingLawTests` |
| standing | v < 0.5 mm/s: f = 0, every leg in stance, amplitude 0 | DeAngelis et al. 2019: "excluding flies moving below 0.5 mm/s" |
| stance-path amplitude | `amp_i = v_i * tau_st(v) / step_ref`, `step_ref` 0.93 mm | **derived from DeAngelis 2019's own confirmed fit, not quoted from it**: the body-frame stance path `v * tau_st(v) = 932.8 ms * v^-0.025 mm` is 0.93 mm at 1 mm/s and 0.86 mm at 30 mm/s, i.e. near-constant across the fitted range. (An earlier version of this row carried a quotation "stance amplitude ... largely kept constant" attributed to DeAngelis 2019; two full-text passes found no such sentence and no statement that the body-frame stance path is constant across speeds -- skeptic pass. What the paper does say is that step length in the camera frame grows roughly linearly with speed, because it includes the body's travel during swing, Mendes 2013 Fig 2E.) |
| turn asymmetry | per-leg ground speed `v_i = |v| - s_i * yaw * b` (s = +1 left, -1 right; yaw > 0 = left turn, left legs inside), `b = half_width_m` 1.0 mm; one shared f; outer legs longer stance paths, inner shorter | rigid-body kinematics of the tarsi about the yaw axis; Strauss & Heisenberg 1990 and DeAngelis et al. 2019 Fig 6C ("outside limbs: step length increased with yaw rate; inside mid/hind: decreased"). **`half_width_m` is NOT a measured number** (no stance-width figure was found in Mendes 2013 / DeAngelis 2019 / Chun et al. 2021; ~1 mm is the body-width order of the mid-leg tarsi): it scales the asymmetry linearly and is a reported constant, never tuned |
| loads | body weight shared equally by the stance legs; `stance_load_L/R` = the share on each side's stance legs (2/3 vs 1/3 alternating within the tripod, 1/2 each standing, 0 airborne); airborne: phase held, no stance, amplitude 0 | the tripod's geometry; no Drosophila per-leg load measurement in a turn exists, so no turn dependence of the load is modelled |

**The yaw rate modulates the cycle, and this is said plainly:** `FlyState.yaw_rate` is `body.Locomotion`'s own realised
yaw scalar (from the DNa02 / leg-MN readout). Here it is read as the body's kinematics -- which leg travels how far over
the ground -- exactly as `speed` is read for the step frequency; neither is injected as a rate anywhere. That is the
difference from the Coriolis stop-gap (arm E, section 1.3), which multiplies an afferent rate by `(1 + g|yaw|)`: a
hand-written scalar fed in as if it were a sense. What is NOT modelled: DeAngelis 2019's per-leg swing-duration
modulation in turns (inner mid/hind swing shorter, outer longer, up to ~25 % net frequency change) -- one frequency for
all six legs; a pivot at v < 0.5 mm/s (the legs stand; the plain fly never goes below its 8 mm/s baseline, and the
pinned compass fly walks at 8-10 mm/s against the pin); backward walking (|v| is used); and **a gait other than a
strict tripod at the realised ~9 mm/s** -- `LegCycle` runs a strict tripod at every speed, which Strauss & Heisenberg
1990 supports only for the fastest walking and which DeAngelis 2019 contradicts at this speed (5-foot stance is its
most frequent category at 7 mm/s, 3-foot stance only at 24 mm/s); the tripod-across-all-speeds reading comes from
eLife 2021;10:e65878, cited above. Strict tripod at 9 mm/s is therefore an ASSUMPTION of the module, not a
measurement.

### 1.2 The side-split haltere readout (`motor.haltere_side_groups`, `motor.read_haltere_sides`)

`WingGroups.haltere` (superclass `vnc_motor`, subclass `hm`) is 16 cells on the shipped cache: **8 L / 8 R by `somaSide`**
(hi2 MN x2 per side; MNhm42, MNhm03, hDVM MN, MNhm43, hi1 MN, hiii2 MN x1 per side; every `instance` carries `_L` / `_R`
and agrees with `somaSide`; 0 unsided) -- `haltere_side_groups(c)`, recorded in every run JSON as
`stimulus.params.haltere_mn_sides`. `read_haltere_sides(brain, groups)` returns the two side means of the same `rate`
tensor `MotorRates.haltere` averages over both sides, at the moment the transducer is fed (before `fb.step`, so it is the
previous frame's readout, the snapshot rule of `proprioception_transducer.md` 1). The task asked for
`MotorRates.haltere_L / haltere_R`; **they are deliberately not fields of `MotorRates`**: `tests/test_bit_identity.py`
hashes `asdict(fb.motor())` field by field into a recorded golden whose regeneration "is an owner decision, not a test
fix", and a new field -- even one that changes no simulated tensor -- moves that digest. The side split therefore lives
beside `MotorRates` as an opt-in function that touches nothing unless called; `MotorRates.haltere` stays the bilateral
mean. Promoting the pair to fields is a one-line change plus a golden regeneration, listed in section 8.

### 1.3 The transducer under the two mechanisms (`senses.Proprioception`, tokens `leg_cycle`, `haltere_sided`)

Spec tokens, both OFF unless named (`'all'` is the round-2 transducer exactly): `'all+leg_cycle'` (arm C),
`'all+leg_cycle+haltere_sided'` (D), `'+haltere_coriolis'` (E, the labelled stop-gap). The sided state reaches the sense
through `Proprioception.take_body(state)` (`BatchSim.step`'s one changed line; `FlyBrain.proprioception`'s five-argument
signature is another file's and is unchanged), or directly through `rates(legs=..., haltere_L=..., haltere_R=...)`.
Cells map to legs by the round-2 side rule and the entry nerve (ProLN / ProAN / VProN / DProN / ProCN -> T1, MesoLN -> T2,
MetaLN -> T3; a cell with no leg nerve -- the 35 PrN neck hair plates -- reads the mean of its side's three legs; an
unsided cell the mean of L and R; `Proprioception.leg_weights`, rows summing to 1). Per leg on the ground, with
`p` = the leg's protraction coordinate (1 at touchdown, 0 at lift-off, rising back through the swing) and `a` = its
stance-path amplitude:

| channel | round-2 law (arms B) | leg-cycle law (arms C, D, E) | bracket (ledger row, `op report`) |
|---|---|---|---|
| chordotonal (FeCO) | `10 + 140 * clip(legMN_side / 30, 0, 1)` | `10 + 140 * clip(a * [1 in swing; \|2p - 1\| in stance], 0, 1)` -- the club / hook movement burst over the whole excursion in swing, the claw position units greatest at the joint extremes in stance (Mamiya, Gurung & Tuthill 2018 for the subtypes; Matheson 1992 / Field & Matheson 1998 for the locust bracket); the annotation does not separate claw / club / hook cells, so one rate per cell carries both | `lit.FeCO.afferent_rate_range_hz` 10-150 |
| hair plate | `5 + 95 * clip(legMN_side / 30, 0, 1)` | `5 + 95 * clip(a * p, 0, 1)` -- deflected at the protracted extreme (Wong & Pearson 1976, cockroach trochanteral hair plate) | `lit.hair_plate.afferent_rate_range_hz` 5-100 |
| leg campaniform | 50 on the ground | `50 * stance * 3 / n_stance` -- 50 Hz per tripod leg, 25 Hz standing on six, 0 in swing (Ridgel et al. 2000 / Zill et al. 2013 tonic-to-sustained-load) | `lit.campaniform_leg.afferent_rate_range_hz` 0-100 |
| haltere | `1 * haltereMN` (one bilateral mean) | `1 * haltereMN_side` per side under `haltere_sided`, unsided cells the mean; `x (1 + 1.0 \|yaw\|)` only under `haltere_coriolis` | `lit.haltere.afferent_rate_range_hz` 0-250; `lit.haltere.coriolis_modulation` (stop-gap) |

Airborne: tonic / tonic / 0 as before. Every rate keeps the round-2 ceiling clip; the tonic / load / ceiling constants
are the round-2 ones; the new dependence is on `a` and `p` only. The turn asymmetry enters the afferents ONLY through `a`
(the outer legs' longer steps) and through the tripod's per-side stance load; no number was chosen by looking at whether
the fly turns.

## 2. Classification under the project rule

| item | classification | why |
|---|---|---|
| `LegCycle` (tripod timing from the realised speed) | **body-model mechanism** (a stance / swing phase derived from the realised speed, the case the rule names as legitimate) | literature laws with citations; feeds nothing back into the walk; exposed state only |
| the yaw term in `LegCycle` (`v_i = v - s_i yaw b`) | **body-model mechanism, stated plainly**: the body's own kinematics read from its own realised yaw scalar | which leg travels how far is the geometry of a rigid body turning; it is not a gain, not a bias, not a rate injected anywhere; `half_width_m` is a reported constant with no measured value (UNCERTAIN) -- **and it is functionally the GAIN OF A CLOSED LOOP**, see below |
| `read_haltere_sides` | **readout that respects the anatomy's sidedness** (8 L / 8 R cells, `somaSide` and `instance` agree) | the same quantity as `MotorRates.haltere`, per side |
| per-leg / per-phase transducer laws | **mechanism the connectome implies** (a transducer on wired, never-driven afferents), body-model limitation now replaced by a body model | brackets unchanged; the claw / club / hook mixing per cell is an annotation limitation, documented |
| `haltere_coriolis` (arm E) | **STOP-GAP, labelled control arm, never a default** (unchanged from round 2) | a hand-written scalar multiplied into a sense |
| swing 30 ms, `step_ref` 0.93 mm, `half_width` 1.0 mm, `v_min` 0.5 mm/s | transducer / body constants with literature brackets, exposed as `LegCycle` fields | none set against behaviour; swing and half-width are UNCERTAIN (section 1.1) |

**`half_width_m` is the loop gain of a brain-body loop, and is named as such** (skeptic pass): the realised yaw enters
the per-leg ground speed, which sets the afferent amplitude, which drives AN04B003 -> DNa02 -> yaw, and `half_width_m`
(1.0 mm, declared UNMEASURED) scales that whole path linearly. Classifying it as a body-model geometric constant is
defensible -- it is a stance width, not a tuned gain, and section 8 item 7 already flags it -- but the loop framing
belongs in the text. What makes a wrong value survivable here is the measurement in section 4.2/4.3: the turn's DC
term on the chordotonal L-R is only 2.5 % of the tripod alternation (-0.3 Hz against a per-frame |L-R| of 12.5 Hz), so
even a 2x error in `half_width_m` would not change any conclusion of this round.

What would be hand-crafting and is not done: any gain on the afferent -> AN or AN -> DNa02 / PS196_b links; a per-side
frequency asymmetry chosen to make PS196_b flip; a load asymmetry in turns without a measurement; driving the cycle from
`yaw_cmd` (the command) instead of the realised `yaw_rate`; keeping arm E as anything but a control.

## 3. The batch

`out/vncd3/batch.sh` (written by `probe_vnc_drive.py plan --family body --dir out/vncd3 --runs 5 --compass-seeds 3
--draws 0 --minutes 90`): ONE `cluster_run.py --name vncd3 --arm-block fam` call with **28 jobs** -- 25 `room` (5 arms x
brain seeds 0-4, 16 flies x 60 s each, environment seeds 100s..100s+15, `probe_walk_straightness.py`'s plain-fly protocol
exactly as in round 2) and 3 `compass` (seeds 0-2; arms A / D / E sequential inside one job each, each arm its own python
process with its own exit code) -- `--fetch out/vncd3/`, console `out/vncd3_cluster.log`. Every room job line ends
`> <file> 2>&1; st=$?; tail -4 <file>; exit $st` and every compass job `exit $((s0 | s1 | s2))`, so a job's status is
python's (the `; tail` shadowing of round 2's batch is gone). No bench jobs: the bench protocol has no body, so arms C-E
cannot run there (section 8 lists it under adoption).

**A first submission was cancelled before any of its jobs started.** The first `batch.sh` wrote the job strings inside
double quotes without escaping `$`, so the LOCAL shell expanded `$?` / `$st` / `$((s0|s1|s2))` before `cluster_run.py`
saw them and the shipped job text read `st=0; tail -4 ...; exit ` -- exactly the exit-code shadowing the line was meant
to remove (the job's status would have been `tail`'s). All 28 jobs of run `vncd3-d494ec` were still `queued` behind the
other threads' work (0 running), so every one was cancelled through the manager's `POST /jobs/<id>/cancel` (0 remaining
on each box), the client was stopped, `plan` now escapes `$` (`quoted()` in `cmd_plan`), a local dry run through an
argv printer confirmed all 25 room lines end `st=$?; tail -4 <file>; exit $st` and the 3 compass lines carry
`s0=$?` ... `exit $((s0 | s1 | s2))`, and the batch was resubmitted under the same name. The cancelled submission's
console is kept as `out/vncd3_cluster_cancelled_d494ec.log`; it produced no artefact and no number comes from it.

**The resubmission (`vncd3-f3bb50`, `out/vncd3_cluster.log`) ran its d-box blocks and was then split.** Block r2 and
compass c0 (6 jobs) went to `r3-h200d`, which was idle, and completed in 349-521 s per room job (4 concurrent) and
96-241 s per compass arm; the other 22 jobs sat `queued` on `r3-h200a` / `r3-h200b` behind 39 / 25 jobs of the other
threads with none of mine started after 56 min (a: 15 completions in that time, b: 6). Since `r3-h200d` and the newly
configured `r3-h200c` (both H200, both idle) were free capacity of the same GPU model, the 22 queued jobs were cancelled
(`POST /jobs/<id>/cancel`, 0 remaining, 0 running; the client then reported `28 job(s), 22 failed` -- the 22 are the
cancellations -- and fetched d's six results into `out/vncd3/`), and the six remaining blocks were submitted as ONE
further call, `out/vncd3/batch_b.sh` (`plan ... --only-seeds 0,1,3,4 --only-compass-seeds 1,2 --targets
r3-h200d,r3-h200c --name vncd3b`, console `out/vncd3b_cluster.log`), which is the second and last submission of this
round. Consequences for the statistics: every block (a seed's five arms; a compass seed's three arms) is still intact on
one box; the five runs of each arm come from two submissions (r2 from `vncd3-f3bb50`, r0 / r1 / r3 / r4 from `vncd3b`),
and section 4 therefore reports each verdict both on the four runs of the single submission `vncd3b` (4 v 4, floor
p 0.029) and on all five (5 v 5, floor 0.0079); the two never disagree on a call in the tables below unless said so.
The box d results were also copied by hand into `out/vncd3_stage_d/` while the rest waited, to develop the robust yaw
statistics of section 4 on real GPU runs; no number is quoted from that staging directory.

**Blocking.** The comparison family is the seed: block `fam_r<s>` holds the five arms A-E of one seed, so every
treatment arm of a seed is compared with its own reference A on ONE box; the compass jobs are blocks `fam_c<s>`. The
fleet at the (second, section below) submission was three H200 boxes (`r3-h200a`, `r3-h200b`, `r3-h200d`, 5 slots each;
the B200s of round 2 are disabled in `.cluster.json`), dealt round-robin largest-first: **r0, r3, c1 -> r3-h200a (11
jobs); r1, r4, c2 -> r3-h200b (11); r2, c0 -> r3-h200d (6)**. All three are the same GPU model, so no arm-vs-A row here
is device-crossed; the realised `device_name` per run is in every JSON (section 4). At submission the boxes already held
**51, 35 and 1 pending / queued / running jobs of the same user** (the other round-3 threads), so this batch queued
behind them; nothing was submitted to the house cluster (disabled) and no second client fetched into `out/vncd3/`.

**What the batch shipped, and what the numbers depend on (docs/INTERP.md 10.1(7b)).** `cluster_run.py` ships every file
that differs from `origin/main` (HEAD d2abf3c = origin/main at submission), 25 files: this task's (section header) and
the other round-3 threads' working-tree edits -- `flyverse/brain.py` (+31 / -1: `LIFParams.w_syn_by_nt`, default
`None`, "executes nothing and the shaped weights are byte-identical"; thread unitary), `flyverse/data/expected_responses.csv`
(+13 `op report` rows; threads unitary / monoamines), their untracked scripts / audits / tests, and
`docs/audits/body_sided_state.md` itself. Neither dependency touches the shipped path (a `None` field and report-only
ledger rows), and every run JSON records the resolved `LIFParams` (with `w_syn_by_nt: null`), the compiled-W md5 and the
44-file `source_fingerprint` as the code identity (`flyverse_commit.commit` is `unknown` on the rented boxes, as in
round 2). None of these cross-task files were committed before submission (this task may not commit), so the honest
statement is the round-2 one: *depended on, authored by another task, reproducible from the fingerprint*.

## 4. The room (25 runs: 5 arms x 5 brain seeds, 16 flies x 60 s, 5-60 s window)

Artefacts: `out/vncd3/room_<arm>_r<seed>.json` (summary, per-fly rows, provenance), `_body.npz` (per-frame heading /
position / commands / the commanded afferent rate per channel, per side and per leg / the cycle state / the haltere MN
readout per side), `_rec.npz` (the watch types' cells at 2-frame resolution), `_flies.npz` / `_max.npz` (per-fly and
per-frame-max rates of every cell). Tables: `out/vncd3/analysis/room_table.csv` (every key, per-run values, vs-A
verdicts; `analysis_console.txt`), `pairwise.csv` (adjacent-arm verdicts), `sided_frames.csv` (section 4.3),
`dna02_decompose_summary.csv` (4.4), and the same under `analysis_4runs/` on seeds 0 1 3 4 alone. **Every room run
reports `device cuda`, `device_name NVIDIA H200`, family `body`, its arm's spec, and its seed's block `fam_r<seed>`**
(`analysis_console.txt` lines 2-7); the five-run and four-run tables differ in **15 of 765 `pairwise.csv` verdict
cells and 7 of 548 `room_table.csv` cells** (skeptic pass, reproduced by diffing the verdict columns; an earlier
version of this sentence said "NO verdict (0 rows ... and 0 ...)" and was wrong). The 22 flips, 5-run -> 4-run:
`pairwise.csv` hops DvB result->null; yaw_p99_abs_clean_deg_s BvA result->null; yaw_pos_flies BvA result->null;
corr_yaw_hairLR EvD result->null; corr_yaw_haltLR CvB null->result; corr_dna02LR_haltLR CvB null->result and EvD
result->null; corr_an04LR_yaw EvD null->result; commanded_haltere_LR_hz EvD null->result; AN06A026_L_hz EvD
result->null; AN07B035_L_hz EvD null->result; AN07B037_a_L_hz EvD result->null; PS059_R_hz EvD result->null;
PS196_b_L_hz DvC result->null and EvD result->null. `room_table.csv` left_table_s_mean_capped D_vs_A and E_vs_A
null->result; hops D_vs_A and E_vs_A result->null; gf_max_hz C_vs_A null->result; yaw_p99_abs_clean_deg_s B_vs_A
result->null; AN07B037_a_L_hz E_vs_A result->null. **None of the 22 is a section-4.1 headline row** (no C-v-B /
D-v-C / E-v-D row quoted in section 0 or 4.1 is among them), so the five-run numbers are the ones quoted below --
with the caveat that the five-run tables span TWO submissions (seed 2 from `vncd3-f3bb50`, seeds 0 / 1 / 3 / 4 from
`vncd3b`) while the four-run table is the single-submission one INTERP asks for.

### 4.1 Behaviour, the motor readout and the yaw artefact

| key | A | B | C (+ cycle) | D (+ sided haltere) | E (+ Coriolis) | C v B | D v C | E v D |
|---|---|---|---|---|---|---|---|---|
| yaw SD, clean frames (deg/s) | 2.64 +- 0.13 | 3.35 +- 0.09 | **7.87 +- 0.14** | 7.73 +- 0.18 | 7.93 +- 0.18 | +4.53 z+51 result | -0.15 null | +0.20 null |
| median \|yaw\|, clean (deg/s) | 0.74 +- 0.01 | 1.13 +- 0.01 | 1.57 +- 0.08 | 1.54 +- 0.05 | 1.53 +- 0.05 | +0.45 z+48 result | null | null |
| p99 \|yaw\|, clean (deg/s) | 15.2 +- 0.8 | 18.0 +- 0.4 | 25.8 +- 0.2 | 24.8 +- 0.8 | 26.4 +- 0.8 | +7.8 z+22 result | null | null |
| mean SIGNED yaw, clean (deg/s; + left) | +0.21 +- 0.04 | +0.59 +- 0.04 | **+1.23 +- 0.16** | +0.99 +- 0.16 | +1.08 +- 0.03 | +0.64 z+15 result | -0.24 null | null |
| flies (of 16) with mean yaw > 0 | 13.2 | 16.0 | 15.6 | 15.4 | 15.4 | | | |
| straightness | 0.995 +- 0.001 | 0.979 +- 0.002 | **0.826 +- 0.032** | 0.876 +- 0.038 | 0.867 +- 0.014 | -0.15 z-98 result | +0.05 null | null |
| net heading change (turns / fly) | 0.034 | 0.090 | 0.216 +- 0.039 | 0.189 +- 0.049 | 0.187 +- 0.017 | +0.13 result | null | null |
| flies (of 16) that left the table top | 0.6 +- 0.9 | 0.2 +- 0.4 | **7.4 +- 1.8** (8 7 5 10 7) | 6.6 +- 2.3 | 7.0 +- 1.9 | +7.2 z+16 result | null | null |
| hops per fly | 0.075 | 0.025 | **0.46 +- 0.24** | 0.38 +- 0.21 | 0.34 +- 0.17 | +0.44 z+8 result | null | null |
| walking frames / fly with \|yaw\| > 720 deg/s | 0 | 0 | 0.96 +- 0.25 | 0.81 +- 0.18 | 0.90 +- 0.13 | undetermined (A, B are exactly 0) | null | null |
| round-2 `yaw_sd_deg_s`, all walking frames | 2.68 | 3.37 | 93.8 +- 17.7 | 80.0 +- 14.2 | 82.4 +- 11.2 | | | |
| DNa02_L / R (Hz) | 0.017 / 0.075 | 0.031 / 0.104 | **0.540 / 0.383** | 0.512 / 0.393 | 0.535 / 0.394 | result / result | null / null | null / null |
| DNa02 L-R (Hz) | -0.059 +- 0.009 | -0.072 +- 0.006 | **+0.158 +- 0.030** | +0.122 +- 0.023 | +0.141 +- 0.010 | +0.23 z+35 result | -0.04 null | null |
| fly-runs (of 80) with DNa02 L-R > 0 | 5 | 8 | 68 | 67 | 68 | | | |
| frames with DNa02 > 5 Hz | 0.006 | 0.010 | 0.066 +- 0.001 | 0.064 | 0.066 | result | null | null |
| leg MN L / R (Hz) | 2.54 / 2.39 | 3.03 / 2.72 | 4.81 / 4.67 | 4.79 / 4.64 | 4.76 / 4.61 | result | null | null |
| leg MN L-R (Hz) | +0.151 +- 0.003 | +0.307 +- 0.009 | +0.141 +- 0.006 | +0.147 +- 0.007 | +0.145 +- 0.007 | -0.17 result | null | null |
| fly-runs (of 80) with leg L-R < 0 | 0 | 0 | 0 | 0 | 0 | | | |
| haltere MN L / R (Hz) | 7.63 / 9.39 | 9.75 / 11.57 | 15.46 / 18.02 | 15.33 / 18.06 | 15.59 / 18.39 | result | null | null |
| haltere MN L-R (Hz) | -1.76 +- 0.02 | -1.82 +- 0.03 | -2.56 +- 0.07 | -2.73 +- 0.14 | -2.80 +- 0.10 | -0.74 result | -0.17 null (p 0.056) | null |
| wing power MN mean (Hz) | 20.3 | 18.0 | 15.3 +- 0.2 | 15.4 | 15.2 | -2.8 result | null | null |
| GF per-frame max (Hz) | 25.6 +- 0.8 | 25.0 +- 0.5 | 29.5 +- 5.0 | 27.6 +- 4.2 | 27.2 +- 2.8 | +4.6 z+9 p0.016 result | null | null |
| step frequency (Hz) / stance fraction | | | 7.86 +- 0.18 / 0.764 | 7.78 / 0.767 | 7.68 / 0.770 | | null | null |
| corr(amp L-R, realised yaw) per fly | | | -0.52 +- 0.09 | -0.57 +- 0.08 | -0.53 +- 0.07 | | null | null |
| \|amp L-R\| / \|stance load L-R\| | | | 0.016 / 0.157 | 0.016 / 0.156 | 0.016 / 0.154 | | null | null |

Per-seed values behind the calls (seed order r0..r4): yaw SD clean C 7.84 / 7.96 / 8.01 / 7.66 / 7.88, D 7.60 / 7.90 /
7.71 / 7.51 / 7.91, E 7.76 / 7.95 / 7.97 / 7.77 / 8.19, B 3.22 / 3.40 / 3.42 / 3.28 / 3.40, A 2.47 / 2.68 / 2.65 / 2.59 /
2.82; DNa02_L C 0.558 / 0.556 / 0.534 / 0.542 / 0.513, D 0.513 / 0.513 / 0.499 / 0.509 / 0.526, B 0.026 / 0.030 /
0.034 / 0.035 / 0.031; DNa02_R C 0.374 / 0.393 / 0.398 / 0.352 / 0.398, D 0.381 / 0.417 / 0.398 / 0.358 / 0.411;
signed yaw C 1.29 / 1.21 / 1.26 / 1.42 / 0.98, D 1.19 / 0.82 / 0.86 / 1.12 / 0.95; straightness C 0.822 / 0.837 /
0.832 / 0.776 / 0.865, D 0.830 / 0.926 / 0.904 / 0.861 / 0.861.

**What "E vs D `null`" is scoped to.** D v C has 14 `result` rows in the five-run `pairwise.csv` and E v D has 15, and
not one of them is a behavioural or a DNa02-rate row -- which is the claim this audit makes. But the parenthetical
"(commanded haltere +1.93 Hz `result`)" names only one of E-v-D's 15. The full set is the commanded and measured
haltere rate per side and their |L-R|, AN06A026_L / _R, AN07B037_a_L, PS059_R +0.65 Hz, PS196_b_L +0.31 Hz, and three
haltere correlations -- all downstream rate rows that the Coriolis multiplier does move by construction (skeptic
pass). The scoped claim is unchanged; the parenthetical should not read as if only one row moved.

**The yaw artefact.** Round 2's `yaw_sd_deg_s` counts every walking frame. Under C-E a fly that walks over the table
edge (7 of 16 do) has its heading representation flip over the edge -- 90-180 deg in one 10 ms frame, 9,000-18,000
deg/s -- and one such frame per fly puts the run SD at 60-125 deg/s (C per seed 92 / 81 / 82 / 124 / 88). `analyse`
therefore computes `robust_room` on every run: a clean frame is on the table top before and after, has no airborne
frame within 0.5 s, and |yaw| <= 720 deg/s (7.2 deg per frame). 92-95 % of window frames are clean under C-E (99.8 %
under A), and the clean SD is the number the task's "yaw-rate SD" is answered with. `yaw_max_abs_deg_s` (5,638 +-
1,243 under C) is that artefact's size; no frame under any arm exceeds 100 deg/s on a clean frame.

**What the cycle did and did not change in the body.** The walk itself is untouched (section 5(a)): the fly's speed,
yaw and position depend on the cycle ONLY through the brain, i.e. through the afferents. The stance fraction 0.76 and
step frequency 7.9 Hz are the laws' values at the realised ~9 mm/s (section 1.1: 7.1 Hz / 0.79 at 8 mm/s, 13.7 / 0.59 at
20). The turn asymmetry is present and has the right sign (corr(amp L-R, yaw) -0.52 to -0.57 per fly; a left turn
shortens the left steps) and is small: |amp L-R| 0.016 of the step against the tripod's per-frame |load L-R| 0.157
and the afferents' per-frame |chordotonal L-R| 12.5 Hz.

**The fixed biases.** Three sided quantities never change sign in any fly-run of any arm: the leg MN L-R (+0.14-0.31
Hz, 0 / 80 negative -- round 2's fact, untouched because the cycle reads the same MNs and feeds nothing back), the
haltere MN L-R (-1.8 to -2.8 Hz, R > L: the same asymmetry under A, so it is the connectome's, not the transducer's),
and, under the cycle, DNa02's L-R (+0.12 to +0.16 Hz, L > R in 67-68 / 80 fly-runs) with the mean signed yaw +1.0 to
+1.2 deg/s (left) in 77-78 / 80 flies. The A fly drifts left too (+0.21 deg/s, 66 / 80 flies) with DNa02_R > DNa02_L:
the sign of the leg-MN bias. Under the cycle the DNa02 bias overrides it: section 4.4 shows why DNa02_L's net input is
positive and DNa02_R's negative.

### 4.2 The afferents per channel, per side and per leg (commanded Hz = the Poisson rate the transducer set; measured = the cells' realised rate)

| channel (n cells; L / R / unsided) | B commanded (measured) | C | D | E | bracket |
|---|---|---|---|---|---|
| chordotonal (615; 271 / 259 / 85) | 23.4 +- 0.1 (23.4) | **88.1 +- 1.8** (88.1) | 87.3 +- 1.1 | 86.5 +- 1.3 | 10-150 |
| chordotonal L / R | 24.1 / 22.7 | 87.9 / 88.3 | 87.2 / 87.5 | 86.3 / 86.6 | |
| chordotonal per leg L1 R1 L2 R2 L3 R3 (C) | | 88.0 88.2 87.9 88.4 88.0 88.2 | | | |
| chordotonal L-R mean / per-frame \|L-R\| | +1.43 / 2.24 | -0.32 / **12.46** | -0.27 / 12.32 | -0.28 / 12.19 | |
| hair plate (113; 57 / 54 / 2) | 14.1 (14.1) | 47.4 +- 1.0 | 47.0 | 46.6 | 5-100 |
| hair plate L-R mean / per-frame \|L-R\| | +0.97 / 1.52 | -0.18 / 8.46 | -0.15 / 8.37 | -0.16 / 8.28 | |
| leg campaniform (12; 6 / 6 / 0; 2 per leg) | 50.0 (50.0) | 24.9 (24.9) | 24.9 | 24.9 | 0-100 |
| campaniform L-R mean / per-frame \|L-R\| | 0 / 0 | -0.01 / 7.86 | -0.01 / 7.78 | -0.01 / 7.68 | |
| haltere (201; 99 / 95 / 7) | 10.7 (10.7) | 16.8 (16.8) | 16.7 | **18.6** | 0-250 |
| haltere L / R | 10.7 / 10.7 | 16.8 / 16.8 | **15.3 / 18.1** | 17.1 / 20.2 | |
| haltere L-R mean / per-frame \|L-R\| | 0 / 0 | 0 / 0 | **-2.73 / 3.92** | -3.06 / 4.33 | |

Cells map to legs by the round-2 side rule and the entry nerve: chordotonal L1 53, R1 25, L2 109, R2 111, L3 109,
R3 123 cells (T1 102, T2 246, T3 267; no cell without a leg nerve), hair plate 8 / 9 / 19 / 14 / 12 / 14 (35 PrN neck
hair plates read their side's mean), campaniform 2 per leg. The per-leg means are equal to 0.5 Hz because every leg
runs the same cycle at the same speed; the sidedness is in the per-frame |L-R| (the tripod alternation: 12.5 Hz of
chordotonal, 8.4 of hair plate, 7.9 of campaniform), and the turn's DC term in the L-R mean is -0.3 Hz (chordotonal)
-- 2.5 % of the alternation. The round-2 B arm had a +1.4 Hz DC L-R and no alternation: the fixed leg-MN bias.

**Where the level comes from.** Under B the leg channels read `clip(legMN_side / 30, 0, 1)`: at 2.7-3.0 Hz that is 0.09-
0.10 of the range, 23 Hz. Under the cycle the chordotonal drive is `a x [1 in swing; |2p - 1| in stance]`: with beta
0.76, amplitude 0.95 and the stance protraction sweeping 1 -> 0, the cycle mean is 0.95 x (0.24 x 1 + 0.76 x 0.5) =
0.59 of the range, 93 Hz; measured 88 (airborne and standing frames at the tonic 10 Hz). Both are inside 10-150 Hz and
neither is a ledger number: the ledger has the bracket only (`lit.FeCO.afferent_rate_range_hz`, Mamiya et al. 2018,
`op report`), and no walking-mean row. The hair plate goes 14 -> 47 Hz the same way (`a x p`, mean ~0.47); the
campaniform go DOWN 50 -> 25 Hz because 50 Hz per tripod leg with three of six legs down is 25 Hz per cell on average
(the round-2 law put 50 Hz on every cell on the ground). This is the whole of the C-vs-B effect that section 4.4 can
account for, and it is a level, not a phase.

### 4.3 Sidedness per frame (`sided_frames.csv`; `probe_vnc_drive.py pairs`; clean walking frames, per fly, then the run mean; SD over 5 runs)

| statistic | B | C | D | E |
|---|---|---|---|---|
| corr(realised yaw, chordotonal cmd L-R) | +0.291 +- 0.007 | **-0.148 +- 0.003** | -0.144 +- 0.005 | -0.151 +- 0.005 |
| corr(realised yaw, hair-plate cmd L-R) | +0.291 | -0.156 | -0.153 | -0.158 |
| corr(realised yaw, haltere cmd L-R) | +0.024 | +0.010 | +0.094 +- 0.006 | +0.127 +- 0.019 |
| corr(realised yaw, haltere MN L-R) | +0.152 | +0.057 | +0.081 | +0.097 |
| corr(realised yaw, leg MN L-R) | +0.265 | +0.041 | +0.046 | +0.041 |
| corr(AN04B003 L-R, chordotonal cmd L-R) | +0.131 +- 0.004 | **-0.295 +- 0.005** | -0.297 | -0.296 |
| E[AN04B003 L-R \| chord L-R > 0] - E[. \| < 0] (Hz) | +0.67 +- 0.03 | **-3.39 +- 0.06** | -3.40 +- 0.04 | -3.36 +- 0.04 |
| corr(DNa02 L-R, chordotonal cmd L-R) | +0.009 | **-0.144 +- 0.002** | -0.140 | -0.146 |
| E[DNa02 L-R \| chord L-R > 0] - E[. \| < 0] (Hz) | +0.02 +- 0.02 | **-0.35 +- 0.02** | -0.33 +- 0.02 | -0.35 +- 0.02 |
| corr(AN04B003 L-R, realised yaw) | +0.040 | -0.068 +- 0.005 | -0.069 | -0.074 |
| corr(DNa02 L-R, haltere cmd L-R) | +0.024 | +0.011 | **+0.136 +- 0.004** | +0.167 +- 0.016 |
| corr(PS059 L-R, haltere cmd L-R) | -0.096 | -0.041 | **+0.082 +- 0.009** | +0.072 |
| corr(PS196_b L-R, haltere cmd L-R) | -0.001 | +0.002 | **+0.150 +- 0.007** | +0.156 |
| corr(PS196_b L-R, realised yaw) | +0.037 | +0.055 | +0.070 +- 0.005 | +0.076 |

Reading it: (i) the kinematic sign is there -- under the cycle a left turn (yaw > 0) comes with a lower chordotonal
L-R (-0.15; the left legs' shorter steps), under B with a higher one (+0.29; the leg-MN bias turns and the MN-scaled
afferents follow it); (ii) AN04B003 and DNa02 follow the afferent L-R per frame, in antiphase to the side sum, and the
tripod-locked swing of DNa02's L-R (-0.35 Hz) is twice its window mean (+0.16); (iii) the sided haltere is seen by
PS059, PS196_b and DNa02 as a per-frame signal (0 -> +0.08 / +0.15 / +0.14) that carries almost no yaw (the haltere
MN L-R's own corr with yaw 0.08); (iv) none of it turns the fly: corr(AN04B003 L-R, yaw) -0.07, corr(PS196_b L-R, yaw)
0.07 -- the same 0.04 they show under A and B.

**Two reducer caveats on this table, both found by the skeptic pass, neither re-run here.** (i) `sided_frames()`
aligns the `_rec.npz` samples one body frame EARLY: `scripts/probe_vnc_drive.py` (~line 1126) uses
`fr = clip(round(t_ms/10) - 2, 0, T-1)` against `cl = chord[1:]`, so rec sample `s` is paired with body frame
`2s - 1`. A lag scan on `room_C_r0` peaks at lag 0, where corr(AN04B003 L-R, chordotonal L-R) is **-0.339** and the
tripod-conditioned swing **-3.49 Hz**, against -0.293 / -3.37 at the shipped lag -1 (which reproduces the -0.295 /
-3.39 printed above) and -0.289 / -2.78 at lag +1. The AN04B003 sidedness is therefore about **13 % LARGER** than the
table reports -- the argument is unchanged (the sided term is real and still a ~30th of the alternation it rides on),
and the DNa02 rows are unaffected because both of their series come from the body npz and are sliced together. The
alignment was NOT changed and the table was NOT re-emitted this round. (ii) Where a z is quoted in an audit table whose
criterion is |z| >= 3, it gets one decimal from here on. `pairs` itself does not round -- `pairwise.csv` stores z at
full precision and `pairs_console.txt` prints `z{:+.1f}` -- but transcribed tables have rounded it to the nearest
integer, and in `round3_integration.md`'s pairwise table the two rows printed `z-3 ... null` are actually z **-2.88**
and **-2.77** (skeptic's numbers); the verdicts are right, the printed z is not. The z values quoted in this audit
(z+51, z+141, z+9.05 ...) are likewise rounded transcriptions of the one-decimal console values.

**The antiphase is the wiring, not a contralateral route**: AN04B003_L
receives 93 synapses from L1, 171 from L2 and 34 from L3 chordotonal afferents (R legs 1 / 27 / 29); AN04B003_R 48
from R1, 218 from R2, 163 from R3 (L legs 0 / 20 / 3) -- ipsilateral, but AN04B003_L's largest input L2 swings with
R1 R3 while AN04B003_R's largest R2 swings with L1 L3, so when the L1 R2 L3 tripod bursts the side-summed L-R rises
(L1 + L3 = 162 cells vs R2 111) and AN04B003 R > L. The ascending neuron reads legs, not sides; the side sum is the
wrong coordinate for it.

### 4.4 DNa02's input per arm (`decompose_DNa02_L / _R_per_type.csv`, rate-weighted input in mV/s per post cell over the window; `dna02_decompose_summary.csv`)

| DNa02_L input | A | B | C | D | E |
|---|---|---|---|---|---|
| E total / I total / **net** | +742 / -1045 / -303 | +817 / -1050 / -233 | +1369 / -1258 / **+110** | +1350 / -1253 / +97 | +1355 / -1267 / +88 |
| AN04B003/L (leg afferents, k1) | +9.3 | +57.9 | **+360.4** | +356.1 | +353.6 |
| SNpp45/? (unsided afferent, direct) | | +40.0 | +129.9 | +128.8 | +127.6 |
| PS013/L, GNG580/L, AN10B021/R, AN10B018/R | | +21 (GNG580) | +52 / +49 / +39 / +33 | +50 / +45 / +39 / +32 | +52 / +50 / +39 / +32 |
| PS059/L (haltere afferents, k1) | -90.2 | -114.1 | **-207.5** | -205.5 | -209.6 |
| IN12B014/R | -90.5 | -85.8 | -64.6 | -64.8 | -65.4 |
| GNG562/L / LT51/L / IN19A003/L | -68.5 / -57.5 / -43.2 | -61.5 / -54.1 / -36.0 | -45.5 / -45.1 / -19.8 | -45.2 / -45.2 / -20.5 | -44.8 / -44.7 / -20.6 |
| LAL046/L, PS077/L, LAL120_a/R (new under C) | | | -48 / -45 / -43 | -45 / -43 / -43 | -47 / -43 / -43 |

| DNa02_R input | A | B | C | D | E |
|---|---|---|---|---|---|
| E total / I total / **net** | +807 / -1169 / -362 | +859 / -1205 / -345 | +1345 / -1500 / **-155** | +1344 / -1497 / -153 | +1338 / -1509 / -171 |
| AN04B003/R | +4.7 | +55.9 | **+360.0** | +356.4 | +352.2 |
| SNpp45/?, AN10B021/L, PS013/R | | +17 | +60 / +55 / +41 | +59 / +54 / +42 | +59 / +53 / +43 |
| PS059/R | -49.3 | -75.1 | **-169.9** | -171.0 | -177.6 |
| IN12B014/L | -125.7 | -120.6 | -98.1 | -98.3 | -98.5 |
| LT51/R / IN19A003/R / GNG562/R | -67.9 / -60.5 / -55.0 | -66.9 / -64.6 / -49.6 | -66.9 / -54.1 / -35.0 | -67.1 / -55.0 / -35.0 | -66.6 / -56.8 / -35.0 |
| LAL126/L, LAL120_a/L, IN09A004/R (new under C) | | | -68 / -57 / -56 | -65 / -57 / -56 | -67 / -57 / -55 |

(All five runs per cell; run SD 1-4 mV/s on the named rows: PS059/L C 207.5 +- 3.9, D 205.5 +- 2.6, E 209.6 +- 3.3;
PS059/R 169.9 +- 3.4 / 171.0 +- 2.0 / 177.6 +- 2.1. Every C / D / E row is `result` against A at p 0.008; D vs C is
inside the run SD on every named row.)

Answers. **Does the PS059 cancellation break when the haltere channel is sided?** No: PS059/L and /R change by < 2 %
from C to D, and by +2 / +4 % from D to E (the Coriolis multiplier raises the haltere afferents 16.7 -> 18.6 Hz and
PS059 with them, 20.0 -> 20.4 / 17.0 -> 17.6 Hz). The sided readout moves the haltere afferents' DC by -1.4 / +1.4 Hz
per side (15.3 / 18.1 against 16.75), and PS059's rates follow the same way (L 20.0, R 17.0 in both C and D -- PS059_L
was already the higher one under the bilateral channel, from the haltere afferents' own side counts 99 / 95 and the
ipsilateral synapse counts 241 / 277). **Does a sided AN04B003 term survive?** In the window mean, no: AN04B003 L 23.1
/ R 23.6 Hz (D), a -0.5 Hz difference that is the two sides' cell counts (L1 + L2 + L3 = 271 cells vs 259) and their
synapse counts; per frame, yes, tripod-locked (section 4.3). **What broke the round-2 cancellation** is the ratio: under
B the haltere-afferent inhibition (PS059/L -114) was twice the leg-afferent excitation (AN04B003/L +58); under the
cycle the leg afferents run at 88 Hz instead of 23 and AN04B003 at 23 Hz instead of 3.7, so its term is +360 against
PS059's -207 -- the haltere MNs, and with them the haltere afferents, rose too (10.7 -> 16.8 Hz: the more active VNC
drives the haltere MNs harder), but by 1.6x, not 3.8x (leg afferents 23 -> 88 Hz; AN04B003 3.7 -> 23 Hz, 6.3x). The tonic VNC inhibitors of DNa02 fall (IN12B014/R -86 ->
-65, IN19A003/L -36 -> -20, GNG562/L -62 -> -45), as they did under B, and new LAL inhibitors appear (LAL046, LAL126,
LAL120_a: the LAL loop answering DNa02's own firing). **Why left:** the excitatory rows are equal (AN04B003/L +360.4,
/R +360.0) and the inhibitory rows are not -- DNa02_R carries IN12B014/L -98 against DNa02_L's IN12B014/R -65, LT51/R
-67 against -45, IN19A003/R -54 against -20 -- so DNa02_L's net is +110 mV/s and DNa02_R's -155, and that is what the
fly's +1.2 deg/s left drift is. **The mechanism is the connectome's sidedness expressed partly through the inhibitors'
own RATES and partly through sub-cap weight differences, with the excitatory rows equalised by `conn_cap`** -- not, as
an earlier version of this paragraph implied, through the inhibitory WIRING alone (skeptic pass):
* the single largest asymmetric row, IN12B014, has IDENTICAL shaped weights on the two sides -- the raw connectome
  carries exactly one -112-synapse edge onto DNa02_L (from IN12B014/R) and exactly one -112-synapse edge onto DNa02_R
  (from IN12B014/L), both clipped by `conn_cap` 60 to -60. Its -98 (on DNa02_R) against -65 (on DNa02_L) is a
  presynaptic RATE asymmetry: IN12B014_L 11.09 Hz vs IN12B014_R 8.38 Hz under C (15.15 vs 12.69 under A).
* genuine wiring asymmetry exists only on the smaller rows: IN19A003 54 vs 70 capped synapse-equivalents, LT51 155 vs
  165 (x the `visual_projection -> DN` path gain 2, i.e. the 310 vs 330 in the shaped matrix).
* conversely the "symmetric" excitation is manufactured by the cap: the raw counts are 165 + 130 + 130 = 425 onto
  DNa02_L against 113 + 84 + 142 = 339 onto DNa02_R (and PS059 476 vs 522), all clipped to the same 180 / 120.
The same rows are asymmetric under A (-126 vs -91, -68 vs -58, -60 vs -43), so the conclusion -- DNa02_L net positive,
DNa02_R net negative, hence the left drift -- is unchanged.

`paths_summary.csv` (arms A, B) and `trace_C_vs_A.json` / `trace_D_vs_A.json` / `trace_E_vs_A.json` were also produced
by `analyse` and are unchanged in kind from round 2: the afferent -> DNa02 walks are the same three (SNpp45 direct
+2.0 mV/volley; SApp -> PS059 -318; SNpp45 -> IN13B001 -> AN04B003 -3.0e4), all live under B-E; the trace's carriers at
depth 1-6 (384 / 1319 / 2050 / 1403 / 493 cells with |z| > 3 vs A under C) are the VNC and the AVLP / LAL / SMP that
the ascending neurons reach.

## 5. Validation (CPU, `CUDA_VISIBLE_DEVICES=-1`; torch 2.10.0+cu128, numpy 2.3.5, python 3.13.2 on this desktop)

`python -m pytest tests/test_body_cycle.py tests/test_proprioception.py`: **23 passed** (10 + 13);
`tests/test_bit_identity.py`: **3 passed, 5 subtests** with the whole working tree (this thread's edits and the other
round-3 threads' `LIFParams.w_syn_by_nt = None`). Nothing was regenerated.

(a) **Bit identity of the shipped path.** `test_shipped_path_is_bit_identical_with_the_cycle_attached_and_the_sense_off`:
two `BatchSim`s, same seed, one with `body.leg_cycle = LegCycle()` attached, 30 frames -- every brain tensor equal at
`atol 0` and every FlyState field (`x y z speed yaw_rate _heading fwd`) equal, while the second sim's `leg_phase` has
moved and the first's is the static default (1.0 stance fraction). `ShippedPathTests` of `test_proprioception.py`
(round 2): the sense is inert unless attached and fed. `test_bit_identity.py`'s recorded golden (which hashes
`asdict(fb.motor())` field by field) passes unchanged -- the reason the side split is a function beside `MotorRates`
(section 1.2).

(b) **The timing laws reproduce the cited curves.** `TimingLawTests`: `tau_st = 0.9328 s x (v / 1 mm/s)^-1.025` to
1e-12 at 5 / 10 / 20 / 28 / 30 mm/s (DeAngelis et al. 2019 Fig 1E); swing constant (`1/f - tau_st = 0.030` exactly);
f rising and beta falling monotonically with speed; **16.5 Hz at 28 mm/s** (Mendes et al. 2013: ~16 cycles / s at
their fast speeds; asserted within 1 Hz); 179 ms stance / 30 ms swing at 5 mm/s (beta 0.857, asserted to 1e-9); beta
floored at 1/2. Standing below 0.5 mm/s: f 0, beta 1, tau infinite, no advance, six legs down, loads 0.5 / 0.5.
Airborne: phase held, no stance, no load, amplitude 0.

(c) **Per-leg phases 180 deg apart across the tripod.** `TripodTests`: 400 frames at 8 / 15 / 28 mm/s, at every frame
`(phase[R1 L2 R3] - phase[L1 R2 L3]) mod 1 == 0.5` to 1e-9, the three legs of a tripod share one phase, at least three
legs are down, the per-side loads sum to 1 and alternate (mean 0.50 +- 0.02, max |L-R| > 0.3 within the cycle).

(d) **The sided channels differ between sides under an imposed yaw.** `TurnAsymmetryTests`: at yaw +2 rad/s (a left
turn) every right leg's amplitude exceeds every left leg's by exactly `2 |yaw| half_width tau_st / step_ref`, the
opposite at -2, equal at 0, the mean amplitude `v tau_st / step_ref` in all three cases; and through the transducer on
the synthetic graph the left and right chordotonal / hair-plate cells of one segment read different rates (L2 < R2 in
the left turn, the unsided T3 cell the mean of L3 / R3), the opposite turn swaps them exactly, yaw 0 makes them equal.
`test_per_leg_per_phase_laws_under_leg_cycle` pins the laws cell by cell (a swing burst at 10 + 140 x 0.5 for a leg at
amplitude 0.5; a stance leg at mid-protraction at the tonic 10 Hz; campaniform 50 Hz per leg with three down, 25 Hz
with six down, 0 airborne; the haltere L cell at `haltere_L`, the R cells at `haltere_R`; every ceiling holds at 1e6;
the `take_body` hold is consumed once and the token without its state raises).

(e) **The batched and scalar cycles agree** to 1e-7 over 50 frames at 12 mm/s, 0.7 rad/s (`test_batched_cycle_matches_the_scalar_cycle`);
`BatchSim` under `all+leg_cycle+haltere_sided` injects exactly `sense.rates(**proprio_state)` x dt into
`brain.poisson_p` for the chordotonal and campaniform cells, the L1 / R1 campaniform cells differ on some frames and
agree on others (the tripod), and the leg state is checkpointed with the FlyState -- a resumed sim reproduces
`poisson_p` at `atol 0` (`test_batch_sim_feeds_per_leg_rates_and_checkpoints_the_leg_state`).

(f) **The side-split haltere readout** equals `MotorRates.haltere` on the synthetic graph's one sided haltere MN and 0
on the empty side (`test_side_split_haltere_readout`); on the shipped cache `haltere_side_groups` returns 8 L / 8 R /
0 unsided (instances `MNhm03 / MNhm42 / MNhm43 / hDVM MN / hi1 MN / hi2 MN / hiii2 MN` x `_L / _R`; recorded in every run
JSON as `haltere_mn_sides`).

Not validated here: the cycle against a walking-kinematics ledger row (none exists; section 8 proposes them), and the
GPU path's bit identity (never claimed: the room runs are H200 rollouts, run-to-run non-reproducible at a fixed seed).

## 6. The compass (`compass_<arm>_r<seed>_{rest,ccw,rest2,cw}.npz` + `_rec.npz` + `_run.json`; arms A / D / E x seeds 0 1 2; `analysis/compass_table.csv`, `compass_chain_<arm>.csv`, `compass_flip_<arm>.csv`, `compass_flip_chain.csv`)

The round-2 efferent protocol exactly: the scalar `FlySim` pinned at `rot.POS`, heading integrated by the body, gains
gE 2 / gD 15, a 10 s phase each of rest, ccw (DNa02_L at 20 Hz), rest2, cw (DNa02_R at 20 Hz), 3 s skipped; under D
and E the scalar body carries `Locomotion.cycle = LegCycle()` and the sense reads `Locomotion.proprio_state` with
`read_haltere_sides` every frame. All nine runs `device cuda NVIDIA H200`, blocks `fam_c<seed>` (each seed's three
arms sequential in one job on one box). The pinned fly walks against the pin at a realised speed that spans
**8.73-10.25 mm/s** across phases, so the cycle runs at `cyc_step_hz` 8.13-8.63 and `cyc_stance_frac` 0.741-0.756
(`compass_table.csv`); "8.4 Hz / beta 0.75" is the central value, not a tight range (skeptic pass: the earlier
"walks at 9.9-10.1 mm/s" was narrower than the data).

| phase (3 seeds) | A: drift w/s, heading deg/s | D: drift, heading | E: drift, heading | D: chord L-R / haltere cmd L-R / haltere MN L-R (Hz) | E: the same |
|---|---|---|---|---|---|
| rest | -0.0037 +- 0.0031, +1.5 | -0.0042 +- 0.0040, +3.0 | -0.0038 +- 0.0033, +3.0 | -1.08 / -1.56 / -1.56 | -0.78 / -1.59 / -1.47 |
| ccw (DNa02_L 20 Hz) | +0.0010 +- 0.0027, **+100.4** | +0.0015 +- 0.0034, **+98.8** | +0.0029 +- 0.0026, +99.1 | **-23.75 / -4.39 / -4.39** | -23.40 / **-14.06** / -6.19 |
| rest2 | +0.0030 +- 0.0028, +1.0 | +0.0021 +- 0.0026, +2.7 | +0.0030 +- 0.0009, +2.2 | -0.94 / -1.26 / -1.26 | -0.52 / -1.36 / -1.27 |
| cw (DNa02_R 20 Hz) | +0.0028 +- 0.0013, **-92.5** | +0.0041 +- 0.0009, **-87.2** | +0.0032 +- 0.0024, -86.5 | **+21.72 / +0.96 / +0.95** | +21.23 / +0.39 / +0.04 |
| ccw vs rests / cw vs rests | +0.0013 / +0.0031 underpowered | +0.0026 / +0.0052 underpowered | +0.0033 / +0.0036 underpowered | | |

Bump vector strength **0.754-0.771** and peak **248-255 Hz** across all 36 phase x seed x arm entries (the pinned bump
of `cx_shift.md`, unchanged; the earlier "0.760-0.765 / 248-253 Hz" was narrower than the data -- skeptic pass); the
ideal drift at the realised +-90 deg/s is +-4.0 w/s and the largest per-phase mean is 0.005. The
body's sided state is unambiguous in the turn: the chordotonal command L-R is -23.8 Hz in the left turn and +21.7 in
the right (per leg: L 78.7 / R 102.5 ccw, L 101.8 / R 80.1 cw -- the outer legs' longer steps), the haltere MN readout
L-R -4.39 / +0.95 (per seed ccw -4.46 / -4.39 / -4.32; cw +1.09 / +1.05 / +0.72; rests -1.6 to -1.0), the haltere
afferent command follows it (D) and is multiplied by 1 + |yaw| under E (-14.1 ccw: the stop-gap's 2.6x).

Flip table (`compass_flip_chain.csv`: L-R at ccw minus L-R at cw, per seed, against the rest2 - rest null; every row
`underpowered` at 3 v 3 by the rule, so the per-seed lists carry the evidence):

| type (depth) | A flip | D flip (seeds) | D null | E flip (seeds) | E null |
|---|---|---|---|---|---|
| AN04B003 (1) | -0.05 +- 0.05 | **-9.81 +- 1.60** (-9.86 / -11.38 / -8.19) | +0.40 +- 0.53 | **-10.70 +- 0.18** | +0.43 +- 0.78 |
| AN06A026 (1) | +0.33 | -2.21 +- 0.25 (-2.07 / -2.50 / -2.07) | -0.21 +- 0.47 | -4.81 +- 0.60 | -0.17 +- 0.30 |
| AN07B035 (1) | +0.60 | -1.88 +- 0.37 | +0.29 +- 0.33 | -1.64 +- 0.72 | +0.40 +- 0.68 |
| PS059 (1; DNa02-driven) | +4.07 | +4.90 +- 0.89 | +0.05 +- 0.86 | +1.55 +- 1.22 | +0.12 +- 0.79 |
| IN19A003 (DNa02-driven) | +6.67 | +5.70 +- 0.91 | +0.51 +- 0.43 | +5.02 +- 0.76 | +1.06 +- 0.89 |
| PS047_b (2) | -0.62 +- 1.08 | -1.71 +- 0.38 (-1.43 / -2.14 / -1.57) | +0.62 +- 1.45 | -3.33 +- 0.33 | +0.33 +- 0.59 |
| PS196_b (2) | +0.05 +- 0.22 | **-1.14 +- 1.31** (0.00 / -2.57 / -0.86) | +0.48 +- 1.15 | **-2.71 +- 1.31** (-3.86 / -1.29 / -3.00) | -0.33 +- 1.30 |
| GLNO (3) | -0.02 +- 0.46 | **+0.52 +- 0.32** (+0.21 / +0.50 / +0.86) | -0.10 +- 0.48 | +0.48 +- 0.15 | -0.10 +- 0.33 |
| LAL184 (3) | +0.10 | +0.71 +- 0.57 | 0.00 +- 0.38 | +0.52 +- 0.36 | +0.14 +- 0.38 |
| PEN_a / PEN_b / EPG (4) | -0.17 / -0.18 / +0.63 | -0.10 / -0.19 / +0.58 | +0.08 / -0.14 / -0.22 | 0.00 / -0.21 / +0.63 | +0.22 / -0.21 / -0.11 |
| DNa02 (the stimulus) | +37.3 | +36.7 | | +36.5 | |

PS196_b's rates by phase (D; `compass_chain_D.csv`): L 7.57 / 6.52 / 7.67 / 6.90 Hz, R 4.14 / 4.19 / 3.76 / 3.43 (rest /
ccw / rest2 / cw). The sense makes PS196_b fire in the pinned compass fly (A: 0.0-0.1 Hz) at 4-8 Hz, sided (L > R by
3.4-3.9 Hz at rest -- a fixed asymmetry, as in the room), and its L-R moves by -1.1 Hz in the left turn and -0.4 in the
right under D: no flip (the null's SD is 1.2). Under E PS196_b is driven HARD and unsigned: **L 20.9 / R 16.3 Hz in the
left turn, 20.8 / 13.5 in the right, against 7.6 / 3.7 at rest** (PS047_b 38.2 / 34.9 and 37.0 / 30.3 against 20.9 /
16.7; GLNO +1.7 Hz on BOTH sides in BOTH turns) -- the same-direction excursion of round 2 (then 11.05 / 7.52 Hz in
the ccw turn), larger now because the pinned fly walks and the haltere MNs run at 15-18 Hz; its L-R moves +0.7 (ccw)
and +3.7 (cw) against the rests, the same way in both directions.
GLNO holds +27 Hz L-R (L 146.9 / R 119.7 Hz at rest under D) at every phase. So the compass question of round 2 has
the same answer with the body sided: **the ring receives no signed report of the self-turn**. The signed report
exists -- AN04B003's L-R follows the sided leg afferents in every seed, AN06A026's the sided haltere -- and it is a
quarter of the ascending neurons' rate at depth 1, 1-3 Hz on PS047_b / PS196_b at depth 2, 0.5 Hz on GLNO at depth 3.
The routes are the ones `paths` named (SApp -> CB0675 / GNG580 -> PS196_b; AN04B003 -> PS047_b -> PS196_b), all
ipsilateral by synapse count (CB0675_L 158 haltere-L synapses / 0 R; GNG580_L 351 / 0; PS059_L 241 / 0), and the
sidedness is diluted by the same fan-in at each step.

## 7. What the two mechanisms did, in one place

* The leg cycle made the never-driven leg afferents fire at literature-typical rates (88 / 47 / 25 Hz) with a
  tripod-locked sidedness (12.5 Hz per-frame |L-R|) and a small kinematic turn asymmetry (-0.3 Hz DC, corr with yaw
  -0.15). That drive fires DNa02 (0.5 / 0.4 Hz, 6.6 % of frames), makes the fly meander (yaw SD 7.9 deg/s, straightness
  0.83), hop (0.46 / fly) and walk off the table (7 / 16), and drifts it left at 1.2 deg/s by the connectome's own
  asymmetry. The sided, body-locked term reaches DNa02 (a -0.35 Hz swing per tripod half-cycle) and does not steer:
  it alternates at 8 Hz under an 80 ms motor filter, and its DC part is 2.5 % of it.
* The side-split haltere readout carried the haltere MN's sidedness (-2.7 Hz DC in the room, -4.4 / +1.0 in the
  imposed turns) into PS059, PS196_b and DNa02 as a per-frame signal, changed no behaviour and no DNa02 rate (D vs C
  `null` on every row), and did not make PS196_b's or GLNO's L-R flip with the turn.
* The Coriolis control (E) raised the haltere afferents (16.7 -> 18.6 Hz in the room; x2.6 in the imposed turn),
  changed nothing in the room (E vs D `null` on every behavioural and DNa02 row) and reproduced round 2's same-
  direction PS196_b excursion.
* The level confound (section 4.2) is the finding that qualifies all of it: this round changed the leg afferents'
  level and their phase structure together, and only the level is needed to account for the DNa02 firing.

Not done, and why: no bench jobs (the bench protocol has no body); no per-leg swing-duration modulation in turns and
no load asymmetry (no measurement to take them from, section 1.1); no house-cluster follow-up (the task is one
submission; the level-matched control is the first thing a follow-up should run, section 8).

## 8. What a default adoption would require (nothing is adopted)

1. **The level-matched control arm**: the round-2 transducer (`'all'`) with `mn_ref_hz` set so its window-mean
   chordotonal rate equals the cycle's 88 Hz (about 3.5 Hz instead of 30), 5 seeds, same block as A / B / C -- a
   LABELLED control (a number chosen to match a level, never a default). If it fires DNa02 as C does, the cycle's
   contribution is its level and the per-leg / per-phase structure is unproven; if it does not, the phase structure
   carries something. Without it no claim about the cycle beyond "the afferents at 88 Hz drive DNa02" is supportable.
2. **The ledger rows** (proposed, not added -- the CSV is another thread's file this round):
   `lit.walk.step_frequency_hz_at_speed` (7.1 at 8 mm/s, 16 at 28-30; Mendes et al. 2013 Fig 2, DeAngelis et al. 2019
   Fig 1E; `op report`), `lit.walk.stance_fraction_at_speed` (0.79 at 8, 0.51 at 28; the same sources),
   `lit.walk.swing_duration_ms` (30-50, UNCERTAIN, Mendes 2013 Fig 2B), `lit.walk.outer_leg_step_ratio_in_turn`
   (DeAngelis et al. 2019 Fig 6C, direction only until a number is taken), and a walking-mean afferent rate for the
   FeCO if one is published (none was found; the bracket 10-150 Hz is all the ledger has, and it is what lets the
   level move 4x without leaving the bracket).
3. **The suite with the cycle on**: `scripts/benchmark.py` / the room ledger rows under `all+leg_cycle+haltere_sided`
   (5 runs, the shipped path as the reference, one block), because the C-E fly leaves the table and hops -- the
   take-off / GF rows (`gf_max_hz` +4.6 Hz, p 0.016; C run SD 5.0, B run SD 0.50, so z +9.05 -- the z follows from the
   NULL arm's SD, not from the C arm's 5.0) and every room row that assumes a straight walk would
   move; the object / loom / feeding rows have not been run with the cycle at all.
4. **`MotorRates.haltere_L / haltere_R`** as fields: a one-line change in `read_motor` plus `haltere_side_groups`
   cached on `WingGroups`, and the regeneration of `tests/test_bit_identity.py`'s golden -- an owner decision (the
   golden hashes `asdict(fb.motor())`).
5. **The compass under the cycle at the fly's own speed**: the pinned fly walks at 10 mm/s against the pin, so the
   cycle ran; a free-walking compass room (`scripts/cx_shift.py` room mode) under D with the bump metrics would
   answer whether the +-4 w/s drift ever appears with sided afferents -- section 6 says the report is gone by GLNO, so
   the expectation is no.
6. **A house-cluster replication** of any row quoted here is a replication across devices (H200 -> B200), not a
   bit-for-bit check, and should be labelled so.
7. **The half_width_m constant** (1.0 mm, not measured) scales the turn asymmetry linearly; a stance-width
   measurement (or the tarsus positions of Chun et al. 2021 / DeAngelis et al. 2019 if extractable) would replace it.
   The swing duration (30 ms, bracket 30-50) shifts beta by 0.05 at 9 mm/s.

## 9. Files, provenance and reproduction

* Code (this thread): `flyverse/body.py` (`TRIPOD_OFFSET`, `FlyState.leg_phase / leg_stance / leg_amp / stance_frac /
  stance_load_L / _R`, `LegCycle`, `Locomotion.cycle` (default `None`), `Locomotion.proprio_state`),
  `flyverse/batch_body.py` (`BatchBody.leg_cycle` (default `None`), `_cycle`, `proprio_state(motor, haltere_sides=None)`),
  `flyverse/motor.py` (`haltere_side_groups`, `read_haltere_sides`), `flyverse/senses.py` (`Proprioception.FLAGS`,
  `parse_flags`, tokens `leg_cycle` / `haltere_sided`, `SEGMENT_OF_NERVE`, `leg_weights`, `leg_of`, `haltere_sides`,
  `take_body`, `rates(legs=, haltere_L=, haltere_R=)`), `flyverse/batch_sim.py` (the one per-frame call),
  `scripts/probe_vnc_drive.py` (`--family body`; `plan --only-seeds / --only-compass-seeds / --targets / --script /
  --name`; `analyse` with `robust_room`, the per-side / per-leg groups, the cycle keys, `--only-seeds`; `pairs`),
  `scripts/probe_walk_straightness.py` (`--leg-cycle`, the tokens), `tests/test_body_cycle.py`,
  `tests/test_proprioception.py` (+2). Defaults: every new field OFF (`cycle = None`, `leg_cycle = None`, the tokens
  unnamed); `'all'` is round 2's transducer to the bit.
* Data: `out/vncd3/` (25 room x 5 files, 9 compass x 17 files, `batch.sh`, `batch_b.sh`), consoles `out/vncd3_cluster.log`
  (`vncd3-f3bb50`: 28 jobs, 22 failed = the 22 cancellations, 6 completed on `r3-h200d`), `out/vncd3b_cluster.log`
  (`vncd3b-c2eeaf`: **22 job(s), 0 failed**, 29.7 min, 11 on `r3-h200d` + 11 on `r3-h200c`), the cancelled first
  submission's `out/vncd3_cluster_cancelled_d494ec.log` (no artefact). Every JSON carries `provenance` (resolved
  `LIFParams` with `w_syn_by_nt: null`, `OpticParams`, `type_path_gain`, the compiled-W md5, the 44-file
  `source_fingerprint` (`provenance.source_fingerprint.n_files == 44` in every room and compass run JSON),
  `execution.device / device_name`, seeds, the block key).
* Analysis: `out/vncd3/analysis/` (`analysis_console.txt`, `room_table.csv/txt`, `summary.json`, `decompose_*`,
  `paths_*`, `trace_*`, `compass_*`, and from `pairs`: `pairs_console.txt`, `pairwise.csv`, `sided_frames.csv`,
  `dna02_decompose_summary.csv`, `compass_flip_chain.csv`, `pairs_summary.json`); `out/vncd3/analysis_4runs/` the
  same on seeds 0 1 3 4.
* Reproduce: `PYTHONIOENCODING=utf-8 python scripts/probe_vnc_drive.py analyse --dir out/vncd3 --out out/vncd3/analysis
  --trace-arms C,D,E` then `... pairs --dir out/vncd3 --out out/vncd3/analysis` (CPU, ~6 min + 6 s); the batch:
  `... plan --family body --dir out/vncd3 --runs 5 --compass-seeds 3 --draws 0 --minutes 90` writes `batch.sh`
  (28 jobs; on the house cluster it needs no `--targets`).
* Session note: the agent that built the mechanisms, submitted the batch, fetched it and ran `analyse` died at a
  session limit after writing sections 1-3; the rented boxes were then stopped for ~5 h and restarted (no job of this
  batch was affected: `vncd3b` had completed and been fetched before the stop). This agent verified the artefact counts
  and devices, ran the tests, added and ran `pairs`, and wrote sections 0 and 4-9.
