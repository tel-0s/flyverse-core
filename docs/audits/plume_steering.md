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

## 2. Findings

Diagnostic source `384898a`, one house submission, both heading providers. Every sampled
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

Results pending. Independent skeptic pending Fable. No new physiological claim or adoption.

## 5. Scalar demo integration declaration

After the compass B=6 room passes with six feeding rows, separately check the user's actual
scalar simulator and backend. No controller parameter/code change: `plume_steering_probe.py
--mode scalar` uses `room_demo.Sim`, seed 0, all fruit/default start, initial energy 0.1,
`compass plume hunger flight`, native CUDA events, warp sparse and captured frames, for 60 s.
Gate: >=1 s feeding; record position, heading, yaw, energy, distance and contact. One additional
house submission then performs a 1 s headless UI launch with brain map and screenshot. This
checks scalar integration and rendering availability, not interactive FPS or generalization.
Freeze both commands before submission; all controller laws remain exactly those of section 3.
