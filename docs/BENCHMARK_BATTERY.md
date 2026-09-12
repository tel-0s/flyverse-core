# Behavioural benchmark battery

A behaviour-level target for the model (and for approximations of it), from a colleague's proposal,
ordered by cost to try. Each assay names the discriminating variable and how the plain model scores
today. `pass` = the connectome model produces it with the plain body; `partial` = the neural signal is
there but the motor path is not, or one branch of the assay works; `program` = only a behaviour
program / module produces it; `gap` = a documented model deficit; `untested` = cheap to run, not run.
Numbers refer to `docs/NOTES.md`. `scripts/benchmark.py` is the harness; assays become sections there.

| # | assay | discriminating variable | status | evidence / what is missing |
|---|---|---|---|---|
| 1 | optomotor response (rotating drum) | sign of the turn vs drum direction | partial | `screen_rotation.py`: HSN d' 4.2, DNp20 3.1, HSE 2.2 flip with sustained yaw. DNa02 is silent (0.01 Hz), so no path from the horizontal system to the legs in the model; the readout can steer, the connectome cannot yet. |
| 2a | looming escape, short mode (giant fibre) | fast expansion -> GF burst -> takeoff, no wing raise | pass | GF 34-56 Hz on 5 of 6 looms, threshold 33 (re-derived after LPi x4); self-motion no longer fakes it (walking 99th percentile 22.5 Hz). `probe_loom.py`. |
| 2b | looming escape, long mode | slow expansion -> GF-independent, wing raise, directed | partial | Expansion-rate sweep (3 cm ball, l/v 15-250 ms, 3 seeds, pre-loom baseline): the wing power MNs rise more for slower looms (+41 -> +53 Hz) and earlier (0.5 s after arrival at l/v 15 ms; 0.9 s and 2.0 s *before* arrival at 120 / 250 ms) -- the wing-raise signature scales the right way -- but the GF still peaks +27-41 Hz 50-140 ms after arrival at every rate and the body takes the short-mode escape in 13 of 15 runs: no mode switch. The model's GF responds to final size, not expansion rate. |
| 3 | landing response | gentle expansion -> leg extension, no takeoff | fail | Stop-short approaches (the ball halts at 8-15 cm, final size 23-41 deg, l/v 60-250 ms, 2 seeds): GF +7.5 to +15 Hz -- under threshold, no takeoff in any run (the short mode is size-gated in the right direction); wing power +35 to +58 Hz without a jump (a wing-raise-like preparation); leg MNs +2.7 to +5.1 Hz -- no leg extension, so no landing. |
| 4 | cast-and-surge in an intermittent plume | surge on contact, cast on loss, alternation | program (compass structure now found) | plume puffs at 1.5 s; the LH apple / berry channels follow the puffs (model level); surge / cast is `AnemotaxisProgram` / `cx.CompassSteering`. The connectome's steering integration (CX) is silent at the defaults, but the session-9 audit found a working ring attractor under Delta7 -> EPG x15 / recurrence x2 / no compass adaptation (`docs/audits/cx_wedge.md`); running it with senses and through PFL3 is the next step. |
| 5 | grooming cascade | anterior -> posterior priority under whole-body dust | untested (new thread) | bristle sensory neurons by segment are in the connectome (class mechanosensory); leg-MN readouts exist; the assay is a stimulation screen (stimulate bristle sets, read leg-MN patterns and their suppression). |
| 6 | local search after food loss | re-looping around the last food site | program (compass structure now found) | `AnemotaxisProgram`'s search; the plain model has no path-integration-like behaviour (CX silent). |
| 7 | hunger-modulated thresholds | starvation sharpens sugar sensitivity, raises plume vigour, crosses a bitter barrier | gap | metabolism exists (`body.Metabolism`), sugar / bitter PER replicates (Shiu rules 124 -> 2 Hz; calibrated 4.6 -> 0). But dopamine / octopamine / serotonin synapses are sign 0 in the model: neuromodulation has no route. `docs/audits/nt_audit.md` quantifies it. |
| + | spontaneous turning (free walking) | yaw-rate SD / path straightness of the plain fly in the room vs free-walking Drosophila (tens of deg/s; frequent saccadic turns) | gap | `scripts/probe_walk_straightness.py`: median yaw SD 1.6-2.8 deg/s, straightness 0.99-1.00 in every weight configuration since session 9 (before that, turns came from an optomotor readout responding to self-motion -- an artefact, removed). No connectome-level source of turning at rest; the compass -> PFL3 -> DNa02 route is the candidate (dynamics round 1). |
| + | flight saccades | straight segments, ~90 deg saccades on expansion | untested | flight model has steering MNs -> yaw and roll; no measurement yet. |

## Order of work implied

2b and 3 first (same stimulus family as 2a, readouts exist, model-level); then 7's prerequisite (a
sign for the monoamines, or an explicit neuromodulation model) since it also blocks any
state-dependence; 5 as the next emergent test; 1, 4 and 6 wait on the central-complex thread.

Session-9 audit notes: `scripts/benchmark.py` now scores assays 1, 2a, 3 (via 2b's readouts), 7's taste half and the
odour gate in one run (`docs/audits/benchmark_suite.md`); the free-walking loom sections predate the `update_loom`
direction fix (the ball now approaches along the fly's left at any heading) and are to be re-measured over six seeds.
