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
| 3 | landing response | gentle expansion -> leg extension, no takeoff | fail (so far) | Leg MNs rise +3.5-4.8 Hz at every rate with no rate dependence; a stop-short approach (small final angular size) is the discriminating condition and is being measured. |
| 4 | cast-and-surge in an intermittent plume | surge on contact, cast on loss, alternation | program | plume puffs at 1.5 s; the LH apple / berry channels follow the puffs (model level); surge / cast is `AnemotaxisProgram` / `cx.CompassSteering`. The connectome's steering integration (CX) is silent -- see the compass entries. |
| 5 | grooming cascade | anterior -> posterior priority under whole-body dust | untested (new thread) | bristle sensory neurons by segment are in the connectome (class mechanosensory); leg-MN readouts exist; the assay is a stimulation screen (stimulate bristle sets, read leg-MN patterns and their suppression). |
| 6 | local search after food loss | re-looping around the last food site | program | `AnemotaxisProgram`'s search; the plain model has no path-integration-like behaviour (CX silent). |
| 7 | hunger-modulated thresholds | starvation sharpens sugar sensitivity, raises plume vigour, crosses a bitter barrier | gap | metabolism exists (`body.Metabolism`), sugar / bitter PER replicates (Shiu rules 124 -> 2 Hz; calibrated 4.6 -> 0). But dopamine / octopamine / serotonin synapses are sign 0 in the model: neuromodulation has no route. `docs/audits/nt_audit.md` quantifies it. |
| + | flight saccades | straight segments, ~90 deg saccades on expansion | untested | flight model has steering MNs -> yaw and roll; no measurement yet. |

## Order of work implied

2b and 3 first (same stimulus family as 2a, readouts exist, model-level); then 7's prerequisite (a
sign for the monoamines, or an explicit neuromodulation model) since it also blocks any
state-dependence; 5 as the next emergent test; 1, 4 and 6 wait on the central-complex thread.
