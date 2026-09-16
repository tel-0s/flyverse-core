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

Pending the frozen diagnostic; no correction selected yet.
