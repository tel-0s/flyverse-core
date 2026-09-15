# Compass stand-in: imposed angular memory

Work started 2026-09-15 at the owner's request after round 7. This is a program-controlled EPG
input, explicitly under `instrumented`, while `raw` keeps the plain brain. The owner permits this
route and permits a future room-demo default; this branch keeps the default raw pending validation.

## Source reading and interpretation

Wang's repository was read at commit `80b94e6608cf927ca2c9f2bbce0577c5a998684c`, including the
requested [findings index](https://pwang724.github.io/fly-circuit-exploration/findings/index.html),
[compass recurrence](https://github.com/pwang724/fly-circuit-exploration/blob/80b94e6608cf927ca2c9f2bbce0577c5a998684c/docs/compass-recurrence.md),
the September 13 audit and the corrected finding-4 page. Its minimal simulator was inspected as
source, not imported or copied. The interpretation used here is deliberately narrower than the
site's headline: a naive rate model can stall when write-position feedback is added at count-based
strength; its updated page also notes that fitted models integrate with these contacts present.
Anatomical counts do not identify the effective gains or prove which biological correction is needed.
The released flat MaleCNS graph cannot separate these contacts by neuropil.

The interleaved EPG wedge order is L1,R8,L2,R7,...,L8,R1, parsed from the released instances and
checked in the previous local compass audits. Only the 46 EPG cells are targeted. Missing labels
or incomplete rings fail explicitly; this is not silently transplanted to another dataset.
[Turner-Evans et al. 2017](https://elifesciences.org/articles/23496) supplies the functional motivation
of maintaining heading and integrating angular velocity. The exact law and constants below are
**unverified engineering choices**, not extracted physiological parameters or an implementation of
Wang's fitted circuit. No claim about ExR6/Delta7 receptors, GLNO's sign, or PEN gain follows from it.

## Implementation contract

`CompassDriver` is an ordinary attached module and named instrument. It holds one phase and one
angular-velocity input per batch row, integrating `phase += yaw_rate * dt` modulo 2*pi. It supplies
`50*exp(-0.5*(wrapped_angle_error/radians(35))**2)` Poisson Hz to each EPG. Initial phase is arbitrary;
positive yaw advances the declared wedge coordinate. The body supplies realized yaw velocity through
the existing `proprioception` entry point. There is no world-heading, fruit-position or goal oracle.
No neuron voltage, rate, receptor, weight or body command is overwritten. Parent neurons still emit
spikes, communicate through the unchanged graph and determine the motor readout.

The continuous drive supplies BOTH angular memory and its neural representation. It is not a small
repair of a discovered natural mechanism, and it is not an attractor in the connectome. It has no
visual cue anchoring, tilt compensation, destination memory or steering policy. Phase follows the
held velocity input until it is updated; reset clears velocity and phase. Checkpoints include both,
partial batch reset affects only selected rows, and detach removes the instrument and its inputs.

Work is O(B*n_EPG) per frame: one batched tensor profile, no neural readback and no per-cell Python
loop. The existing extension scheduler and core CUDA graphs are retained. Their actual overhead is
measured separately; a small formula alone does not establish low integration overhead.

CPU development checks before the house declaration: 11 focused/golden tests and 10 subtests passed;
49 existing integration tests and 200 subtests passed, 4 skipped. A 1 s full-brain CPU smoke gave a
clear stationary bump. A seed-0, B=1, 12 s turn pilot at -180 deg/s gave mean vector strength 0.815,
stationary error p95 5.72 deg and moving error p95 24.41 deg. The Poisson law was not changed after
these observations. The legacy motion/loom benchmark smoke passed all four checks with the real
instrument scheduler active. These are development observations; house contract seeds 10-15 are fresh.

## House declaration (before submission)

`scripts/compass_driver_batch.py` freezes commands, parameters, source hashes and the criteria below.
The batch is engineering validation, not a hypothesis test about the animal. No gain fitting, model
default adoption or mechanistic receptor inference is licensed by a pass.

1. Functional contract: raw and instrumented, seeds 10-15, each B=8, full brain, default LIF/receptors,
   no optic input or edge holds, 12 s. Angular speeds -180,-90,-45,0,45,90,180,90 deg/s, held at zero
   for [0,2), forward on [2,5), zero [5,7), reversed [7,10), zero [10,12). Last row starts at 90 deg;
   others at zero. Readout averages cells within each wedge first, then computes the circular moment.
   Ignore the first 0.5 s; omit the first 0.5 s after turn cessation for stationary error. Each driven
   run/row must have mean vector strength >=0.7, fraction below 0.6 <=0.05, stationary error p95 <=15 deg,
   moving error p95 <=max(22.5, 0.1*abs(speed)+15) deg, and each sustained-turn slope within
   max(5, 0.15*abs(speed)) deg/s of the imposed speed (opposite on reversal). The error allowance
   explicitly includes the existing 100 ms rate-estimator lag. No eligibility-based dropping of rows.
   Bump rate/width are descriptive; these criteria are not the prior physiological compass ledger.
2. Legacy benchmark: three draws per preset, seeds 0,1,2, full `all` sections with matched seed
   overrides. Every legacy direct-Brain probe uses `InstrumentedBenchmarkBrain` to run the actual
   FlyBrain extension scheduler; room probes receive the same preset/instrument. Controller provenance
   is saved. No previously passing benchmark row may worsen for admission; otherwise keep the driver
   experimental and name the failures. The earlier handoff's `benchmark_suite.py` filename does not
   exist; the real script is `scripts/benchmark.py`.
3. Room observation: two matched B=6 batches, environment seeds 10-15, neural seed 10, fenced apple
   room, 60 s, no body program. Record movement, yaw, hops, EPG activity and frame timing. This short
   engineering run cannot establish food finding or replace the 300 s adoption rate-half. No default
   change is made on its strength.
4. Performance: B=1,8,32, four alternating repeats of 100 frames per preset after 50 warmup frames.
   Both receive the same tonic EPG forcing, so differences in Poisson activation do not confound
   scheduler overhead. Report synchronized wall and CUDA event time, plus module-only time. Shared
   hardware timings are observations, not idle-device guarantees. Target: <=10% median frame overhead;
   if missed, optimize the scheduler or clearly record the limitation before calling the route fast.
5. UI smoke: headless room with `--preset instrumented --instrument compass --brain-map`, save a frame.
   The console must name the preset/instrument and the existing map must display actual neural rates.

## Results

The immutable declaration and all generated records stay under `out/compass_standin/`.

### Scheduler correction declaration (before the second submission)

The first paired profile missed the <=10% target: median synchronized frame overhead was
91.80%, 85.44%, 16.76% at B=1,8,32; the formula alone cost 114.76,119.48,132.87 CUDA microseconds.
The shared-device timings are noisy, but they do not support calling the first scheduler integration fast.
Two explicit host waits sit in the module path: finite-output checking and reduction of Poisson activity.
The body's yaw upload also used a blocking copy. No gain or dynamical law is changed in the correction.

`out/compass_standin_r2/` freezes the following correction tests before running them:

- The compass opts into an asynchronous CUDA invariant assertion. Its ordinary user inputs (shape,
  finite yaw and representable gain/width, 0-10 ms frame, checkpoint shape/range) are still validated.
  A nonfinite internal CUDA output is a bug and may abort the CUDA context on a later launch; other
  modules keep synchronous error checking. The known positive Gaussian lower bound, including dt,
  avoids reading back whether Poisson forcing is active. The proof is used only after a full output,
  cleared on any reset/load/detach, and unavailable for a narrow profile whose bound underflows.
  Host yaw copies are nonblocking; fields retain clone ownership, but stop allocating an unused zero tensor.
- Rerun all six instrumented contract seeds with unchanged parameters. Require **every saved array
  to equal the first submission exactly**, then apply its original engineering gates without changes.
- Repeat the 60 s, B=6 instrumented room with the original backend and require identical saved body
  and EPG trajectories and hop counts. CPU tests also cover matched raw/compass forcing and all-row
  partial resets. A house EPG-subgraph fixture compares the asynchronous and original checked paths
  through pulse expiry, partial reset, checkpoint replay and detach, every brain tensor exactly.
- Repeat the original paired profile, and add a paired native-CUDA profile (CUDA kernels, event driven,
  graphs, warp sparse at B=1 and torch CSR in batches). Four alternating repeats, B=1,8,32, same <=10%
  median overhead target. Matched-input final brain tensors must agree exactly. Native room arms at
  B=6, 60 s add actual body/sensory timing; they remain observations, not the 300 s admission rate-half.

The first submission stays intact. Its full benchmark pairs are still running at this declaration;
no suite outcome or compass gain has been used to select the scheduler correction.

### Capture declaration (before the third submission)

The second submission completed 11/11 jobs. All six instrumented turn traces reproduce every first-batch
NPZ array exactly, and the nine small-graph CUDA lifecycle checks pass. Its **room replay is not exact**.
The matched-input profile also fails exact equality on voltage/conductance, and on additional spike-state
tensors at native B=32. These failed gates are retained, not rounded away. The room repeat control below
will test whether this is specific to the scheduler change; no numerical tolerance is chosen from them.

Native median frame timings (raw/instrumented) are 0.5776/0.9371, 3.0920/3.4978,
12.7838/13.1810 ms at B=1,8,32. The native B=1 and B=8 <=10% targets still fail. The plain Torch
profile is visibly affected by changing contention (B=8 raw repeat times 185.6,185.6,173.7,23.3 ms).
The remaining small-batch cost motivates capturing the explicit, read-free compass module together with
the existing neural frame. This does not fuse or change its arithmetic or gains. Timed pulses, hooks and
general modules retain the eager scheduler; cache replay must restore the correct module input buffers.

`out/compass_standin_r3/` freezes **one sequential house job**, after this experiment's other jobs end:

- Extended CUDA lifecycle fixture: compare captured and checked paths, including returning from a timed
  pulse, new external drive, different frame lengths, module-input snapshots, partial reset and checkpoint.
  Also run the existing opt-in CUDA test file. Any error stops the job before performance claims.
- Native paired profiles with and without module capture, same B=1,8,32 and four alternating repeats;
  target remains <=10%. Use an external held 50 Hz setter in both arms instead of an unexpired 100 s
  pulse, because a live timed pulse deliberately disables module capture. The represented forcing is
  identical throughout the 4.5 s profile. Record per-tensor maximum differences as well as exact equality.
- CPU/CUDA traces for raw, synchronous-check control, eager-module and captured-module frames. Verify
  actual module capture and count scalar readbacks; all modes receive the same tonic input. The check
  control also uses the original blocking yaw upload. This identifies work independently of wall time.
- Repeat all six fixed turn seeds under capture and report every original functional gate and array
  equality result. No parameter is fitted and no failed row is dropped.
- Two **same-code eager** 60 s B=6 instrumented rooms, then a captured room, same neural/environment
  seeds and no program. Report exactness, maximum differences and first divergence for the repeat and
  capture comparisons. Repeat native raw/instrumented rooms for actual integrated timing. These are
  reproducibility/timing controls, not additional draws for an adoption or food-finding claim.

The initial suite has 87 paired check rows with no passing-row regression, but a taste row changes
FAIL to PASS. That passes the initial declaration's narrower screen but **fails PRESETS_SPEC's stricter
no-status-change-outside-the-gap rule**. This is an author self-review correction of the acceptance
interpretation, not a retroactive change to data or thresholds. The driver remains experimental and
the raw default remains unchanged regardless of the timing outcome.
