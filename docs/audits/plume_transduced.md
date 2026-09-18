# A transduced-contrast plume law (Astra, 2026-09-17)

## 0. Answer and status

Implemented as an opt-in computation stand-in: `plume:bilateral=orn`. The bilateral cue
comes from the brain's antennal ORN population rates. No physical concentration receiver
is installed on this variant. The default `plume`, `raw`, sensory transduction, weights
and body constants are unchanged. Nothing adopted. The independent skeptic refuted the
original runtime pairing at 4258468; the correction and regression tests are recorded below.
Review of the correction remains pending Fable.

The CPU characterization is unfavorable for reliable instantaneous lateralization. At
the historical median total concentration and 0.57% physical contrast, four explicit
odor-mixture scenarios give **0.022-0.056 Hz** weighted L-R signal, against **0.224-0.273 Hz**
SD in a 250 ms counting window. Including the existing 100 ms rate filter and the new
250 ms contrast filter improves the approximate signal/noise ratio to **0.16-0.34**,
still below one. This is an independent sensory-input model, not observed full-brain
ORN activity. More gain amplifies noise as well as signal.

Eighteen independent room runs are prepared, **NOT submitted**. Fable must first release
the GPU pool. No GPU work, behavioral result, performance claim or default adoption is
part of this delivery. `docs/audits/determinism_gate.md` was not present at this branch's
base; no bit-reproducibility exception is assumed for the eventual rooms.

## 1. Population choice and side bias

Use `senses.Smell`'s assignments, including its reference-connectome laterality lookup,
not soma side. `somaSide` is NaN for every olfactory cell; the dataset's `instance` suffix does carry a side for 2,226 of them (883 L / 1,343 R) and disagrees with the laterality assignment for 222 cells while 526 labelled cells fall below the 0.2 threshold. Repeating the characterisation on the instance labels gives cascade SNR 0.44 instead of 0.34, so the conclusion does not depend on the choice.
That instance-label sensitivity calculation is from the independent skeptic, quoted below.
Per-glomerulus L/R ratios span 0.08-13, i.e. the imbalance is a reconstruction artefact rather than biology.
MaleCNS has 2,639 olfactory-class cells: **708 left, 1,396 right, 535
unsided**. All 54 transducer labels have both sides (53 named labels plus an empty label
containing two cells per side). Unsided cells receive averaged physical input and are
excluded from the directional cue. No active glomerulus is selected from the room's
physical smell dictionary.

Two unweighted antenna means are unsuitable: unequal glomerulus composition creates a
1.6-3.8 Hz false L-R difference under exactly equal odor input in the four median-total
scenarios, larger than the intended signal. For each glomerulus g with both sides, set:

```
h_g = n_Lg * n_Rg / (n_Lg + n_Rg)
q_g = h_g / sum_g(h_g)
r_L = sum_g(q_g * mean(brain.rate[ORN_Lg]))
r_R = sum_g(q_g * mean(brain.rate[ORN_Rg]))
```

Both antennas receive the same glomerulus mass. The pooled indices are returned in ascending model order so the weights stay paired with their cells after the scheduler's `resolve()`; an end-to-end `FlyBrain` test on a fixture with non-ascending read order confirms zero L-R under symmetric stimulation. This removes the count-composition bias in the independent-input model; full-circuit asymmetries can remain.
**Withdrawn as a claim about the pre-fix attached instrument:** "Both antennas receive the same glomerulus mass. This removes the count-composition bias in the independent-input model; full-circuit asymmetries can remain."
Before the fix, the scheduler sorted only the indices, producing +3.57679 Hz under
symmetric MaleCNS stimulation while the characterization and room logger paired correctly.
The fixed weights
minimize difference variance if independent cells have equal rates. Actual glomeruli
have different rates, so this is an **unverified pooling rule**, not an optimal decoder
for this scene. Effective cell counts (`1/sum(cell_weight^2)`) are about 656 left and
1,127 right. Counts, per-label weights and exact target IDs are recorded in `describe()`.

ORNs provide a large directly stimulated population with the transducer's explicit side
assignment. Switching to PNs would introduce a downstream circuit transfer and correlations;
no evidence here establishes improved signal/noise there. This experiment therefore does
not choose PNs or fit a decoder to scene responses. Static CPU evaluations of the actual
room generator (six environment seeds, fixed default pose, times 0/2/4/6 s) produce expected
weighted sensory forcing rates of **3.9-5.5 Hz per side**, L-R **-0.016 to +0.033 Hz** and
cascade SNR **0.016-0.27**. These are sensory-field evaluations, not neural room runs.

The old room traces log antenna concentration sums but **not** glomerular mixtures or ORN/PN
rates. They cannot establish historical `brain.rate` values. The planned rooms explicitly
record ORN rates; full-brain firing rates, correlations and possible side bias remain open.

## 2. Law and boundary

At each 10 ms module frame, the ordinary extension scheduler supplies `brain.rate` at
the frame boundary. Compute matched rL/rR above, then:

```
d = (rL-rR)/(rL+rR)                   # zero if both rates are zero
b = exp(-dt/0.25)*b + (1-exp(-dt/0.25))*d
goal = heading + atan(200*atanh(clamp(b, -0.999, 0.999)))
```

`2*atanh(d) = log(rL/rR)`, so this is a response to the log *rate* ratio. The old
concentration law had small-contrast slope `2*0.1/0.001 = 200`; this experiment explicitly
retains that dimensionless slope on the new rate contrast. **The gain 200 is unverified and, for a dimensionless rate contrast, underived**: the concentration law's 200 came from `gradient_length_m / antenna_separation_m`, which has no counterpart here; it is retained only so the two arms share a small-contrast slope.
It is not an inversion of the ORN transducer and no distance is inferred from neural rates.
For one homogeneous concentration c, small contrasts obey
`d_rate ~= [c*f'(c)/f(c)]*d_concentration`, where `f(c)=1+150*c/(c+0.5)`;
saturation attenuates contrast. A heterogeneous population does not admit a single c
inverse. Measured on the four scenarios, the rate contrast is 0.37-0.64 of the physical contrast (lime 0.371, apple 0.429, banana 0.447, all-fruit 0.639), so at gain 200 the transduced arm's small-contrast turn response is 0.37-0.64x the shipped arm's for the same physical contrast; `transduced vs full` therefore varies effective gain as well as cue source.
Neither the concentration gain nor this retained rate gain is physiological.

The walking goal is local whenever weighted rates are nonzero. Airborne or zero-rate rows
use the existing wind/entry-memory fallback. Basal spikes can therefore produce a noisy
walking goal even without physical odor; the existing LH/confidence/feeding/hunger gate
still controls output. No hidden physical availability flag suppresses this noise.

Existing EPG heading, LH gate, PFL3 comparator, DNp09 output and DNa02 feedback are retained.
Only the bilateral cue changes in the transduced arm. Wind and internal-state receivers
remain explicit; "rate-only" refers to this bilateral cue. Output remains Poisson drive
on biological PFL3 and DNp09, with no weight, receptor, body or sensory-law edits.

The opt-in constructor is `PlumeNavigation(c, bilateral="orn")`; the named syntax is
`plume:bilateral=orn`. `describe()` records `replaces="computation"`, `law="unverified"`,
the input, pooling, gain, target IDs, audit and inherited removal condition. Per-row
weighted rates reuse the checkpointed `odor_L/R` buffers (Hz in this mode); `bilateral`
holds the filtered rate contrast. Configuration identity prevents loading a physical-cue
checkpoint into the rate-cue configuration. The new mode has no `observe_smell` callback.
The default path keeps its prior arithmetic, state fields and description.

Runtime work is two fixed weighted reductions over 2,104 ORNs, followed by existing
tensor arithmetic. No per-frame CPU readback, extra Poisson sampler or dense neural
matrix is added. CUDA replay compatibility follows the existing module structure but
is **not tested in this CPU-only delivery**; no timing or capture-equality claim.

## 3. CPU signal/noise, before any rooms

Generator: `scripts/plume_transduced_characterise.py`. All conditions, counts, hashes,
assumptions and Monte Carlo seeds are in
[characterisation.json](data/plume_transduced/characterisation.json); the generated
[table](data/plume_transduced/characterisation.md) lists the median-total cases.

The 3,600 historical compass samples yield total antenna concentrations (mean of L/R)
at p05/p50/p95 of about **0.64 / 1.80 / 5.15**. We combine those totals with apple, lime,
banana and an explicitly equal-source mixture of the six fruit odor profiles. These are
bracketing scenarios, **not recovered historical glomerular inputs**. Each uses
`cL_g=c_g*(1+d)` and `cR_g=c_g*(1-d)` for d = 0, 0.0026, 0.0057, 0.007, 0.011, matching
the review's typical and p95 contrast ranges. No sensory response or goal gain is fitted.

For independent sensory spikes with per-cell rates lambda and normalized weights w:

```
V = sum_L(w_i^2 * lambda_i) + sum_R(w_i^2 * lambda_i)
SD_250ms_box(L-R) = sqrt(V/0.25)
SD_100ms_plus_250ms_EMAs(L-R) ~= sqrt(V/[2*(0.1+0.25)])
```

The second formula is the squared-impulse integral for two first-order filters. Contrast
noise uses the delta-method derivatives `2*rR/(rL+rR)^2` and `-2*rL/(rL+rR)^2`;
filtering the nonlinear ratio is only approximately a linear cascade. The shipped
forcing is Bernoulli per 0.5 ms step; its variance correction is `lambda*(1-lambda*dt)`.
Both the Poisson estimate and that correction are recorded. Forced events can bypass
the LIF refractory state (`brain.py`); native synaptic spikes and correlations are not
included in this analytical input model; every model ORN receives synaptic input (mean in-degree 57; the largest presynaptic cells contact 1,800-2,100 of the 2,639), and both shared within-side noise and added common-mode rate lower the ratio, so the estimate above is conservative.

The exact isolated-forcing check draws grouped Bernoulli counts at 0.5 ms, updates the
100 ms rate EMA, reads the contrast every 10 ms and applies its 250 ms EMA. Eight CPU
seeds, 30 s each, discarding the first 2 s; glomeruli and antennae are independent. At
the median-total equal-source mixture and 0.57% contrast, the filtered contrast's temporal
SD is **0.0104 +/- 0.0010**, and correct turn sign occupies **0.667 +/- 0.069** of samples (an independent eight-stream draw with different seeds gives 0.646 +/- 0.067; the analytic prediction is Phi(0.00364/0.01057) = 0.635, so 0.667 is one draw-set of a ~0.64 quantity).
Here +/- is across-draw sample SD (eight independent streams); samples within a stream
are autocorrelated. No binomial significance calculation treats them as replicates.

The inherited gain yields mean absolute goal offsets **50.7 +/- 3.1 deg** in this signal
condition, versus **48.3 +/- 3.1 deg** in the zero-contrast control. The latter is noise
alone; no steering or brain circuit ran. Here mean absolute goal offset is the magnitude of the commanded turn relative to the current heading, not an error against the true source bearing. This is the most direct warning against
interpreting a high-gain rate cue as accurate lateralization. Gains and filters were
left fixed after this result. An eventual room improvement could involve integration,
exploration or other loop dynamics; it would not retroactively validate the sensory law.

## 4. Prepared comparison, not submitted

Generator: `scripts/plume_transduced_batch.py plan --out out/plume_transduced_rooms_v3 --published-plan docs/audits/data/plume_transduced/predeclared_v3.json`.
The corrected frozen [JSON](data/plume_transduced/predeclared_v3.json) and guarded
[batch_v3.sh](data/plume_transduced/batch_v3.sh) are copied beside the CPU artifacts. The user
confirmed the goal-only definition on 2026-09-17:

| Arm | Bilateral goal | DNa02 feedback | Instruments |
|---|---|---|---|
| full | existing physical contrast | on | compass plume hunger flight |
| transduced | matched ORN rate contrast | on | compass plume:bilateral=orn hunger flight |
| goal_only | existing physical contrast | off | compass plume:feedback=off hunger flight |

Six **independent processes/runs per arm**, B=1; matched environment seeds 0-5 and neural
seeds 31-36 across arms. Seed-drawn starts cover the tabletop with a 3 cm edge margin,
rejecting positions already within 1.5 cm of fruit. Start RNG seeds 20260917-20260922;
heading uniform [-180,180). All six exact positions/headings are frozen before results.
This fixture construction uses geometry; no controller receives it. Arm order rotates
across seeds. This changes the prior B=6/shared-start diagnostic to independent draws
with broader starting poses; it does not promise old contact times will reproduce.

Otherwise use the steering configuration: full MaleCNS, compass/hunger/flight, 60 s,
initial energy 0.1, all fruit, no fence/program/escape gating, base GF threshold 33 Hz,
0.5 ms LIF, 1 ms optic, native CUDA/event-driven/captured frames, torch sparse backend.
No local GPU or cluster room was run. The shell has `set -euo pipefail`, house target,
family blocking, preserved per-job exit codes and a named fetch directory. The shell
and room entry point refuse execution without the explicit pool-release environment
flag; it must not be set until Fable authorizes submission. Source hashes must match
the declaration or the room runner refuses.
The declaration is also published under `docs/audits/data/plume_transduced/` so the source
overlay ships it; ignored `out/` files are not assumed to be present on a worker. The v1
local draft was superseded during CPU preflight (declaration shipping and family grouping),
before any submission. The v2 draft used the refuted runtime pairing and is superseded;
its frozen hashes correctly refuse the fixed tree. The fresh v3 declaration carries
`stamped_utc`; v1 and v2 were never submitted. A later source change requires a fresh
plan directory and `--published-plan` path, preserving this declaration.

**A v4 re-freeze is required before submission.** The round-8 merge (`feat/round8`) reconciled the two plume
constructors into one -- `PlumeNavigation(c, *, bilateral="concentration", feedback_gain_per_s=None,
walking_goal=True)`, where `feedback_gain_per_s=0.0` is the arm this declaration spells `plume:feedback=off` -- so
`flyverse/navigation.py` and `flyverse/instruments.py` no longer carry the hashes in this section's
`source_sha256_lf` set. The v3 room runner therefore refuses the tree ("source or protocol differs from frozen
plan; do not run"), exactly as designed. The arms, the starts and the measures are unchanged and `plume:feedback=off`
still names the same arm; only the declaration's hashes are stale. The re-freeze is Astra's, not this branch's: no
v4 plan is generated here. The CPU characterisation was re-run on the merged tree and every measured leaf of
`data/plume_transduced/characterisation.json` is unchanged (4,120 of 4,127 leaves; the differences are
`history.sha256`, `provenance.commit`, the two moved `source_sha256_lf` entries, and three new `describe()`
parameter keys -- `variant`, `variant_spec`, `walking_goal_enabled` -- that the unified API records);
`characterisation.md` reproduces byte-for-byte. The re-run could not read the original history input
(`out/plume_validation_v1/compass.json`, sha256 `e7e2a9bc...`): `out/` is ignored and that file is in no local
tree, so a stand-in history carrying this section's three frozen `total_quantiles` exactly was used instead --
which is all the script reads from it (`np.quantile(total, [0.05, 0.5, 0.95])`), hence the unchanged scenarios,
and which is why `history.sha256` differs.

Report six feeding durations and contact indicators per arm (>=1 s feeding), censored
first-contact times, energy, distance, airborne duration, neural rates, goal offsets and
target/measured DNa02. Each metric is a run-level summary; report mean plus across-run SD,
not trace-sample pseudo-replication. Keep all failures and exact machine records, but quote
no unsupported decimal precision. No gain search, pass/fail adoption gate, physiological
claim or significance claim is predeclared. Full/transduced tests cue replacement;
full/goal-only isolates the feedback bridge. Analysis rejects mismatched provenance,
non-CUDA or incomplete runs. Native-path determinism remains unestablished here.

## 5. Author self-review and pending work

Ten CPU tests cover default versus explicit legacy equality, unequal populations,
signed rate responses, actual frame-start `brain.rate` reads, physical-input isolation,
checkpoint replay/configuration/partial reset, input validation, feedback-off behavior,
draft generation/guards and run-level analysis. The two scheduler-order regressions fail
at 4258468 and pass with the indices and weights sorted together. The reversed fixture
sets DM1=10 Hz and DM2=100 Hz on both sides through actual `brain.rate`, then steps
`FlyBrain`: the old +62.31 Hz bias becomes equal 42.3077 Hz populations and zero contrast
within float32 roundoff. The initial direct-module test and ascending fixture missed this.
A separate before/after replay against
the pre-edit branch snapshot matches **all 40 state arrays byte for byte** after 120
frames. The pre-review full CPU suite, including the original MaleCNS golden, was
**516 passed / 19 skipped / 220 subtests**; it did not detect the ordering bug.
The corrected tree passes **518 tests / 19 skipped / 220 subtests**, including the
MaleCNS bit-identity tests; all ten plume-specific tests are included. Both new
ordering tests were demonstrated failing before the runtime fix.
MaleCNS cache MD5s remain `c50c598a708b5b373cbaffca7d6a9d82` (neurons),
`ac131529cebf98decde58d0c227b7954` (W) and `bf01d724acf2a1fec8fdb60ef8a9e066` (sign-0 counts).

This is the author's self-review. The independent skeptic is quoted below; review of
the correction remains pending Fable. Pending experimental
work: GPU lifecycle/capture checks and the authorized room comparison after pool release;
measure actual neural rates and correlations, then assess whether any approach improvement
survives the low sensory SNR; the 29-check `instrumented` suite column (PRESETS_SPEC
section 2 item 5) remains required before any adoption. No GPU result is implied by a prepared harness.


## Skeptic pass (independent, Opus, 2026-09-17)

The verdict and claims below are quoted verbatim from the independent review of
4258468. The defect was fixed at `9c74945`; that is an author correction, not a revised
independent verdict. All requested prose corrections above are applied. Re-review of
the fix remains pending; no merge or cluster submission.

> VERDICT: refuted (the section-1 mechanism; the SNR arithmetic itself is sound and reproduces exactly)
>
> CLAIMS
>
> 1. Population/weighting: counts reproduce exactly -- `c.select(class="olfactory")` = 2,639, side from `senses.Smell`'s PN-laterality lookup (somaSide is NaN for all 2,639), threshold 0.2 -> 708 L / 1,396 R / 535 unsided, 54 labels both-sided incl. an empty label with 2/side; effective counts 655.51/1127.32. The R bias is a reconstruction/proxy artefact, not biology: the laterality histogram is broad not bimodal, the ~1.9:1 bias survives every threshold (0.0 -> 952/1681, 0.8 -> 263/511) while PN targets are balanced (312/313), and per-glomerulus ratios run 0.077 (VM7v 13L/1R) to 13 (VA5 1L/13R). The dataset's own `instance` suffix (883 L / 1,343 R / 413 none) is never mentioned by the audit and disagrees with the transducer for 222 cells (opposite antenna), drops 526 labelled cells, adds 404 unlabelled ones -- inherited from main, conclusion robust to it (SNR 0.443 vs 0.345). The matched-mass rule does remove the composition bias *in the analysis code* (d=0 difference ~1e-8; naive 1.589-3.809 Hz at d=0, median total, = the audit's "1.6-3.8"), and `q_g ∝ nLnR/(nL+nR)` is provably the min-variance choice. **But not at runtime -- see the refutation below.**
>
> 2. Law/boundary: contrast is `(rL-rR)/(rL+rR)` on the two weighted population rates; rates >= 0 and the value is clamped to +-0.999, so `atanh` is well defined (|d| ~ 0.004 in practice). Nothing was "re-derived": `atan(0.1*2*atanh(d)/0.001) == atan(200*atanh(d))`, the identical function with a new input -- the audit says so correctly. The gain is **unjustified, not merely unverified**: for concentrations 200 came from a dimensional construction (0.1 m / 1 mm); for a dimensionless rate contrast there is no derivation at all (the audit's body covers this, its headline sentence does not). Unstated consequence: the transducer compresses contrast by 0.371/0.429/0.447/0.639 (lime/apple/banana/all-fruit), so retaining 200 gives the transduced arm 0.37-0.64x the shipped arm's turn response to the same physical contrast. Boundary is clean: `observe_smell = None` and `fly.py:269/327` dispatch on `callable(...)`, so no concentration reaches the instrument while `FlyBrain.smell` still drives the ORN Poisson input; inputs are `b.rate[:, idx]` gathered before any module steps (`modules.py:458-471`). `feedback=off` is exactly the DNa02 integral bridge and nothing else (gain -> 0 + `steer_integral.zero_()`; DNa02 still read, no effect).
>
> 3. SNR finding: reproduced twice. Re-running the generator gives a JSON differing from the committed one in **one field** (`provenance.commit`); an independent reimplementation matches every number to 4-5 s.f. (0.02192-0.05620 Hz vs SD 0.2237-0.2728, cascade SNR 0.164-0.345). History p05/p50/p95 = 0.6421/1.7995/5.1504, sha256 matches, and its columns really are only `antenna_L/antenna_R` sums. The four mixtures are labelled scenarios in section 0, section 3 and the JSON. The cascade factor `1/(2(tau1+tau2))` is exact (I derived it), `rate_tau=100 ms` confirmed in `brain.py:40`. Independence is exact for the *forcing* (`brain.py:794/848`, per-cell Bernoulli), but every model ORN has synaptic input (mean in-degree 57; top presynaptic cells touch 1,800-2,100 of 2,639), and both shared-drive effects push SNR **down** -- the estimate is conservative, and the audit flags the caveat without its direction.
>
> 4. Spike streams: committed values reproduce (0.66696 +/- 0.06867; 50.670 +/- 3.050 vs 48.349 +/- 3.130). Independent implementation with different seeds: 0.6455 +/- 0.0670 vs null 0.4947 +/- 0.0756; analytic P(correct) = Phi(0.00364/0.01057) = 0.635. 0.667 **is** distinguishable from 0.5 at n=8 (one-sample t=6.88, p=2.4e-4; paired vs matched null t=11.2, p=1.0e-5), and the audit claims *less* than that -- it explicitly refuses a significance statement and names the within-stream autocorrelation. No over-claim. Nit: `mean_abs_goal_offset_deg` is |goal-heading|, the commanded-turn magnitude, not an error against the true bearing.
>
> 5. Raw/defaults: independently confirmed. Same 120-frame instrumented replay run in worktrees at a455bcf and 4258468 -> **47 arrays (39 instrument + 8 brain), 0 byte-differing**, 28 non-trivially non-zero; default `describe()` of all four instruments byte-identical (sha256 dfb407da...), so the prose reflowing changed nothing; cache MD5s c50c598a.../ac131529.../bf01d724... unchanged before and after; no constructor or CLI default moved (bare `plume` still -> `PlumeNavigation(c)`).
>
> 6. Batch: `plan()` regenerates the committed declaration exactly (starts bit-identical) apart from the paths I redirected and `commit`; the frozen 72-file `source_sha256_lf` matches branch head; `batch_sha256` matches `batch.sh`. All four refusals exercised in-process with nothing written: guard-unset RuntimeError (before `import torch`), no-CUDA RuntimeError, tampered navigation.py hash -> ValueError, tampered arms -> ValueError. 18 jobs, rotation by seed%3, `set -euo pipefail`, one `fam_plume` block (10.4 item 15), 10.3 job-line form with `exit $st`, named fetch dir. Measures are all recorded (`DNa02_L/R` = `motor.turn_L/R` = `type=DNa02` per `motor.py:49`); declared config matches BatchSim defaults and `body.py:469` gf_hz=33. n=6 runs as replicate unit, mean + across-run SD only, no verdict/significance rule. **Deviation: no `stamped_utc`** although 10.4 item 10 names that field and all three sibling declarations carry it. **Hand-off quotes the wrong declaration hash** (d463805c... is the CRLF copy in Astra's worktree; the committed LF file is b1db620c322f2e075f9c8263a44688dec79d34ff031b50bef83f51fe77e2afb5).
>
> 7. Tests: targeted 8 passed; full CPU suite 515 passed / 1 failed / 40 skipped / 220 subtests -- the failure is my cache junction (`c.cache_dir == cn.CACHE_DIR`), so **516 non-skipped outcomes = Astra's 516/19**; the 21 extra skips are test_cuda.py(10)+test_metal.py(11), which Astra excluded rather than skipped. Tests are substantive, not tautological -- **but none runs the ORN read path through the module scheduler on a graph whose read order isn't already sorted, which is why the bug below ships green.**
>
> 8. Prose: section 0 is the Answer, no separate REPORT block, its numbers agree with section 3 and the artefacts; no ALL-CAPS placeholders; vocabulary clean (four uses of "result" are negations or refer to the CPU output; no "null"/"significant"; "byte for byte" only of a CPU replay); no infrastructure identifiers in the diff (only `source .venv/bin/activate` and `--target house`, both established repo conventions); self-review labelled in section 5. PRESETS_SPEC section 2 items 1/2/3/6 and section 5 `replaces="computation"` met; item 5 (the 29-check instrumented suite column) is neither done nor listed as pending.
>
> 9. Bottom line: yes, the CPU pass answers the narrow TODO B question, and answers "no" -- at the shipped ORN law, population, 100 ms rate filter and 250 ms contrast filter, 0.26-1.1 % physical contrast arrives as 0.010-0.108 Hz against 0.13-0.16 Hz of filter noise (SNR 0.08-0.67, below 1 across the whole historical range), and saturation makes the earlier "~0.1 Hz/ORN" estimate optimistic by 2-5x. What the rooms would add: whether a moving, integrating, casting closed loop can use a 64 %-correct sign at 4 Hz (an SNR-per-window number does not settle a sequential-decision question); whether AL/PN processing expands the contrast; what full-brain ORN rates and correlations actually are (no record in the repo has them); and whether the DNa02 feedback bridge rather than the cue is what finds food (the `goal_only` arm).

Author freeze record: `data/plume_transduced/predeclared_v3.json`,
`stamped_utc=2026-09-18T00:06:24Z`, fixed runtime source `9c74945`.
