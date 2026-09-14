# TODO — public release

Ordered by what blocks a release, then by what makes the release worth reading. "Model-side" means a
mechanism the connectome or physiology data imply, tested with the interpretability toolkit
(`docs/INTERP.md`); never a gain tuned to a behaviour. Status of every behaviour: `docs/BENCHMARK_BATTERY.md`.

## A. Release blockers (days)

- [x] **LICENSE** (code: MIT, 2026-09-13) and a data-licence note: MaleCNS v1.0 (Janelia FlyEM; cite the release paper and its
      licence), the external expression tables (`scripts/fetch_data.py`; not redistributed — `data/external/` is
      git-ignored), FlyWire NT predictions where used. `CITATION.cff` for flyverse itself.
- [ ] **Infrastructure scrub** — done at the tip (2026-09-13: cluster host / IP / filesystem paths replaced by
      `<cluster-host>`, `$CLUSTER_RUNS`, … in 19 files); decide whether to rewrite history before the announcement
      (`git filter-repo` on the same patterns) — the identifiers are internal hostnames and paths, nothing secret,
      but they were committed. `docs/CLUSTER.md` and `.cluster.json` stay git-ignored.
- [~] **Packaging** (2026-09-13: pyproject metadata, extras, `flyverse.interp` now shipped in wheels, `CITATION.cff` validated, `docs/INSTALL.md`; still owed: a clean-clone CUDA run-through and a lock file): `pip install -e .` from a clean clone works (pyproject has the deps; add `python_requires`,
      optional extras `[cuda]`, `[ui]`, `[interp]`), `python scripts/fetch_data.py --malecns` then
      `python scripts/room_demo.py` runs on CPU-only and on CUDA; pin torch/numpy minimums; a `requirements-lock`.
- [x] **CI** (2026-09-13: `.github/workflows/ci.yml`, data-free subset via `tests/conftest.py` markers — 236 passed locally, first GitHub run green in 1m43s; `ruff --select F821` reports 12 undefined names in `scripts/cx_shift.py` / `cx_wedge.py` nested closures — verify whether those paths are dead or rely on an enclosing scope, then fix). Original item: GitHub Actions running the CPU test subset (`tests/test_control.py`, `test_world.py`,
      `test_nt_readout.py`, `test_interp.py -k "not cluster"`, `test_receptor_model.py`) on a synthetic /
      subset connectome so it needs no 3 GB download; lint.
- [ ] **README pass**: one page a newcomer can follow — run it, what is simulated, what emerges unprompted,
      what we added and why (every stop-gap named), the benchmark table with statuses, the three localized
      deficits stated plainly (straight walking, small object, compass rotation input), how to embed the brain,
      how to run the toolkit when a behaviour fails. Move the long form to `docs/`.
- [ ] **Demo media**: a 20-30 s GIF/MP4 of the room (observatory UI, loom escape, wind orientation, feeding
      approach) and one figure of the toolkit output (a `trace` stage map or the atlas), committed under
      `docs/media/` (git-ignore rule currently excludes `*.gif`/`*.mp4` — carve out `docs/media/`).
- [ ] **Reproducibility statement**: the shipped default model (LIFParams / OpticParams / gains) with the cache
      fingerprint (sum|W| 121,460,584; W md5) and the exact commit the benchmark table was produced at; note
      that the GPU rollout is not seed-reproducible (round-1 finding) and that runs are the replicate unit,
      and that a bit-identity claim about the shipped output is a CPU claim only: two identical runs of one
      tree at one seed on a B200 diverge from frame 500 of 6,000 (`out/proprio_bitid/compare.txt`).
- [~] **Owed bookkeeping before the numbers are quoted publicly** (all four edits made 2026-09-13, in the
      working tree): de-score `walk.power_max` — **decided from the data** (`docs/audits/anti_runaway.md`
      round 6; 12/12 draws at 48.48, margin inside the worst single-arm scatter, non-monotone, Spearman
      −0.600 against the room take-off rate) and **DONE**: `scripts/benchmark.py:85` now carries the
      `notnone` form of `loom.escape_cm`, and the row stays in the pass tally as a report
      (`benchmark.py:135`); the no-op pair-gain entry (`DEFAULT_PAIR_GAIN[4]`) **REMOVED** and the T5 pair
      gain re-labelled a drive gain and the "x4" comment fixed in `optic.py` — **DONE**; record the
      `drive_clip_mv` decision — **RECORDED: 7 draws, 10 PASS / 0 FAIL ×6, `walk.power_max` 49.2480,
      `walk.GF_max` 4.63 → 13.26, `motion.min_dsi` 0.0040 below the lowest shipped draw on record; NOT
      adopted — the adopt-alone rule's 29-check suite × ≥ 3 and the room take-off protocol were not run.**
- [ ] **Neurome acknowledgement** and the interchange note (`docs/NEUROME_INTERFACE.md`): agree with Astra what
      is public (their reports are in their repo).

## B. Core behaviours — model-side (the clock; needs the rented B200s)

Where each stands, and the data-implied route (from `docs/audits/deficit_*.md`, skeptic-corrected):

- [ ] **Spontaneous turning / the straight walker.** DNa02 sits under tonic sign-correct inhibition
      (−1.6 / −2.0 mV steady vs a 7 mV gap) while every lateralised excitatory route is silent at source: PFL3
      (compass at 0 Hz by default), AOTU015 (object route), most LLPC1; the VNC runs open-loop — every
      `vnc_sensory` cell at 0 Hz. Routes, in order:
  - [~] the **proprioceptive / haltere transducer** (dynamics round 2, done and measured; shipped as an
        opt-in module, default OFF — `docs/audits/proprioception_transducer.md`, `vnc_drive.md`). Answer:
        the ascending chain **does** carry it — AN04B003 0.589/0.314 → 3.713/3.690 Hz, PS196_b → 1.44/1.51,
        GLNO → 0.66/0.66, all `result` at 5 runs/arm — and **neither deficit closes**: DNa02's operating
        point moves −1.52 → −1.17 mV against a 7.0 mV gap (2–3 % of the required dose, the ascending
        excitation cancelled on the spot by PS059), and the compass report arrives **unsigned** (PS196_b's
        L−R moves the same way in both turn directions even when driven to 11 Hz). Still owed before any
        default: `rest` redefined with a ledger row (the arm reads 6.0 against `< 5` by construction), the
        full 29-check suite × ≥ 3 with the sense on through `BatchSim`, and a measured Drosophila rate per
        channel. **Next mechanisms, both body-side:** a leg cycle in `body.py` (makes the leg channels sided
        during a turn) and a side-split haltere MN readout in `motor.py`.
  - [ ] **neuromodulator signs**: 3,312 presynaptic bodies (2.7 M synapses: dopamine, octopamine, serotonin,
        unknown) are silenced (sign 0). Receptor-expression tiers for DA / OA / 5-HT receptors per postsynaptic
        type (same sources as `receptors_by_type.csv`; `docs/NT_INTEGRATION.md`) would un-silence the arousal /
        locomotor-state system the animal's spontaneous walking depends on. Largest single missing input.
  - [ ] **the compass at the defaults**: the ring is silent unless EPG/PEN/Δ7 gains are raised (experiment
        overrides); with them the bump persists but is pinned, heading-blind and steering-inert. Data question:
        the per-transmitter unitary strength (mV per synapse) and the CX receptor tiers; a bump in darkness
        (Seelig & Jayaraman 2015) is the ledger target. Rotation input needs the transducer above (GLNO's
        inputs are efference / proprioceptive territory; the visual route is direction-blind).
  - [ ] **intrinsic / spontaneous activity**: the model's only noise is sensory Poisson; per-type baseline
        rates (DNs in walking flies — Aymanns 2022; ANs — Chen 2018; CX) as ledger rows and as a documented
        Poisson-background mechanism per type if the data support it.
- [~] **Small-object pathway (LC11).** Lost by opposite-signed carrier convergence at T2/T3, the residual
      scrambled by spiking feedback, pooled away at LC11/LC10a. **Object round 2 done
      and measured** (`docs/audits/object_matched_assay.md`, `object_synthetic_stimuli.md`,
      `optic_stream_hooks.md`, `object_baseline_r2.md`, `object_compare_r2.md`,
      `object_export_r2.md`; the three rate-lobe hooks ship opt-in, default `None`, bit-identical
      off on a deterministic backend). Answers: the **matched assay** removes the elevation /
      speed / diameter confound (realised deviation 0.0 deg, 1.4e-14 deg, 40.000 deg/s) and
      **no size preference is called for either LC type at 6 v 6** — LC11 12/12 `null`, LC10a's
      one `result` at 30 deg fails Holm (p_holm 0.104) and lies inside its own 15–30 deg target;
      the **model comparison finds no passing mechanism** — rectification carries the T3/T2
      carrier figure at 6–10× base but releases bar / grating / flicker at LC11 (the Keleş 2020
      constraint), creates a size-increasing figure, moves only the max over cells and shifts the
      operating point; adaptation is inert; spatial suppression costs the escape benchmark
      (GF peak 50.0 → 31.2 Hz). **Nothing adopted, no default changed.** Still owed before
      anything here can advance: (a) a **stimulus-driven LC11 localizer** — 0 of 143 bodies fit
      at `z_min` 5, so 405 of 418 LC windows are anatomical boxes; try a smaller probe square,
      more passes, or a lower `z_min` with its false-fit rate quoted; (b) a **same-device `base`
      re-run** (5 runs + 5 nulls on the box that hosted `rectify` and `suppress`) to de-confound
      arm from GPU model; (c) the **contrast-matched
      synthetic rectangle ladders** (height and width separately, as Keleş & Frye did), recorded
      on the boxes but never fetched (150 expected outputs missing); (d) **export of the compare
      arms** (`object_round2_export.py compare`), which no directory under `out/export/` carries.
      **Closed since the critic's list:** the pre-fix `spearman_perm` floor (`p = 5e-05` with an
      undefined rho on four primary rows) — `out/interp/objr2/baseline.json` was re-analysed and
      the ladder re-exported on 2026-09-14 (`objr2-ladder-20260914T024906Z-035363c0`), the
      2026-09-13 directories remaining as the superseded delivery; and the **ON/OFF transition
      split**, re-derived on CPU from the stored spec recordings
      (`out/interp/objr2c/spec_transitions.json`) — both transitions survive in every arm
      (no window below 0.56× base's), what rectification moves is the ON/OFF asymmetry, and almost
      every 0.3 s window sits at its own blank/blank floor (`object_compare_r2.md` 6.1).
      Data: receptor tiers for Tm5Y / TmY21 / TmY13 / LC11 (Neurome); per-body LC11/LC10a
      recordings at the six matched sizes; the Keleş 2020 Rdl constraint in quantitative form so
      it can be a **scored** ledger row rather than the magnitude rule this round used.
- [ ] **Feeding / terminal approach.** Not a model-default question: the cx program's approach fails in the
      last centimetres (plume downwind-only, no concentration-change rule). Model-side alternative worth one
      round: bilateral antennal sampling → AL → LH → DN route already exists; test whether a data-implied
      odour-gradient encoding (ORN adaptation, Nagel & Wilson 2011) makes klinotaxis emerge; keep `--program`
      modules as the documented fallback. Re-instrument the assay (ground-only closest approach, first_contact).
- [~] **Take-off cost of the receptor signs** — split done with a dose control and replicated at fresh
      seeds (dynamics round 2, `receptor_integration.md` G.6): the **histamine silencings** carry the hop,
      voluntary and GF-median cost (null vs the default, 7.21× off on hops pooled over 7 runs/arm) and the
      larger glutamate class carries none; entry/|W| dose is refuted. **Correction to this item as
      written:** the class is a product of the **silencing rule**, not the contested-flip rule
      (`fast_net_abs none`, `flip_contested` empty on 45/45), so `docs/NT_INTEGRATION.md` gets an item on
      the *silencing* rule's premise for photoreceptor → medulla edges — and first as an `optic.py`
      question (`optic.py:165-225` applies the receptor factor to photoreceptor → rate edges too), never
      decided on the room take-off rate. Still open: which rows within the class (per-row split), and
      **dose in postsynaptic cells touched**, which this design structurally cannot close.
- [ ] **Escape and wind orientation** pass today; keep them in every suite run as regression guards.

Most useful experimental data, ranked by leverage: (1) receptor / conductance profiles for the unprofiled
types and for DA / OA / 5-HT receptors; (2) per-type baseline firing in behaving flies (DN / AN / CX imaging);
(3) unitary synaptic strengths by transmitter (mV per synapse; the global scale is the one number every
attractor depends on); (4) proprioceptor firing ranges (Mamiya 2018; Agrawal 2020); (5) behavioural
kinematics ground truth for the ledger (DeAngelis 2019 walking; Katsov 2017 turning statistics; Álvarez-Salvado
2018 plume navigation; von Reyn 2014 escape latency); (6) **[now actionable]** per-body LC11 / LC10a recordings at
matched sizes — object round 2 built and exported the matched geometry, so this is no longer hypothetical: the six
rungs (4.5 / 8.8 / 11 / 15 / 20 / 30 deg, elevation / distance / diameter / speed held per frame) are in
`out/export/objr2-{ship,fb0}-d*/readout_per_body.csv`, keyed by `bodyId` decimal string with `n_trials` 6 and both
`upstream_drive_mV` and `output_Hz` rows per body (`docs/NEUROME_INTERFACE.md` 3c, ask 2).

## C. Extensibility (make "glue anything on" a supported path)

- [ ] **Step hooks** on `FlyBrain`: `fb.add_input_hook(fn)` / `fb.add_readout_hook(fn)` called every frame with
      `(fb, t_ms)`; hooks may `set_drive` / `set_poisson` on any `Connectome.select` population and read rates —
      the primitives exist (`brain.set_drive`, `set_poisson`, `stimulate`, `rates`); this is a thin wrapper +
      docs + tests (days).
- [ ] **Module protocol**: `class Module: reads: selection; writes: selection; step(dt_ms, reads) -> writes;
      state_dict()` — an SNN graph, a torch net, or a plain function run inside the frame loop, batched;
      `programs.Composite` already composes body-side modules the same way. Register by name; recorded in
      provenance (`interp.common.model_record`) so the toolkit knows what is glued on.
- [ ] **Graph extension**: `Connectome.extend(nodes, edges)` (synthetic bodies under a `dataset='synthetic'`
      key — the Neurome join key already carries `(dataset, release, bodyId)`), rebuilding the CSR and cache;
      `brain.set_weights` exists; `prune` is the inverse. Needed for additional SNN graphs wired into the
      connectome rather than run beside it (~a week incl. tests, interp provenance, CUDA-graph recapture).
- [ ] **Trainable encoders / decoders**: sense side (radiance → photoreceptor drive; odour → ORN rates) and
      motor side (rates → commands) as torch modules trained by RL through `flyverse/env.py` (already
      vectorised) or by supervision; document the pattern with one example (a learned motor decoder that
      does not touch the brain).
- [ ] **Gradients through the brain**: opt-in surrogate-gradient LIF on the Torch path only (the CUDA / Metal
      kernels and CUDA graphs do not backprop); feasible for `modules=` subsets; document as slow and
      experimental. Do not promise end-to-end training of the full 167k-cell model.
- [ ] **Stable public API surface**: `flyverse.FlyBrain`, `flyverse.body.*`, `flyverse.programs.*`,
      `flyverse.env.FlyRoomEnv`, `flyverse.batch_sim.BatchSim`, `flyverse.interp.*`, `flyverse.nt_readout`;
      semantic version it; deprecation policy for the old adapters (`body.motor_groups`, `air.WindSense`).

## D. Toolkit and process

- [ ] `flyverse/interp` open contract defects (`docs/INTERP.md` §11, items 9-16): the front door
      (`interp_deficit.py` running the §10 procedure), the `edges` lesion kind, a live null and vision context for
      the atlas, one validation semantics, CPU tests for the apply scripts, graded units' drive in mV in exports.
- [ ] `cluster_run.py`: refuse the bare `--fetch out/`; verify-batch built in; the console log's failure line
      surfaced; a per-run `provenance.json`.
- [ ] Batch-sustain / probe JSON headers: done for `batch_sustain.py`; do the same for every probe still
      writing `options.device = None`.
- [ ] Observatory UI: NT readout of the `health` tool; a "why did it do that" panel that runs `decompose` on
      the current frame's DN inputs.

## E. Nice to have for the announcement

- [ ] A results page (artifact / docs site): the benchmark table, the three deficit maps, the toolkit's
      stage map for the object, the compass-in-the-room figure — every number with its audit link.
- [ ] A 5-minute "run the fly" notebook (CPU subset: antennal lobe + MB + central + DN, smell → turn).
- [ ] Contributing guide: the project rule in one paragraph; how to propose a default change (audit +
      skeptic + suite run); how to add a ledger row.
