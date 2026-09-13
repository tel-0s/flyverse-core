# TODO — public release

Ordered by what blocks a release, then by what makes the release worth reading. "Model-side" means a
mechanism the connectome or physiology data imply, tested with the interpretability toolkit
(`docs/INTERP.md`); never a gain tuned to a behaviour. Status of every behaviour: `docs/BENCHMARK_BATTERY.md`.

## A. Release blockers (days)

- [ ] **LICENSE** (code) and a data-licence note: MaleCNS v1.0 (Janelia FlyEM; cite the release paper and its
      licence), the external expression tables (`scripts/fetch_data.py`; not redistributed — `data/external/` is
      git-ignored), FlyWire NT predictions where used. `CITATION.cff` for flyverse itself.
- [ ] **Infrastructure scrub** — done at the tip (2026-09-13: cluster host / IP / filesystem paths replaced by
      `<cluster-host>`, `$CLUSTER_RUNS`, … in 19 files); decide whether to rewrite history before the announcement
      (`git filter-repo` on the same patterns) — the identifiers are internal hostnames and paths, nothing secret,
      but they were committed. `docs/CLUSTER.md` and `.cluster.json` stay git-ignored.
- [ ] **Packaging**: `pip install -e .` from a clean clone works (pyproject has the deps; add `python_requires`,
      optional extras `[cuda]`, `[ui]`, `[interp]`), `python scripts/fetch_data.py --malecns` then
      `python scripts/room_demo.py` runs on CPU-only and on CUDA; pin torch/numpy minimums; a `requirements-lock`.
- [ ] **CI**: GitHub Actions running the CPU test subset (`tests/test_control.py`, `test_world.py`,
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
      that the GPU rollout is not seed-reproducible (round-1 finding) and that runs are the replicate unit.
- [ ] **Owed bookkeeping before the numbers are quoted publicly**: de-score `walk.power_max` (unfit bound;
      `docs/audits/optic_measures.md`), remove the no-op pair-gain entry (`DEFAULT_PAIR_GAIN[4]`), re-label the
      T5 pair gain as a drive gain in `optic.py`'s comment, fix the `optic.py:89` "x4" comment; record the
      `drive_clip_mv` decision once its 3 draws exist.
- [ ] **Neurome acknowledgement** and the interchange note (`docs/NEUROME_INTERFACE.md`): agree with Astra what
      is public (their reports are in their repo).

## B. Core behaviours — model-side (the clock; needs the rented B200s)

Where each stands, and the data-implied route (from `docs/audits/deficit_*.md`, skeptic-corrected):

- [ ] **Spontaneous turning / the straight walker.** DNa02 sits under tonic sign-correct inhibition
      (−1.6 / −2.0 mV steady vs a 7 mV gap) while every lateralised excitatory route is silent at source: PFL3
      (compass at 0 Hz by default), AOTU015 (object route), most LLPC1; the VNC runs open-loop — every
      `vnc_sensory` cell at 0 Hz. Routes, in order:
  - [ ] the **proprioceptive / haltere transducer** (dynamics round 2, `scratchpad/vnc_round2.js`, paused): leg
        chordotonal / hair-plate / campaniform from the leg-MN rates and ground contact, haltere from the
        haltere-MN rate; literature-ranged; the Coriolis term only as a labelled control. Decides whether the
        ascending chain (AN04B003 → DNa02's VNC inhibitors; AN07B037_a → PS196_b → GLNO → PEN) carries anything.
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
- [ ] **Small-object pathway (LC11).** Lost by opposite-signed carrier convergence at T2/T3, the residual
      scrambled by spiking feedback, pooled away at LC11/LC10a. Object round 2 (`scratchpad/object_round2.js`,
      paused): the matched assay Neurome asked for + the three opt-in rate-lobe hooks (per-stream
      rectification with signs preserved, per-stream adaptation, spatial suppression) with the specificity
      battery. Data: receptor tiers for Tm5Y / TmY21 / TmY13 / LC11 (Neurome); Keleş 2020 / Tanaka & Clark 2020
      constraints are in the ledger.
- [ ] **Feeding / terminal approach.** Not a model-default question: the cx program's approach fails in the
      last centimetres (plume downwind-only, no concentration-change rule). Model-side alternative worth one
      round: bilateral antennal sampling → AL → LH → DN route already exists; test whether a data-implied
      odour-gradient encoding (ORN adaptation, Nagel & Wilson 2011) makes klinotaxis emerge; keep `--program`
      modules as the documented fallback. Re-instrument the assay (ground-only closest approach, first_contact).
- [ ] **Take-off cost of the receptor signs**: the histamine / glutamate class split with a dose control
      (dynamics round 2); if one class carries it, the contested-flip rule in `docs/NT_INTEGRATION.md` gets
      re-examined.
- [ ] **Escape and wind orientation** pass today; keep them in every suite run as regression guards.

Most useful experimental data, ranked by leverage: (1) receptor / conductance profiles for the unprofiled
types and for DA / OA / 5-HT receptors; (2) per-type baseline firing in behaving flies (DN / AN / CX imaging);
(3) unitary synaptic strengths by transmitter (mV per synapse; the global scale is the one number every
attractor depends on); (4) proprioceptor firing ranges (Mamiya 2018; Agrawal 2020); (5) behavioural
kinematics ground truth for the ledger (DeAngelis 2019 walking; Katsov 2017 turning statistics; Álvarez-Salvado
2018 plume navigation; von Reyn 2014 escape latency); (6) per-body LC11 / LC10a recordings at matched sizes.

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
