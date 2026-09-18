# TODO — public release

Ordered by what blocks a release, then by what makes the release worth reading. "Model-side" means a
mechanism the connectome or physiology data imply, tested with the interpretability toolkit
(`docs/INTERP.md`); never a gain tuned to a behaviour. Status of every behaviour: `docs/BENCHMARK_BATTERY.md`.

## A. Release blockers (days)

- [x] **LICENSE** (code: MIT, 2026-09-13) and a data-licence note: MaleCNS v1.0 (Janelia FlyEM; cite the release paper and its
      licence), the external expression tables (`scripts/fetch_data.py`; not redistributed — `data/external/` is
      git-ignored), FlyWire NT predictions where used. `CITATION.cff` for flyverse itself.
- [x] **Infrastructure scrub** — done at the tip (2026-09-13: cluster host / IP / filesystem paths replaced by -- DONE 2026-09-17: the whole history was rewritten with git-filter-repo (161 commits; hostnames, the cluster user, the shared-filesystem root, the scheduler name, the rented-box IPs and the workstation home path replaced by `<cluster-host>` / `<cluster-node>` / `<cluster-user>` / `<cluster-fs>` / `<scheduler>` / `<rented-box-ip>` / `<workstation-home>` in every blob and commit message); a pre-rewrite bundle is kept outside the repo; the ten remote branches were force-pushed. GitHub may retain the old objects until its own GC -- ask support to purge before the public flip.
      `<cluster-host>`, `$CLUSTER_RUNS`, … in 19 files); decide whether to rewrite history before the announcement
      (`git filter-repo` on the same patterns) — the identifiers are internal hostnames and paths, nothing secret,
      but they were committed. `docs/CLUSTER.md` and `.cluster.json` stay git-ignored.
      **Still owed at the tip (the 2026-09-17 code skeptic's list; nothing on the instrument branch added any of
      them):** `docs/audits/receptor_verification.md` and `docs/NOTES.md` (cluster node names, hosts and
      `<workstation>` home-directory paths quoted inside older verbatim blocks -- scrub the identifier, keep the quotation
      intact and say it was scrubbed), `docs/media/README.md` lines 41 / 63 / 109. The five ignored
      `out/*/predeclared.json` families (`out/cx5`, `cx6`, `cx7`, `objr3sd*`, `vncd4-7`) carry `host` / `ssh`
      fields; they are git-ignored today and the check is that they **stay** ignored, not that they are edited.
      That is eight files in all; do them in one pass with a `grep -n` receipt before the announcement.
- [~] **Packaging** (2026-09-13: pyproject metadata, extras, `flyverse.interp` now shipped in wheels, `CITATION.cff` validated, `docs/INSTALL.md`; still owed: a clean-clone CUDA run-through and a lock file): `pip install -e .` from a clean clone works (pyproject has the deps; add `python_requires`,
      optional extras `[cuda]`, `[ui]`, `[interp]`), `python scripts/fetch_data.py --malecns` then
      `python scripts/room_demo.py` runs on CPU-only and on CUDA; pin torch/numpy minimums; a `requirements-lock`.
- [x] **CI** (2026-09-13: `.github/workflows/ci.yml`, data-free subset via `tests/conftest.py` markers — 236 passed locally, first GitHub run green in 1m43s; `ruff --select F821` reports 12 undefined names in `scripts/cx_shift.py` / `cx_wedge.py` nested closures — verify whether those paths are dead or rely on an enclosing scope, then fix). Original item: GitHub Actions running the CPU test subset (`tests/test_control.py`, `test_world.py`,
      `test_nt_readout.py`, `test_interp.py -k "not cluster"`, `test_receptor_model.py`) on a synthetic /
      subset connectome so it needs no 3 GB download; lint.
- [x] **README pass** (2026-09-14, at `28e862f`): one page a newcomer can follow — run it, what is simulated, what emerges unprompted,
      what we added and why (every stop-gap named), the benchmark table with statuses, the three localized
      deficits stated plainly (straight walking, small object, compass rotation input), how to embed the brain,
      how to run the toolkit when a behaviour fails. Move the long form to `docs/`.
      DONE: README rewritten as 12 sections (run it / what is simulated / what emerges unprompted / where the
      model stands with the assay statuses and the 27 PASS / 0 FAIL / 2 KNOWN GAP suite tally / what we added
      with every stop-gap named in a table / the three localized deficits with round 3's state / the two female
      connectomes in one paragraph / embed the brain / run the toolkit / reproducibility / map / licence).
      Long form moved, not deleted, to **`docs/OVERVIEW.md`** (controls, flags, speed, backends, BatchSim, the
      per-stage table, the full "what we added" argument, food-finding and programs, RL, the file-by-file map).
      Every CPU command in the README was run and passes: `fetch_data.py --list`, `python -m flyverse.connectome`
      (cache md5s unchanged), five `room_demo.py` forms headless, the embedding snippet, the three CPU toolkit
      commands, and `pytest -m "not gpu and not data and not cluster"` (**364 passed, 7 skipped, 29 deselected**).
      Four corrections made in passing: the optomotor readout is DNp20 + HSN/HSE, not DNp04 + LPT27/30
      (`motor.py:52`); the LIF partition is **71,618**, not 71,625; the taste/bitter numbers are now the round-3
      suite's (Shiu 139.9 -> 0.8 Hz, calibrated 5.5 -> 0), not the superseded 2026-09-11 run's; `CITATION.cff`
      gained the two FlyWire releases. `docs/INSTALL.md` made ASCII and its `data` marker row completed.
- [x] **Demo media** (2026-09-15, `docs/media/`: `loom.gif` / `loom.mp4` (25 s, loom at 8 s, giant-fibre jump and re-landing),
      `wind_apple.gif` / `.mp4` (25 s; honestly captioned: no wind orientation and no feeding can be claimed from the clip),
      `toolkit_loom_gf.png` (`paths` stage map + `atlas` readout); every command line, commit and device in
      `docs/media/README.md`; generator `scripts/make_demo_media.py`; rendered on the house B200 at 67c76a4). Original item: a 20-30 s GIF/MP4 of the room (observatory UI, loom escape, wind orientation, feeding
      approach) and one figure of the toolkit output (a `trace` stage map or the atlas), committed under
      `docs/media/` (git-ignore rule currently excludes `*.gif`/`*.mp4` — carve out `docs/media/`).
- [x] **Reproducibility statement** (2026-09-14, `docs/REPRODUCIBILITY.md`, linked from the README): the shipped default model (LIFParams / OpticParams / gains) with the cache
      fingerprint (sum|W| 121,460,584; W md5) and the exact commit the benchmark table was produced at; note
      that the GPU rollout is not seed-reproducible (round-1 finding) and that runs are the replicate unit,
      and that a bit-identity claim about the shipped output is a CPU claim only: two identical runs of one
      tree at one seed on a B200 diverge from frame 500 of 6,000 (`out/proprio_bitid/compare.txt`).
      **The round-3 commit is the reproducibility anchor**: commit the whole round-3 working tree in one commit
      (so every batch's uncommitted cross-task dependencies have a history) and quote that tree's identity with
      the numbers -- `provenance.source_fingerprint` **44 files** (not 43), compiled-W md5
      **ef23cc27bea13be7f6a96f3c04fd3737**, receptor-table md5 **0381a446107e6050e75cc87b16d7f830**
      (`scratchpad/r3_results/critique.json`, owner notes 0(d)). Every number in
      `docs/NOTES.md` "Session 11" is quoted against that tree; **nothing was adopted in round 3 and no default
      moved**, so the shipped model the statement describes is unchanged.
      DONE: 8 sections. Every fingerprint RECOMPUTED at `28e862f`, not copied -- sum|W| **121,460,584**, W md5
      **ef23cc27bea13be7f6a96f3c04fd3737**, cache md5s neurons.parquet `c50c598a...` / W_post_pre.npz
      `ac131529...` / sign0_counts.npz `bf01d724...`, receptor table `0381a446...`; the **44 files** of the
      round-3 `source_fingerprint` verified by enumerating `f9e9fea` against its own `export.SOURCE_PATTERNS`
      (the current tree is 51, the backends having added six patterns). Carries: LIFParams / OpticParams /
      every DEFAULT_* gain as field-by-field tables with the stop-gaps marked; the female fingerprints;
      MaleCNS / FAFB / BANC file names and sha256s; the GPU non-reproducibility with `compare.txt` quoted
      verbatim; bit-identity as a CPU-only claim; regeneration commands; and a table of which README number
      was measured at which commit. **The benchmark table predates the anchor and this is stated**: the
      14-section table in `docs/audits/benchmark_suite.md` is 2026-09-11 on an RTX 4090 at `069deb0`, and its
      own JSON `config` proves it is not the shipped model (GF x0.3 damping still present, retired at
      `560aaf3`; `gf_hz` 38 not 33; receptor model not yet default, adopted at `79769c3`). The authoritative
      per-check numbers are the round-3 guard suite (`guard7-97ce35`, tree `d2abf3c` + inert `LegCycle` =
      `f9e9fea`'s defaults).
- [x] **Owed bookkeeping before the numbers are quoted publicly** (all four edits made 2026-09-13, in the
      working tree): de-score `walk.power_max` — **decided from the data** (`docs/audits/anti_runaway.md`
      round 6; 12/12 draws at 48.48, margin inside the worst single-arm scatter, non-monotone, Spearman
      −0.600 against the room take-off rate) and **DONE**: `scripts/benchmark.py:85` now carries the
      `notnone` form of `loom.escape_cm`, and the row stays in the pass tally as a report
      (`benchmark.py:135`); the no-op pair-gain entry (`DEFAULT_PAIR_GAIN[4]`) **REMOVED** and the T5 pair
      gain re-labelled a drive gain and the "x4" comment fixed in `optic.py` — **DONE**; record the
      `drive_clip_mv` decision — **RECORDED: 7 draws, 10 PASS / 0 FAIL ×6, `walk.power_max` 49.2480,
      `walk.GF_max` 4.63 → 13.26, `motion.min_dsi` 0.0040 below the lowest shipped draw on record; NOT
      adoptable -- the adopt-alone suites were RUN (round 7, `guard_suites_r3.md`): suite 27/0/2 x3, room
      voluntary take-offs 3.96 vs 1.94 per 1,000 fly-s (`result` at 4 v 4 on the B200); the clip binds on the
      wing-power route. LPi x1 also run in the room: 34.6 per 1,000 fly-s, 48/48 flies above the escape
      threshold -- NOT adoptable.** Both are closed as NOT adoptable in `anti_runaway.md`; the next submission
      on that thread is only for a candidate REPLACEMENT mechanism.
- [ ] **Neurome acknowledgement** and the interchange note (`docs/NEUROME_INTERFACE.md`): agree with Astra what
      is public (their reports are in their repo).

## B. Core behaviours — model-side (the clock; needs the rented B200s)

Where each stands, and the data-implied route (from `docs/audits/deficit_*.md`, skeptic-corrected):

- [x] **Experimental imposed compass memory** (owner-authorized control arm, 2026-09-15):
      `--preset instrumented --instrument compass` uses an ordinary module with held realized yaw,
      supplies a continuous EPG Poisson bump and passes all 48 controlled turn/reversal/hold trajectories.
      Final native overhead 54.22 us/frame at B=1 (11.01%, misses <=10% target), 3.19% at B=8, 1.22% at B=32.
      This does not close the biological compass or steering items below. One taste status changes in
      the three-draw suite, rejecting strict admission; the full room rate-half is not run. Raw remains
      default, no food-finding claim. See `docs/audits/compass_standin.md`; independent skeptic pass
      (Opus, 2026-09-17) mostly sound -- the claims match the code and no unearned physiology was found.
- [~] **Review the compass experiment before any adoption**: the independent skeptic pass is **done**
      (Opus, 2026-09-17, mostly sound; quoted in `compass_standin.md`). Still owed: account for the
      taste-row change, then the full room gate if an admissible candidate exists. Do not retune
      gains to pass the suite or treat short room rollouts as evidence of food finding.

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
        **Round 3: leg cycle + side-split haltere built and measured** (`body_sided_state.md`): DNa02 fires
        (0.54/0.38 Hz), yaw SD 7.9, a fixed left drift, no frame > 100 deg/s, the report gone at GLNO; a LEVEL
        effect (23 -> 88 Hz) not separated from the phase structure. Owed: the level-matched control (`all`,
        `mn_ref_hz` ~3.5), `lit.walk.*` ledger rows, hops + room at >= 6 runs/arm, `half_width_m` measured,
        `MotorRates.haltere_L/_R` (owner -- **deferred**, session 11). Both mechanisms ship opt-in, default OFF;
        nothing adopted and no default changed.
    - [x] **the level-matched control -- RUN** (`docs/audits/level_matched_control.md`; ONE house submission
          `vncd4-8dd183`, 24 jobs, 0 failed, 32.3 min, B200; 4 arms x 5 seeds in the room + the same 4 arms x 4
          seeds on the efferent compass; the `lit.walk.*` rows, the `sided_frames` lag and the DNa02 frame mask
          done first on CPU). `mn_ref_hz` is **8.84**, not ~3.5: 8.84 is the fixed point of the afferent -> leg-MN
          loop, while ~3.5 ignores the clip and the loop and would have saturated at **143-150 Hz**. The
          chordotonal level matched -- L **86.2 +- 0.5** vs C **87.4 +- 1.0 Hz** (C v L +1.2, `null`, inside the
          predeclared 10 Hz tolerance) -- and C v L is `result` on **DNa02_L +0.127 Hz** (z +9.2), **DNa02_R
          +0.103** (z +16.7) and the **clean yaw SD +0.57 deg/s** (z +7.1), Holm-called at m 5: SOMETHING other
          than the chordotonal mean separates the cycle from the round-2 transducer. But an independent Opus
          skeptic pass (verdict **mostly sound**, 16 corrections applied; not one of the 15 family verdicts flips)
          shows **"the per-leg / per-phase STRUCTURE contributes beyond the LEVEL" is NOT yet supported**: L differs
          from C in three ways at once -- per-phase modulation, a **+9.7 / -24.8 Hz** hair-plate / campaniform
          mismatch whose dominant route into DNa02 is **sign-negative** (`SNpp45 -> IN13B001 -| AN04B003`, and an
          additive level model with an inhibitory hair plate reproduces the whole AN04B003 gap with a zero structure
          term), and a **+13.6 Hz DC chordotonal L-R**. DNa02_L is the robust row (L's left side is over-driven on
          all three channels and still fires less); straightness reads as sidedness (82-92 % of it is the drift);
          DNa02_R is partly an 8.3 Hz per-side level deficit. The level, sidedness included, reaches **76 / 67 /
          89 %** of the cycle's DNa02_L / DNa02_R / yaw-SD rise. **Nothing adopted, no default changed.**
    - [x] **the UNSIDED level control -- RUN** (control (a) of `level_matched_control.md` 7.1): the round-2 law
          reading the side-MEAN leg-MN rate on every cell at `mn_ref_hz` 8.84. Removes the +13.6 Hz DC chordotonal
          L-R and the +9.2 Hz hair-plate L-R at the same means; the control that separates the round-2 law's
          sidedness from its level.
          **Outcome** (`vncd5-2ffbc3`, `docs/audits/level_controls.md` F1): U v L is `null` on DNa02_L, DNa02_R and
          the clean yaw SD while straightness goes +0.231 and the drift halves (+3.48 -> +1.74 deg/s) -- the DC bias
          owns the drift and the straightness and **none of the DNa02 rate**.
    - [x] **the MODULATION-ONLY cycle arm -- RUN** (control (b)): the cycle's per-phase law with the per-leg
          amplitude held at 1 (no turn kinematics in the afferents), `body.LegCycle(flat_amplitude=True)` behind the
          opt-in `leg_cycle_flat` token.
          **Outcome** (F4, and NOT the clean contrast it was meant to be): amplitude 1.000 is above the law's
          realised **0.948**, so M bought **+6.0 Hz of chordotonal**; about 60 % of its AN04B003 excess over C is
          that level and the rest is not -- M's structure residual **exceeds** C's by +0.659 / +0.745 Hz (z +3.8 /
          +3.5) and on DNa02_L by +0.069 (z +4.0) -- so **"the amplitude law adds no drive" is NOT supported** and
          F4 stays open. What the turn term does own is the per-frame sided DNa02 signal (-0.334 C vs -0.107 M).
    - [x] **the CHANNEL-MATCHED level control -- RUN** (control (c), added by the skeptic and not in the audit's
          original list): the round-2 law at `mn_ref_hz` 8.84 with `hair_plate_max_hz` **86.71** /
          `campaniform_load_hz` **25.05**, derived on CPU as a fixed point of the afferent -> leg-MN loop; realised
          hair plate **44.34** and campaniform **24.78** against C's 46.36 / 24.91.
          **Outcome** (F2, F3): matching the two channels costs K **8.7 Hz of chordotonal** through the same loop
          (the predeclared side effect P1b), so K v L is `result` NEGATIVE, the pair settles nothing alone and
          C v K is an **upper bound**; the hair-plate route is sized from the level model (**-0.070 Hz/Hz**) and the
          single cell (**-0.149**) instead -- it **accounts for +0.72 Hz, 14 %, of the +5.22 Hz pooled C-over-L
          AN04B003 difference**, and the per-phase modulation owns the remaining ~4.49 Hz.
    - [x] **the next-round arm: M at the cycle's own realised amplitude 0.948 -- RUN** (`level_controls.md` 10 item
          2): `flat_amplitude` with the per-leg amplitude set to the cycle's realised mean (or an `mn_ref`-style
          compensation) instead of 1, so that M and C sit at ONE chordotonal level. As run, M buys +6.0 Hz and F4 /
          F6 cannot be read as level contrasts; at 0.948 they become the clean contrasts they were meant to be and
          the open question "does the amplitude law add drive?" gets an answer.
          **Outcome** (`vncd6-2ad71d`, `docs/audits/level_controls_r2.md` F1): the new opt-in
          `body.LegCycle.flat_amplitude_value` = 0.948 put M2 at C's level on all three leg channels (-0.76 / -0.41 /
          +0.03 Hz) with |amp L-R| 0.000000, and **M2 v C is `null` on all seven primaries** -- the amplitude / turn
          law adds NO drive at the relay or at DNa02 beyond the per-phase modulation, so **F4 is closed**.
    - [x] **the next-round predeclaration: a Holm family that can be satisfied -- DONE** (`level_controls.md` 10 item
          1): at n v n the exact-U floor is `p_floor(n, n)` and a family of m members can only be called if
          `p_floor x m <= alpha` -- **m <= 6 at 5 v 5** (0.0079365 x 6 = 0.0476) and **m <= 23 at 6 v 6**
          (0.0021645 x 23 = 0.0498). Round 4b declared m = 7 at 5 runs and called nothing, which `docs/INTERP.md`
          10.2 already forbade. Either size the family to m <= 6, or run **6 runs per arm**.
          **Outcome** (round 4c): **6 runs per arm** at m = 7, floor `2/C(12, 6)` = 0.0021645 and 0.0021645 x 7 =
          **0.0152 <= 0.05** -- rows were CALLED for the first time in this thread (F2 four of seven, F3 four,
          F4 all seven; F1 calls none, which is the answer).
    - [x] **derive the three-channel matched fixed point** -- round 4d (`level_fixed_point.md`), L3 at
          8.23 / 81.09 / 25.10 matches C on both sides; direct C-L3 relay +2.6687 / +3.8782 Hz, pooled
          +3.2734. M2-L3 +2.8521 / +4.0290; M2-C null on all seven primaries. The older level model
          leaves +1.2028 / +1.2344 Hz in modulation-free L3: its extrapolation overestimated the size.
          Nothing adopted. **Independent skeptic pass done** (Opus, 2026-09-17, mostly sound): all seven claims
          reproduced, no conclusion moved, eleven corrections applied in the audit. What it changed in the record
          is bookkeeping only -- the receipt time, two line-ending "source-stamp differences" that are not edits,
          the driver script missing from the 52-file fingerprint, and the rule-7 breach in commit ad2efc0.
    - [ ] **more runs of the REFERENCE arm for DNa02_R / the clean yaw SD** (`level_controls_r2.md` 9 item 6): both
          rows reproduce as DIFFERENCES in all three batches (DNa02_R +0.103 / +0.107 / +0.094 Hz; clean yaw SD
          +0.572 / +0.490 / +0.588 deg/s) and as verdicts only sometimes (DNa02_R `result` in rounds 4 and 4b at
          z +16.6 / +7.8, `null` at 6 v 6 at z +2.9; the yaw SD `result` only in round 4), because L's own between-run
          SD is the denominator every time (0.081 -> 0.172 -> 0.224 deg/s). A claim on either row needs more runs of
          **L**, not of C.
    - [x] **the single-cell AN04B003 check** (CPU, `interp_atlas`-style, no room run): AN04B003 under (i) steady vs -- RUN 2026-09-15 (`scripts/probe_an04b003_single_cell.py`, `docs/audits/level_controls.md` section 13): +1.63 Hz at a matched per-cell mean with IN13B001 clamped (z +7.2), hair plate -1.00 Hz per +6.7 Hz (z -5.9), campaniform null; reproduced bit-for-bit on an independent rerun.
          8 Hz-modulated chordotonal input at the same mean with IN13B001's rate clamped, and (ii) the hair-plate
          level varied alone at a fixed chordotonal level. The two together settle whether the C-over-L AN04B003
          difference (19.1 / 16.5 -> 23.1 / 23.6 Hz) is modulation or hair-plate disinhibition.
    - [ ] **the adoption-licensing run for the module** (next-round item 2; ONE submission, one block `fam_lic`):
          shipped vs `all+leg_cycle+haltere_sided` on `--sections hops` x 6 draws and the `batch_sustain` room
          take-off protocol x 6 batches at seed-matched seeds, plus the 29-check suite x 3 for the record (28 of
          29 checks cannot carry the sense) and the room ledger rows under the sided spec. Run it **only** if the
          level-matched control shows the phase structure matters; otherwise the module stays a module.
  - [~] **neuromodulator signs**: 3,312 presynaptic bodies (2.7 M synapses: dopamine, octopamine, serotonin,
        unknown) are silenced (sign 0). Receptor-expression tiers for DA / OA / 5-HT receptors per postsynaptic
        type (same sources as `receptors_by_type.csv`; `docs/NT_INTEGRATION.md`) would un-silence the arousal /
        locomotor-state system the animal's spontaneous walking depends on. Largest single missing input.
        **Round 3: the monoamine slow class measured** (`monoamine_slow_term.md`) -- inert at 0.02 (and FAIL on
        `taste.MN9_hz` on CPU), runaway at 0.2/1.0 additive, MBONs zeroed in gain mode; VNC targets have 0
        receptor rows. Nothing adopted; the parked module stays parked and `sign` stays the default. Next: split
        the class per transmitter (code, thread B/C), a KC>MBON plasticity module, VNC receptor rows.
    - [ ] **the monoamine class split** (next-round item 4; code first, then ONE submission): `receptor_signs`
          slow_class per presynaptic transmitter (DA / OA / 5-HT) and `LIFParams.slow_gain_by_class` /
          `slow_tau_by_class` with three keys, default None, CPU bit-identity test, separate E/I accumulators in
          gain mode; then 5 runs per arm -- off / DA 0 + OA gain 1-3 / 5-HT additive 0.02-0.2 -- with health, the
          plain-fly room, the suite on the GPU **and** the same sections on the CPU, and the compass rows under
          the OA arm. New source of signs for the same set: FAFB per-cell probabilities and BANC verified
          transmitters (`flywire_banc_survey.md` 4, `docs/NT_INTEGRATION.md` item 11) -- about 400 MaleCNS
          `unknown` cells carry a classical-transmitter prediction in BANC.
  - [~] **the compass at the defaults**: the ring is silent unless EPG/PEN/Δ7 gains are raised (experiment
        overrides); with them the bump persists but is pinned, heading-blind and steering-inert. Data question:
        the per-transmitter unitary strength (mV per synapse) and the CX receptor tiers; a bump in darkness
        (Seelig & Jayaraman 2015) is the ledger target. Rotation input needs the transducer above (GLNO's
        inputs are efference / proprioceptive territory; the visual route is direction-blind).
        **Round 3: the per-transmitter unitary is answered NO** (0/48 bumps at shipped gains under ACh x0.5-1.0,
        I/E 0.25-0.75; `unitary_strength.md`): a transmitter scale multiplies the tuned Delta7 inhibition and the
        untuned ring feedback by the same factor and cannot set the Delta7 : ring ratio. `LIFParams.w_syn_by_nt`
        is **kept as an opt-in instrument** (owner decision, session 11) and no bracket of it is adoptable.
        Owed: a type-level ring mechanism the data imply, and 4 seeds per compass arm so the rows are callable.
        **Round 5: answered NO on the type level, and nothing adopted** (`compass_ring_mechanism.md`, batch `cx5-5cde5e`,
        8 jobs / 0 failed / 48 runs; `glno_relabel.md`, batch `cx5b-d08be3`, 22 jobs / 0 failed; both skeptic passes
        mostly sound): at gE 1 / gD 1 no arm meets the "working compass" rule and every surviving bump fails on rate,
        the binding constraint being the ExR6 / ER6 / ER4m DC balance on PEN, not a missing ring mode. **GLNO ->
        glutamate is NOT adopted**: the suite half of the round-2 rule is met (27/0/2 in all six draws, no status
        change), the rate half is unrun, the source standard is unmet, every compass row is null and the signed
        self-turn report still stops at AN04B003. Nothing adopted, no default moved.
    - [ ] **the ACh-only unitary family** (next-round item 3; ONE submission, `--arm-block-map` so the family is
          ONE block): `w_syn_by_nt` acetylcholine x0.8 and x0.5 with inhibition x1, through the suite x 4 draws,
          the wedge compass x 4 seeds and the room x 4 runs with the transducer OFF **and** ON; `taste.MN9_hz`
          re-read as the re-calibration it is, not re-passed. Before submission, on CPU: pin Kazama & Wilson
          2008's primary EPSP (5 vs 7 mV), add the Periplaneta unitary I/E 0.28 as a ledger row, cite or relabel
          `unitary.IoverE.chloride_driving_force`, and fix `probe_unitary`'s rounding-before-compare.
    - [ ] **a type-level ring mechanism, or none** (next-round item 5): before any compass batch, a CPU structure
          pass (`interp_paths` / `structure.json`) naming what data-implied fact could change the Delta7 : ring
          ratio (receptor tiers on ER / ExR -> EPG, the GLNO transmitter, a conductance-based synapse). If none
          exists, park the compass at "no attractor at shipped gains" and run the free-walking compass room under
          the transducer at the experiment gains once (4 seeds, one block, bump metrics + `circ_corr_heading`) to
          close `body_sided_state.md` 8 item 5 -- expected negative: the report is gone by GLNO.
    - [x] **the ExR6 / ER6 / ER4m -> PEN, EPG hold arm** (2026-09-15, round 6: `compass_dc_balance.md`, batch
          `cx6-995cd5`, 10 jobs / 0 failed / 40 runs, skeptic pass mostly sound) -- **necessary but not sufficient**:
          the hold lifts the driven PEN from 0.29-0.66 to 40.2-48.3 Hz (`result`, Holm 0.0317) and buys no bump
          (survival 0.00, confined 0.000 in 5/5) because the same DC term also holds the unstimulated ring at rest, with
          ExR6 carrying most of it, ER6 some and ER4m `null` on PEN; the structure tool's defects were fixed first
          (`--legacy` reproduces 5A), and **nothing was adopted** -- `--hold-edges` defaults to `None`.
    - [x] **record per-type ring rates (ExR6 / ER6 / ER4m) in `cx_wedge`** -- closed by compass 6B
          (`compass_local_recurrence.md`): all 40 runs carry per-type rates and per-cell maxima; S ExR6
          41.3-42.8 Hz / ER6 23.1-23.9 post-pulse, below the fixed point's 107.7 / 51.9.
    - [x] **measure the LIF's effective input noise sigma** -- 6B: never-spiked relay membrane 4.638 mV,
          128 cells in two CPU seeds; state dependent. EPG 5.521 is a spiking-cell reading; input 11.800 is
          a quasi-static sensitivity scenario, not a measured upper bound. Scalar f-I calibration remains open.
    - [x] **the hold PLUS a wedge-local recurrence** -- 6B answers NO for these interventions: 0/5 joint
          working-compass successes in every arm. H3E is wide, high-rate and systematically offset; H3F
          nearly flat. Withdrawn: missing inhibition everywhere else; the far half is already near background.
          No default changed or intervention adopted. Independent completed notes and their limits are in the audit.
          **Round 7's skeptic pass sharpens the same answer** (2026-09-17): under the hold plus the GLNO relabel a
          hump DOES form -- vector strength ~0.72, ~160 Hz, ~3.9 wedges -- and does not rotate. In the three
          gate-passing runs the confined-frame centre slope is +0.137 / -0.086 / -0.069 wedges/s against an ideal
          +4.0, and the gate fails mostly on its `out_above <= 3` clause, not on an absent bump. So the open
          question is not "is there a bump" but "why does a bump that exists not follow the side signal".
    - [x] **ExR6's transmitter and receptor -- a relabel with sources, not a gain**: 2 cells, MaleCNS `nt` glutamate, -- ANSWERED by Astra (2026-09-16, `docs/audits/exr6_evidence.md`, merged ad4f9dd): ExR6 glutamate and ER6 GABA have direct EASI-FISH support (Wolff et al. 2025 eLife 104764 Fig. 9 source data: SS53617 vGlut strong, SS58833 Gad1 weak -- worksheet rows verified locally); the model's labels stand; peptide / monoamine co-transmission untested; the open question is receptor placement and kinetics at the EB/GA contacts (a 2026 Turner-Evans-lab preprint reports glutamate inhibiting E-PG but exciting P-EN in the PB -- not isolated to ExR6). Nothing adopted.
    - [x] **PS196_b, from the compass side too** -- ANSWERED by round 7 and its one follow-up; the remaining
          bump question moved to the instrumented-preset follow-ups at the end of this section.
          (2026-09-15, literature note in `docs/NOTES.md`): Wang's MaleCNS mining (`fly-circuit-exploration`, finding 3) names PS196_b as GLNO's largest input outside the ring (1,801 syn, 19-21 %; every count reproduced to the synapse in our cache) and FB3A as a PFNd input; rounds 1-4 reach the same cell from the body (AN04B003 -> PS196_b is where the self-turn report breaks; its L-R is unsigned under the Coriolis stop-gap). Owed before any compass room is called: PS196_b's afferent budget by side (AN07B037_a/_b 419 / 52, CB0675 / GNG580 / PS047_b) and a sided ascending-stimulus arm, so a signed PS196_b L-R exists to test whether GLNO carries it into PEN -- a labelled diagnostic, never a default. His "compass brake" (the EPG -> PEN write-position recurrence, 3 : 1 in hemibrain; not checkable in our cache, no per-ROI edges) is the other term of 6A's balance: both readings predict the saturated ring H3 gives. hDelta path integration (Janke 2025 hDeltaG / vDeltaE; Avritzer 2026 hDeltaA, ~7-10 s) goes on the expectation side once a bump exists.
          Withdrawn older continuation: ExR6 transmitter wholly UNKNOWN. The checked evidence is in the completed ExR6 item above.
          **Round 7 answered the afferent budget / sided-input item** (`compass_velocity_route.md`, cx8r):
          AN07B037 drives a signed PS196_b report and V-S GLNO L-R is +2.2183 Hz, result. HGV has 0/6
          eligible follow runs, so following/sign reversal are undetermined; PEN and DNa02 L-R are null.
          No compass room unlocked, no gain selected or default changed. The one predeclared GLNO-to-PEN
          transfer follow-up is complete (cx8t, audit section 6): direct 90 Hz forcing gives PEN L-R
          HL-HR +2.9939 Hz, result, while lowering PEN mean. The cut changes the baseline; the right-edge
          contrast has the opposite sign to prediction. Transfer at this strong challenge is established,
          but a stable following bump remains open. No second batch. **Independent skeptic pass done**
          (Opus, 2026-09-17, mostly sound): every quantitative claim in both documents reproduced, and it added
          two things the round did not claim -- cx8 and cx8r are bit-identical (6,966/6,966 metrics,
          1,560/1,560 arrays), a free replication recorded as a qualification of INTERP 10.4 item 2 for the
          cx_wedge protocol only; and cx8t's failed HR-CR sign is the operating-state shift (the cut moves PEN
          L-R +11.4 Hz, the challenge about -1.7), so the challenge-attributable part carries the predicted sign
          in both edge contrasts. No unitary transfer or receptor sign is inferred either way.
    - [ ] **the room rate-half for GLNO, if adoption is ever wanted** (`glno_relabel.md` 4.3(b);
          `guard_suites_r3.md` 4): the room take-off protocol on `out/cache_glno_glu` at **>= 6 runs per arm** in ONE
          submission (`guard_suites.sh`'s room jobs with `FLYVERSE_CACHE` / `--cache-dir`). The suite half is already met
          (27/0/2 x 6, no status change); this is the only half still runnable, and the source standard of the three
          existing `TYPE_NT_OVERRIDE` entries (a transcriptome or EASI-FISH profile) would remain unmet either way.
    - [ ] **the `struct.GLNO_PEN.sign` ledger row's polarity** (`flyverse/data/expected_responses.csv`;
          `glno_relabel.md` 4.6 item 1): the row is written `expected 0`, `op ==`, `bound 0`, `gap 1` with the note
          "known gap: the largest single input of PEN carries no sign and is silent in the model", so meeting `== 0`
          prints "PASS (gap closed)" for what IS the gap -- the polarity is inverted. It also has no `check_key` (the
          29-check suite cannot see it), and its companion `struct.GLNO_PEN.synaptic_pair_count` row is still MISSING a
          `model_reference` until `scripts/interp_paths.py` emits a per-type row for the GLNO -> PEN block; any GLNO
          adoption must re-score the sign row.
    - [ ] **the body-model question at PS196_b** (`vnc_drive.md` 6, `body_sided_state.md` 6; `glno_relabel.md` 4.2): the
          signed self-turn report's break is **AN04B003 -> PS196_b** -- AN04B003 is `result` (-11.2 / -11.8 / -11.5 /
          -11.5 Hz, p 0.0286 = the exact-U floor) in 4/4 arms and PS196_b is `null` in 4/4, its L-R the same sign in both
          turn directions. A **sided PS196_b** is what a signed GLNO needs to carry anything; until then the GLNO sign
          changes rates, not reports.
  - [ ] **saccade generator / signed steering command**: named missing by round 3
        (`round3_integration.md` 9): nothing in the wired graph produces a clean frame above 100 deg/s or a
        DNa02 sided rate that leads the yaw. Round 3's largest yaw numbers (12.8-13.4 deg/s under transducer +
        unitary-high) rest on 12-17 % of the window with every fly off the table and are a fixed one-sign bias
        in 80/80 fly-runs -- not steering.
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
      **Object round 3 closed all four owed items and changed no answer** (`object_samedevice_r3.md`,
      `object_rectangles_r3.md`, `object_localizer_r3.md`, `object_export_r3.md`; six batches, 0 failed, every
      original H200 and every fresh-seed replication B200, one host per batch): (a) the **localizer** ran and
      **neither LC population localizes under a static 4.5-deg probe on either lobe in either batch** -- LC11
      zero fits at the fixed z = 5 throughout, the B200 `fb0` blank-selected z* = 4 giving 4/143 against 1/143
      blank and still failing coverage and enrichment, with the Mi1 positive control on `optic_dr` and not on
      the `drive_mv` quantity the negative is measured in; (b) the **same-device re-run** is done on two GPU
      models -- H200 `REPRODUCES`, B200 **`PARTIAL`** (rectify T2 at 4.5 deg z +2.66825 under the z >= 3 gate,
      separation complete) -- and it also showed that the large-rung companion effect is a three-batch effect
      and that the **base arm itself crosses the gate on the B200 and not the H200**; (c) the **rectangle
      ladders** ran in two batches, **40/40 primary verdicts `null`**, no animal-shape call, with retinal
      contrast matched only at width >= 8.8 deg and the upstream size-monotonicity call a two-directional
      knife-edge; (d) the **compare arms are exported** (41 of 95 directories in
      `out/export/objr3_index.json`; 1,712 tables, 0 problems, four native B200 replication Results linked
      under `replication_evidence`). **Nothing adopted, no default changed.** Still open, and both are protocol
      decisions rather than owed work: a **separately declared moving-probe RF assay** with blank controls and a
      smaller, level-matched probe, and **more LC10a runs under a new declaration** to settle the width-15 row
      the two batches sign-reverse on. `docs/NEUROME_INTERFACE.md` 3d.
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

**Instrumented-preset follow-ups (2026-09-17, from the four independent skeptic passes).** All five are
questions, not plans; none of them is authorized as a batch yet, and none licenses an adoption.

- [x] **A goal-only plume arm, with no DNa02 feedback.** Mean |demand| did not rise between
      the diagnostic and the validation (0.057-0.074 -> 0.036-0.083) while mean |yaw| tripled (0.038-0.067 ->
      0.106-0.245 rad/s), so the feedback bridge looks like the load-bearing change -- but that is an inference,
      and a goal-only arm is what would measure it. Same rooms, same seeds, feedback off.
      **ANSWERED by round 8 item 4** (`docs/audits/plume_goal_only.md`, batch plume-go, 36 runs): the inference was
      wrong -- the walking goal is the load-bearing part. Shipped rooms, six runs per arm: feedback-only (upwind goal
      closed on DNa02) feeds 0.17 +/- 0.41 rows of six with the LARGEST DNa02 |L-R| (1.97 Hz); goal-only (walking goal,
      integral gain 0) feeds 4.67 +/- 0.52 at 0.62 Hz; full 6/6 in every run. full - feedback-only +5.83 rows (result);
      full - goal-only +1.33 rows (null under the reference-SD rule, all runs separate) -- the feedback's share is the
      fly that starts facing away (185 deg: goal-only 0/6 runs, full 6/6). Drawn starts: 6 / 4.33 / 4.00 (two of the
      six drawn rows start on a fruit). `plume:feedback=0` / `plume:walking_goal=0` are opt-in, recorded in describe();
      bare `plume` unchanged. No admission claim. The grammar is one after the merge of Astra's branch:
      `plume:feedback=off` is the canonical spelling of the gain-0 arm and `plume:feedback=0` the accepted synonym,
      beside `plume:bilateral=orn` and `plume:walking_goal=0`.
- [x] **A transduced-contrast plume law.** ANSWERED on the CPU (Astra, `plume_transduced.md`, merged ca23768 after an independent skeptic refuted-then-fixed a weight/index pairing bug): the model's own ORNs deliver the rooms' 0.26-1.1 % contrast at SNR < 1 (0.16-0.34), correct sign in 0.667 of windows; the `plume:bilateral=orn` and `plume:feedback=off` options ship opt-in. Still owed: the 18-room v3 batch (full / transduced / goal_only x 6 seed-drawn starts; frozen, `predeclared_v3.json`) once the GPU pool is free, quoted as mean +- across-run SD only. The shipped correction reads the bilateral contrast off the physical
      concentration field (noise-free, pre-transduction) at a gain of 200, and the observed contrast is
      0.26-0.57 %. Through the model's own ORN law (1 + 150c/(c+0.5)) that is ~0.1 Hz per ORN, under the Poisson
      noise of the 0.25 s window. The question: does a law that reads the ORN RATES still find food? If not, the
      200x gradient is doing the navigating and the instrument should say so in one sentence.
      CPU implementation/characterization: `plume:bilateral=orn`, [audit](docs/audits/plume_transduced.md).
      The 18-run full/transduced/goal-only comparison is prepared, not submitted; await pool release -- and the v3
      freeze now refuses on source hashes after the round-8 merge changed `flyverse/navigation.py` and
      `flyverse/instruments.py`, so it needs a v4 re-freeze (Astra's) before submission.
- [x] **A V- arm for the GLNO sign.** Round 7 declared no sign-flipped counterpart of V, and in the held arms
      GLNO L-R does not reverse (+0.6862 +- 1.9132 Hz, positive in four of six seeds), so primary 3's sign
      specificity is untested where it was measured. One arm: the afferent at k 0.5, sign -1, no hold, no relabel.
      **ANSWERED by round 8 item 2** (`docs/audits/compass_sign_control.md`, batch cx9, 18 runs): V- minus S GLNO
      L-R -2.0665 +/- 0.6461 Hz (result, Holm p 0.0065), V minus V- +4.2848 Hz (result), PS196_b L-R -36.1679 Hz
      (result), all predicted signs, all six seeds separated; the flipped afferent drives GLNO_R (2.09 Hz) and leaves
      GLNO_L at S's exact values seed for seed. The unheld ring's side report follows the afferent's sign; the held
      arms' non-reversal (HGV-) stands and is the remaining sign question. cx9's S and V equal cx8r's in 76/76 rows
      across two nodes. Nothing adopted.
- [x] **A deterministic-kernel gate before any room number is quoted again.** The native event path is not
      reproducible run to run: per-cell rate differences to 44.9 Hz (voltages to 77.6 mV, conductances to 247)
      under bit-identical `poisson_p` and `drive`. Either an exact-workload pass, or the torch-sparse path for
      the quoted rooms, or repeated draws with the spread reported -- until one of those, first-contact times and
      end energies are one draw and not a number.
      **ANSWERED by round 8 item 1** (`docs/audits/determinism_gate.md`, batch det1): the cx_wedge protocol (B=1
      FlyBrain, torch path, no world) repeats exactly on one B200 (31/31 arrays, 546/546 metrics); every B=6 room
      pair is one draw on BOTH paths (first differing frame 4 / 30 native, 86 / 60 torch). Frozen decision: rooms
      are >= 6 draws with the run as the replicate unit and nothing quoted beyond its across-run SD; every room
      number quoted before the gate stays one draw. Still open: which room stage (batched SpMM, optic lobe, ray
      tracer, readout) does not repeat -- two more pairs (FlyBrain B=6 without optics; optics with a frozen body)
      would split it. Tooling from this item: `cluster_run.py --gpu-ids` (a strict per-job pin) and `--ship`
      (the box checkout is behind origin/main after the history rewrite; without `--ship flyverse,scripts` a
      batch runs the box's stale copy of every file outside the local diff), and the scheduler's `completed`
      status is no receipt (it lists a crashed job as completed with `exit_code None`).
- [x] **The 29-check suite under `instrumented`, three draws, with the three-instrument list**, beside the `raw`
      column, as PRESETS_SPEC section 2 item 5 requires. The compass stand-in's own run already rejected
      admission on `taste.MN9_hz` (seed 1, 1.690456 -> 4.295961 Hz, FAIL -> PASS outside the declared heading
      gap, rejected regardless of its favourable direction), so the expectation is another rejection; the point
      is to have the instrumented column on the record.
      **ANSWERED by round 8 item 3** (`docs/audits/instrumented_suite.md`, batches suite-inst and suite-inst-room):
      the expected rejection did not come. No row changes status in 3 + 3 draws (27/0/2, 26/1/2, 27/0/2 under
      both presets; taste.MN9_hz is raw's own instability, identical under the instruments; the compass row stays
      KNOWN GAP), and the room rate-half at six seed-matched runs per arm passes (110 vs 101 take-offs over 28,800
      fly-s each, one-sided exact p 0.291, run-level null). The list is ADMISSIBLE by the PRESETS_SPEC gate and
      nothing is adopted; a fewer-meals descriptive (instrumented 1-4 vs raw 0-7 per run, null at 6 v 6) is on the
      record. Tooling: `benchmark.py --hold-edges / --nt-override` (instrumented only), `instrumented_room.py`;
      the benchmark adapter's drive setter no longer assumes a scheduler module (the first submission lost eight
      checks per instrumented draw to that).

Most useful experimental data, ranked by leverage: (1) receptor / conductance profiles for the unprofiled
types and for DA / OA / 5-HT receptors; (2) per-type baseline firing in behaving flies (DN / AN / CX imaging);
(3) unitary synaptic strengths by transmitter (mV per synapse; the global scale is the one number every
attractor depends on) (round 3: one insect unitary I/E on record, 0.28, Periplaneta -- J Neurosci 34:13039; none
for Drosophila, and no Drosophila central fast IPSP at all); (4) proprioceptor firing ranges (Mamiya 2018;
Agrawal 2020) (round 3 needs a walking-mean FeCO rate; the 10-150 Hz bracket lets the level move 4x, and the
level is what the round could not separate from the phase structure); (5) behavioural
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
- [ ] **Cluster tooling the round-3 process debts name** (next-round item 7; no GPU, `docs/INTERP.md` 10.4 items
      13, 15, 19, 21): a `cluster_run.py --attach <run>` mode so a new client can resume the wait / fetch of an existing
      run (four of six behaviour batches lost their client); the scheduler's own completion receipt fetched as a
      file into `out/<dir>/` instead of trusting the client console; a guard that refuses (or reddens and
      requires `--confirm`) an `--arm-block` that yields more blocks than jobs/2 or any block of size 1, plus a
      family token that is the same string on every arm; `fetch_run.py` per-file **sha256 receipts** (hash
      computed on the box before transfer, compared locally) so an audit may write "verified"; and a
      verdict-agreement script that diffs two analysis CSVs and prints the count and the flipped keys verbatim.
- [ ] Batch-sustain / probe JSON headers: done for `batch_sustain.py`; do the same for every probe still
      writing `options.device = None`.
- [ ] **`benchmark.Context.new_brain` provenance records carry `execution.host`** (2026-09-17 code skeptic,
      claim 2): it now builds a `provenance()` record per legacy section even under `raw`, and emits a new
      `controllers` array. Brain construction is unchanged and no default moved, but the records carry the
      execution host, so **those JSONs must not be committed raw** -- strip or placeholder `execution.host`
      (and any path-bearing field) in whatever writes them to a tracked file, and add a test that a committed
      benchmark JSON carries no host.
- [ ] Observatory UI: NT readout of the `health` tool; a "why did it do that" panel that runs `decompose` on
      the current frame's DN inputs.

## E. Nice to have for the announcement

- [ ] A results page (artifact / docs site): the benchmark table, the three deficit maps, the toolkit's
      stage map for the object, the compass-in-the-room figure — every number with its audit link.
- [ ] A 5-minute "run the fly" notebook (CPU subset: antennal lobe + MB + central + DN, smell → turn).
- [ ] Contributing guide: the project rule in one paragraph; how to propose a default change (audit +
      skeptic + suite run); how to add a ledger row.

## F. Cross-connectome: FlyWire FAFB v783 and BANC v888 (female brain; female brain + VNC)

Survey: `docs/audits/flywire_banc_survey.md` (2026-09-14); implementation spec: `docs/CONNECTOME_BACKENDS_SPEC.md`
(Astra, 2026-09-14: items 1-3 and the BANC walking replicate delivered on `feat/connectome-backends`, merged as cfaa694.
Headline: the female CNS walks straighter than the male at the shipped defaults (clean yaw SD 0.28 vs 2.64 deg/s,
DNa02 silent bilaterally); the leg-cycle yaw increase replicates qualitatively (0.38 -> 2.65 deg/s, `result`
within-dataset) but the neural pattern does not (BANC DNa02_L stays silent, PS059 ~0 Hz vs 20 Hz) -- descriptive,
not a sex test; `docs/audits/connectome_backends.md`.) Both releases are public; type names overlap MaleCNS
exactly for 59 % (FAFB) / 72 % (BANC) of MaleCNS cells and `type_aliases.csv` already bridges the rest.

- [x] `scripts/cross_connectome.py` (merged cfaa694): every anatomical claim of rounds 1-3 (DNa02's inhibitory budget, PS049 /
      PS059 / VES051 / AOTU019, AN04B003 and LT51 excitation, IN12B014's symmetric contralateral pair, the
      PS196a -> PS059 loop, the haltere-afferent route, LC11 / LC10a inputs) printed MaleCNS / FAFB / BANC side by
      side with a per-release synapse scale (counts run ~1 : 0.6 : 0.3). CPU, a day. The README's "not a
      reconstruction artefact" table.
- [x] `Connectome.load(dataset="fafb" | "banc")` (merged cfaa694; review `docs/audits/connectome_backends_review.md`, merge with fixes B1-B4 applied; nits 2/4/5/6/7/8/10/11 resolved in Astra's acdfd91, reviewed in `connectome_backends_followups_review.md` and merged 8050971): `root_id` -> `bodyId` (int64 fits), alias-normalised types,
      vocabulary maps for superclass / NT / side / neuropil; the receptor table and ledger transfer by type name.
      BANC = the female CNS as a whole-animal replicate of the walking result (either outcome is a finding);
      FAFB = the complete female optic lobe. ~2 days; the biggest scientific payoff and the best release story.
- [x] NT sources 4 and 5 for `docs/NT_INTEGRATION.md` (section 8, merged cfaa694): FAFB per-cell probabilities, BANC verified transmitters
      (65,369 cells). Conflict rows to add: PFL3 (ACh in MaleCNS / FAFB, TYR predicted in BANC; PFL2 verified
      tyramine), Delta7 (`glutamate,serotonin` verified), LAL074; ~400 MaleCNS `unknown` (silenced) cells carry a
      classical-transmitter prediction in BANC; MaleCNS `serotonin` splits SER / DA / tyramine across sources.
- [x] FAFB `column_assignment` as ground truth for `trace.column_of_cells` (2026-09-15, `docs/audits/column_ground_truth.md`,
      skeptic mostly sound): columnar types recovered to one ommatidium (T2 92 %, T3 97 % within 4.6 deg; chance 0.1 %);
      LC11 / LC4 get a single column from a wide-field partner at the permutation distance from their input field -- the
      round-2/3 LC windows were mis-centred, but re-windowing the same runs on the input-weighted centroid reproduces the
      LC11 null (caveat added to object_baseline_r2.md and object_rectangles_r3.md). Rule: never window a visual_projection
      cell on that single column. The transform hex1 = q + 18, hex2 = p + 20 is in (1,581 columns; mirror <= 1.6 deg; T4 offsets cos >= 0.98; DRA strict rim check FAILS 100/126 and is recorded as an expected failure) -- still to use it as ground truth for
      `trace.column_of_cells` and the LC anatomical windows; LC11 / LC10a themselves are not column-assigned there.
- [ ] **A female fly that sees and walks (BANC).** BANC's optic lobes are reconstructed and already simulating as LIF
      cells (72,574 `ol_intrinsic`, the optic -> central -> VNC chain wired) but it has NO column map and NO R1-R6 at
      all (its lamina's presynaptic partners are Tm3 / Dm6 / C2 / L5 -- feedback only). Path, in order: (A) recover
      BANC's own hex lattice from its columnar tiling (Mi1 / L1 / L5 / T4 one-per-column; nearest-neighbour graph via
      shared partners embedded in 2D), assessed against the four spec-2.5 checks (T4a-d
      offsets, L/R mirror, DRA rim, ~800 columns/side); right eye only (the left is under-reconstructed: Mi1 560 vs
      878); then (B) add the stereotyped R1-R6 input layer through `Connectome.extend` (negative bodyIds,
      `dataset=synthetic`, column-local neural-superposition cartridge onto L1/L2/L3) -- the only invented part,
      auditable. NOT a FAFB chimera: the two animals share zero body IDs and the optic -> central seam would have to
      be fabricated by type matching, which is the hypothesis a two-connectome comparison exists to test.
      Astra's candidate reconstruction is in `docs/audits/banc_column_reconstruction.md`: 877 right-eye sites,
      T4 direction check passes, DRA fails, anatomical L/R mirror unavailable. **A remains partial.** The owner
      re-scoped **B as an explicit experiment**: `load(dataset='banc', vision='candidate')`, right eye only,
      synthetic R1-R6 through `Connectome.extend`, default BANC unchanged. One house comparison passed all eight
      preferred motion directions and loom→GF; T5 modulation is weak and BANC TTM stayed silent. See
      `docs/audits/banc_candidate_experiment.md`. This is a **synthetic input layer on a candidate lattice**,
      not anatomical validation or a walking improvement. `scripts/recover_banc_columns.py` remains diagnostic.
- [ ] Do not: mix counts across releases unscaled; treat BANC's optic lobes as complete (T2 853 vs 1,466);
      prefer BANC predicted monoamine labels to its verified column; assume `fru` / `dsx` circuits are sex-shared.


### Navigation instruments follow-up (2026-09-15, experimental only)

- [x] Plural `--instruments`, composition/dependency errors, recurrent ring experiment, published PFL3
      comparator, explicit metabolic gain, wing-MN power/steering control. Raw default unchanged.
      Evidence and limits: `docs/audits/navigation_instruments.md`; independent review done (Opus,
      2026-09-17, mostly sound for all three navigation audits; quoted in each).
- [ ] Validate plume return with controlled odor/wind ablations and multiple neural draws before
      claiming robust food finding. After the explicit bilateral-goal/feedback correction, six
      of six compass and five of six ring rows feed in one 60 s family; no admission claim.
- [x] Close the optional PFL3 -> DNa02 steering bridge with feedback and use bilateral smell for
      local walking goals. Scalar room demo reaches food; `docs/audits/plume_steering.md` records
      the unverified laws, controlled neural checks, all trajectories and remaining failures.
- [x] Interrupt the optional flight servo for low reserves and strong neural odor; bound its bouts
      and allow ground search. Engineering policy only: `docs/audits/flight_foraging_priority.md`.
- [ ] Source/validate neural flight-state and landing control, altitude/visual/tilt controls and
      flight energy costs as explicit instruments/model proposals. No precision landing is supplied.
- [ ] Declare a recurrent velocity-calibration study; EB=0 is a textbook counterfactual, 2.7 is a
      count-ratio brake diagnostic, neither identifies biological effective coupling.
- [ ] Obtain reproducible runtime profiles before kernel fusion; native full-brain exact-workload
      matching failed even with matched forcing. Separate UI, sensory, module and brain timings.
