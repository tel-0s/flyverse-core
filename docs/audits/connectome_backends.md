# Connectome backends: FAFB v783 and BANC v888

Implemented on `feat/connectome-backends`, based on `f9e9fea`. The loader preserves the MaleCNS model and cache; female graphs use independent annotations and no MaleCNS transmitter overrides.

## Reproduce

```
python scripts/fetch_data.py --fafb --banc
python scripts/check_connectome_backends.py
python scripts/cross_connectome.py --out out/connectome_backends/anatomy
python scripts/audit_connectome_sex_labels.py --audit-table docs/audits/connectome_sex_matches.csv
python -m pytest tests -q --ignore=tests/test_cuda.py --ignore=tests/test_metal.py
python scripts/probe_vnc_drive.py plan --family body --dataset banc --runs 5 --name cbwalk --dir out/connectome_backends/walking
```

GPU checks and walking runs use the house cluster. Results live under `out/connectome_backends/` (ignored); `acceptance.json`, `anatomy/report.json`, and the simulation results carry dataset/release and model provenance. Source file URLs and SHA-256s are in the public data manifest. All nine download URLs returned 200 and their ETags matched local file MD5s; the FAFB neuron table also matched a streamed SHA-256.

## MaleCNS compatibility

The baseline at f9e9fea was **338 passed / 19 skipped**. With backend tests added, the CPU suite is **357 passed / 19 skipped**, including the unchanged recorded golden in `test_bit_identity.py`. A fresh MaleCNS compilation produced an exactly equal neuron DataFrame and complete legacy fingerprint. No cache file was rewritten.

| File | MD5 before and after |
|---|---|
| neurons.parquet | `c50c598a708b5b373cbaffca7d6a9d82` |
| W_post_pre.npz | `ac131529cebf98decde58d0c227b7954` |
| sign0_counts.npz | `bf01d724acf2a1fec8fdb60ef8a9e066` |

The CSR fingerprint remains `ef23cc27bea13be7f6a96f3c04fd3737` (167,106 neurons, 25,578,600 stored entries). To satisfy spec 4.1, the **legacy fingerprint dictionary is unchanged**, including its key set. Female fingerprints carry dataset/release plus the compile manifest; `provenance().model` carries dataset/release for every graph. This is the explicit compatibility interpretation of sections 1.3 and 4.1.

## Compiled graphs

| Dataset | Neurons | Stored pairs | Raw synapses | Observed compile time |
|---|---:|---:|---:|---:|
| FAFB, pairs >= 5 | 139,255 | 3,732,460 | 50,666,648 | 14.0 s |
| BANC, pairs >= 3 | 157,789 | 3,036,600 | 23,553,107 | 7.0 s |
| FAFB, no threshold | 139,255 | 19,773,733 | 76,944,499 | 25.8 s |

Times are single observations on a shared desktop, not performance comparisons. Per-neuropil rows are summed before the compiler min_weight filter. The input files have more rows than the resulting unique pairs. Raw synapse totals use float64 accumulation (float32 console sums can round whole-graph totals). Sign-zero contacts remain explicit CSR entries and their original counts are saved alongside W.

**BANC count correction:** the source has 158,262 rows. The specified exclusions remove 458 glia, 8 trachea and 7 non-neurons, leaving 157,789. Its 11,499 rows with no superclass are retained, including 97 with primary types. The spec estimate of ~147k would require an additional exclusion it does not specify. These cells remain unclassified in the source vocabulary and are included in all budgets.

| NT | FAFB | BANC |
|---|---:|---:|
| acetylcholine | 74,204 | 89,248 |
| gaba | 14,650 | 21,178 |
| glutamate | 17,380 | 24,443 |
| histamine | 11,153 | 6,092 |
| dopamine | 467 | 7,601 |
| octopamine | 41 | 1,537 |
| serotonin | 696 | 1,496 |
| tyramine | 0 | 173 |
| unknown | 20,664 | 6,021 |

The BANC verified-first co-transmitter rule and FAFB score threshold 0.5 are recorded in each manifest. Glycine/nitric-oxide-only verified labels remain unknown; tyramine stays sign zero without an invented receptor response. The source labels and full verified strings remain available. See `docs/NT_INTEGRATION.md` section 8 for conflicts, including PFL3, PFL2, Delta7, LAL074, and FAFB PS059.

## Alias and selection coverage

Normalization uses exact/alias rows in `type_aliases.csv` and its caveat flags. Exact names take priority; unresolved multiple targets retain the source name. PS196a/b and PEN_a/PEN1, PEN_b/PEN2 have explicit notation rows. Generic R7/R8 use the existing unclear-subtype aggregate, selected by a documented `backend_primary` CSV flag. DNg02 subtype splits are not collapsed.

| Female graph | MaleCNS cells represented before aliases | After aliases | Unresolved source names |
|---|---:|---:|---:|
| fafb | 99,203 / 167,106 (59.37%) | 132,239 / 167,106 (79.13%) | 257 |
| banc | 120,413 / 167,106 (72.06%) | 148,340 / 167,106 (88.77%) | 202 |

Every ambiguous source name and candidate list is in the cache manifest. `type_presence.csv` and `type_absence.csv` enumerate counts and missing matches in both directions. **A missing type match is not established sex specificity.** Different naming, reconstruction coverage and true sex differences are all possible; no fru/dsx circuit is assumed to be sex-shared from this join.

The separate [sex-annotation table](connectome_sex_matches.csv) uses the original MaleCNS `dimorphism` and `fruDsx` columns without changing its compiled neuron table. **1,258 cells across 266 types** carry the explicit `male-specific` source annotation; **263 of those type names are absent from both female graphs**. All names and counts are listed in the CSV. Three annotated names also occur in BANC: AN08B059 (5 annotated MaleCNS cells; 2 BANC cells), IN14A006 (2 annotated of 6 MaleCNS cells; 6 BANC), and INXXX217 (2 annotated of 10 MaleCNS cells; 11 BANC). These are annotation/name conflicts or subtype distinctions to resolve, not evidence that their circuitry is sex-shared.

The reverse list contains **2,310 female names with no MaleCNS match**. The female tables lack the corresponding structured dimorphism/fruDsx columns, so these rows are explicitly labelled **sex specificity unestablished**. This does not certify female-specific types from absence alone. `sex_labels/report.json` records the annotation source hash and provenance for all three graphs.

| Selected proprioceptive channel | MaleCNS actual selection | BANC |
|---|---:|---:|
| chordotonal (including leg category) | 615 | 985 |
| hair plate | 113 | 320 |
| leg campaniform | 12 | 61 |
| haltere | 201 | 392 |

These are the transducer selections after class and nerve filters, not counts of all cells sharing a subclass string. The handoff table mixes those definitions: e.g. MaleCNS has 426 campaniform-labelled cells but 12 in the selected leg nerves. BANC motor subclasses are fl 139, ml 126, hl 128, wm 64, hm 25. Those 64 `wm` cells reach the *selected* wing groups only through the alias table: BANC names them `b1`, `hg1`, `DLM1-4`, `DVM1a-c` and so on, without the MaleCNS ` MN` / `DLMn` notation, so before the 21 wing-MN alias rows the steering and power groups came back silently empty (connectome_backends_review B1). With them BANC gives steer_L/steer_R/power **16 / 16 / 24**, the MaleCNS counts exactly; `wing_groups` now raises `NotAvailable` naming the dataset rather than returning an empty group whenever `vnc_motor`/`wm` cells carry type names it does not recognise. BANC's `iii4` has no MaleCNS counterpart and stays unmapped. Sides use source soma side, with single-sided nerve fallback; multi-nerve names retain all mapped codes. No rate law or gain changes.

## CPU and optic acceptance

Both female graphs ran a deterministic 200-cell DNa02/input subset for 100 ms on CPU with finite state. BANC also ran FlyBrain with optic=None and the body/proprioception smoke with two flies. Vocabulary, alias ambiguity, duplicate-pair summation, sign-zero recovery, subset/extension save-load, cache namespace guards and unavailable capabilities have synthetic tests. External-cache checks carry the existing data marker.

| Receptor tier | FAFB stored pairs | BANC stored pairs |
|---|---:|---:|
| fallback | 2,222,685 | 2,459,054 |
| pre_unknown | 383,458 | 943 |
| nt_class | 0 | 0 |
| class | 116,403 | 74,290 |
| fuzzy | 145,426 | 84,448 |
| alias | 32,596 | 17,780 |
| exact | 831,892 | 400,085 |

`regions.labels()` assigns every cell:

| Region | FAFB | BANC |
|---|---:|---:|
| optic | 89,024 | 74,419 |
| central | 25,281 | 37,443 |
| visual_projection | 8,206 | 7,793 |
| mushroom_body | 5,608 | 4,661 |
| mechanosensory | 3,580 | 10,070 |
| antennal_lobe | 3,433 | 4,137 |
| vnc | 2,286 | 16,490 |
| descending | 1,305 | 1,316 |
| gustatory | 532 | 1,460 |

The existing `vnc` region includes ascending neurons; FAFB's 2,286 labelled cells are their brain portions, not a reconstructed VNC. BANC optic-intrinsic cells run as LIF cells without the optic module. Direct retina/optic construction raises NotAvailable. FAFB VNC/proprioceptive/motor calls raise NotAvailable. Female sweet-taste input remains unavailable because the shipped taste table uses MaleCNS body IDs; no unsupported cross-animal ID transfer is made.

The FAFB affine transform is **hex1 = q + 18, hex2 = p + 20** in each hemisphere. Column hemisphere supplies hex_side while somaSide retains its source meaning. Annotated columns are preserved on R7/R8; unannotated R1-6 cells use the existing strongest hexed-partner vote. The retina retains all annotated columns, even those lacking a reconstructed R axon.

- **Columns:** 1,581 = 785 left + 796 right.
- **Mirror:** 772 shared hex coordinates; reflected viewing directions differ by at most 1.583 degrees (below the 4.6-degree interommatidial spacing). Eye centering reflects their slightly different coverage.
- **T4 directions:** per-postsynaptic Mi4 minus Mi1 input-centroid offsets agree with MaleCNS for all four subtypes on both sides; minimum cosine 0.9818. Per-cell normalization avoids a population-coverage bias in pooled centroids.
- **DRA orientation:** non-putative community-labelled R7/R8 cells have median dorsal heights 62.0/61.5 versus whole-grid 80th-percentile boundaries 53/54. Fractions on that dorsal band are 80.6%/78.1%; 26 of 126 labels are outside the flat band, which is not the same as being off a curved rim. Against the **local dorsal envelope at equal anterior coordinate**, median depth is one lattice row in both eyes, **118/126 are within two rows**, and **125/126 within four rows**. The distant exception, body 720575940624946103, is 22 rows below the envelope and obtained its column by the strongest-partner vote. All eight cells deeper than two rows are listed with source/depth in acceptance.json. Orientation and dorsal localization pass at the population level; strict localization of every label does not. No transform or annotation was changed to force a pass. The strict spec 2.5 (i) check -- every label on the rim -- therefore **fails** (100/126 on the flat height band; 118/126 within two rows of the dorsal envelope), and `scripts/check_connectome_backends.py` now carries that result in its `checks` dict as the recorded expected failure `i_dra_all_labels_on_rim`, asserted to be exactly `False`, so the gate tests the same thing the prose reports.

The full FAFB controller rendered one room sensory frame and stepped 10 ms on a house B200 with native CUDA kernels and warp sparse products. Brain and optic states were finite; `gpu/fafb.json` records the realised device and model. Batch `cboptic-cf1505`: 1 job, 0 failed.

## Cross-connectome anatomy

`cross_connectome.py` reads only the Connectome API and its raw-count accessor. It generates 491 side-resolved edge rows, 100 NT budgets and 300 ranked input rows, including named inhibitory/ascending/haltere paths and LC11/LC10a inputs. Missing populations are distinguished from a present population with no retained edge.

DNa02 total raw input is 48,125 / 28,497 / 13,536 synapses (MaleCNS/FAFB/BANC), giving a descriptive anchor scale **1 : 0.59215 : 0.28127**. Fractions are normalized within each release. This scale is not a universal correction and is never used to tune the dynamics.

| Connection (L/R, anatomical counts) | MaleCNS | FAFB | BANC |
|---|---|---|---|
| PS049 -> DNa02, ipsilateral | 492 / 528 | 395 / 414 | 132 / 139 |
| PS059 -> DNa02, ipsilateral | 476 / 522 | 311 / 351 (fast sign 0) | 104 / 115 |
| AN04B003 -> DNa02, ipsilateral | 425 / 339 | named VNC population absent | 56 / 133 |
| IN12B014 -> DNa02, contralateral | 112 / 112 | named VNC population absent | 58 / 59 |
| PS196_a -> PS059, contralateral | 331 / 306 | 325 / 363 | 245 / 183 |

FAFB PS059 has no per-cell NT label and score 0 (GABA probabilities 0.37-0.44): its anatomical edges remain stored, with fast sign zero under the specified rule. Anatomical replication alone therefore does not guarantee functional replication.

## Walking replicate

Batch `cbwalk-384afd`: five house jobs, each with five body arms and the plain-walking probe (30 simulations total), five independent process seeds, 16 flies per run, 60 simulated seconds, shipped parameters. All arms of a seed share its job, with rotated order across seeds. `--arm-block fam` records the seed blocks; all jobs are in one submission. Compass and benchmark protocols are explicitly omitted from this walking-only plan because their retinal/wedge capabilities need separate validation.

**All five jobs and all 30 simulations completed with exit 0.** Recordings are fetched. Regenerate the comparison with:

```
python scripts/summarize_connectome_walk.py --malecns /path/to/historical/out/vncd3
```

The historical MaleCNS comparison is **four runs, seeds 0/1/3/4, from vncd3b-c2eeaf**. Its seed 2 belongs to a different submission and is excluded. BANC has five runs, seeds 0-4, from cbwalk-384afd. Means below are means of the 16-fly run means; uncertainty is **SD across runs**, not across individual flies. Clean yaw uses the existing body audit's mask after the first 5 seconds, removing airborne/edge-crossing frames. Both cohorts use the same analysis function. JSON sources and body recordings are hashed in `walking_comparison/report.json`.

| Body arm | MaleCNS clean yaw SD, deg/s | BANC clean yaw SD, deg/s | MaleCNS straightness | BANC straightness |
|---|---:|---:|---:|---:|
| A: shipped | 2.641 +/- 0.146 | 0.275 +/- 0.004 | 0.9943 +/- 0.0007 | 0.9984 +/- 0.0001 |
| B: all proprioception | 3.327 +/- 0.091 | 0.381 +/- 0.003 | 0.9786 +/- 0.0013 | 0.9956 +/- 0.0001 |
| C: B + leg cycle | 7.837 +/- 0.127 | 2.651 +/- 0.161 | 0.8249 +/- 0.0371 | 0.8655 +/- 0.0172 |
| D: C + sided haltere | 7.731 +/- 0.205 | 2.833 +/- 0.137 | 0.8693 +/- 0.0405 | 0.8519 +/- 0.0143 |
| E: D + Coriolis control | 7.915 +/- 0.202 | 2.709 +/- 0.107 | 0.8696 +/- 0.0143 | 0.8614 +/- 0.0091 |

Within **each** dataset, B-A and C-B are `result` on clean yaw and straightness under common.compare (BANC p=0.00794; MaleCNS p=0.02857). D-C and E-D are `null` on both measures in each cohort. BANC C-B yaw increases by 2.269 deg/s; its enormous z=655.9 reflects B's tiny run SD, not a corresponding biological effect size. All BANC body arms retain 100% clean walking frames, with zero hops or table departures. Shipped BANC path length is 0.49494 +/- 0.00002 m over 60 s.

**The qualitative increase in turning with the leg-cycle input replicates; the MaleCNS neural activity pattern does not.** BANC DNa02_L and DNa02_R are both silent in A/B. Under C, L stays silent and R reaches 0.0985 +/- 0.0133 Hz (D 0.1132, E 0.1035). MaleCNS C is L 0.5421 +/- 0.0208 / R 0.3792 +/- 0.0210 Hz. BANC C-B DNa02_R is formally `undetermined` under the project's rule because the baseline SD is zero, despite p=0.00794; the absolute change is reported rather than inventing a z. Both BANC DNa02 cells and all four PS059 cells exist with sided annotations, so silence is not an empty-selection artefact.

The BANC leg L-R mean is negative (A -0.0552 Hz, C -0.3792) versus positive in MaleCNS (A +0.1515, C +0.1395). BANC AN04B003 reaches L/R 10.74/10.27 Hz under C from 0.021/0.116 under B; chordotonal command rises 14.37 -> 84.78 Hz. Its PS059 remains almost silent (C L 0.00087 / R 0 Hz), unlike MaleCNS C 20.06/16.77 Hz. These observations do not establish a shared cancellation or steering mechanism; the level-versus-phase confound of the original leg-cycle experiment also remains.

The five **plain-walking** BANC runs independently give straightness **0.998427 +/- 0.000063**, path **0.494932 +/- 0.000018 m**, zero hops, and untrimmed yaw SD 0.2771 +/- 0.0039 deg/s. Their original table-exit counters are **invalid**: the existing probe interpreted `(xmin,xmax,ymin,ymax)` as half-extents and labelled every initial pose off-table. This branch corrects that reporter and records `metrics_version=2` for future runs. The summary retains and explicitly excludes the old counters; no replacement GPU batch was submitted. All 9,600 retained trajectory samples (0.5 s cadence) are on the table under the corrected bounds, while the body-arm recordings provide the valid full-frame table metrics above.

This is a **descriptive comparison across reconstructions**, not a causal sex test. Individual animal, sex, lab, annotation, synapse threshold/yield, optic capabilities and GPU generation differ (historical MaleCNS H200; BANC B200). Each statistical call stays within its own submission; no cross-dataset row pairing or p-value is used. No gain, default or sensory law was tuned to the female graph.
