# BANC right-eye functional experiment

Author follow-up to the independent [column reconstruction review](banc_column_reconstruction_review.md).
This is a **synthetic input layer on a candidate lattice**. The owner authorised this
experiment after the diagnostic merge; the anatomical validation gates remain closed.

## Predeclared model and acceptance

The pinned map is the clean `3deec1b` diagnostic, generated with NumPy 2.3.5,
SciPy 1.17.0 and no OMP/OPENBLAS/MKL thread override. Its LF-normalised CSV SHA-256 is
`60181451e52435ff31ca950d63d71833eb43740c470ab32d78721c2c74716e3d`.
It has 64 initial collisions and 136 displaced seeds. The full diagnostic checks,
independent controls and source fingerprints travel in the candidate metadata.
The review estimates **80–85% exact / 97–98% within one column** for BANC, with errors
concentrated at the rim and T4 home columns. The 94.38% FAFB score is the tuned k=6
optimum (k=5: 78.54%; k=7: 67.82%), not an estimate of BANC accuracy.

`load(dataset='banc')` remains a graph without optical columns. The explicit
`load(dataset='banc', vision='candidate', vision_cache_dir=<scratch>)` adds coordinates
only to mapped right-eye neurons and appends six negative-ID, `dataset='synthetic'`,
histaminergic R1–R6 nodes per cartridge through `Connectome.extend`. Each receptor in
a cartridge receives that column's radiance and connects to each mapped native
L1/L2/L3 there. This represents the neural-superposition cartridge, not six facets
pointing in different directions. Existing biological synapses are unchanged,
including stored sign-zero edges. Native mapped R7/R8 are retained; the left eye
gets no coordinates or direct visual input. Pruning all synthetic cells restores
the original BANC graph and its unavailable optical capability, even after reload.

The input weights were fixed before functional testing. MaleCNS right-eye native
R1–R6 edges have median raw counts L1=30 (2,180 pairs), L2=30 (2,167), L3=5 (2,004).
The BANC/MaleCNS raw DNa02 input-budget ratio is 13,536/48,125. Multiplying gives
**8.438025974 / 8.438025974 / 1.406337662** per synthetic receptor→target edge.
This is an **unmeasured synthetic assumption**, using a global sampling-yield ratio,
not a calibrated BANC lamina strength. No dynamics, gains or biological edge weights
are tuned. There are 877 cartridges, 5,262 synthetic cells, 12,762 synthetic edges,
and 865/687/575 mapped L1/L2/L3 targets. Six cartridges have no mapped lamina target.

One house/node1 submission runs `scripts/run_banc_candidate_gate.py`. It fetches and
compiles a fresh isolated BANC cache using the committed aliases, then runs these
four measurements sequentially on the same GPU:

| Dataset | Probe | Fixed conditions |
|---|---|---|
| MaleCNS | `probe_motion.py` | right eye, four directions, 60 deg/s, 30 deg period, contrast 0.5, 1.5 s/direction |
| BANC candidate | `probe_motion.py` | identical stimulus and model parameters |
| MaleCNS | `probe_loom.py` | right eye, sphere from right, seed 0, 1 m/s, radius 0.03 m |
| BANC candidate | `probe_loom.py` | identical stimulus and model parameters |

Both probes retain their own existing parameter defaults: motion uses the default
OpticParams; loom uses gain_out=80, out_norm=l1 and receptor_model=off. JSON records
the resolved parameters. Right-eye selection keeps the original full-eye viewing
directions and selects right-side T4/T5 populations; it does not recenter the eye.

The mandatory motion criterion is the correct **unique, finite, positive preferred
direction in all eight T4a-d/T5a-d populations**: a front→back, b back→front, c up,
d down. DSI is reported without a newly fitted threshold. The male comparator must
also pass. Any failure shelves step B without a second tuning batch. Loom reports
the giant-fibre peak separately from TTM: walking GF must stay below the escape
threshold and looming GF must reach it. A TTM-only escape does not pass that row.
The wrapper's exit 0 means completed measurements; `acceptance.json` records the
functional pass/fail decision. Infrastructure errors produce a nonzero exit.

This is a cross-dataset comparison, not a sex-effect estimate. The loom seed is one
engineering acceptance observation, not a replicated behavioural claim. Any later
behavioural statement must retain **synthetic input layer on a candidate lattice**.

## Reproduction

On CPU, reconstruct and pin (the first command is expected to exit 2):

```powershell
python scripts/recover_banc_columns.py --out out/banc_candidate_source --fafb-control --review-controls
python scripts/build_banc_candidate.py out/banc_candidate_source
```

The diagnostic must come from a clean commit. Exact coordinates depend on BLAS/thread
settings; the packaged CSV, its fingerprint, and matching BANC graph/annotation hashes
pin the experiment across machines. Runtime loading does not reconstruct the lattice.
The controls and the DRA gate's interpretation are recorded in the sibling
[author audit](banc_column_reconstruction.md), without editing the reviewer's file.

## Results

**Both functional gates passed**, on the first and only submission. The tested code
is `59edac0`, descended from main `b4ebf9a` through the review-fix commit `3deec1b`.
House run `banc-candidate-gate-3b6b23`, job `205215ec9038`, ran on **node1 / NVIDIA B200 /
PyTorch 2.11.0+cu128**: one job, four sequential probe processes, zero job failures.
No parameter was changed after seeing these results. These are functional results
for a **synthetic input layer on a candidate lattice**.

| Subtype | Male right cells | BANC right cells | Preferred direction, both | Male DSI | BANC DSI |
|---|---:|---:|---|---:|---:|
| T4a | 849 | 811 | front→back | 0.228987 | 0.128753 |
| T4b | 846 | 816 | back→front | 0.330266 | 0.111756 |
| T4c | 883 | 807 | up | 0.309110 | 0.126262 |
| T4d | 859 | 804 | down | 0.334832 | 0.123962 |
| T5a | 838 | 801 | front→back | 0.430114 | 0.089971 |
| T5b | 852 | 813 | back→front | 0.420194 | 0.056832 |
| T5c | 858 | 797 | up | 0.307647 | 0.042548 |
| T5d | 808 | 763 | down | 0.244265 | 0.058681 |

All eight directions are correct in both graphs. BANC's modulation is weaker,
especially T5: **all four BANC T5 DSIs are below 0.1**. The owner's gate was preferred
direction, not equal selectivity strength or the older ledger's DSI≥0.1 criterion.
This passes that restricted functional gate; it does not recover male-like motion
coding or resolve the per-column assignment errors. Counts include all right-side
cells of each subtype, including those without a candidate home column: their native
inputs still connect them to the rest of BANC's optic lobe.

| Loom readout (seed 0) | MaleCNS | BANC candidate |
|---|---:|---:|
| Walking GF maximum, Hz | 0 | 0 |
| Loom GF peak, Hz | 44.096458 | 53.298664 |
| GF threshold, Hz | 33 | 33 |
| First GF-triggered escape readout, s | 0.53 | 0.46 |
| Object distance at that readout, cm | 3.5 | 5.0 |
| Loom TTM peak, Hz | 38.360828 | 0 |

Both GF rows pass. **BANC TTM remains silent**: the GF result does not establish
recovery of the downstream jump motor pathway. The earlier threshold crossing and
larger GF peak are single-run observations under this synthetic model, not evidence
of a female/male behavioural difference. No food-finding or walking improvement is
claimed, and the anatomical DRA/mirror validation remains unresolved.

The fetched Results are at `out/banc_candidate_gate/results/`. The durable house copy
is `/mnt/beegfs/neurome/runs/banc-candidate-gate-3b6b23/out/banc_candidate_gate/results/`.
Each Result passed `Result.check()`. An independent read of the saved response
vectors reproduced all sixteen direction decisions and both GF peaks. Male/BANC
LIF, optic, body and receptor-table parameters match within each probe. The cluster
snapshot has no `.git`, so each Result carries source-content fingerprints; the
local committed source used for shipping is `59edac0`.
`export.match_sources` verified every loaded source for all five Results (19–21
files each, no differing or missing files). After attaching the completed evidence,
a fresh CPU build reproduced the same candidate CSR and unchanged biological block.

```
MaleCNS CSR MD5       ef23cc27bea13be7f6a96f3c04fd3737
BANC biological MD5  27e330891b62641686f64fd7a1b66138
BANC candidate MD5   9281a7fb3586cce55bc90a906ab09416

Result JSON SHA-256
acceptance      f1201775d7a5e0e7bec15fa100010fc2474bff047ce3e082fdecbec22e829f18
malecns_motion  2dcbca008c20b55d5fec6806867898516c73ee3cd4f2ff69c265581d6fcdfe32
banc_motion     60f161720d5318753b94532b75d7642b489e0cbe17d07e09ef829f436e8cbffd
malecns_loom    6850e37657c64eeebbec6771e54a649ccf201cff51e152ce460a94afd338cda8
banc_loom       a8cd7f470faf693eaac1f078dd5550069fed5952ecd88e394b487b504770901d
```

The post-run metadata includes this functional evidence without changing the map,
weights or dynamics. Regenerate that metadata with:

```powershell
python scripts/build_banc_candidate.py out/banc_candidate_source --acceptance out/banc_candidate_gate/results/acceptance.json
```

The generator rejects evidence whose map, biological identity, checks or input layer
differs. The original Results retain the pre-run metadata; they have not been rewritten.
Future candidate Results carry the recorded functional outcome as well as the failed
anatomical checks and approximate 80–85% exact-column estimate.

The full CPU suite passed **385 tests / 19 skipped / 215 subtests**. The seven focused
candidate/bit-identity tests also passed. All three MaleCNS cache MD5s remain exactly
the values in the column-reconstruction audit. Tests cover synthetic IDs/transmitter,
column-local edges, missing targets and left-eye exclusion, preservation of explicit
zero entries, save/load/prune restoration, provenance and source-identity guards.
