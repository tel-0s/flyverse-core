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

Pending the single predeclared house submission. Candidate integration is not accepted
by the existence of the opt-in implementation. Functional outcomes will be appended here.
