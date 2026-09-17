# BANC right-eye column reconstruction: candidate, validation blocked

Author: Astra. Base: `67c76a4` on `feat/banc-vision`. This is an author experiment
record, not an independent review. The owner explicitly requested that R1-R6
integration remain gated on anatomical validation.

**Result: a useful candidate lattice, not an accepted column map.** The BANC
right-eye shared-partner graph contains 877 of its 878 Mi1 cells in one component.
Its T4 directions agree closely with MaleCNS. The strict DRA check is unpassable as written, including on FAFB's published map; its BANC result primarily exposes missing annotations. The
left-eye reconstruction does not support an anatomical mirror validation. No
synthetic photoreceptors or edges were added; BANC's `has_optic_columns` remains
false. `Connectome.load`, `Retina`, `OpticLobe`, and all caches are unchanged.

## Reproduce

```
python scripts/recover_banc_columns.py --fafb-control --plot --out out/banc_columns
```

Use a new, empty output directory. CPU only; `--plot` additionally needs
matplotlib. **Every completed diagnostic run exits 2 by construction**: the literal DRA and
mirror checks have no passing state in this script. Exit 0 is unreachable. The
exit code marks the closed integration boundary, not new evidence against a lattice. There is no
extension flag or way to load these CSVs as an accepted backend annotation.

The JSON includes dataset/release fingerprints, source manifests, the generator's
SHA-256, package versions, all check statuses, per-cell DRA evidence and output
hashes. CSVs preserve biological body IDs, include unassigned rows, mark the
source `connectivity_candidate`, and expose seed displacements required to make
the drawing injective. The right-eye figure is labelled diagnostic only.

## Method and boundaries of the claim

1. Start with BANC's own Mi1 cells on one side. Form an affinity between two Mi1s
   from the dot product of their nonnegative synapse counts onto shared T4
   partners. All four T4 subtypes contribute, without their subtype names being
   used at this stage. Keep six mutual nearest neighbours; embed the largest
   connected component with classical MDS of its graph distances. Disconnected
   seeds remain unassigned.
2. Estimate the six lattice directions, solve edge displacements with 12 robust
   least-squares iterations, and round to integer coordinates. A minimum-cost
   one-to-one assignment to nearby lattice sites removes collisions. This last
   step is an **imposed constraint**, not validation: the report exposes the
   initial collisions and every resulting displacement.
3. Associate L1/L5 with Mi1, L2 with L5, L3 with Mi1, then Mi4/Mi9 and R8/R7
   through named same-animal anchor populations. Each type uses a maximum-count
   one-to-one match; matches with zero supporting counts are discarded. T4 home
   assignments are also emitted, with their uncertainty visible in the FAFB
   control. No wide-field neuron is assigned a home column by this procedure.
4. Choose among the twelve hex-lattice rotations/reflections using T4a/b
   Mi4-minus-Mi1 input offsets relative to MaleCNS. T4c/d subtype directions are
   held out from this choice. This is a limited holdout: all unordered T4 inputs
   contributed to the neighbour graph. The validation uses input centroids, not
   the inferred T4 home columns.
5. Test R7->Dm-DRA1 and R8->Dm-DRA2 inputs against the inferred dorsal envelope,
   keeping unassigned photoreceptors in the denominator. These edges were not
   used to orient the lattice. This is a **connectivity proxy**, not a new DRA
   label. Even a passing proxy would need corroboration before satisfying the
   literal specification.

The one-to-one connectivity matching has precedent in the published FAFB
[H. S. Seung (2024), *Predicting visual function by interpreting a neuronal wiring diagram*](https://www.nature.com/articles/s41586-024-07953-5).
The DRA proxy follows the specific photoreceptor/target pairings in the
[Matsliah et al. (2024), *Neuronal parts list and wiring diagram for a visual system*](https://www.nature.com/articles/s41586-024-07981-1).
These sources justify testing the method; they do not validate a BANC body ID's
inferred position. No FAFB edges, body IDs or column assignments enter the BANC
reconstruction. MaleCNS supplies the direction convention only.

## Right-eye results

| Quantity | Result |
|---|---:|
| Mi1 seeds | 878 |
| Main connected component / distinct candidate sites | 877 |
| Mutual neighbour edges | 2,374 |
| Collisions before imposing injectivity | 64 |
| Seeds displaced from their rounded location to impose injectivity | 136 |
| Maximum assignment displacement | 1.323 column spacings |
| Neighbour-edge step agreement after assignment | 82.39% |

The table above was regenerated from committed `0b3668f` (unchanged reconstruction
code from `ebd6f26`) with the following fingerprint block:

```
generator SHA256 a0630eca7b898c454f332c013322f84eccbcd3b5085a6295e75bf06e6374ce89
BANC v888 CSR MD5 27e330891b62641686f64fd7a1b66138
neurons.csv.gz SHA256 40a2201554a8c34d2c4b07c8322543a07c1b3faf5acafbac363fd1a3d0fa617f
connections_princeton.csv.gz SHA256 8772298eb69455759c7b4a23ee1d1843f5b389965e312524e652415f8a09b312
Python 3.13.2; NumPy 2.3.5; SciPy 1.17.0; pandas 3.0.1
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS: unset
```

A controlled rerun of the same follow-up script with OMP/OPENBLAS set to 4
reproduces the original 65/139 right and 116/179 left counts; unset reproduces
64/136 and 114/175. The IRLS/rounding boundary is sensitive to BLAS thread-dependent
floating point differences. The left solution is particularly unstable. These are
numeric-environment differences, not changed connectivity. Runtime reports now
include the thread environment; a drawing is identified by its CSV hash as well.

The disconnected Mi1 is `720575941689026011`; the script leaves it unassigned.
The distinction between 877 *candidate sites* and 877 *validated columns* matters:
this run establishes the former only.

| Type | Cells | Positive matches with coordinates |
|---|---:|---:|
| L1 | 874 | 865 |
| L5 | 878 | 850 |
| L2 | 810 | 687 |
| L3 | 699 | 575 |
| Mi4 | 830 | 767 |
| Mi9 | 796 | 670 |
| R8_unclear | 797 | 709 |
| R7_unclear | 799 | 708 |
| T4a | 811 | 784 |
| T4b | 816 | 786 |
| T4c | 807 | 743 |
| T4d | 804 | 762 |

Counts refer to the backend's normalized types, rather than the raw release's
primary-type strings. A positive match does not certify anatomical correctness.

## The four gates

| Spec 2.5 check | Result | Evidence |
|---|---|---|
| (i) DRA dorsal rim | **Fail / direct labels unavailable** | No direct DRA photoreceptor labels in the released neuron table. Of 28 R7->Dm-DRA1 / R8->Dm-DRA2 proxy cells, 11 have candidate coordinates and only 7 are within two rows of the dorsal envelope; 5 are in the upper height quintile. |
| (ii) L/R mirror | **Unavailable** | Left reconstruction is fragmented and has no validated correspondence/origin to the right. Reflecting a generated eye would test the code's convention, not the animal's anatomy. |
| (iii) T4 preferred directions | **Pass for the right-eye offset statistic** | Cosines: a 0.9999846, b 0.9996227, c 0.9996896, d 0.9996263. a/b selected orientation; c/d held out from that choice. |
| (iv) Column count | **Pass as a population sanity check** | 877 right-eye sites; explicit approximate range 750-950. This cannot validate individual positions. |

The strict rim criterion uses the same two-row dorsal-envelope depth as the
existing FAFB acceptance script. This strict all-cells criterion is not meetable even on FAFB's published map:
80/90 proxy cells are within two rows, and 60/64 real labelled DRA cells are within
two rows. The reconstructed FAFB control scores 75/87 mapped proxy cells. The
proxy has precision 61/90 = 67.8% and recall 61/64 = 95.3% against real labels.
BANC's mapping coverage (11/28 versus FAFB 87/90) is the strong incompleteness
signal; the rim result conditional on mapping (7/11) is too small and proxy-limited
to establish lattice error. The gate is retained as an integration boundary, not
interpreted as a discriminating anatomical test. These calibrations reproduce the
[independent review](banc_column_reconstruction_review.md), sections 3-4. The script records missing evidence explicitly;
it does not drop unavailable checks from `integration_ready`.

The exploratory union of both R7/R8 onto either Dm-DRA type selected 31 cells.
The final generator uses only the two literature pairings, yielding the 28-cell
denominator above. This selection is based on anatomy, not which cells pass.

## Left-eye diagnostic

Of 560 left Mi1 seeds, the largest component contains only 321. The initial
integer solution has 114 collisions; enforcing injectivity displaces 175 seeds.
Its T4c/d offset cosines are -0.973 / -0.819 after the same a/b orientation
procedure. The one DRA proxy cell is unassigned. This is not a usable left-eye
map and cannot establish the requested mirror check.

## FAFB control: measured errors, not training labels

Run the identical method on the right FAFB optic lobe, withholding its published
hex coordinates until evaluation. Then align only lattice symmetry and integer
origin to count exact and neighbouring-column matches.

| Type | Total cells | Evaluated | Exact | Within one column |
|---|---:|---:|---:|---:|
| Mi1 | 796 | 783 | 739 | 777 |
| L1 | 793 | 766 | 727 | 762 |
| L5 | 785 | 734 | 696 | 729 |
| L2 | 791 | 723 | 683 | 718 |
| L3 | 738 | 671 | 640 | 666 |
| Mi4 | 766 | 744 | 702 | 738 |
| Mi9 | 770 | 722 | 691 | 718 |
| R8_unclear | 661 | 640 | 623 | 638 |
| R7_unclear | 669 | 640 | 622 | 638 |
| T4a | 737 | 714 | 598 | 703 |
| T4b | 748 | 729 | 651 | 715 |
| T4c | 842 | 725 | 582 | 700 |
| T4d | 777 | 718 | 629 | 703 |

**94.38% is the tuned optimum on the FAFB development control at neighbours=6,
not an accuracy estimate for BANC.** The independent review's neighbour sweep gives
78.54% exact at 5, 94.38% at 6, and 67.82% at 7 (within one: 97.57%, 99.23%,
97.32%). Six is anatomically motivated, but the narrow peak has no robustness margin.

For Mi1 this is 94.38% exact and 99.23% within one column **among the 783 evaluated
seeds**; 13 seeds are outside the main component. It is 739/796 exact over the
whole population. The weaker T4 home assignments show why a correct population
offset is insufficient to certify every cell's position.

This is a development control, not an untouched test set: exploratory method
choices were inspected against FAFB. Alternative normalized/shared-partner and
local-relaxation attempts did not eliminate boundary ambiguity. The committed
generator fixes the documented raw-T4 method and exposes its residual errors.
The BANC coordinates themselves still come solely from BANC connectivity.

## BANC-specific accuracy estimate and review controls

The independent review's label-free split-half calibration puts BANC at **about
80-85% exact / 97-98% within one column**, roughly one in six misplaced. This is
an estimate, not BANC ground truth: it assumes effective evidence budget explains
the difference between animals. It may miss structural errors. It must accompany
any use of the candidate; the tuned 94.38% FAFB result must not replace it.

`--review-controls` reproduces the proxy calibration and adds two genuinely
independent c/d neighbour-graph controls (no c/d partners in reconstruction):
Tm3-only FAFB 576/765 = 75.29% exact; T4a+b-only 585/782 = 74.81%. The held-out
BANC c/d cosines remain positive: Tm3 .99836/.99659, T4a+b .99998/.99876.
These test orientation independently, at lower exact column accuracy.

It also reports DRA threshold sensitivity in FAFB-equivalent synapse counts,
scaled by the raw DNa02 input budgets (BANC 13,536 / FAFB 28,497 = .474997).
This is a global release-yield assumption, not a calibrated local DRA model:

| FAFB-equivalent minimum | FAFB raw minimum; selected/mapped/rim | BANC raw minimum; selected/mapped/rim |
|---|---|---|
| 5 | 5; 90/90/80 (published map) | 2.375; 28/11/7 (candidate) |
| 10 | 10; 85/85/77 (published map) | 4.750; 21/7/6 (candidate) |

These are sensitivity diagnostics; no threshold replaces the closed gate.

## Integration boundary and next evidence needed

TODO F(A) remains partial; F(B) is blocked. **Zero synthetic nodes/edges**, no
`Connectome.extend` call, no capability override, and no GPU or house batch.

The next useful input is an independently registered BANC scaffold: Mi1 or
lamina positions/column traces, particularly around the dorsal boundary, plus
left-eye anatomical landmarks if the mirror criterion is retained. The emitted
DRA body IDs and unmatched seed provide concrete targets for that examination.
A small number of anchors might repair global orientation/registration, but
local boundary assignments still require validation. Lowering a threshold,
forcing lattice uniqueness, or copying FAFB coordinates by type would not supply
that evidence.

MaleCNS cache MD5s remain:

```
neurons.parquet  c50c598a708b5b373cbaffca7d6a9d82
W_post_pre.npz   ac131529cebf98decde58d0c227b7954
sign0_counts.npz bf01d724acf2a1fec8fdb60ef8a9e066
```

The generator asserts unchanged MaleCNS and BANC fingerprints. Synthetic CPU
tests verify exact pairwise geometry on a perfect lattice, invariance to seed
ordering, orphan exclusion, sign-safe positive matching, nonmutation, missing
evidence, and retention of unassigned DRA cells in the gate's denominator.
The full CPU suite passed **381 tests / 19 skipped / 215 subtests**; the focused
reconstruction and unchanged MaleCNS golden tests passed **13 tests**.

## Subsequent owner scope

After the independent review, the owner authorized a separate, explicitly opt-in
experiment: synthetic R1-R6 input on a candidate BANC lattice, with motion and
loom functional acceptance on one house submission. That authorization supersedes
the earlier instruction to block all experiments on the anatomical gate; it does
not make the anatomical map validated. The diagnostic gate above remains closed.
