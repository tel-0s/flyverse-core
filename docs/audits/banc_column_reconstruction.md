# BANC right-eye column reconstruction: candidate, validation blocked

Author: Astra. Base: `442c420` on `feat/banc-vision`. This is an author experiment
record, not an independent review. The owner explicitly requested that R1-R6
integration remain gated on anatomical validation.

**Result: a useful candidate lattice, not an accepted column map.** The BANC
right-eye shared-partner graph contains 877 of its 878 Mi1 cells in one component.
Its T4 directions agree closely with MaleCNS. The dorsal-rim check fails, and the
left-eye reconstruction does not support an anatomical mirror validation. No
synthetic photoreceptors or edges were added; BANC's `has_optic_columns` remains
false. `Connectome.load`, `Retina`, `OpticLobe`, and all caches are unchanged.

## Reproduce

```
python scripts/recover_banc_columns.py --fafb-control --plot --out out/banc_columns
```

Use a new, empty output directory. CPU only; `--plot` additionally needs
matplotlib. **Exit code 2 is the failed anatomical gate**, after writing the
diagnostics. A zero exit is reserved for all four gates passing. There is no
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
[column-coordinate method](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446822/).
The DRA proxy follows the specific photoreceptor/target pairings in the
[visual-system inventory](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446827/).
These sources justify testing the method; they do not validate a BANC body ID's
inferred position. No FAFB edges, body IDs or column assignments enter the BANC
reconstruction. MaleCNS supplies the direction convention only.

## Right-eye results

| Quantity | Result |
|---|---:|
| Mi1 seeds | 878 |
| Main connected component / distinct candidate sites | 877 |
| Mutual neighbour edges | 2,374 |
| Collisions before imposing injectivity | 65 |
| Seeds displaced from their rounded location to impose injectivity | 139 |
| Maximum assignment displacement | 1.323 column spacings |
| Neighbour-edge step agreement after assignment | 82.01% |

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
| (iii) T4 preferred directions | **Pass for the right-eye offset statistic** | Cosines: a 0.9999939, b 0.9996495, c 0.9997653, d 0.9996363. a/b selected orientation; c/d held out from that choice. |
| (iv) Column count | **Pass as a population sanity check** | 877 right-eye sites; explicit approximate range 750-950. This cannot validate individual positions. |

The strict rim criterion uses the same two-row dorsal-envelope depth as the
existing FAFB acceptance script. FAFB's known strict-rim expected failure is not
an exemption for a new BANC map. The script records missing evidence explicitly;
it does not drop unavailable checks from `integration_ready`.

The exploratory union of both R7/R8 onto either Dm-DRA type selected 31 cells.
The final generator uses only the two literature pairings, yielding the 28-cell
denominator above. This selection is based on anatomy, not which cells pass.

## Left-eye diagnostic

Of 560 left Mi1 seeds, the largest component contains only 321. The initial
integer solution has 116 collisions; enforcing injectivity displaces 179 seeds.
Its T4c/d offset cosines are -0.144 / -0.981 after the same a/b orientation
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

For Mi1 this is 94.38% exact and 99.23% within one column **among the 783 evaluated
seeds**; 13 seeds are outside the main component. It is 739/796 exact over the
whole population. The weaker T4 home assignments show why a correct population
offset is insufficient to certify every cell's position.

This is a development control, not an untouched test set: exploratory method
choices were inspected against FAFB. Alternative normalized/shared-partner and
local-relaxation attempts did not eliminate boundary ambiguity. The committed
generator fixes the documented raw-T4 method and exposes its residual errors.
The BANC coordinates themselves still come solely from BANC connectivity.

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
