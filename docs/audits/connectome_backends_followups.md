# Connectome backend author follow-ups

Author: Astra. This records implementation follow-ups to the independent
[backend review](connectome_backends_review.md); it is not part of the reviewer's verdict. B1-B4 and nits
1/3/9/12-15 were resolved by Fable in 34c2eb0. In particular, the original decision 11.7 was superseded:
MaleCNS load/save defaults remain pinned to CACHE_DIR; only female caches use FLYVERSE_CACHE.

| Remaining nit | Resolution |
|---|---|
| 2 | An explicit capability registry validates dataset names during Connectome construction and every capability lookup. Unknown names raise ValueError. The properties describe source-release availability, retained by subsets; CONTROL_SURFACE.md distinguishes this from the selected population. |
| 4 | FAFB entryNerve is restricted to sensory/sensory_ascending rows (N1 in the subsequent review removed sensory_descending, which is not a v783 superclass). This clears 3,250 non-sensory annotations; exitNerve remains motor/endocrine-only. |
| 5 | The alias normalizer and controller docs explicitly state that flywire/banc identify shared source vocabularies, not dataset restrictions. Ambiguous mappings remain unresolved. |
| 6 | The controller docs explain synthesized female instance suffixes and their precedence over connectivity laterality. BANC's zero `both` count is not evidence of absent bilateral anatomy. |
| 7 | The retina module and controller docs explain that generic FAFB R7/R8 use unclear aggregates; pale/yellow/DRA-specific spectral, receptor and tau entries cannot match them. |
| 8 | Retina.coverage() and retina.summarize() report photoreceptor-free columns. Results with such a live retina gain retina.coverage, including column indices and side counts. FAFB: 1,530 columns with photoreceptors, 51 without (42 L / 9 R). These lack direct input; recurrent optic activity remains possible. |
| 10 | The controller docs identify model.dataset/model.release as additive provenance fields for all datasets and preserve the legacy MaleCNS fingerprint key set. |
| 11 | The controller docs record that subsets retain their source cache_dir, including explicit MaleCNS paths and environment-selected female roots. Normalization is unchanged. |

The optional haltere cleanup adds four CSV aliases: hi1 -> hi1 MN, hi2 -> hi2 MN, hDVM -> hDVM MN,
hiii2 -> hiii2 MN. They rename 11 BANC cells (4/3/2/2 respectively); the selected haltere group remains 25 cells.
Source names remain in flywireType; no other haltere names are guessed.

Reproduce the cache comparison against a retained copy of the merged caches:

```
python scripts/check_connectome_backends.py --compare-cache /path/to/merged/cache --rebuild --out out/connectome_review_followups/cache_comparison.json
python scripts/check_connectome_backends.py --out out/connectome_review_followups/acceptance.json
```

The comparison includes MaleCNS, FAFB, BANC and unthresholded FAFB. Every W_post_pre.npz and sign0_counts.npz
is byte-identical to its merged baseline. MaleCNS's neuron table and complete fingerprint are identical after
accounting for the two cache directories. The only neuron-value changes are FAFB entryNerve (3,250 values in
each variant) and BANC type/instance (11 each); column order is unchanged. On the two default female graphs,
fast/slow receptor signs, gain classes, slow classes, tiers and default shaped weights all have **zero changed
entries**. Manifests record the new alias hash and compilation time. The acceptance gate retains the explicit
strict-DRA expected failure; no geometry was changed.

The merged female caches were copied to main before these follow-ups and all 14 copied files checked by
SHA-256. Main's CPU suite then passed 360 tests with 19 skipped. The follow-up worktree carries the freshly
rebuilt annotations; copy those or recompile after merging this follow-up. Original MaleCNS file MD5s remain
c50c598a708b5b373cbaffca7d6a9d82 / ac131529cebf98decde58d0c227b7954 / bf01d724acf2a1fec8fdb60ef8a9e066.

The complete follow-up CPU suite passes **364 tests / 19 skipped / 215 subtests**, including the unchanged
MaleCNS golden. New regressions cover invalid dataset/capability names, sensory-only nerve mapping, missing
photoreceptors in summaries/provenance (including an empty eye), and the real BANC haltere aliases. No GPU
batch was submitted for this follow-up.

## Integration status

Fable merged `281a68b` in `347c800`, refreshed main's female caches from this worktree, and recorded the
[independent follow-up review](connectome_backends_followups_review.md) in `442c420`. The cache-copy and
360/364-test counts above describe the pre-merge checkpoints, not current main. At integration Fable reported
371 passed / 19 skipped. The earlier main neuron tables/manifests were superseded by the refreshed caches.
