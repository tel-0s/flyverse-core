# Typing source: cross-connectome cell-type names and aliases for MaleCNS

Source-acquisition and mapping record for the "typing" step of `docs/NT_INTEGRATION.md` (step 2: type map with a
confidence tier, so that FlyWire-, hemibrain- and optic-inventory-keyed tables can be joined to MaleCNS by name).
Built 2026-09-11 by `scripts/build_type_map_typing.py`; outputs `flyverse/data/type_aliases.csv` (=
`flyverse/data/type_aliases_typing.csv`, a collision-proof copy: a concurrently run per-source builder also writes
`type_aliases.csv`) and `flyverse/data/type_map_typing.csv`. Raw files live in `data/external/typing/` (git-ignored).

## 1. Files acquired

All downloads on 2026-09-11 with `curl`. Sizes in bytes; SHA-256 of the file as stored.

| file (data/external/typing/) | source / URL | accession / version | size | sha256 |
|---|---|---|---|---|
| schlegel2024_Supplemental_file1_neuron_annotations.tsv | https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv | flywire_annotations repo, commit 8587524 (2026-07-21), annotations v3.0.0 (extended with Berg et al. 2025); FlyWire release 783 | 31,718,505 | 9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be |
| schlegel2024_Supplemental_file2_non_neuron_annotations.tsv | same repo, `Supplemental_file2_non_neuron_annotations.tsv` | as above | 127,761 | 2349fd5b4b9e53fd1b4a84aa73230bb7bebe66faa5d6a7490155aa2128861de1 |
| schlegel2024_Supplemental_file3_summary_with_ngl_links.csv | same repo, `Supplemental_file3_summary_with_ngl_links.csv` | as above (hemilineage summary) | 758,670 | 8e9d92c0263c19863da3d537400bc2fe5ab6ed19e8327b0237e1e8a60ef9cfa5 |
| schlegel2024_Supplemental_file4_hemilineages_clustering.csv | same repo, `Supplemental_file4_hemilineages_clustering.csv` | as above | 2,315,552 | daa1ebbede4ff3299d36d3224c3463f4b5a36116c057261d4f6e9a55ac4cbdc2 |
| schlegel2024_Supplemental_file5_hemibrain_meta.csv | same repo, `Supplemental_file5_hemibrain_meta.csv` | as above; hemibrain v1.2.1 neuPrint metadata, 25,397 bodies, 5,629 (connectivity) types | 2,634,367 | c9ca8421a5b6a5382348e06a7f62ef452bc78562983deff9ff479e30be2bd2a2 |
| schlegel2024_README.md | same repo, `supplemental_files/README.md` (column definitions) | as above | 7,525 | 49e884a1273105427ded3c110d48266702a3f90148006cf02c1a0fa50919eb71 |
| schlegel2024_41586_2024_7686_MOESM5_ESM.tsv | https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07686-5/MediaObjects/41586_2024_7686_MOESM5_ESM.tsv | Nature paper release of Supplementary Data 1 (neuron annotations, 2024 naming; 5,634 cell types) | 27,015,208 | 30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6 |
| schlegel2024_41586_2024_7686_MOESM9_ESM.csv | https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07686-5/MediaObjects/41586_2024_7686_MOESM9_ESM.csv | Nature paper release of Supplementary Data 5 (hemibrain meta) | 2,618,808 | cf1c6d1821cf9179584e2ba89a614bd1b184234e547b660d378c964ed46fd9e1 |
| nature_s41586-024-07686-5.html | https://www.nature.com/articles/s41586-024-07686-5 | article HTML (supplementary listing, data / code availability) | 950,104 | 0f6b7a1ab62270e53e1e06dea985a4cf21b8ab7c69c2741159b10be8d2481518 |
| nern2025_41586_2025_8746_MOESM4_ESM.zip | https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-08746-0/MediaObjects/41586_2025_8746_MOESM4_ESM.zip | Nern 2025 Supplementary Tables 1-7 + guide (zip, extracted to `nern2025_MOESM4/`) | 687,758 | ef34f64c74e157fdec06312e3a1d6e567ae48a383944dd99a3f854e7ba593c93 |
| nern2025_MOESM4/Sup_Table_1_Cell-types_and_counts_final.xlsx | (from the zip) | 732 types x sides, counts, predicted NT | 29,923 | 095a80142923b09e17e5025f6d0e8b8aa0faed0adf5fbad17aff793e30b284a8 |
| nern2025_MOESM4/Sup_Table_5_Neurotransmitter_validation_final.xlsx | (from the zip) | NT ground truth / validation, 147 types | 17,321 | 958284e7a46ba7e638eca75408532c644bf791476b0feba9bba6abecdfc6d5ee |
| nern2025_MOESM4/Sup_Table_7_MatchingCellTypes_final.xlsx | (from the zip) | optic lobe <-> FlyWire (Matsliah / Schlegel names) <-> hemibrain, 733 rows, "matched as" cardinality, NT predictions both sides | 655,805 | d9e1b35282abb7da6cb5ec6ffa84bed4e9231bcde5994d7e41711e2751b61c8c |
| nern2025_MOESM4/Sup_Table_{2,3,4,6}_*.xlsx, Supplementary Tables Guide.pdf | (from the zip; not used) | | 13,984 / 123,972 / 64,250 / 59,360 / 85,891 | e6d0d66e... / f18662c6... / 0b82f07c... / eb7977e0... / 5a333b1a... |
| nern2025_Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx | https://raw.githubusercontent.com/reiserlab/male-drosophila-visual-system-connectome-code/main/params/Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx | code repo HEAD dbafc73 (2025-06-17); byte-identical to the Nature zip's Table 1 | 29,923 | 095a80142923b09e17e5025f6d0e8b8aa0faed0adf5fbad17aff793e30b284a8 |
| nern2025_Nern-et-al_SuppTable05_Neurotransmitter_validation.xlsx | same repo, `params/Nern-et-al_SuppTable05_Neurotransmitter_validation.xlsx` | as above | 17,322 | 38baa04d0d8b13eccbfb28193d7d38451679ccb09674341ec21ff897598b0602 |
| nern2025_Primary_cell_type_table.xlsx | same repo, `params/Primary_cell_type_table.xlsx` | 733 types with main group / figure group | 28,048 | 5ed863fbe1ccb6e2f2c8e9f2271fdc5861394df6bb47b25b30174d3451c7b501 |
| nern2025_cell_types.html, nern2025_explorer_type_list.json | https://reiserlab.github.io/male-drosophila-visual-system-connectome/cell_types.html | Visual System Cell Type Explorer index (732 types; the json is the parsed list) | 250,511 / 6,585 | fb4553fac82d4344a027da2d7f42d749e4c31f8146928d22dcc0aac0d382c3bb / 04c7700134d9fa34c414ec298ac2d9b0e0e9642294bd78b17032e9edd92c18fc |
| nature_s41586-025-08746-0.html | https://www.nature.com/articles/s41586-025-08746-0 | article HTML | 1,128,118 | 33f3e24c42f6e7ff8f363c87a5e5db108b6ecf957b9b25ba1b7aae65d2d7c7cf |
| reiser_malecns_explorer_neurons.json | https://raw.githubusercontent.com/reiserlab/celltype-explorer-drosophila-male-cns/main/data/neurons.json | MaleCNS Cell Type Explorer, repo HEAD 789cc6c (2026-09-10), generated 2026-09-09 from neuPrint male-cns:v1.0; 11,751 types with `flywire` and `synonyms` fields | 1,610,653 | cbbb8a35d0aac077f64d5f98d51d19f6d3ed40331b5c49fb4edb7e2f1e6b1180 |
| reiser_malecns_explorer_README.md | same repo, `README.md` | field documentation | 11,425 | a2ec6ee24bc3699f703d005622b8970571e12fb2695f2f43d99dd83f1e236eb6 |
| (local, not downloaded) `cache/neurons.parquet` | MaleCNS v1.0 flat tables, `body-annotations-male-cns-v1.0-minconf-0.5.feather` (gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/) | columns `flywireType`, `hemibrainType`, `mancType`, `class` (see `D:\Datasets\male-cns-connectome-v1.0\README.md` for md5 provenance) | | |

Not obtainable: the full text of the Neuroinformatics note (Springer, exclusive licence, paywalled: the PDF endpoint
returns an HTML login page). Its abstract and metadata were read from the article page and Crossref
(`api.crossref.org/works/10.1007/s12021-026-09783-4`); "No datasets were generated in the current study."
The Nature article pages are rendered through a cookie redirect; the HTML was fetched with a browser user agent.

Citations and licences:

* Schlegel P, Yin Y, Bates AS, et al. (2024) Whole-brain annotation and multi-connectome cell typing of *Drosophila*.
  Nature 634, 139-152. doi:10.1038/s41586-024-07686-5. Open access, CC BY 4.0. Supplementary files also at
  github.com/flyconnectome/flywire_annotations (v3.0.0 extends them with Berg et al. 2025 bioRxiv: `synonyms`,
  `fru_dsx`, `dimorphism`, `supertype`, and optic types renamed to the Matsliah et al. 2024 convention).
  Data availability: codex.flywire.ai, FAFB-FlyWire CATMAID, Virtual Fly Brain. Numbers quoted by the paper:
  8,453 cell types, 3,643 previously proposed in the hemibrain; 56 % of hemibrain types unambiguously found, 13 %
  matched but merged / split, 1,651 (32 %) not re-identified.
* Nern A, Loesche F, Takemura S-y, et al. (2025) Connectome-driven neural inventory of a complete visual system.
  Nature 641, 1225-1237. doi:10.1038/s41586-025-08746-0. Open access, CC BY 4.0. neuPrint dataset `optic-lobe:v1.1`;
  code at github.com/reiserlab/male-drosophila-visual-system-connectome-code. 732 cell types; the optic-lobe dataset
  is the right optic lobe of the same male CNS volume as MaleCNS v1.0, so its names are MaleCNS names verbatim.
* Olivera RA (2026) Cross-Platform Neurotransmitter & Alias Ambiguity for OA-AL2b1 and OA-AL2b2 Neurons in
  *Drosophila melanogaster*. Neuroinformatics 24:25 (published 2026-04-27). doi:10.1007/s12021-026-09783-4.
  Exclusive licence to Springer (not redistributable; only two alias pairs and the transmitter observations are used).
* MaleCNS v1.0 (Janelia FlyEM / Google, Cell 2026, doi:10.1016/j.cell.2026.08.015), CC BY 4.0; neuPrint
  `male-cns:v1.0`. The MaleCNS Cell Type Explorer (Reiser lab, github.com/reiserlab/celltype-explorer-drosophila-male-cns)
  re-exports the neuPrint `flywireType` and `synonyms` fields; its `flywire` lists equal the annotation column for
  11,746 / 11,751 types (5 differ by typographic variants such as `VP2+_adPN` vs `VP2_adPN`), so only `synonyms`
  (905 types) is new information from it.

## 2. What the sources say about MaleCNS names

* MaleCNS `type` names: 11,751 types on 164,501 typed cells (98.4 % of the 167,106-cell node set); 2,605 untyped
  cells carry 0.77 % of raw output synapses.
* MaleCNS's own cross-dataset columns: `flywireType` on 143,153 cells (equal to `type` on 95,345; 4,321 distinct
  differing (type, alias) pairs, comma-separated lists = one MaleCNS type matched to several FlyWire types, 420
  such values; `/` on 2), `hemibrainType` on 32,919 cells (equal on 25,676; 1,051 differing pairs; 124 values are
  untyped hemibrain body ids `(hb...)`), `mancType` on 22,743 cells (equal on 19,881; 719 differing pairs).
* FlyWire names: SD1 v3.0.0 has 8,840 `cell_type` values; 4,671 MaleCNS type names occur in it verbatim (the 2024
  paper release: 5,634 types, 1,708 MaleCNS names). Optic-lobe types were renamed between the two releases
  (Tm5Y -> Tm5f, LPi12 -> LPi14, Tm6 1,297 -> 1 cell), and MaleCNS `flywireType` carries the v3 / Matsliah name.
* hemibrain names: 5,899 (SD5 connectivity types plus SD1 `hemibrain_type` values); 3,899 MaleCNS names verbatim.
* Nern 2025: all 732 optic-inventory types are MaleCNS names verbatim (same volume, same naming). Its Table 7 gives
  the FlyWire (Matsliah and Schlegel-2024 names) and hemibrain matches with cardinality: 1-to-1 644, 2-to-1 32,
  3-to-1 19, 4-to-1 4, 6-to-1 6, 6-to-6 6, 1-to-2 4, 1-to-3 1, 1-to-8 1, 1-to-many 2, unmatched 14.
* Name collisions (a MaleCNS type name exists in the source, but the MaleCNS authors' own column maps the type to a
  different name): 182 types vs FlyWire v3.0.0, 18 vs hemibrain. 48 of the FlyWire ones are splits (`AOTU050`
  -> `AOTU050a`,`AOTU050b`); most of the rest are central-complex renames (`FB1C` -> `CB.FB1H0` while FlyWire also
  has a type called `FB1C`) and `CBxxxx` placeholders. These are **not** given tier exact; their alias rows carry a
  `NAME COLLISION` note. For 9 optic types (LPi12, LT82a, LT82b, Li12, Li16, Li19, Tm12, Tm6, TmY18) the collision is
  a release artefact: the name exists in the 2024 paper release and Nern Table 7 confirms it, while v3.0.0 reuses
  the name for a different Matsliah-convention type; these get tier exact with an explicit CAUTION in the evidence.
* Olivera 2026 (checked against the data): `LoVCLo3` (MaleCNS) = `OA-AL2b1` (FlyWire) = `PLP244` (hemibrain);
  octopamine in MaleCNS (consensus), FlyWire top_nt (0.77-0.86) and Busch 2009 immunostaining. `MeVCMe1` (MaleCNS)
  = `OA-AL2b2` (FlyWire) = `aMe14b` (hemibrain); **acetylcholine** in MaleCNS (all 4 cells) and FlyWire (top_nt
  0.83-0.86), FlyWire `known_nt` = "tyramine (Chiang et al. 2011, MCFO, unsure)", Nern Table 7 OL = FW =
  acetylcholine. So the "OA-" name of OA-AL2b2 is historical; every EM prediction says cholinergic. The model's
  sign for MeVCMe1 (+1, acetylcholine) is consistent with the predictions; LoVCLo3 stays sign 0 (octopamine).

## 3. Mapping rules (tiers)

| tier | rule | rows |
|---|---|---|
| exact | MaleCNS name is a type name in the source and the MaleCNS cross-dataset column does not contradict it (or the 2024-release + Nern Table 7 case above) | flywire 4,493; hemibrain 3,899; optic_inventory 732; manc 3,433 |
| alias | one documented hop: MaleCNS `flywireType` / `hemibrainType` / `mancType` (each comma-split element, cardinality in evidence), Nern 2025 Table 7 columns, explorer / Schlegel `synonyms`, Olivera 2026, parenthetical in the MaleCNS name (`PEN_a(PEN1)`) | flywire 4,589; hemibrain 1,210; hemibrain_body 123; manc 853; literature 1,679 |
| fuzzy | (1) transitive MaleCNS -> FlyWire (exact or documented alias) -> Schlegel SD1 `hemibrain_type` (majority over FlyWire cells); (2) MaleCNS connectivity suffix `_a/_b` stripped, base is a hemibrain morphology type; (3) case-insensitive, punctuation-stripped equality with a unique source name. Only emitted when the (type, system) has no exact / alias row | hemibrain 154; flywire 1 |
| class | MaleCNS `class` label of the type (majority over cells), the key for class-level transcriptomes | 1,034 |

Systems: `flywire` (Schlegel 2024 / Matsliah 2024 names), `hemibrain` (Scheffer 2020 names), `hemibrain_body`
(untyped hemibrain body id), `optic_inventory` (Nern 2025), `manc` (Takemura 2024 VNC names), `literature`
(published synonyms; "Author year: name" strings as given by the source), `malecns_class`.

Caveats: (a) Schlegel 2024 warns that ~1/3 of hemibrain types could not be re-identified; MaleCNS `hemibrainType`
inherits that uncertainty, and a name match between two datasets typed by different groups is not proof of
identity. (b) Nern Table 7 is labelled "preliminary" by its authors. (c) Sex: FlyWire is female, MaleCNS male;
dimorphic types (FlyWire `dimorphism` column) are matched by name like all others. (d) Synapse weights below use
`c.W` (|W| column sums; sign-0 presynaptic cells -- monoamines, unknown -- contribute 0) and, alongside, the raw
`weight` sum from the `-significant-only` flat table restricted to the node set (all cells count).

## 4. Coverage (output of `scripts/build_type_map_typing.py`)

Per-type best tiers are in `out/type_map_typing_per_type.csv` (git-ignored). "Group optic" is superclass
`ol_intrinsic` + `visual_projection` + `ol_sensory` as the task defines it; visual centrifugal cells are separate.

MaleCNS node set: 167,106 cells, 121,427,136 output synapses with a nonzero sign (abs W), 124,025,046 raw output synapses; typed cells 164,501 (98.4%) in 11,751 types; untyped 2,605 cells (1.6%) carrying 900,071 abs-W / 961,116 raw output synapses (0.77%) -- untyped cells are 'none' in every table and are included in the whole-CNS denominators below.

Name collisions suppressed from tier exact: FlyWire 182 types, hemibrain 18 types (the MaleCNS cross-dataset column names a different type; their documented alias rows carry a NAME COLLISION note).

### Whole CNS, best tier over FlyWire / hemibrain / optic inventory (denominators include untyped cells)

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 6,178 (52.6%) | 135,199 (80.9%) | 88,191,744 (72.6%) | 90,100,826 (72.6%) |
| alias | 1,873 (15.9%) | 6,860 (4.1%) | 10,682,421 (8.8%) | 10,959,706 (8.8%) |
| none | 3,700 (31.5%) | 22,442 (13.4%) | 21,652,904 (17.8%) | 22,003,398 (17.7%) |
| total (denominator) | 11,751 | 167,106 | 121,427,136 | 124,025,046 |

### Whole CNS, best tier including the class tier

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 6,178 (52.6%) | 135,199 (80.9%) | 88,191,744 (72.6%) | 90,100,826 (72.6%) |
| alias | 1,873 (15.9%) | 6,860 (4.1%) | 10,682,421 (8.8%) | 10,959,706 (8.8%) |
| class | 198 (1.7%) | 6,221 (3.7%) | 3,353,422 (2.8%) | 3,436,901 (2.8%) |
| none | 3,502 (29.8%) | 16,221 (9.7%) | 18,299,482 (15.1%) | 18,566,497 (15.0%) |
| total (denominator) | 11,751 | 167,106 | 121,427,136 | 124,025,046 |

### Whole CNS, best tier over FlyWire / hemibrain / optic inventory / MANC

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 9,414 (80.1%) | 153,834 (92.1%) | 110,064,800 (90.6%) | 112,245,984 (90.5%) |
| alias | 1,925 (16.4%) | 8,320 (5.0%) | 8,197,047 (6.8%) | 8,457,512 (6.8%) |
| none | 412 (3.5%) | 2,347 (1.4%) | 2,265,220 (1.9%) | 2,360,434 (1.9%) |
| total (denominator) | 11,751 | 167,106 | 121,427,136 | 124,025,046 |

### Whole CNS, system flywire

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 4,493 (38.2%) | 99,998 (59.8%) | 65,631,276 (54.0%) | 67,310,653 (54.3%) |
| alias | 3,465 (29.5%) | 41,435 (24.8%) | 32,749,424 (27.0%) | 33,247,339 (26.8%) |
| fuzzy | 1 (0.0%) | 2 (0.0%) | 3,236 (0.0%) | 3,236 (0.0%) |
| none | 3,792 (32.3%) | 23,066 (13.8%) | 22,143,132 (18.2%) | 22,502,702 (18.1%) |
| total (denominator) | 11,751 | 167,106 | 121,427,136 | 124,025,046 |

### Whole CNS, system hemibrain

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 3,899 (33.2%) | 27,998 (16.8%) | 31,614,696 (26.0%) | 33,011,092 (26.6%) |
| alias | 911 (7.8%) | 6,655 (4.0%) | 7,446,501 (6.1%) | 7,778,063 (6.3%) |
| fuzzy | 137 (1.2%) | 488 (0.3%) | 774,065 (0.6%) | 818,971 (0.7%) |
| none | 6,804 (57.9%) | 129,360 (77.4%) | 80,691,808 (66.5%) | 81,455,804 (65.7%) |
| total (denominator) | 11,751 | 167,106 | 121,427,136 | 124,025,046 |

### Whole CNS, system optic_inventory

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 732 (6.2%) | 104,270 (62.4%) | 53,778,228 (44.3%) | 54,139,973 (43.7%) |
| none | 11,019 (93.8%) | 60,231 (36.0%) | 66,748,840 (55.0%) | 68,923,957 (55.6%) |
| total (denominator) | 11,751 | 167,106 | 121,427,136 | 124,025,046 |

### Whole CNS, system manc

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 3,433 (29.2%) | 19,128 (11.4%) | 23,645,618 (19.5%) | 24,061,431 (19.4%) |
| alias | 749 (6.4%) | 3,664 (2.2%) | 3,938,428 (3.2%) | 4,019,765 (3.2%) |
| none | 7,569 (64.4%) | 141,709 (84.8%) | 92,943,024 (76.5%) | 94,982,734 (76.6%) |
| total (denominator) | 11,751 | 167,106 | 121,427,136 | 124,025,046 |

### Group optic (superclasses: ['ol_intrinsic', 'ol_sensory', 'visual_projection']), best tier over the three systems; 37 untyped cells added to the denominator

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 592 (94.3%) | 103,600 (99.0%) | 51,752,132 (99.7%) | 51,682,161 (99.7%) |
| alias | 5 (0.8%) | 863 (0.8%) | 122,979 (0.2%) | 122,972 (0.2%) |
| none | 31 (4.9%) | 191 (0.2%) | 47,743 (0.1%) | 47,850 (0.1%) |
| total (denominator) | 628 | 104,691 | 51,923,456 | 51,854,364 |

### Group optic, system flywire

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 229 (36.5%) | 77,789 (74.3%) | 37,685,472 (72.6%) | 37,711,527 (72.7%) |
| alias | 360 (57.3%) | 26,527 (25.3%) | 14,100,548 (27.2%) | 14,004,513 (27.0%) |
| none | 39 (6.2%) | 338 (0.3%) | 136,836 (0.3%) | 136,943 (0.3%) |
| total (denominator) | 628 | 104,691 | 51,923,456 | 51,854,364 |

### Group optic, system hemibrain

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 105 (16.7%) | 4,830 (4.6%) | 4,782,064 (9.2%) | 4,796,862 (9.3%) |
| alias | 176 (28.0%) | 3,815 (3.6%) | 2,975,636 (5.7%) | 2,975,636 (5.7%) |
| fuzzy | 6 (1.0%) | 125 (0.1%) | 138,963 (0.3%) | 138,963 (0.3%) |
| none | 341 (54.3%) | 95,884 (91.6%) | 44,026,192 (84.8%) | 43,941,522 (84.7%) |
| total (denominator) | 628 | 104,691 | 51,923,456 | 51,854,364 |

### Group optic, system optic_inventory

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 583 (92.8%) | 103,580 (98.9%) | 51,710,096 (99.6%) | 51,640,125 (99.6%) |
| none | 45 (7.2%) | 1,074 (1.0%) | 212,758 (0.4%) | 212,858 (0.4%) |
| total (denominator) | 628 | 104,691 | 51,923,456 | 51,854,364 |

### Group visual_centrifugal (superclasses: ['visual_centrifugal']), best tier over the three systems; 1 untyped cells added to the denominator

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 105 (97.2%) | 559 (99.3%) | 1,759,823 (99.9%) | 2,137,132 (99.9%) |
| none | 3 (2.8%) | 3 (0.5%) | 2,044 (0.1%) | 2,044 (0.1%) |
| total (denominator) | 108 | 563 | 1,762,428 | 2,139,737 |

### Group visual_centrifugal, system flywire

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 25 (23.1%) | 105 (18.7%) | 399,402 (22.7%) | 634,775 (29.7%) |
| alias | 80 (74.1%) | 454 (80.6%) | 1,360,421 (77.2%) | 1,502,357 (70.2%) |
| none | 3 (2.8%) | 3 (0.5%) | 2,044 (0.1%) | 2,044 (0.1%) |
| total (denominator) | 108 | 563 | 1,762,428 | 2,139,737 |

### Group visual_centrifugal, system hemibrain

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 22 (20.4%) | 95 (16.9%) | 428,672 (24.3%) | 524,514 (24.5%) |
| alias | 47 (43.5%) | 230 (40.9%) | 840,274 (47.7%) | 1,093,237 (51.1%) |
| none | 39 (36.1%) | 237 (42.1%) | 492,921 (28.0%) | 521,425 (24.4%) |
| total (denominator) | 108 | 563 | 1,762,428 | 2,139,737 |

### Group visual_centrifugal, system optic_inventory

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 101 (93.5%) | 533 (94.7%) | 1,757,998 (99.7%) | 2,135,218 (99.8%) |
| none | 7 (6.5%) | 29 (5.2%) | 3,869 (0.2%) | 3,958 (0.2%) |
| total (denominator) | 108 | 563 | 1,762,428 | 2,139,737 |

### Group central (superclasses: ['cb_intrinsic']), best tier over the three systems; 880 untyped cells added to the denominator

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 4,872 (73.8%) | 25,226 (78.4%) | 28,834,388 (77.0%) | 30,215,363 (77.2%) |
| alias | 1,366 (20.7%) | 4,171 (13.0%) | 5,962,815 (15.9%) | 6,165,053 (15.7%) |
| none | 367 (5.6%) | 1,883 (5.9%) | 2,079,407 (5.5%) | 2,147,694 (5.5%) |
| total (denominator) | 6,605 | 32,160 | 37,466,908 | 39,150,716 |

### Group central, system flywire

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 3,636 (55.0%) | 16,329 (50.8%) | 21,701,008 (57.9%) | 22,906,476 (58.5%) |
| alias | 2,520 (38.2%) | 12,605 (39.2%) | 12,691,827 (33.9%) | 13,060,693 (33.4%) |
| fuzzy | 1 (0.0%) | 2 (0.0%) | 3,236 (0.0%) | 3,236 (0.0%) |
| none | 448 (6.8%) | 2,344 (7.3%) | 2,480,541 (6.6%) | 2,557,705 (6.5%) |
| total (denominator) | 6,605 | 32,160 | 37,466,908 | 39,150,716 |

### Group central, system hemibrain

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 3,671 (55.6%) | 20,363 (63.3%) | 24,630,702 (65.7%) | 25,892,170 (66.1%) |
| alias | 557 (8.4%) | 2,147 (6.7%) | 2,389,444 (6.4%) | 2,416,538 (6.2%) |
| fuzzy | 117 (1.8%) | 335 (1.0%) | 496,323 (1.3%) | 541,229 (1.4%) |
| none | 2,260 (34.2%) | 8,435 (26.2%) | 9,360,142 (25.0%) | 9,678,173 (24.7%) |
| total (denominator) | 6,605 | 32,160 | 37,466,908 | 39,150,716 |

### Group central, system optic_inventory

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 40 (0.6%) | 130 (0.4%) | 235,640 (0.6%) | 268,708 (0.7%) |
| none | 6,565 (99.4%) | 31,150 (96.9%) | 36,640,972 (97.8%) | 38,259,402 (97.7%) |
| total (denominator) | 6,605 | 32,160 | 37,466,908 | 39,150,716 |

### Group descending (superclasses: ['descending_neuron']), best tier over the three systems; 4 untyped cells added to the denominator

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 445 (92.7%) | 1,182 (90.0%) | 3,696,659 (94.7%) | 3,882,984 (94.9%) |
| alias | 32 (6.7%) | 122 (9.3%) | 141,951 (3.6%) | 141,951 (3.5%) |
| none | 3 (0.6%) | 6 (0.5%) | 54,898 (1.4%) | 54,898 (1.3%) |
| total (denominator) | 480 | 1,314 | 3,901,715 | 4,092,292 |

### Group descending, system flywire

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 445 (92.7%) | 1,182 (90.0%) | 3,696,659 (94.7%) | 3,882,984 (94.9%) |
| alias | 32 (6.7%) | 122 (9.3%) | 141,951 (3.6%) | 141,951 (3.5%) |
| none | 3 (0.6%) | 6 (0.5%) | 54,898 (1.4%) | 54,898 (1.3%) |
| total (denominator) | 480 | 1,314 | 3,901,715 | 4,092,292 |

### Group descending, system hemibrain

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 41 (8.5%) | 98 (7.5%) | 399,595 (10.2%) | 423,615 (10.4%) |
| alias | 121 (25.2%) | 293 (22.3%) | 1,093,299 (28.0%) | 1,144,015 (28.0%) |
| fuzzy | 10 (2.1%) | 20 (1.5%) | 89,201 (2.3%) | 89,201 (2.2%) |
| none | 308 (64.2%) | 899 (68.4%) | 2,311,413 (59.2%) | 2,423,002 (59.2%) |
| total (denominator) | 480 | 1,314 | 3,901,715 | 4,092,292 |

### Group descending, system manc

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 197 (41.0%) | 517 (39.3%) | 1,843,304 (47.2%) | 1,966,671 (48.1%) |
| alias | 279 (58.1%) | 785 (59.7%) | 2,037,426 (52.2%) | 2,090,044 (51.1%) |
| none | 4 (0.8%) | 8 (0.6%) | 12,778 (0.3%) | 23,118 (0.6%) |
| total (denominator) | 480 | 1,314 | 3,901,715 | 4,092,292 |

### Group ascending (superclasses: ['ascending_neuron']), best tier over the three systems; 5 untyped cells added to the denominator

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 4 (0.7%) | 19 (1.0%) | 7,499 (0.1%) | 24,026 (0.5%) |
| alias | 410 (72.8%) | 1,324 (70.8%) | 4,320,317 (84.2%) | 4,340,360 (83.7%) |
| none | 149 (26.5%) | 522 (27.9%) | 796,742 (15.5%) | 810,353 (15.6%) |
| total (denominator) | 563 | 1,870 | 5,132,504 | 5,182,685 |

### Group ascending, system flywire

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 2 (0.4%) | 4 (0.2%) | 7,499 (0.1%) | 16,000 (0.3%) |
| alias | 412 (73.2%) | 1,339 (71.6%) | 4,320,317 (84.2%) | 4,348,386 (83.9%) |
| none | 149 (26.5%) | 522 (27.9%) | 796,742 (15.5%) | 810,353 (15.6%) |
| total (denominator) | 563 | 1,870 | 5,132,504 | 5,182,685 |

### Group ascending, system hemibrain

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| alias | 2 (0.4%) | 4 (0.2%) | 66,369 (1.3%) | 66,369 (1.3%) |
| fuzzy | 4 (0.7%) | 8 (0.4%) | 49,578 (1.0%) | 49,578 (1.0%) |
| none | 557 (98.9%) | 1,853 (99.1%) | 5,008,611 (97.6%) | 5,058,792 (97.6%) |
| total (denominator) | 563 | 1,870 | 5,132,504 | 5,182,685 |

### Group ascending, system manc

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 493 (87.6%) | 1,675 (89.6%) | 4,794,599 (93.4%) | 4,844,780 (93.5%) |
| alias | 68 (12.1%) | 183 (9.8%) | 324,272 (6.3%) | 324,272 (6.3%) |
| none | 2 (0.4%) | 7 (0.4%) | 5,687 (0.1%) | 5,687 (0.1%) |
| total (denominator) | 563 | 1,870 | 5,132,504 | 5,182,685 |

### Group vnc (superclasses: ['vnc_efferent', 'vnc_endocrine', 'vnc_intrinsic', 'vnc_motor', 'vnc_sensory']), best tier over the three systems; 980 untyped cells added to the denominator

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| alias | 1 (0.0%) | 27 (0.1%) | 12,809 (0.1%) | 12,809 (0.1%) |
| none | 3,118 (100.0%) | 19,312 (95.0%) | 18,323,560 (98.5%) | 18,587,225 (98.5%) |
| total (denominator) | 3,119 | 20,319 | 18,593,376 | 18,871,984 |

### Group vnc, system flywire

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| alias | 1 (0.0%) | 27 (0.1%) | 12,809 (0.1%) | 12,809 (0.1%) |
| none | 3,118 (100.0%) | 19,312 (95.0%) | 18,323,560 (98.5%) | 18,587,225 (98.5%) |
| total (denominator) | 3,119 | 20,319 | 18,593,376 | 18,871,984 |

### Group vnc, system hemibrain

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| none | 3,119 (100.0%) | 19,339 (95.2%) | 18,336,368 (98.6%) | 18,600,034 (98.6%) |
| total (denominator) | 3,119 | 20,319 | 18,593,376 | 18,871,984 |

### Group vnc, system manc

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 2,722 (87.3%) | 16,727 (82.3%) | 16,835,028 (90.5%) | 17,046,187 (90.3%) |
| alias | 389 (12.5%) | 2,497 (12.3%) | 1,450,513 (7.8%) | 1,476,198 (7.8%) |
| none | 8 (0.3%) | 115 (0.6%) | 50,829 (0.3%) | 77,649 (0.4%) |
| total (denominator) | 3,119 | 20,319 | 18,593,376 | 18,871,984 |

### Group vnc, best tier including MANC

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 2,722 (87.3%) | 16,727 (82.3%) | 16,835,028 (90.5%) | 17,046,187 (90.3%) |
| alias | 389 (12.5%) | 2,497 (12.3%) | 1,450,513 (7.8%) | 1,476,198 (7.8%) |
| none | 8 (0.3%) | 115 (0.6%) | 50,829 (0.3%) | 77,649 (0.4%) |
| total (denominator) | 3,119 | 20,319 | 18,593,376 | 18,871,984 |

### Group sensory_motor_other (superclasses: ['cb_efferent', 'cb_endocrine', 'cb_motor', 'cb_sensory', 'efferent_ascending', 'efferent_descending', 'sensory_ascending', 'sensory_descending']), best tier over the three systems; 135 untyped cells added to the denominator

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 160 (64.5%) | 4,613 (82.0%) | 2,141,239 (81.1%) | 2,159,160 (79.3%) |
| alias | 59 (23.8%) | 353 (6.3%) | 121,550 (4.6%) | 176,561 (6.5%) |
| none | 29 (11.7%) | 525 (9.3%) | 348,510 (13.2%) | 353,334 (13.0%) |
| total (denominator) | 248 | 5,626 | 2,639,001 | 2,722,524 |

### Group sensory_motor_other, system flywire

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 156 (62.9%) | 4,589 (81.6%) | 2,141,239 (81.1%) | 2,158,891 (79.3%) |
| alias | 60 (24.2%) | 361 (6.4%) | 121,550 (4.6%) | 176,630 (6.5%) |
| none | 32 (12.9%) | 541 (9.6%) | 348,510 (13.2%) | 353,534 (13.0%) |
| total (denominator) | 248 | 5,626 | 2,639,001 | 2,722,524 |

### Group sensory_motor_other, system hemibrain

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 60 (24.2%) | 2,612 (46.4%) | 1,373,662 (52.1%) | 1,373,931 (50.5%) |
| alias | 8 (3.2%) | 166 (3.0%) | 81,479 (3.1%) | 82,268 (3.0%) |
| none | 180 (72.6%) | 2,713 (48.2%) | 1,156,158 (43.8%) | 1,232,856 (45.3%) |
| total (denominator) | 248 | 5,626 | 2,639,001 | 2,722,524 |

### Group sensory_motor_other, system manc

| tier | types | cells | out synapses (abs W) | out synapses (raw) |
|---|---|---|---|---|
| exact | 21 (8.5%) | 209 (3.7%) | 172,688 (6.5%) | 203,793 (7.5%) |
| alias | 13 (5.2%) | 199 (3.5%) | 126,217 (4.8%) | 129,251 (4.7%) |
| none | 214 (86.3%) | 5,083 (90.3%) | 2,312,394 (87.6%) | 2,356,011 (86.5%) |
| total (denominator) | 248 | 5,626 | 2,639,001 | 2,722,524 |

### Per superclass, best tier over the three systems (typed cells only)

| superclass | types | cells | exact | alias | fuzzy | none | cells exact | cells alias | cells fuzzy | cells none | raw out-syn exact+alias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ascending_neuron | 563 | 1,865 | 4 | 410 | 0 | 149 | 19 | 1,324 | 0 | 522 | 84.3% |
| cb_efferent | 1 | 4 | 0 | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 100.0% |
| cb_endocrine | 10 | 65 | 10 | 0 | 0 | 0 | 65 | 0 | 0 | 0 | 100.0% |
| cb_intrinsic | 6605 | 31,280 | 4872 | 1366 | 0 | 367 | 25,226 | 4,171 | 0 | 1,883 | 94.4% |
| cb_motor | 43 | 106 | 4 | 39 | 0 | 0 | 16 | 90 | 0 | 0 | 100.0% |
| cb_sensory | 158 | 4,756 | 141 | 17 | 0 | 0 | 4,516 | 240 | 0 | 0 | 100.0% |
| descending_neuron | 480 | 1,310 | 445 | 32 | 0 | 3 | 1,182 | 122 | 0 | 6 | 98.7% |
| efferent_ascending | 5 | 8 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 8 | 0.0% |
| efferent_descending | 1 | 4 | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 100.0% |
| ol_intrinsic | 271 | 89,354 | 248 | 1 | 0 | 22 | 89,265 | 4 | 0 | 85 | 99.9% |
| ol_sensory | 11 | 6,098 | 8 | 2 | 0 | 1 | 5,167 | 846 | 0 | 85 | 99.9% |
| sensory_ascending | 26 | 536 | 0 | 2 | 0 | 24 | 0 | 19 | 0 | 517 | 8.8% |
| sensory_descending | 4 | 12 | 4 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 100.0% |
| visual_centrifugal | 108 | 562 | 105 | 0 | 0 | 3 | 559 | 0 | 0 | 3 | 99.9% |
| visual_projection | 346 | 9,202 | 336 | 2 | 0 | 8 | 9,168 | 13 | 0 | 21 | 99.9% |
| vnc_efferent | 27 | 78 | 0 | 0 | 0 | 27 | 0 | 0 | 0 | 78 | 0.0% |
| vnc_endocrine | 4 | 22 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 22 | 0.0% |
| vnc_intrinsic | 2777 | 12,942 | 0 | 1 | 0 | 2776 | 0 | 27 | 0 | 12,915 | 0.1% |
| vnc_motor | 142 | 699 | 0 | 0 | 0 | 142 | 0 | 0 | 0 | 699 | 0.0% |
| vnc_sensory | 169 | 5,598 | 0 | 0 | 0 | 169 | 0 | 0 | 0 | 5,598 | 0.0% |

### The 40 unmatched types (no FlyWire / hemibrain / optic-inventory name at any tier) with the most raw output synapses

| type | superclass | class | cells | out_syn_raw | out_syn_W |
|---|---|---|---|---|---|
| SNta29 | vnc_sensory | mechanosensory_tactile | 260 | 130,475 | 130,475 |
| SNta37 | vnc_sensory | mechanosensory_tactile | 228 | 129,363 | 129,363 |
| SNta02,SNta09 | vnc_sensory | mechanosensory_tactile | 241 | 128,166 | 128,166 |
| SNta20 | vnc_sensory | mechanosensory_tactile | 156 | 84,933 | 84,933 |
| LgLG1a | vnc_sensory | gustatory | 136 | 80,913 | 80,913 |
| SApp | sensory_ascending | mechanosensory_proprioceptive | 148 | 79,510 | 79,510 |
| IN12B002 | vnc_intrinsic |  | 6 | 75,197 | 75,197 |
| WG4 | vnc_sensory | gustatory | 96 | 74,911 | 74,911 |
| SNppxx | vnc_sensory | mechanosensory_proprioceptive | 80 | 74,182 | 74,182 |
| SNta04 | vnc_sensory | mechanosensory_tactile | 83 | 71,296 | 71,296 |
| WG3 | vnc_sensory | gustatory | 96 | 66,339 | 39,519 |
| SNxx03 | vnc_sensory | unknown_sensory | 186 | 64,052 | 64,052 |
| SNxx04 | vnc_sensory | unknown_sensory | 130 | 63,607 | 63,607 |
| IN19A002 | vnc_intrinsic |  | 6 | 60,107 | 60,107 |
| LgLG3 | vnc_sensory | gustatory | 162 | 58,460 | 58,460 |
| SNch01 | vnc_sensory | chemosensory | 35 | 57,615 | 57,615 |
| IN09A001 | vnc_intrinsic |  | 6 | 57,060 | 57,060 |
| IN17A001 | vnc_intrinsic |  | 6 | 56,481 | 56,481 |
| SNta38 | vnc_sensory | mechanosensory_tactile | 122 | 56,006 | 56,006 |
| LgLG1b | vnc_sensory | gustatory | 134 | 55,937 | 36,926 |
| IN19B003 | vnc_intrinsic |  | 6 | 53,326 | 53,326 |
| SNta11 | vnc_sensory | mechanosensory_tactile | 67 | 51,363 | 51,363 |
| SNta21 | vnc_sensory | mechanosensory_tactile | 117 | 50,015 | 50,015 |
| SNta11,SNta14 | vnc_sensory | mechanosensory_tactile | 42 | 49,337 | 49,337 |
| IN14A002 | vnc_intrinsic |  | 6 | 48,213 | 48,213 |
| IN07B002 | vnc_intrinsic |  | 6 | 45,896 | 45,896 |
| IN13B001 | vnc_intrinsic |  | 6 | 44,759 | 44,759 |
| IN13A003 | vnc_intrinsic |  | 6 | 44,690 | 44,690 |
| INXXX044 | vnc_intrinsic |  | 8 | 44,114 | 44,114 |
| SNxx14 | vnc_sensory | unknown_sensory | 66 | 43,760 | 43,760 |
| IN26X001 | vnc_intrinsic |  | 6 | 43,316 | 43,316 |
| SNxx33 | vnc_sensory | unknown_sensory | 72 | 42,556 | 42,556 |
| ANXXX084 | ascending_neuron |  | 8 | 42,257 | 42,257 |
| SNta18 | vnc_sensory | mechanosensory_tactile | 55 | 42,211 | 42,211 |
| SApp10 | sensory_ascending | mechanosensory_proprioceptive | 37 | 41,829 | 41,829 |
| WG2 | vnc_sensory | gustatory | 97 | 39,878 | 39,878 |
| IN19A001 | vnc_intrinsic |  | 6 | 39,415 | 39,415 |
| SNpp45 | vnc_sensory | mechanosensory_proprioceptive | 53 | 39,390 | 39,390 |
| SNta28 | vnc_sensory | mechanosensory_tactile | 74 | 38,108 | 38,108 |
| IN08A002 | vnc_intrinsic |  | 6 | 38,081 | 38,081 |

### Alias-table rows by system and tier

| system          |   alias |   class |   exact |   fuzzy |
|:----------------|--------:|--------:|--------:|--------:|
| flywire         |    4589 |       0 |    4493 |       1 |
| hemibrain       |    1210 |       0 |    3899 |     154 |
| hemibrain_body  |     123 |       0 |       0 |       0 |
| literature      |    1679 |       0 |       0 |       0 |
| malecns_class   |       0 |    1034 |       0 |       0 |
| manc            |     853 |       0 |    3433 |       0 |
| optic_inventory |       0 |       0 |     732 |       0 |


wrote D:\Projects\flyverse\flyverse\data\type_aliases.csv (22,200 rows) and D:\Projects\flyverse\flyverse\data\type_map_typing.csv (22,187 rows)

### By-product: Nern 2025 Table 7 transmitter predictions, optic lobe (male) vs FlyWire (female), where both are called

Rows with a call on both sides: 545; agree 503; disagree 42 (a further 179 rows have `unclear` on at least one side). The systematic pattern is glutamate (OL) vs GABA (FW) in Dm / Pm / Cm / LPi / Li types, where Nern Table 5 validation (Davis 2020 TAPIN) sides with glutamate for Dm1 / Dm4 / Dm11 / Dm12. MaleCNS model NT (`nt` column of cache/neurons.parquet, majority per type) added for reference.

| OL_type | FlyWire (Schlegel) | matched as | OL NT pred | FW NT pred | MaleCNS model nt | cells |
|---|---|---|---|---|---|---|
| 5-HTPMPV03 | 5-HTPMPV03 | 1-to-1 | serotonin | dopamine | serotonin | 2 |
| 5thsLNv_LNd6 | s-LNv_a + LNd_a | 1-to-2 | acetylcholine | glutamate | acetylcholine | 4 |
| Cm25 | nan | 1-to-1 | glutamate | gaba | glutamate | 7 |
| Cm34 | nan | 1-to-1 | glutamate | gaba | glutamate | 2 |
| Cm7 | Dm21 | 1-to-1 | glutamate | gaba | glutamate | 184 |
| Cm9 | Dm21 | 1-to-1 | glutamate | gaba | glutamate | 143 |
| Dm1 | Dm1 | 1-to-1 | glutamate | gaba | glutamate | 80 |
| Dm12 | Dm12 | 1-to-1 | glutamate | gaba | glutamate | 276 |
| Dm16 | CB3849 | 1-to-1 | glutamate | gaba | glutamate | 151 |
| Dm19 | Dm19 | 1-to-1 | glutamate | gaba | glutamate | 30 |
| Dm20 | Dm20 | 1-to-1 | glutamate | gaba | glutamate | 98 |
| Dm6 | Dm6 | 1-to-1 | glutamate | gaba | glutamate | 62 |
| Dm9 | Dm9 | 1-to-1 | glutamate | acetylcholine | glutamate | 273 |
| HBeyelet | HBeyelet | 1-to-1 | histamine | glutamate | histamine | 7 |
| LPi34 | CB3857 | 1-to-1 | glutamate | gaba | glutamate | 119 |
| LPi3412 | CB3845 | 1-to-1 | glutamate | gaba | glutamate | 104 |
| LPi43 | CB3826 | 1-to-1 | glutamate | gaba | glutamate | 61 |
| LT68 | LT68 | 1-to-1 | glutamate | gaba | glutamate | 4 |
| LT88 | cM10 | 1-to-1 | glutamate | gaba | glutamate | 2 |
| Lai | Am | 1-to-1 | glutamate | gaba | glutamate | 84 |
| Li36 | nan | 1-to-1 | glutamate | gaba | glutamate | 2 |
| LoVC16 | cL21 | 1-to-1 | glutamate | gaba | glutamate | 4 |
| LoVC26 | cL02a | 1-to-1 | glutamate | gaba | glutamate | 6 |
| MeVC21 | cM08a | 1-to-1 | glutamate | serotonin | glutamate | 6 |
| MeVPMe10 | MeMe_e10 | 1-to-1 | glutamate | gaba | glutamate | 4 |
| MeVPMe11 | aMe19b | 1-to-1 | glutamate | gaba | glutamate | 2 |
| Mi13 | Mi13 | 1-to-1 | glutamate | gaba | glutamate | 910 |
| Mi19 | CB3842 | 1-to-1 | serotonin | dopamine | unknown | 12 |
| OA-ASM1 | OA-ASM1 | 1-to-1 | octopamine | dopamine | octopamine | 4 |
| Pm12 | nan | 1-to-1 | glutamate | gaba | gaba | 4 |
| Pm13 | nan | 1-to-1 | glutamate | gaba | glutamate | 2 |
| R1-R6 | R1-6 | 1-to-1 | histamine | acetylcholine | histamine | 3377 |
| R7d | nan | 3-to-1 | histamine | glutamate | histamine | 82 |
| R7p | nan | 3-to-1 | histamine | glutamate | histamine | 332 |
| R7y | nan | 3-to-1 | histamine | glutamate | histamine | 482 |
| R8d | nan | 3-to-1 | histamine | acetylcholine | histamine | 76 |
| R8p | nan | 3-to-1 | histamine | acetylcholine | histamine | 330 |
| R8y | nan | 3-to-1 | histamine | acetylcholine | histamine | 481 |
| Tm29 | CB3851 | 1-to-1 | glutamate | acetylcholine | glutamate | 544 |
| TmY16 | TmY16 | 1-to-1 | glutamate | gaba | glutamate | 168 |
| aMe17c | aMe17c | 1-to-1 | glutamate | gaba | glutamate | 4 |
