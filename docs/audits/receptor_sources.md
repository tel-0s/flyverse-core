# Receptor integration: data sources (index)

One page per source, each with URL / size / SHA-256 of every file, citation, licence, the mapping rules and a
corrections table from the adversarial verification. Raw files live under `data/external/<key>/` (git-ignored;
`python scripts/fetch_data.py --external all` reproduces them from `flyverse/data/manifest.json`); derived per-type
tables in `flyverse/data/`; builders in `scripts/build_*`.

| key | source | doc | builder | derived tables | optic-lobe coverage (cells / input syn, any tier) |
|---|---|---|---|---|---|
| ozel2021 | Ozel et al. 2021 Nature, adult optic-lobe scRNA-seq (GSE142787) | `receptor_sources_ozel2021.md` | `build_ozel2021_tables.py`, `search_ozel_unannotated_targets.py` | `type_map_ozel2021.csv`, `expression_ozel2021.csv`, `expression_ozel2021_mm.csv` | 71 % / 50 % (exact 46 % / 36 %) |
| davis2020 | Davis et al. 2020 eLife, driver-sorted bulk RNA-seq (GSE116969) | `receptor_sources_davis2020.md` | `build_davis2020_tables.py` | `type_map_davis2020.csv`, `expression_davis2020.csv` | 59 % / 48 % (QC-pass) |
| kurmangaliyev2020 | Kurmangaliyev et al. 2020 Neuron, pupal optic lobe (GSE156455) | `receptor_sources_profiles.md` | `build_kurmangaliyev2020_tables.py` | `type_map_kurmangaliyev2020.csv`, `expression_kurmangaliyev2020.csv` | 66 % / 40 % (exact 56 % / 37 %) |
| nern2025 | Nern et al. 2025 Nature, optic-lobe inventory (NT predictions, synonyms) | `receptor_sources_nern2025.md` | `build_type_map_nern2025.py` | `type_map_nern2025.csv`, `type_aliases_nern2025.csv` | NT label for 732 types (no expression) |
| central | Fly Cell Atlas 2022 head 10x + Davie et al. 2018 brain (GSE107451) | `receptor_sources_central.md` | `build_central_agg_fca.py`, `build_central_agg_davie.py`, `build_central_map.py` | `type_map_central.csv`, `expression_central.csv` | central brain 26 % of cells / 15 % of input syn (mostly class tier) |
| typing | Schlegel et al. 2024 FlyWire / hemibrain typing (flywire_annotations v3.1.0), aliases | `receptor_sources_typing.md` | `build_type_map_typing.py` | `type_map_typing.csv`, `type_aliases_typing.csv`, `type_aliases.csv` | exact + alias names for 81 % of synapses (97 % with MANC) |

Combined (`scripts/build_receptor_table.py` -> `receptors_by_type.csv`, `nt_by_type_transcriptome.csv`; rules and
coverage tables in `receptor_rules.md` sections 4-6): 609 profiled types = 51.3 % of typed cells, 26.2 % of input
synapses; the edge lookup decides the sign on 29.9 % of stored entries / 26.0 % of |W| synapses (exact 17.1 %, fuzzy
4.7 %, class 3.0 %, alias 1.2 %); the remaining 74.0 % keep `NT_SIGN`. By postsynaptic module: optic 53 %, visual
projection 38 %, mushroom body 93 % (class priors), antennal lobe 48 % (17 % of its edges on mixed-pool priors),
central 5 %, descending 2 %, gustatory 0.1 %, VNC 0. Types with no profile in any source that matter for the open
model questions: Tm5Y, TmY21, TmY13, LC11, Y3, Li19, Tm32 (object pathway); DNp04, LPT27/30, DNp20, DNa02 (optomotor /
steering); PFL3, hDelta, PEG, ExR1/4/6, GLNO (compass output and loop); MN9 and the sweet interneurons.
