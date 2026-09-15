"""Build the cross-connectome alias table for MaleCNS cell-type names (the "typing" source of docs/NT_INTEGRATION.md).

Round 2 (verify:tables:typing corrections applied, see docs/audits/receptor_verification.md): the coverage tables count
the group's untyped cells in the 'none' row; the FlyWire annotation release is labelled v3.1.0; every alias row carries a
`flag` column (conflict / resolvability); the alias-tier coverage is reported twice, with and without unresolvable rows;
the Nern 2025 Table 7 transmitter by-product is generated here (544 / 503 / 41 / 175).

Inputs (data/external/typing/, git-ignored; provenance and hashes in docs/audits/receptor_sources_typing.md):
  * cache/neurons.parquet                                  MaleCNS v1.0 node table (type, flywireType, hemibrainType, mancType, class, ...)
  * schlegel2024_Supplemental_file1_neuron_annotations.tsv  FlyWire v783 typing (cell_type, hemibrain_type, synonyms, known_nt);
                                                            github.com/flyconnectome/flywire_annotations tag v3.1.0 = commit 8587524
                                                            (2026-07-21, "matching the release of version 1.0 of the MaleCNS dataset")
  * schlegel2024_41586_2024_7686_MOESM5_ESM.tsv             the 2024 Nature paper release of the same table (5,634 cell types, old optic names)
  * schlegel2024_Supplemental_file5_hemibrain_meta.csv      hemibrain v1.2.1 type list (type, morphology_type)
  * nern2025_MOESM4/Sup_Table_1_Cell-types_and_counts_final.xlsx   optic-lobe inventory (732 types, predicted NT)
  * nern2025_MOESM4/Sup_Table_7_MatchingCellTypes_final.xlsx       optic-lobe <-> FlyWire (Matsliah / Schlegel) <-> hemibrain matches
  * reiser_malecns_explorer_neurons.json                    MaleCNS Cell Type Explorer (neuPrint male-cns:v1.0 flywireType + published synonyms)
  * Olivera 2026 Neuroinformatics 10.1007/s12021-026-09783-4 (two aliases, hard-coded below with the citation)

Outputs:
  * flyverse/data/type_aliases.csv      malecns_type, alias, system, tier, flag, evidence
  * flyverse/data/type_aliases_typing.csv   byte-identical copy (collision-proof name)
  * flyverse/data/type_map_typing.csv   source_name, malecns_type, tier, flag, evidence, n_cells_malecns, system
  * out/type_map_typing_report.md, out/type_map_typing_per_type.csv (git-ignored; the report is section 4 of
    docs/audits/receptor_sources_typing.md)

Tiers:
  exact  the MaleCNS type name itself is a type name in the source, and the MaleCNS authors' own cross-dataset
         column (flywireType / hemibrainType / mancType) does not contradict it.
  alias  a documented one-hop match: MaleCNS flywireType / hemibrainType / mancType columns, Nern 2025 Supplementary
         Table 7, the explorer / Schlegel `synonyms` fields, Olivera 2026, or a parenthetical in the MaleCNS name.
  fuzzy  a rule, named in `evidence`: (1) transitive MaleCNS -> FlyWire -> hemibrain through Schlegel SD1;
         (2) hemibrain connectivity-type suffix `_a/_b` stripped to the morphology type; (3) punctuation- and
         case-insensitive name equality with a unique target. Never emitted when an exact / alias row exists for
         the same (type, system).
  class  the MaleCNS `class` label of the type (Kenyon_Cell, DAN, MBON, ALPN, visual, CX, ...), the only key that
         class-level transcriptomes (FCA, Davie 2018) can be joined on.

Flags (`flag` column; several are joined with ';'; empty = no caveat):
  unresolvable          the alias names no type in the source: FlyWire alias absent from SD1 v3.1.0 AND from the 2024
                        paper release (347 of these are 'CB####' placeholders, sub-flag cb_placeholder); hemibrain alias
                        absent from SD5 / SD1 hemibrain_type (bare body ids, strings truncated at '(').
  paper_release_only    FlyWire name present only in the 2024 paper release of SD1 (renamed / dropped in v3.1.0).
  unmatched             the '(unmatched)' Nern Table 7 rows (dropped from type_map_typing.csv).
  conflict_nern7_vs_malecns           FlyWire rows of a type whose Nern Table 7 FlyWire name (Schlegel_type / Matsliah_type)
                        and MaleCNS majority flywireType disagree as strings (47 types); the sub-flag `_notation` marks
                        the 11 where the two are the same names written differently or a parent / subtype split
                        (LTe49a,b,d,e,f vs LTe49a,LTe49b,...; MeTu2 vs MeTu2a), the other 36 name different types
                        (Cm -> Sm off-by-one series, Li shifts, AOTU056 CB1558 vs CB2216, LoVP19 LTe49* vs LC46).
  minority_agreement    exact row whose MaleCNS cross-dataset column names the type on fewer than half of the cells.
  merged_list_only      exact row whose MaleCNS cross-dataset column carries the name only inside a merged comma list.
  name_only             exact row resting on the name alone (MaleCNS cross-dataset column empty for every cell).
  v3_name_collision     exact row from the 2024 paper release whose name denotes a different type in SD1 v3.1.0.

Usage:  PYTHONIOENCODING=utf-8 python scripts/build_type_map_typing.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external" / "typing"
OUT_DIR = ROOT / "flyverse" / "data"
REPORT_DIR = ROOT / "out"
sys.path.insert(0, str(ROOT))
from flyverse.connectome import DATA_DIR  # noqa: E402

RAW_WEIGHTS = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather"

SD1_RELEASE = "flywire_annotations v3.1.0 (commit 8587524, 2026-07-21)"
SD1_SHORT = "SD1 v3.1.0"

OPTIC_SUPERCLASSES = {"ol_intrinsic", "visual_projection", "ol_sensory"}
TIER_RANK = {"exact": 0, "alias": 1, "fuzzy": 2, "class": 3}
UNRESOLVABLE_FLAGS = ("unresolvable", "unmatched")


def group_of(superclass: str) -> str:
    if superclass in OPTIC_SUPERCLASSES:
        return "optic"
    if superclass == "visual_centrifugal":
        return "visual_centrifugal"
    if superclass == "cb_intrinsic":
        return "central"
    if superclass == "descending_neuron":
        return "descending"
    if superclass == "ascending_neuron":
        return "ascending"
    if isinstance(superclass, str) and superclass.startswith("vnc"):
        return "vnc"
    if isinstance(superclass, str) and ("sensory" in superclass or "motor" in superclass or "endocrine" in superclass
                                        or "efferent" in superclass):
        return "sensory_motor_other"
    return "other"


def norm(s: str) -> str:
    return re.sub(r"[\s_\-()+'/.]", "", s).lower()


def split_list(v: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,/]", v) if x.strip()]


def nt_called(v) -> bool:
    return isinstance(v, str) and v.strip() != "" and not v.strip().lower().startswith("unclear")


def main() -> None:
    # ------------------------------------------------------------------ MaleCNS
    n = pd.read_parquet(ROOT / "cache" / "neurons.parquet")
    W = sp.load_npz(ROOT / "cache" / "W_post_pre.npz").tocsc()
    n["type"] = n["type"].fillna("")
    n["out_syn_W"] = np.asarray(abs(W).sum(axis=0)).ravel()  # column = presynaptic; sign-0 cells contribute 0
    n["group"] = n.superclass.map(group_of)
    # raw output synapse counts (sign-0 cells included) from the flat table, restricted to the node set
    if RAW_WEIGHTS.exists():
        w = pf.read_table(RAW_WEIGHTS, columns=["body_pre", "body_post", "weight"]).to_pandas()
        node = set(n.bodyId.to_numpy())
        w = w[w.body_pre.isin(node) & w.body_post.isin(node)]
        raw = w.groupby("body_pre").weight.sum()
        n["out_syn_raw"] = n.bodyId.map(raw).fillna(0).astype(np.int64)
        del w
    else:
        print(f"WARNING: {RAW_WEIGHTS} not found; out_syn_raw = out_syn_W", file=sys.stderr)
        n["out_syn_raw"] = n.out_syn_W
    typed = n[n.type != ""]
    n_cells = typed.groupby("type").size()
    types = list(n_cells.index)
    type_set = set(types)
    col_fw = typed.groupby("type").flywireType.agg(lambda s: Counter(s.dropna()))
    col_hb = typed.groupby("type").hemibrainType.agg(lambda s: Counter(s.dropna()))
    col_manc = typed.groupby("type").mancType.agg(lambda s: Counter(s.dropna()))
    col_class = typed.groupby("type")["class"].agg(lambda s: Counter(s.dropna()))
    nt_mode = typed.groupby("type").nt.agg(lambda s: s.mode().iat[0] if s.notna().any() else "")
    superclass_of = typed.groupby("type").superclass.agg(lambda s: s.mode().iat[0] if s.notna().any() else "")

    # ------------------------------------------------------------------ sources
    s1 = pd.read_csv(EXT / "schlegel2024_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False)
    fw_count = s1.cell_type.value_counts()
    fw_set = set(fw_count.index)
    s1_paper = pd.read_csv(EXT / "schlegel2024_41586_2024_7686_MOESM5_ESM.tsv", sep="\t", low_memory=False,
                           usecols=["cell_type"])
    fw_count_paper = s1_paper.cell_type.value_counts()
    fw_set_paper = set(fw_count_paper.index)
    # FlyWire cell_type -> hemibrain_type (majority over cells), and -> synonyms
    fw_to_hb = {}
    for ct, sub in s1[s1.hemibrain_type.notna()].groupby("cell_type").hemibrain_type:
        fw_to_hb[ct] = sub.value_counts().index[0]
    fw_syn = {}
    for ct, sub in s1[s1.synonyms.notna()].groupby("cell_type").synonyms:
        fw_syn[ct] = sorted(set(sub))
    h5 = pd.read_csv(EXT / "schlegel2024_Supplemental_file5_hemibrain_meta.csv", low_memory=False)
    hb_conn_types = set(h5.type.dropna())
    hb_morph_types = set(h5.morphology_type.dropna())
    hb_from_s1 = set()
    for v in s1.hemibrain_type.dropna():
        hb_from_s1 |= set(split_list(v))
    hb_set = hb_conn_types | hb_from_s1
    hb_count = h5.type.value_counts()

    nern1 = pd.read_excel(EXT / "nern2025_MOESM4" / "Sup_Table_1_Cell-types_and_counts_final.xlsx")
    nern1 = nern1.groupby("cell type").agg(n=("no. of cells", "sum"), group=("main groups", "first"),
                                          nt=("predicted neurotransmitter", "first"))
    nern7 = pd.read_excel(EXT / "nern2025_MOESM4" / "Sup_Table_7_MatchingCellTypes_final.xlsx")
    nern7.columns = [c.strip() for c in nern7.columns]
    nern7 = nern7[nern7.OL_type.notna()]
    nern7_schlegel = {r.OL_type: str(r.Schlegel_type) for r in nern7.itertuples() if pd.notna(r.Schlegel_type)}

    explorer = json.load(open(EXT / "reiser_malecns_explorer_neurons.json", encoding="utf-8"))
    ex_syn = {e["name"]: e.get("types", {}).get("synonyms", []) for e in explorer["neurons"]}

    olivera = [  # Olivera 2026, Neuroinformatics 24:25, doi:10.1007/s12021-026-09783-4 (abstract; full text paywalled)
        ("LoVCLo3", "OA-AL2b1", "octopaminergic in every queried resource; the OA name is not the searchable label"),
        ("MeVCMe1", "OA-AL2b2", "cholinergic in neuPrint / Neuroglancer (predicted and consensus) but grouped as "
                                "octopaminergic elsewhere; Busch 2009 did not confirm OA immunoreactivity"),
    ]

    # ------------------------------------------------------------------ rows
    rows: list[dict] = []
    have: dict[tuple[str, str], str] = {}  # (type, system) -> best tier seen among exact/alias

    def add(t, alias, system, tier, evidence, flag=""):
        if not alias or alias == "":
            return
        rows.append({"malecns_type": t, "alias": alias, "system": system, "tier": tier, "flag": flag, "evidence": evidence})
        key = (t, system)
        if tier in ("exact", "alias") and (key not in have or TIER_RANK[tier] < TIER_RANK[have[key]]):
            have[key] = tier

    def exact_flags(t, col: Counter, nc: int) -> str:
        """Caveat flags of an exact row from the MaleCNS cross-dataset column of type t."""
        if not col:
            return "name_only"
        agree_cells = sum(k for v, k in col.items() if t in split_list(v))
        f = []
        if agree_cells * 2 < nc:
            f.append("minority_agreement")
        if col.get(t, 0) == 0 and agree_cells > 0:
            f.append("merged_list_only")
        return ";".join(f)

    fw_conflict = hb_conflict = 0
    for t in types:
        nc = int(n_cells[t])
        # ---- FlyWire: exact
        fw_doc = col_fw[t]
        fw_doc_elems = set()
        for v in fw_doc:
            fw_doc_elems |= set(split_list(v))
        if t in fw_set and (not fw_doc or t in fw_doc_elems):
            ev = f"name is a FlyWire v783 cell_type (Schlegel 2024 {SD1_SHORT}, {int(fw_count[t])} FlyWire cells)"
            ev += "; MaleCNS flywireType agrees" if fw_doc else "; MaleCNS flywireType empty"
            add(t, t, "flywire", "exact", ev, exact_flags(t, fw_doc, nc))
        elif nern7_schlegel.get(t) == t and t in fw_set_paper:
            # Nern 2025 Table 7 uses the 2024 paper-release FlyWire names; SD1 v3 (v3.0.0 and v3.1.0) renamed optic types to
            # the Matsliah 2024 convention (Tm5Y -> Tm5f, LPi12 -> LPi14), and the MaleCNS flywireType column carries the v3 name.
            ev = (f"name is a FlyWire cell_type in the 2024 paper release of Schlegel SD1 ({int(fw_count_paper[t])} FlyWire cells) and "
                  f"Nern 2025 Supplementary Table 7 Schlegel_type confirms it; {SD1_SHORT} renamed the type to the Matsliah 2024 name "
                  f"given by MaleCNS flywireType (see the alias row)")
            flag = "paper_release_only"
            if t in fw_set:
                ev += f"; CAUTION: in {SD1_SHORT} the name '{t}' denotes a different type ({int(fw_count[t])} cells)"
                flag += ";v3_name_collision"
            add(t, t, "flywire", "exact", ev, flag)
        elif t in fw_set:
            fw_conflict += 1
        # ---- FlyWire: documented alias (MaleCNS annotation column)
        for v, k in fw_doc.items():
            elems = split_list(v)
            card = f"; MaleCNS 1 : FlyWire {len(elems)} (merged types)" if len(elems) > 1 else ""
            for e in elems:
                if e == t:
                    continue
                ev = f"MaleCNS v1.0 body-annotations flywireType='{v}' on {k}/{nc} cells{card}"
                if e not in fw_set:
                    ev += f"; name not in Schlegel 2024 {SD1_SHORT} cell_type list"
                if t in fw_set and t not in fw_doc_elems:
                    ev += f"; NAME COLLISION: a FlyWire type named '{t}' exists but MaleCNS maps this type elsewhere"
                add(t, e, "flywire", "alias", ev)
        # ---- hemibrain: exact
        hb_doc = col_hb[t]
        hb_doc_elems = set()
        for v in hb_doc:
            hb_doc_elems |= set(split_list(v))
        if t in hb_set:
            if not hb_doc or t in hb_doc_elems:
                cnt = int(hb_count.get(t, 0))
                ev = f"name is a hemibrain v1.2.1 type (Schlegel 2024 SD5, {cnt} hemibrain bodies)"
                ev += "; MaleCNS hemibrainType agrees" if hb_doc else "; MaleCNS hemibrainType empty"
                add(t, t, "hemibrain", "exact", ev, exact_flags(t, hb_doc, nc))
            else:
                hb_conflict += 1
        for v, k in hb_doc.items():
            elems = split_list(v.strip("()"))
            card = f"; MaleCNS 1 : hemibrain {len(elems)} (merged types)" if len(elems) > 1 else ""
            for e in elems:
                if e == t:
                    continue
                m = re.fullmatch(r"(?:\(?)(hb\d+)\)?", e)
                if m or e.startswith("hb"):
                    add(t, e.strip("()"), "hemibrain_body", "alias",
                        f"MaleCNS v1.0 hemibrainType='{v}' on {k}/{nc} cells: untyped hemibrain body id, no hemibrain type")
                    continue
                ev = f"MaleCNS v1.0 body-annotations hemibrainType='{v}' on {k}/{nc} cells{card}"
                if e not in hb_set:
                    ev += "; name not in hemibrain type list (Schlegel 2024 SD5 / SD1)"
                if t in hb_set and t not in hb_doc_elems:
                    ev += f"; NAME COLLISION: a hemibrain type named '{t}' exists but MaleCNS maps this type elsewhere"
                add(t, e, "hemibrain", "alias", ev)
        # ---- MANC
        for v, k in col_manc[t].items():
            for e in split_list(v):
                tier = "exact" if e == t else "alias"
                add(t, e, "manc", tier, f"MaleCNS v1.0 body-annotations mancType='{v}' on {k}/{nc} cells")
        # ---- optic inventory (Nern 2025)
        if t in nern1.index:
            r = nern1.loc[t]
            add(t, t, "optic_inventory", "exact",
                f"Nern 2025 Supplementary Table 1 row: {int(r.n)} cells, group {r.group}, predicted NT {r.nt}")
        # ---- parenthetical alias in the MaleCNS name
        m = re.fullmatch(r"(.+?)\((.+)\)", t)
        if m:
            add(t, m.group(1), "literature", "alias", "prefix of the MaleCNS type name before the parenthetical")
            add(t, m.group(2), "literature", "alias", "parenthetical alias inside the MaleCNS type name")
        # ---- explorer synonyms (neuPrint male-cns:v1.0 `synonyms`)
        for syn in ex_syn.get(t, []):
            for part in [p.strip() for p in syn.split(";") if p.strip()]:
                add(t, part, "literature", "alias", "MaleCNS Cell Type Explorer (neuPrint male-cns:v1.0 synonyms, 2026-09-09)")
        # ---- class
        cl = col_class[t]
        if cl:
            c, k = cl.most_common(1)[0]
            add(t, c, "malecns_class", "class", f"MaleCNS v1.0 body-annotations class on {k}/{nc} cells")

    # ---- Nern 2025 Supplementary Table 7 (optic lobe <-> FlyWire <-> hemibrain)
    for _, r in nern7.iterrows():
        t = r.OL_type
        if t not in type_set:
            continue
        base = f"Nern 2025 Supplementary Table 7 (preliminary), matched as {r['matched as']}"
        nt = ""
        if pd.notna(r.OL_transmitter_pred) and pd.notna(r.FW_transmitter_pred):
            nt = f"; NT pred OL={r.OL_transmitter_pred} FW={r.FW_transmitter_pred}" + \
                 (" (MISMATCH)" if r.OL_transmitter_pred != r.FW_transmitter_pred else "")
        for col, system in (("Schlegel_type", "flywire"), ("Matsliah_type", "flywire"), ("hemibrain_type", "hemibrain")):
            v = r[col]
            if pd.isna(v) or str(r["matched as"]) == "unmatched":
                continue
            v = str(v)
            for e in [x.strip() for x in re.split(r"[,+]", v) if x.strip()]:
                if e == t:
                    continue
                add(t, e, system, "alias", f"{base}, column {col}='{v}'{nt}")
        if str(r["matched as"]) == "unmatched":
            add(t, "(unmatched)", "flywire", "alias",
                f"{base}: no FlyWire match; {str(r['Notes']).strip()[:160] if pd.notna(r['Notes']) else ''}", "unmatched")

    # ---- Olivera 2026
    for t, a, note in olivera:
        add(t, a, "literature", "alias",
            "Olivera 2026 Neuroinformatics 24:25 doi:10.1007/s12021-026-09783-4 (cross-platform alias / NT ambiguity): " + note)

    # ---- Schlegel synonyms through the FlyWire link (exact or documented alias)
    fw_links = defaultdict(set)
    for r in rows:
        if r["system"] == "flywire" and r["tier"] in ("exact", "alias") and r["alias"] != "(unmatched)":
            fw_links[r["malecns_type"]].add(r["alias"])
    for t, fws in fw_links.items():
        for f in sorted(fws):
            for syn in fw_syn.get(f, []):
                for part in [p.strip() for p in syn.split(";") if p.strip()]:
                    add(t, part, "literature", "alias", f"Schlegel 2024 {SD1_SHORT} synonyms of FlyWire type {f}")

    # ---- fuzzy 1: transitive hemibrain through FlyWire (only where no exact / alias hemibrain row)
    for t, fws in fw_links.items():
        if (t, "hemibrain") in have:
            continue
        for f in sorted(fws):
            hb = fw_to_hb.get(f)
            if hb:
                for e in split_list(hb):
                    if e != t:
                        add(t, e, "hemibrain", "fuzzy",
                            f"rule transitive: MaleCNS->FlyWire '{f}' ({'exact' if f == t else 'MaleCNS flywireType'}) -> "
                            f"Schlegel 2024 SD1 hemibrain_type '{hb}' (majority over FlyWire cells)")
    # ---- fuzzy 2: connectivity-type suffix stripped (hemibrain morphology type)
    for t in types:
        if (t, "hemibrain") in have:
            continue
        m = re.fullmatch(r"(.+)_[a-z]$", t)
        if m and m.group(1) in hb_morph_types:
            add(t, m.group(1), "hemibrain", "fuzzy",
                "rule suffix: MaleCNS connectivity-type suffix _a/_b stripped; base name is a hemibrain morphology type "
                "(Schlegel 2024 SD5 morphology_type)")
    # ---- fuzzy 3: normalised-name equality with a unique target
    fw_norm = defaultdict(set)
    for f in fw_set:
        fw_norm[norm(f)].add(f)
    hb_norm = defaultdict(set)
    for h in hb_set:
        hb_norm[norm(h)].add(h)
    for t in types:
        nt_ = norm(t)
        if (t, "flywire") not in have:
            cands = fw_norm.get(nt_, set()) - {t}
            if len(cands) == 1:
                add(t, next(iter(cands)), "flywire", "fuzzy",
                    "rule normalised: case-insensitive, punctuation-stripped name equality with a unique FlyWire cell_type")
        if (t, "hemibrain") not in have:
            cands = hb_norm.get(nt_, set()) - {t}
            if len(cands) == 1:
                add(t, next(iter(cands)), "hemibrain", "fuzzy",
                    "rule normalised: case-insensitive, punctuation-stripped name equality with a unique hemibrain type")

    aliases = pd.DataFrame(rows).drop_duplicates(subset=["malecns_type", "alias", "system", "tier"])
    aliases["_r"] = aliases.tier.map(TIER_RANK)
    aliases = aliases.sort_values(["malecns_type", "system", "_r", "alias"]).drop(columns="_r").reset_index(drop=True)

    # ------------------------------------------------------------------ flags on alias rows
    def add_flag(mask, flag):
        cur = aliases.loc[mask, "flag"]
        aliases.loc[mask, "flag"] = np.where(cur == "", flag, cur + ";" + flag)

    fw_alias = (aliases.system == "flywire") & (aliases.tier == "alias") & (aliases.alias != "(unmatched)")
    absent_v3 = fw_alias & ~aliases.alias.isin(fw_set)
    add_flag(absent_v3 & aliases.alias.isin(fw_set_paper), "paper_release_only")
    unres_fw = absent_v3 & ~aliases.alias.isin(fw_set_paper)
    add_flag(unres_fw, "unresolvable")
    add_flag(absent_v3 & aliases.alias.str.startswith("CB"), "cb_placeholder")  # CB#### / CB.FB* placeholders, either release
    hb_alias = (aliases.system == "hemibrain") & (aliases.tier == "alias")
    unres_hb = hb_alias & ~aliases.alias.isin(hb_set)
    add_flag(unres_hb, "unresolvable")
    # Nern Table 7 FlyWire name vs MaleCNS majority flywireType (string comparison, as in the verification record)
    fw_major = {t: c.most_common(1)[0][0] for t, c in col_fw.items() if c}
    conflict_types, notation_types = [], []
    n_compared = 0
    for r in nern7.itertuples():
        t = r.OL_type
        cols = [str(v).strip() for v in (r.Schlegel_type, r.Matsliah_type) if pd.notna(v) and str(v).strip()]
        if t not in fw_major or not cols:
            continue
        n_compared += 1
        mv = fw_major[t]
        if mv in cols:
            continue
        t7_elems = set()
        for c in cols:
            t7_elems |= {x.strip() for x in re.split(r"[,+]", c) if x.strip()}
        if set(split_list(mv)) & t7_elems or any(re.fullmatch(re.escape(c) + r"[a-z]", mv) for c in cols):
            notation_types.append((t, cols, mv))
        else:
            conflict_types.append((t, cols, mv))
    conf_set = {t for t, _, _ in conflict_types}
    nota_set = {t for t, _, _ in notation_types}
    fw_alias_all = (aliases.system == "flywire") & (aliases.tier == "alias")
    add_flag(fw_alias_all & aliases.malecns_type.isin(conf_set), "conflict_nern7_vs_malecns")
    add_flag(fw_alias_all & aliases.malecns_type.isin(nota_set), "conflict_nern7_vs_malecns_notation")

    # ------------------------------------------------------------------ coverage
    def best_tiers(df: pd.DataFrame) -> dict:
        best = {}  # (type, system) -> best tier over the rows given
        for r in df.itertuples():
            if r.alias == "(unmatched)":
                continue
            key = (r.malecns_type, r.system)
            if key not in best or TIER_RANK[r.tier] < TIER_RANK[best[key]]:
                best[key] = r.tier
        return best

    main_systems = ["flywire", "hemibrain", "optic_inventory"]

    def any_of(best, systems):
        return lambda t: min((best[(t, sy)] for sy in systems if (t, sy) in best), key=lambda x: TIER_RANK[x], default="none")

    best = best_tiers(aliases)
    is_unres = aliases.flag.str.contains("unresolvable|unmatched", regex=True)
    is_strict_bad = is_unres | aliases.flag.str.contains("paper_release_only|conflict_nern7_vs_malecns(?!_notation)", regex=True)
    best_res = best_tiers(aliases[~is_unres])       # rows resolvable in SD1 v3.1.0 or the 2024 paper release / SD5
    best_strict = best_tiers(aliases[~is_strict_bad])  # additionally drops paper-release-only names and Nern-vs-MaleCNS conflicts
    per_type = pd.DataFrame({"type": types})
    for sy in main_systems + ["manc"]:
        per_type[sy] = per_type.type.map(lambda t: best.get((t, sy), "none"))
    per_type["any"] = per_type.type.map(any_of(best, main_systems))
    per_type["any_incl_manc"] = per_type.type.map(any_of(best, main_systems + ["manc"]))
    per_type["any_or_class"] = per_type.apply(lambda r: r["any"] if r["any"] != "none" else ("class" if (r.type, "malecns_class") in best else "none"), axis=1)
    per_type["flywire_resolvable"] = per_type.type.map(lambda t: best_res.get((t, "flywire"), "none"))
    per_type["hemibrain_resolvable"] = per_type.type.map(lambda t: best_res.get((t, "hemibrain"), "none"))
    per_type["any_resolvable"] = per_type.type.map(any_of(best_res, main_systems))
    per_type["any_strict"] = per_type.type.map(any_of(best_strict, main_systems))
    per_type["flywire_strict"] = per_type.type.map(lambda t: best_strict.get((t, "flywire"), "none"))
    per_type["n_cells"] = per_type.type.map(n_cells).astype(int)
    per_type["out_syn_W"] = per_type.type.map(typed.groupby("type").out_syn_W.sum())
    per_type["out_syn_raw"] = per_type.type.map(typed.groupby("type").out_syn_raw.sum())
    per_type["group"] = per_type.type.map(superclass_of).map(group_of)
    per_type["superclass"] = per_type.type.map(superclass_of)

    tot_cells = len(n)
    tot_W = float(n.out_syn_W.sum())
    tot_raw = int(n.out_syn_raw.sum())
    untyped = n[n.type == ""]

    def untyped_of(u: pd.DataFrame) -> dict:
        return {"cells": len(u), "out_syn_W": float(u.out_syn_W.sum()), "out_syn_raw": int(u.out_syn_raw.sum())}

    def table(df: pd.DataFrame, col: str, label: str, denom: dict, untyped_add: dict) -> str:
        """Tier rows of `df[col]`; the group's untyped cells (`untyped_add`) are added to the 'none' row and are part of
        the denominators (round-1 omitted them from 'none', so the rows did not sum to the denominator)."""
        order = ["exact", "alias", "fuzzy", "class", "none"]
        g = df.groupby(col).agg(types=("type", "size"), cells=("n_cells", "sum"), out_syn_W=("out_syn_W", "sum"),
                                out_syn_raw=("out_syn_raw", "sum")).reindex(order).fillna(0)
        for k in ("cells", "out_syn_W", "out_syn_raw"):
            g.loc["none", k] += untyped_add[k]
        d = denom
        lines = [f"### {label}", "", "| tier | types | cells | out synapses (abs W) | out synapses (raw) |", "|---|---|---|---|---|"]
        for tier, r in g.iterrows():
            if r.types == 0 and tier != "none":
                continue
            lines.append(f"| {tier} | {int(r.types):,} ({r.types / max(d['types'], 1):.1%}) | {int(r.cells):,} ({r.cells / max(d['cells'], 1):.1%}) "
                         f"| {int(r.out_syn_W):,} ({r.out_syn_W / max(d['out_syn_W'], 1):.1%}) | {int(r.out_syn_raw):,} ({r.out_syn_raw / max(d['out_syn_raw'], 1):.1%}) |")
        lines.append(f"| total (denominator) | {d['types']:,} | {int(d['cells']):,} | {int(d['out_syn_W']):,} | {int(d['out_syn_raw']):,} |")
        assert abs(g.cells.sum() - d["cells"]) < 1 and abs(g.out_syn_raw.sum() - d["out_syn_raw"]) < 1, (label, g.cells.sum(), d)
        return "\n".join(lines) + "\n"

    report = []
    report.append(f"MaleCNS node set: {tot_cells:,} cells, {int(tot_W):,} output synapses with a nonzero sign (abs W), "
                  f"{tot_raw:,} raw output synapses; typed cells {len(typed):,} ({len(typed) / tot_cells:.1%}) in {len(types):,} types; "
                  f"untyped {len(untyped):,} cells ({len(untyped) / tot_cells:.1%}) carrying {int(untyped.out_syn_W.sum()):,} abs-W / "
                  f"{int(untyped.out_syn_raw.sum()):,} raw output synapses ({untyped.out_syn_raw.sum() / tot_raw:.2%}) -- untyped cells are "
                  f"counted in the 'none' row of every table (type count 0) and in the denominators; the rows of every table sum to its denominator.\n")
    report.append(f"Name collisions suppressed from tier exact: FlyWire {fw_conflict} types, hemibrain {hb_conflict} types "
                  f"(the MaleCNS cross-dataset column names a different type; their documented alias rows carry a NAME COLLISION note).\n")
    whole = {"types": len(types), "cells": tot_cells, "out_syn_W": tot_W, "out_syn_raw": tot_raw}
    u_all = untyped_of(untyped)
    report.append(table(per_type, "any", "Whole CNS, best tier over FlyWire / hemibrain / optic inventory (denominators include untyped cells)", whole, u_all))
    report.append(table(per_type, "any_or_class", "Whole CNS, best tier including the class tier", whole, u_all))
    report.append(table(per_type, "any_incl_manc", "Whole CNS, best tier over FlyWire / hemibrain / optic inventory / MANC", whole, u_all))
    for sy in main_systems + ["manc"]:
        report.append(table(per_type, sy, f"Whole CNS, system {sy}", whole, u_all))
    for grp in ["optic", "visual_centrifugal", "central", "descending", "ascending", "vnc", "sensory_motor_other"]:
        sub = per_type[per_type.group == grp]
        ung = untyped[untyped.group == grp]
        u_g = untyped_of(ung)
        d = {"types": len(sub), "cells": sub.n_cells.sum() + len(ung), "out_syn_W": sub.out_syn_W.sum() + ung.out_syn_W.sum(),
             "out_syn_raw": sub.out_syn_raw.sum() + ung.out_syn_raw.sum()}
        report.append(table(sub, "any", f"Group {grp} (superclasses: {sorted(set(sub.superclass))}), best tier over the three systems; "
                                        f"{len(ung):,} untyped cells in the denominator and the 'none' row", d, u_g))
        for sy in main_systems + ["manc"]:
            if sy == "optic_inventory" and grp not in ("optic", "visual_centrifugal", "central"):
                continue
            if sy == "manc" and grp not in ("vnc", "descending", "ascending", "sensory_motor_other"):
                continue
            report.append(table(sub, sy, f"Group {grp}, system {sy}", d, u_g))
        if grp == "vnc":
            report.append(table(sub, "any_incl_manc", f"Group {grp}, best tier including MANC", d, u_g))
    # per superclass, any tier
    lines = ["### Per superclass, best tier over the three systems (typed cells only)", "",
             "| superclass | types | cells | exact | alias | fuzzy | none | cells exact | cells alias | cells fuzzy | cells none | raw out-syn exact+alias |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc, sub in per_type.groupby("superclass"):
        c = sub.groupby("any").size()
        cc = sub.groupby("any").n_cells.sum()
        ss = sub.groupby("any").out_syn_raw.sum()
        tot = sub.out_syn_raw.sum()
        lines.append(f"| {sc} | {len(sub)} | {sub.n_cells.sum():,} | {c.get('exact', 0)} | {c.get('alias', 0)} | {c.get('fuzzy', 0)} | {c.get('none', 0)} "
                     f"| {cc.get('exact', 0):,} | {cc.get('alias', 0):,} | {cc.get('fuzzy', 0):,} | {cc.get('none', 0):,} "
                     f"| {(ss.get('exact', 0) + ss.get('alias', 0)) / max(tot, 1):.1%} |")
    report.append("\n".join(lines) + "\n")
    # biggest unmatched types
    un = per_type[per_type["any"] == "none"].sort_values("out_syn_raw", ascending=False).head(40)
    lines = ["### The 40 unmatched types (no FlyWire / hemibrain / optic-inventory name at any tier) with the most raw output synapses", "",
             "| type | superclass | class | cells | out_syn_raw | out_syn_W |", "|---|---|---|---|---|---|"]
    for r in un.itertuples():
        cl = col_class[r.type].most_common(1)[0][0] if col_class[r.type] else ""
        lines.append(f"| {r.type} | {r.superclass} | {cl} | {r.n_cells} | {int(r.out_syn_raw):,} | {int(r.out_syn_W):,} |")
    report.append("\n".join(lines) + "\n")
    # system / tier row counts
    cnt = aliases.groupby(["system", "tier"]).size().unstack(fill_value=0)
    report.append("### Alias-table rows by system and tier\n\n" + cnt.to_markdown() + "\n")

    # ------------------------------------------------------------------ flags: counts and resolvable-only coverage
    fl = aliases[["system", "tier", "flag"]].assign(flag=aliases.flag.str.split(";")).explode("flag")
    fl = fl[fl.flag != ""]
    lines = ["### Flag counts (rows; a row can carry several flags)", "", "| flag | rows | systems / tiers |", "|---|---|---|"]
    for f, k in fl.flag.value_counts().items():
        sub = fl[fl.flag == f]
        lines.append(f"| {f} | {k:,} | {', '.join(f'{s}/{t} {c}' for (s, t), c in sub.groupby(['system', 'tier']).size().items())} |")
    report.append("\n".join(lines) + "\n")
    n_fw_alias = int(((aliases.system == "flywire") & (aliases.tier == "alias")).sum())
    n_absent = int((absent_v3 | ((aliases.system == "flywire") & (aliases.alias == "(unmatched)"))).sum())
    n_cb = int((absent_v3 & aliases.alias.str.startswith("CB")).sum())
    n_cb_unres = int((unres_fw & aliases.alias.str.startswith("CB")).sum())
    n_paper = int((absent_v3 & aliases.alias.isin(fw_set_paper)).sum())
    n_evflag = int((absent_v3 & aliases.evidence.str.contains("name not in")).sum())
    fw_alias_types = per_type[per_type.flywire == "alias"]
    lost_fw = fw_alias_types[fw_alias_types.flywire_resolvable == "none"]
    any_alias = per_type[per_type["any"] == "alias"]
    lost_any = any_alias[any_alias.any_resolvable == "none"]
    lost_strict = any_alias[any_alias.any_strict == "none"]
    fw_exact_types = per_type[per_type.flywire == "exact"]
    fw_exact_lost_strict = fw_exact_types[fw_exact_types.flywire_strict != "exact"]
    lost_fw_strict = fw_alias_types[fw_alias_types.flywire_strict == "none"]
    report.append(
        f"### Resolvability of the alias tier\n\n"
        f"* FlyWire alias rows: {n_fw_alias:,}; {n_absent:,} name a type absent from Schlegel SD1 v3.1.0 cell_type "
        f"({int(((aliases.system == 'flywire') & (aliases.alias == '(unmatched)')).sum())} '(unmatched)' rows included; {n_evflag:,} carry the "
        f"'name not in ... cell_type list' note in `evidence`, the Nern-Table-7-derived rows do not). Of the absent names {n_cb:,} are "
        f"'CB' placeholders (flag `cb_placeholder`; {n_cb_unres:,} of them in neither release), {n_paper:,} exist only in the 2024 paper release "
        f"(flag `paper_release_only`, resolvable with that release), and {int(unres_fw.sum()):,} are in neither release (flag `unresolvable`).\n"
        f"* hemibrain alias rows: {int(hb_alias.sum()):,}; {int(unres_hb.sum())} name a type absent from SD5 / SD1 hemibrain_type "
        f"({', '.join(sorted(aliases[unres_hb].alias))}) -> flag `unresolvable`.\n"
        f"* Nern Table 7 FlyWire name vs MaleCNS majority flywireType: {n_compared} types compared, {len(conf_set) + len(nota_set)} differ as strings; "
        f"{len(conf_set)} name different types (flag `conflict_nern7_vs_malecns`: {', '.join(f'{t} T7={cols} MaleCNS={mv}' for t, cols, mv in conflict_types)}); "
        f"{len(nota_set)} are notation / parent-subtype differences (flag `conflict_nern7_vs_malecns_notation`: {', '.join(t for t, _, _ in notation_types)}).\n"
        f"* Exact-row caveats (FlyWire / hemibrain): minority_agreement "
        f"{int(((aliases.system == 'flywire') & (aliases.tier == 'exact') & aliases.flag.str.contains('minority_agreement')).sum())} / "
        f"{int(((aliases.system == 'hemibrain') & (aliases.tier == 'exact') & aliases.flag.str.contains('minority_agreement')).sum())} types, merged_list_only "
        f"{int(((aliases.system == 'flywire') & (aliases.tier == 'exact') & aliases.flag.str.contains('merged_list_only')).sum())} / "
        f"{int(((aliases.system == 'hemibrain') & (aliases.tier == 'exact') & aliases.flag.str.contains('merged_list_only')).sum())}, name_only "
        f"{int(((aliases.system == 'flywire') & (aliases.tier == 'exact') & aliases.flag.str.contains('name_only')).sum())} / "
        f"{int(((aliases.system == 'hemibrain') & (aliases.tier == 'exact') & aliases.flag.str.contains('name_only')).sum())}, paper_release_only "
        f"{int(((aliases.system == 'flywire') & (aliases.tier == 'exact') & aliases.flag.str.contains('paper_release_only')).sum())} (of which v3_name_collision "
        f"{int(((aliases.system == 'flywire') & (aliases.tier == 'exact') & aliases.flag.str.contains('v3_name_collision')).sum())}); the merged_list_only "
        f"FlyWire types hold {int(per_type.set_index('type').n_cells.reindex(aliases[(aliases.system == 'flywire') & aliases.flag.str.contains('merged_list_only')].malecns_type).sum()):,} cells. "
        f"MANC 'exact' ({int(((aliases.system == 'manc') & (aliases.tier == 'exact')).sum()):,} types) is mancType == type from the MaleCNS column alone (no independent MANC type list).\n"
        f"* FlyWire alias-tier types: {len(fw_alias_types):,}; {len(lost_fw):,} of them ({int(lost_fw.n_cells.sum()):,} cells, {int(lost_fw.out_syn_raw.sum()):,} raw out-syn) "
        f"have no alias resolvable in either SD1 release. Best-of-three alias-tier types: {len(any_alias):,}; {len(lost_any):,} "
        f"({int(lost_any.n_cells.sum()):,} cells, {int(lost_any.out_syn_raw.sum()):,} raw) owe the tier solely to unresolvable names and drop to 'none' "
        f"when those rows are excluded. Under the strict rule (also excluding paper-release-only names and the {len(conf_set)} Nern-vs-MaleCNS conflicts) "
        f"{len(lost_strict):,} best-of-three alias-tier types ({int(lost_strict.n_cells.sum()):,} cells, {int(lost_strict.out_syn_raw.sum()):,} raw) drop to 'none'; "
        f"in the FlyWire system alone {len(lost_fw_strict):,} alias-tier types ({int(lost_fw_strict.n_cells.sum()):,} cells, {int(lost_fw_strict.out_syn_raw.sum()):,} raw) drop to 'none' "
        f"and the {len(fw_exact_lost_strict)} paper-release exact types ({', '.join(fw_exact_lost_strict.type)}; {int(fw_exact_lost_strict.n_cells.sum()):,} cells) fall back to their "
        f"MaleCNS-flywireType alias row (their optic_inventory exact row keeps the best-of-three tier at exact).\n")
    report.append(table(per_type, "any_resolvable", "Whole CNS, best tier over the three systems, unresolvable rows excluded "
                                                    "(flags unresolvable / unmatched dropped; paper-release-only names kept)", whole, u_all))
    report.append(table(per_type, "any_strict", "Whole CNS, best tier over the three systems, strict: unresolvable, paper-release-only and "
                                                "Nern-vs-MaleCNS conflict rows excluded", whole, u_all))
    report.append(table(per_type, "flywire_resolvable", "Whole CNS, system flywire, unresolvable rows excluded", whole, u_all))
    report.append(table(per_type, "hemibrain_resolvable", "Whole CNS, system hemibrain, unresolvable rows excluded", whole, u_all))

    # ------------------------------------------------------------------ by-product: Nern Table 7 NT predictions OL vs FlyWire
    t7 = nern7.rename(columns={"matched as": "matched_as"}).copy()
    ol_called = t7.OL_transmitter_pred.map(nt_called)
    fw_called = t7.FW_transmitter_pred.map(nt_called)
    both = t7[ol_called & fw_called]
    agree = both[both.OL_transmitter_pred.str.strip() == both.FW_transmitter_pred.str.strip()]
    dis = both[both.OL_transmitter_pred.str.strip() != both.FW_transmitter_pred.str.strip()]
    ol_unc = t7.OL_transmitter_pred.map(lambda v: isinstance(v, str) and v.strip().lower().startswith("unclear"))
    fw_unc = t7.FW_transmitter_pred.map(lambda v: isinstance(v, str) and v.strip().lower().startswith("unclear"))
    one_unclear = int(((ol_unc & fw_called) | (fw_unc & ol_called)).sum())
    any_unclear = int((ol_unc | fw_unc).sum())
    both_unclear = int((ol_unc & fw_unc).sum())
    unclear_vs_blank = int(((ol_unc & t7.FW_transmitter_pred.isna()) | (fw_unc & t7.OL_transmitter_pred.isna())).sum())
    lines = ["### By-product: Nern 2025 Table 7 transmitter predictions, optic lobe (male) vs FlyWire (female), where both are called", "",
             f"Rows with an OL_type ({len(t7)}; the OL_type-less LTe12 row is excluded) and a call on both sides: {len(both)}; agree {len(agree)}; "
             f"disagree {len(dis)}; a further {one_unclear} rows have `unclear` on one side and a call on the other "
             f"({any_unclear} rows have `unclear` on at least one side: OL unclear {int(ol_unc.sum())}, FW unclear {int(fw_unc.sum())}, "
             f"{both_unclear} both unclear, {unclear_vs_blank} unclear against a blank FW cell; {any_unclear} - {unclear_vs_blank} = "
             f"{any_unclear - unclear_vs_blank} is the verification record's count, which keeps the {both_unclear} both-unclear rows; "
             f"FW blank {int(t7.FW_transmitter_pred.isna().sum())}). "
             "The systematic pattern is glutamate (OL) vs GABA (FW) in Dm / Pm / Cm / LPi / Li types. MaleCNS model NT "
             "(`nt` column of cache/neurons.parquet, majority per type) added for reference.", "",
             "| OL_type | FlyWire (Schlegel) | matched as | OL NT pred | FW NT pred | MaleCNS model nt | cells |", "|---|---|---|---|---|---|---|"]
    for r in dis.sort_values("OL_type").itertuples():
        lines.append(f"| {r.OL_type} | {r.Schlegel_type} | {r.matched_as} | {r.OL_transmitter_pred.strip()} | {r.FW_transmitter_pred.strip()} "
                     f"| {nt_mode.get(r.OL_type, '')} | {int(n_cells.get(r.OL_type, 0))} |")
    report.append("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = ("# MaleCNS v1.0 cell-type aliases across connectomes. Built by scripts/build_type_map_typing.py from: MaleCNS v1.0 "
              "body-annotations (flywireType / hemibrainType / mancType / class; CC BY 4.0), Schlegel et al. 2024 Nature "
              f"doi:10.1038/s41586-024-07686-5 Supplementary Data 1 and 5 (github.com/flyconnectome/{SD1_RELEASE}, 'matching MaleCNS v1.0'; "
              "plus the 2024 paper release of SD1 for the old optic names; CC BY 4.0), "
              "Nern et al. 2025 Nature doi:10.1038/s41586-025-08746-0 Supplementary Tables 1 and 7 (CC BY 4.0), the MaleCNS Cell Type "
              "Explorer synonyms (github.com/reiserlab/celltype-explorer-drosophila-male-cns, 2026-09-09), and Olivera 2026 "
              "Neuroinformatics doi:10.1007/s12021-026-09783-4. Tiers: exact | alias | fuzzy | class (rules in the script docstring and "
              "docs/audits/receptor_sources_typing.md). Systems: flywire, hemibrain, hemibrain_body, optic_inventory, manc, literature, malecns_class. "
              "flag: ';'-joined caveats -- unresolvable (name absent from the source), cb_placeholder, paper_release_only, unmatched, "
              "conflict_nern7_vs_malecns[_notation], minority_agreement, merged_list_only, name_only, v3_name_collision (definitions in the script docstring); "
              "empty = no caveat. Downstream joins that trust tier 'alias' should drop rows flagged unresolvable / unmatched / conflict_nern7_vs_malecns.\n")
    # type_aliases.csv is the name the integration plan gives this table; type_aliases_typing.csv is a copy that a
    # concurrently running per-source builder cannot clobber (build_type_map_nern2025.py writes type_aliases_nern2025.csv).
    cols = ["malecns_type", "alias", "system", "tier", "flag", "evidence"]
    # The backend review records explicit notation/aggregate decisions in the
    # canonical CSV. Preserve that evidence when regenerating upstream aliases.
    # Names and targets still live only in the CSV, never in normalization code.
    canonical = OUT_DIR / "type_aliases.csv"
    if canonical.exists():
        old = pd.read_csv(canonical, comment="#").fillna("")
        keep = old.flag.map(lambda s: "backend_primary" in s.split(";"))
        keep |= old.evidence.str.contains("CONNECTOME_BACKENDS_SPEC section 3", regex=False)
        manual = old.loc[keep, cols]
        if len(manual):
            aliases = pd.concat([aliases, manual], ignore_index=True).drop_duplicates(
                ["malecns_type", "alias", "system"], keep="last")
    for name in ("type_aliases.csv", "type_aliases_typing.csv"):
        with open(OUT_DIR / name, "w", encoding="utf-8", newline="") as f:
            f.write(header)
            aliases[cols].to_csv(f, index=False)
    tm = aliases[aliases.alias != "(unmatched)"].rename(columns={"alias": "source_name"})
    tm["n_cells_malecns"] = tm.malecns_type.map(n_cells).astype(int)
    tm = tm[["source_name", "malecns_type", "tier", "flag", "evidence", "n_cells_malecns", "system"]]
    with open(OUT_DIR / "type_map_typing.csv", "w", encoding="utf-8", newline="") as f:
        f.write(header.replace("cell-type aliases across connectomes", "type map, source name -> MaleCNS type (same rows as type_aliases.csv minus the 13 '(unmatched)' rows, source-keyed)"))
        tm.to_csv(f, index=False)
    report.append(f"wrote flyverse/data/type_aliases.csv (= type_aliases_typing.csv; {len(aliases):,} data rows) and "
                  f"flyverse/data/type_map_typing.csv ({len(tm):,} data rows)\n")
    if REPORT_DIR.exists():
        per_type.to_csv(REPORT_DIR / "type_map_typing_per_type.csv", index=False)
        (REPORT_DIR / "type_map_typing_report.md").write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))


if __name__ == "__main__":
    main()
