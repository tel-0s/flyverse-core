"""Build the cross-connectome alias table for MaleCNS cell-type names (the "typing" source of docs/NT_INTEGRATION.md).

Inputs (data/external/typing/, git-ignored; provenance and hashes in docs/audits/receptor_sources_typing.md):
  * cache/neurons.parquet                                  MaleCNS v1.0 node table (type, flywireType, hemibrainType, mancType, class, ...)
  * schlegel2024_Supplemental_file1_neuron_annotations.tsv  FlyWire v783 typing (cell_type, hemibrain_type, synonyms, known_nt)
  * schlegel2024_Supplemental_file5_hemibrain_meta.csv      hemibrain v1.2.1 type list (type, morphology_type)
  * nern2025_MOESM4/Sup_Table_1_Cell-types_and_counts_final.xlsx   optic-lobe inventory (732 types, predicted NT)
  * nern2025_MOESM4/Sup_Table_7_MatchingCellTypes_final.xlsx       optic-lobe <-> FlyWire (Matsliah / Schlegel) <-> hemibrain matches
  * reiser_malecns_explorer_neurons.json                    MaleCNS Cell Type Explorer (neuPrint male-cns:v1.0 flywireType + published synonyms)
  * Olivera 2026 Neuroinformatics 10.1007/s12021-026-09783-4 (two aliases, hard-coded below with the citation)

Outputs:
  * flyverse/data/type_aliases.csv      malecns_type, alias, system, tier, evidence
  * flyverse/data/type_map_typing.csv   source_name, malecns_type, tier, evidence, n_cells_malecns, system
  * stdout: coverage tables (also pasted into docs/audits/receptor_sources_typing.md)

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
sys.path.insert(0, str(ROOT))
from flyverse.connectome import DATA_DIR  # noqa: E402

RAW_WEIGHTS = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather"

OPTIC_SUPERCLASSES = {"ol_intrinsic", "visual_projection", "ol_sensory"}
TIER_RANK = {"exact": 0, "alias": 1, "fuzzy": 2, "class": 3}


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

    def add(t, alias, system, tier, evidence):
        if not alias or alias == "":
            return
        rows.append({"malecns_type": t, "alias": alias, "system": system, "tier": tier, "evidence": evidence})
        key = (t, system)
        if tier in ("exact", "alias") and (key not in have or TIER_RANK[tier] < TIER_RANK[have[key]]):
            have[key] = tier

    fw_conflict = hb_conflict = 0
    for t in types:
        nc = int(n_cells[t])
        # ---- FlyWire: exact
        fw_doc = col_fw[t]
        fw_doc_elems = set()
        for v in fw_doc:
            fw_doc_elems |= set(split_list(v))
        if t in fw_set and (not fw_doc or t in fw_doc_elems):
            ev = f"name is a FlyWire v783 cell_type (Schlegel 2024 SD1 v3.0.0, {int(fw_count[t])} FlyWire cells)"
            ev += "; MaleCNS flywireType agrees" if fw_doc else "; MaleCNS flywireType empty"
            add(t, t, "flywire", "exact", ev)
        elif nern7_schlegel.get(t) == t and t in fw_set_paper:
            # Nern 2025 Table 7 uses the 2024 paper-release FlyWire names; SD1 v3.0.0 renamed optic types to the
            # Matsliah 2024 convention (Tm5Y -> Tm5f, LPi12 -> LPi14), and the MaleCNS flywireType column carries the v3 name.
            ev = (f"name is a FlyWire cell_type in the 2024 paper release of Schlegel SD1 ({int(fw_count_paper[t])} FlyWire cells) and "
                  f"Nern 2025 Supplementary Table 7 Schlegel_type confirms it; SD1 v3.0.0 renamed the type to the Matsliah 2024 name "
                  f"given by MaleCNS flywireType (see the alias row)")
            if t in fw_set:
                ev += f"; CAUTION: in SD1 v3.0.0 the name '{t}' denotes a different type ({int(fw_count[t])} cells)"
            add(t, t, "flywire", "exact", ev)
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
                    ev += "; name not in Schlegel 2024 SD1 v3.0.0 cell_type list"
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
                add(t, t, "hemibrain", "exact", ev)
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
            add(t, "(unmatched)", "flywire", "alias", f"{base}: no FlyWire match; {str(r['Notes']).strip()[:160] if pd.notna(r['Notes']) else ''}")

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
                    add(t, part, "literature", "alias", f"Schlegel 2024 SD1 v3.0.0 synonyms of FlyWire type {f}")

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

    # ------------------------------------------------------------------ coverage
    best = {}  # (type, system) -> best tier over all rows
    for r in aliases.itertuples():
        if r.alias == "(unmatched)":
            continue
        key = (r.malecns_type, r.system)
        if key not in best or TIER_RANK[r.tier] < TIER_RANK[best[key]]:
            best[key] = r.tier
    main_systems = ["flywire", "hemibrain", "optic_inventory"]
    per_type = pd.DataFrame({"type": types})
    for sy in main_systems + ["manc"]:
        per_type[sy] = per_type.type.map(lambda t: best.get((t, sy), "none"))
    per_type["any"] = per_type.apply(
        lambda r: min((r[sy] for sy in main_systems if r[sy] != "none"), key=lambda x: TIER_RANK[x], default="none"), axis=1)
    per_type["any_incl_manc"] = per_type.apply(
        lambda r: min((r[sy] for sy in main_systems + ["manc"] if r[sy] != "none"), key=lambda x: TIER_RANK[x], default="none"), axis=1)
    per_type["any_or_class"] = per_type.apply(lambda r: r["any"] if r["any"] != "none" else ("class" if (r.type, "malecns_class") in best else "none"), axis=1)
    per_type["n_cells"] = per_type.type.map(n_cells).astype(int)
    per_type["out_syn_W"] = per_type.type.map(typed.groupby("type").out_syn_W.sum())
    per_type["out_syn_raw"] = per_type.type.map(typed.groupby("type").out_syn_raw.sum())
    per_type["group"] = per_type.type.map(superclass_of).map(group_of)
    per_type["superclass"] = per_type.type.map(superclass_of)

    tot_cells = len(n)
    tot_W = float(n.out_syn_W.sum())
    tot_raw = int(n.out_syn_raw.sum())
    untyped = n[n.type == ""]

    def table(df: pd.DataFrame, col: str, label: str, denom: dict | None = None) -> str:
        order = ["exact", "alias", "fuzzy", "class", "none"]
        g = df.groupby(col).agg(types=("type", "size"), cells=("n_cells", "sum"), out_syn_W=("out_syn_W", "sum"),
                                out_syn_raw=("out_syn_raw", "sum")).reindex(order).fillna(0)
        d = denom or {"types": len(df), "cells": df.n_cells.sum(), "out_syn_W": df.out_syn_W.sum(), "out_syn_raw": df.out_syn_raw.sum()}
        lines = [f"### {label}", "", "| tier | types | cells | out synapses (abs W) | out synapses (raw) |", "|---|---|---|---|---|"]
        for tier, r in g.iterrows():
            if r.types == 0 and tier != "none":
                continue
            lines.append(f"| {tier} | {int(r.types):,} ({r.types / max(d['types'], 1):.1%}) | {int(r.cells):,} ({r.cells / max(d['cells'], 1):.1%}) "
                         f"| {int(r.out_syn_W):,} ({r.out_syn_W / max(d['out_syn_W'], 1):.1%}) | {int(r.out_syn_raw):,} ({r.out_syn_raw / max(d['out_syn_raw'], 1):.1%}) |")
        lines.append(f"| total (denominator) | {d['types']:,} | {int(d['cells']):,} | {int(d['out_syn_W']):,} | {int(d['out_syn_raw']):,} |")
        return "\n".join(lines) + "\n"

    report = []
    report.append(f"MaleCNS node set: {tot_cells:,} cells, {int(tot_W):,} output synapses with a nonzero sign (abs W), "
                  f"{tot_raw:,} raw output synapses; typed cells {len(typed):,} ({len(typed) / tot_cells:.1%}) in {len(types):,} types; "
                  f"untyped {len(untyped):,} cells ({len(untyped) / tot_cells:.1%}) carrying {int(untyped.out_syn_W.sum()):,} abs-W / "
                  f"{int(untyped.out_syn_raw.sum()):,} raw output synapses ({untyped.out_syn_raw.sum() / tot_raw:.2%}) -- untyped cells are "
                  f"'none' in every table and are included in the whole-CNS denominators below.\n")
    report.append(f"Name collisions suppressed from tier exact: FlyWire {fw_conflict} types, hemibrain {hb_conflict} types "
                  f"(the MaleCNS cross-dataset column names a different type; their documented alias rows carry a NAME COLLISION note).\n")
    whole = {"types": len(types), "cells": tot_cells, "out_syn_W": tot_W, "out_syn_raw": tot_raw}
    report.append(table(per_type, "any", "Whole CNS, best tier over FlyWire / hemibrain / optic inventory (denominators include untyped cells)", whole))
    report.append(table(per_type, "any_or_class", "Whole CNS, best tier including the class tier", whole))
    report.append(table(per_type, "any_incl_manc", "Whole CNS, best tier over FlyWire / hemibrain / optic inventory / MANC", whole))
    for sy in main_systems + ["manc"]:
        report.append(table(per_type, sy, f"Whole CNS, system {sy}", whole))
    for grp in ["optic", "visual_centrifugal", "central", "descending", "ascending", "vnc", "sensory_motor_other"]:
        sub = per_type[per_type.group == grp]
        ung = untyped[untyped.group == grp]
        d = {"types": len(sub), "cells": sub.n_cells.sum() + len(ung), "out_syn_W": sub.out_syn_W.sum() + ung.out_syn_W.sum(),
             "out_syn_raw": sub.out_syn_raw.sum() + ung.out_syn_raw.sum()}
        report.append(table(sub, "any", f"Group {grp} (superclasses: {sorted(set(sub.superclass))}), best tier over the three systems; "
                                        f"{len(ung):,} untyped cells added to the denominator", d))
        for sy in main_systems + ["manc"]:
            if sy == "optic_inventory" and grp not in ("optic", "visual_centrifugal", "central"):
                continue
            if sy == "manc" and grp not in ("vnc", "descending", "ascending", "sensory_motor_other"):
                continue
            report.append(table(sub, sy, f"Group {grp}, system {sy}", d))
        if grp == "vnc":
            report.append(table(sub, "any_incl_manc", f"Group {grp}, best tier including MANC", d))
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

    # ------------------------------------------------------------------ write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = ("# MaleCNS v1.0 cell-type aliases across connectomes. Built by scripts/build_type_map_typing.py from: MaleCNS v1.0 "
              "body-annotations (flywireType / hemibrainType / mancType / class; CC BY 4.0), Schlegel et al. 2024 Nature "
              "doi:10.1038/s41586-024-07686-5 Supplementary Data 1 and 5 (github.com/flyconnectome/flywire_annotations v3.0.0; CC BY 4.0), "
              "Nern et al. 2025 Nature doi:10.1038/s41586-025-08746-0 Supplementary Tables 1 and 7 (CC BY 4.0), the MaleCNS Cell Type "
              "Explorer synonyms (github.com/reiserlab/celltype-explorer-drosophila-male-cns, 2026-09-09), and Olivera 2026 "
              "Neuroinformatics doi:10.1007/s12021-026-09783-4. Tiers: exact | alias | fuzzy | class (rules in the script docstring and "
              "docs/audits/receptor_sources_typing.md). Systems: flywire, hemibrain, hemibrain_body, optic_inventory, manc, literature, malecns_class.\n")
    # type_aliases.csv is the name the integration plan gives this table; type_aliases_typing.csv is a copy that a
    # concurrently running per-source builder cannot clobber (build_type_map_nern2025.py writes type_aliases.csv too).
    for name in ("type_aliases.csv", "type_aliases_typing.csv"):
        with open(OUT_DIR / name, "w", encoding="utf-8", newline="") as f:
            f.write(header)
            aliases[["malecns_type", "alias", "system", "tier", "evidence"]].to_csv(f, index=False)
    tm = aliases[aliases.alias != "(unmatched)"].rename(columns={"alias": "source_name"})
    tm["n_cells_malecns"] = tm.malecns_type.map(n_cells).astype(int)
    tm = tm[["source_name", "malecns_type", "tier", "evidence", "n_cells_malecns", "system"]]
    with open(OUT_DIR / "type_map_typing.csv", "w", encoding="utf-8", newline="") as f:
        f.write(header.replace("cell-type aliases across connectomes", "type map, source name -> MaleCNS type (same rows as type_aliases.csv, source-keyed)"))
        tm.to_csv(f, index=False)
    per_type.to_csv(ROOT / "out" / "type_map_typing_per_type.csv", index=False) if (ROOT / "out").exists() else None

    print("\n".join(report))
    print(f"\nwrote {OUT_DIR / 'type_aliases.csv'} ({len(aliases):,} rows) and {OUT_DIR / 'type_map_typing.csv'} ({len(tm):,} rows)")


if __name__ == "__main__":
    main()
