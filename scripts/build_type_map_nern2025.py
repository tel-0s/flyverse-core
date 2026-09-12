"""Nern et al. 2025 (Nature, optic-lobe inventory) -> MaleCNS type map, alias table and coverage.

Inputs (git-ignored, see docs/audits/receptor_sources_nern2025.md for URLs and hashes):
  data/external/nern2025/nature_esm/MOESM4_unzipped/Sup_Table_1_Cell-types_and_counts_final.xlsx
      one row per instance (681 right-side + 97 left-side instances of 732 types): cell type, no. of cells,
      main group, predicted neurotransmitter (ACh / Glu / GABA / His / OA / 5HT / Dop / unclear).
  .../Sup_Table_5_Neurotransmitter_validation_final.xlsx
      148 rows / 147 distinct names (Tm29 twice) of experimentally validated types (FISH / bulk RNA-seq / antibody),
      the classifier's ground truth (Part_of_training_data = yes, 60 rows; the paper and gt_count.csv say 59
      types -- unreconciled) plus new validation data (no, 88 rows).
  .../Sup_Table_7_MatchingCellTypes_final.xlsx
      optic-lobe type -> FlyWire (Matsliah 2024 optic-lobe names, Schlegel 2024 whole-brain names) and
      hemibrain names, match cardinality, and both datasets' transmitter predictions.
  cache/neurons.parquet, cache/W_post_pre.npz (flyverse.connectome cache; W[post, pre] signed counts,
      sign-0 (monoamine / unknown) edges are explicit zeros).
  optional: the raw MaleCNS weights feather (uncapped, unsigned synapse counts) for a second weighting.

  optional: data/external/typing/schlegel2024_Supplemental_file1_neuron_annotations.tsv (flywire_annotations v3.1.0,
      commit 8587524, 2026-07-21) to flag FlyWire alias names absent from that release.

Outputs:
  flyverse/data/type_map_nern2025.csv   source_name, malecns_type, tier, evidence, n_cells_malecns + NT columns
  flyverse/data/type_aliases_nern2025.csv   malecns_type, alias, system, tier, flag, evidence (Sup_Table_7 + MaleCNS annotation columns)
      flag (round 2, verify:tables:nern2025): conflict_nern7_vs_malecns = the type's Sup_Table_7 FlyWire name and its
      MaleCNS majority flywireType name different types (25 types: Cm -> Sm off-by-one series, Li shifts, AOTU056, LoVP19, ...);
      conflict_nern7_vs_malecns_notation = the two differ as strings but are the same names written differently or a
      parent / subtype split (22 types; 47 string mismatches in all); absent_sd1_v3.1.0 = no element of the alias is a
      cell_type of flywire_annotations v3.1.0 (only when the SD1 file above is present); '' = no caveat.
  stdout: coverage by tier (types / cells / output synapses) and the NT cross-check tables, pasted into
          docs/audits/receptor_sources_nern2025.md.

CPU only (pandas + scipy); ~30 s with the raw weights, ~10 s without.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external/nern2025/nature_esm/MOESM4_unzipped"
T1_FILE = EXT / "Sup_Table_1_Cell-types_and_counts_final.xlsx"
T5_FILE = EXT / "Sup_Table_5_Neurotransmitter_validation_final.xlsx"
T7_FILE = EXT / "Sup_Table_7_MatchingCellTypes_final.xlsx"
RAW_WEIGHTS = Path(os.environ.get("FLYVERSE_DATA_DIR", r"D:\Datasets\male-cns-connectome-v1.0\flat-connectome")) / \
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
OUT_MAP = ROOT / "flyverse/data/type_map_nern2025.csv"
OUT_ALIAS = ROOT / "flyverse/data/type_aliases_nern2025.csv"  # type_aliases.csv is owned by build_type_map_typing.py
SD1_V31 = ROOT / "data/external/typing/schlegel2024_Supplemental_file1_neuron_annotations.tsv"  # flywire_annotations v3.1.0

OL_SUPERCLASSES = ["ol_intrinsic", "visual_projection", "ol_sensory", "visual_centrifugal"]
SOURCE_CITATION = ("Nern A, Loesche F, Takemura S-y, et al. (2025) Connectome-driven neural inventory of a complete "
                   "visual system. Nature 641, 1225-1237. doi:10.1038/s41586-025-08746-0 (CC BY 4.0)")

# Source vocabularies -> the model's (flyverse.connectome NT_SIGN keys).
NT_SHORT = {"ACh": "acetylcholine", "Ach": "acetylcholine", "Glu": "glutamate", "GABA": "gaba", "His": "histamine",
            "OA": "octopamine", "5HT": "serotonin", "Dop": "dopamine", "unclear": "unclear"}


def norm_nt(x) -> str:
    """Map a Nern label to the model vocabulary; multi-labels ('His, ACh') become 'a+b'; free text 'unclear(...)'
    becomes 'unclear'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s.lower().startswith("unclear"):
        return "unclear"
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        if p in NT_SHORT:
            out.append(NT_SHORT[p])
        elif p.lower() in {"acetylcholine", "glutamate", "gaba", "histamine", "octopamine", "serotonin", "dopamine"}:
            out.append(p.lower())
        else:
            raise ValueError(f"unknown NT label {x!r}")
    return "+".join(out)


def load_source():
    t1 = pd.read_excel(T1_FILE)
    t1.columns = [c.strip() for c in t1.columns]
    t1 = t1.rename(columns={"cell type": "type", "no. of cells": "n_cells", "main groups": "main_group",
                            "predicted neurotransmitter": "nt"})
    t1["side"] = t1.instance.astype(str).str[-1]
    t1["row_T1"] = t1.index + 2  # Excel row number (1-based, header = 1)
    rr = t1[t1.side == "R"].copy()
    l = t1[t1.side == "L"].copy()
    assert rr.type.is_unique and l.type.is_unique and t1.type.nunique() == 732, (len(rr), len(l), t1.type.nunique())
    # left-side instances: same type name, must not contradict the right-side NT call
    lr = l.merge(rr[["type", "nt"]], on="type", suffixes=("_L", "_R"))
    contradictions = lr[(lr.nt_L != lr.nt_R)]
    # one row per type: the right-side instance where it exists (681 types), else the left-only instance (51 types)
    r = pd.concat([rr, l[~l.type.isin(rr.type)]]).set_index("type")
    r["nt_norm"] = r.nt.map(norm_nt)
    r["n_cells_L"] = l.groupby("type").n_cells.sum().reindex(r.index).fillna(0).astype(int)
    r.loc[r.side == "L", "n_cells"] = 0  # n_cells = right-side count; left-only types have 0 there

    t5 = pd.read_excel(T5_FILE)
    t5.columns = [c.strip() for c in t5.columns]
    t5 = t5.rename(columns={"Cell Type": "type", "Driver line used": "driver", "Inferred transmitter": "nt",
                            "Reference(s)": "reference", "Part_of_training_data": "training",
                            "Observed  signal (new FISH data only)": "fish_signal", "Method": "method"})
    t5["type"] = t5.type.astype(str).str.strip()
    t5["nt_norm"] = t5.nt.map(norm_nt)
    t5["row_T5"] = t5.index + 2

    t7 = pd.read_excel(T7_FILE)
    t7.columns = [c.strip() for c in t7.columns]
    t7 = t7.rename(columns={"matched as": "matched_as", "OL_transmitter_pred": "ol_nt", "FW_transmitter_pred": "fw_nt"})
    t7["row_T7"] = t7.index + 2
    t7 = t7[t7.OL_type.notna()].copy()
    t7["OL_type"] = t7.OL_type.astype(str).str.strip()
    assert t7.OL_type.is_unique
    return r, l, contradictions, t5, t7


def load_malecns():
    neurons = pd.read_parquet(ROOT / "cache/neurons.parquet")
    W = sp.load_npz(ROOT / "cache/W_post_pre.npz").tocsr()
    neurons["out_syn_W"] = np.asarray(abs(W).sum(axis=0)).ravel()  # column sums: output of each presynaptic cell
    neurons["out_syn_raw"] = np.nan
    if RAW_WEIGHTS.exists():
        w = pd.read_feather(RAW_WEIGHTS, columns=["body_pre", "body_post", "weight"])
        nodes = pd.Series(np.arange(len(neurons)), index=neurons.bodyId.to_numpy())
        pre = nodes.reindex(w.body_pre.to_numpy()).to_numpy()
        post = nodes.reindex(w.body_post.to_numpy()).to_numpy()
        m = ~np.isnan(pre) & ~np.isnan(post)
        out = np.bincount(pre[m].astype(np.int64), weights=w.weight.to_numpy()[m], minlength=len(neurons))
        neurons["out_syn_raw"] = out
        raw_edges = (pre[m].astype(np.int64), post[m].astype(np.int64), w.weight.to_numpy()[m].astype(np.float64))
        del w
    else:
        raw_edges = None
        print(f"[warn] raw weights not found at {RAW_WEIGHTS}; synapse weighting uses c.W only", file=sys.stderr)
    neurons["type"] = neurons.type.fillna("")
    return neurons, W, raw_edges


CLASS_BIN_RE = re.compile(r"^(?P<prefix>.+)_unclear$")


def class_members(prefix: str, source_types: set[str]) -> list[str]:
    """Nern types that are subtypes of `prefix` (the MaleCNS '<prefix>_unclear' bins).

    prefix ending in a digit  ('T4', 'Tm3', 'LC10'):   prefix itself, or prefix + lowercase subtype letters
                                                        (+ optional '-n'): T4a..T4d, Tm3, LC10a..LC10e, LC10c-1.
    prefix ending in a letter ('TmY', 'Pm', 'T5a'):    the prefix itself or prefix + digits + optional subtype letters: T5a, TmY3, TmY19a,
                                                        Pm2a, LPi3412; 'Tm' does not capture 'TmY3' (letter
                                                        before the digits) and 'Li' does not capture 'Lai'.
    """
    if prefix[-1].isdigit():
        pat = re.compile(r"^" + re.escape(prefix) + r"([a-z]+(-[0-9]+)?)?$")
    else:  # the prefix itself is allowed too: 'T5a_unclear' -> T5a
        pat = re.compile(r"^" + re.escape(prefix) + r"([0-9]+[A-Za-z]*(-[0-9]+)?)?$")
    return sorted(t for t in source_types if pat.match(t))


def build_type_map(src, t5, t7, neurons):
    src_types = set(src.index)
    by_type = neurons.groupby("type")
    n_cells = by_type.size()
    out_W = by_type.out_syn_W.sum()
    out_raw = by_type.out_syn_raw.sum(min_count=1)
    sc_mode = by_type.superclass.agg(lambda s: s.fillna("").mode().iloc[0] if len(s) else "")
    nt_mode = by_type.nt.agg(lambda s: s.mode().iloc[0])
    nt_share = by_type.nt.agg(lambda s: (s == s.mode().iloc[0]).mean())
    n_unknown = by_type.nt.agg(lambda s: int((s == "unknown").sum()))

    t5i = t5.drop_duplicates("type").set_index("type")
    t7i = t7.set_index("OL_type")

    def nt_cols(nern_types: list[str], nern_nt: str, evidence_extra: str = ""):
        return dict(nern_nt=nern_nt)

    rows = []
    # --- tier exact: identical type name in both inventories (all 732 Nern types)
    for t, s in src.iterrows():
        assert t in n_cells.index, t
        v = t5i.loc[t] if t in t5i.index else None
        m7 = t7i.loc[t] if t in t7i.index else None
        rows.append(dict(
            source_name=t, malecns_type=t, tier="exact",
            evidence=f"identical name; Sup_Table_1 row {s.row_T1} (instance {s.instance}, {s.n_cells} cells, {s.main_group})",
            n_cells_malecns=int(n_cells[t]),
            nern_main_group=s.main_group, nern_n_cells_R=int(s.n_cells), nern_n_cells_L=int(s.n_cells_L),
            nern_nt=s.nt_norm,
            fw_nt=norm_nt(m7.fw_nt) if m7 is not None else "", t7_matched_as=m7.matched_as if m7 is not None else "",
            validated_nt=v.nt_norm if v is not None else "",
            validated_method=str(v.method).strip() if v is not None else "",
            validated_reference=str(v.reference).strip() if v is not None and pd.notna(v.reference) else "",
            validated_in_training=str(v.training).strip() if v is not None else "",
        ))
    # --- MaleCNS optic-lobe-superclass types absent from the Nern inventory
    ol = neurons[neurons.superclass.isin(OL_SUPERCLASSES)]
    missing = sorted(set(ol.type.unique()) - src_types - {""})
    for t in missing:
        m = CLASS_BIN_RE.match(t)
        members, prefixes = [], []
        if m:
            prefixes = ["R7", "R8"] if m["prefix"] == "R7R8" else [m["prefix"]]
            for p in prefixes:
                members += class_members(p, src_types)
        if members:
            nts = src.loc[members, "nt_norm"]
            counts = nts.value_counts().to_dict()
            called = nts[nts != "unclear"]
            # class label only if every *called* subtype agrees and at least half of the subtypes are called
            # (an 'unclear' subtype is absence of evidence, not contrary evidence)
            if called.nunique() == 1 and len(called) * 2 >= len(members):
                unclear = sorted(nts.index[nts == "unclear"])
                ev = (f"MaleCNS '{t}' bin = unresolved subtype of {'/'.join(prefixes)}; all {len(called)} called "
                      f"Nern subtypes ({', '.join(called.index)}) predicted {called.iloc[0]} (Sup_Table_1)")
                if unclear:
                    ev += f"; {len(unclear)} subtype(s) unclear ({', '.join(unclear)})"
                rows.append(dict(source_name="|".join(members), malecns_type=t, tier="class", evidence=ev,
                                 n_cells_malecns=int(n_cells[t]), nern_main_group=src.loc[members[0], "main_group"],
                                 nern_n_cells_R=int(src.loc[members, "n_cells"].sum()),
                                 nern_n_cells_L=int(src.loc[members, "n_cells_L"].sum()), nern_nt=called.iloc[0],
                                 fw_nt="", t7_matched_as="", validated_nt="", validated_method="",
                                 validated_reference="", validated_in_training=""))
                continue
            ev = (f"MaleCNS '{t}' bin spans {len(members)} Nern {'/'.join(prefixes)} subtypes whose predictions "
                  f"disagree ({', '.join(f'{k} {v}' for k, v in counts.items())}); no class-level label")
        elif m:
            ev = f"MaleCNS '{t}' is a region / class bin with no Nern type of that prefix"
        elif t.startswith(("OCG", "OCC")):
            ev = "ocellar type; the ocelli are outside the Nern 2025 optic-lobe inventory"
        elif t == "LoVP109":
            ev = "Sup_Table_7 row 734 note: 'LoVP109 (named based on LM data, not found in OL dataset)'"
        elif t == "Pm7_Li28":
            ev = (f"MaleCNS composite of Pm7 ({src.loc['Pm7', 'nt_norm']}) and Li28 ({src.loc['Li28', 'nt_norm']}); "
                  "not a Nern type")
        else:
            ev = "no Nern type of this name or prefix"
        rows.append(dict(source_name="", malecns_type=t, tier="unmatched", evidence=ev,
                         n_cells_malecns=int(n_cells[t]), nern_main_group="", nern_n_cells_R=0, nern_n_cells_L=0,
                         nern_nt="", fw_nt="", t7_matched_as="", validated_nt="", validated_method="",
                         validated_reference="", validated_in_training=""))
    df = pd.DataFrame(rows)
    df["malecns_superclass"] = df.malecns_type.map(sc_mode)
    df["malecns_nt"] = df.malecns_type.map(nt_mode)
    df["malecns_nt_share"] = df.malecns_type.map(nt_share).round(3)
    df["malecns_n_unknown_nt"] = df.malecns_type.map(n_unknown).astype(int)
    df["out_syn_W"] = df.malecns_type.map(out_W).round(0).astype(int)
    df["out_syn_raw"] = df.malecns_type.map(out_raw)
    return df


def t7_vs_malecns_conflicts(t7, neurons):
    """Types whose Sup_Table_7 FlyWire name (Schlegel_type / Matsliah_type) and MaleCNS majority flywireType differ as
    strings (the verify:tables:nern2025 rule, 47 types). Returns (conflict, notation): `conflict` = no name element in
    common and not a parent / subtype pair (different types named); `notation` = same names written differently
    ('LTe49a,b,d,e,f' vs 'LTe49a,LTe49b,...') or a parent / subtype split ('MeTu2' vs 'MeTu2a')."""
    typed = neurons[(neurons.type != "") & neurons.flywireType.notna()]
    fw_major = {t: s.value_counts().index[0] for t, s in typed.groupby("type").flywireType}
    conflict, notation = {}, {}
    n_compared = 0
    for r in t7.itertuples():
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
        mv_elems = {x.strip() for x in re.split(r"[,/]", mv) if x.strip()}
        if mv_elems & t7_elems or any(re.fullmatch(re.escape(c) + r"[a-z]", mv) for c in cols):
            notation[t] = (cols, mv)
        else:
            conflict[t] = (cols, mv)
    return conflict, notation, n_compared


def build_aliases(t7, neurons):
    rows = []
    systems = [("Matsliah_type", "flywire_matsliah2024", "Matsliah"), ("Schlegel_type", "flywire_schlegel2024", None),
               ("hemibrain_type", "hemibrain", None)]
    conflict, notation, n_compared = t7_vs_malecns_conflicts(t7, neurons)
    fw_v31 = None
    if SD1_V31.exists():
        fw_v31 = set(pd.read_csv(SD1_V31, sep="\t", low_memory=False, usecols=["cell_type"]).cell_type.dropna())

    def flag_of(t, alias, system):
        f = []
        if system in ("flywire_matsliah2024", "flywire_schlegel2024", "malecns_flywireType"):
            if t in conflict:
                f.append("conflict_nern7_vs_malecns")
            elif t in notation:
                f.append("conflict_nern7_vs_malecns_notation")
            if fw_v31 is not None and alias != t:
                elems = {x.strip() for x in re.split(r"[,+/]", alias) if x.strip()} | {alias}
                if not (elems & fw_v31):
                    f.append("absent_sd1_v3.1.0")
        return ";".join(f)

    for _, r in t7.iterrows():
        for col, system, _ in systems:
            a = r[col]
            if pd.isna(a) or str(a).strip() == "":
                continue
            a = str(a).strip()
            composite = bool(re.search(r"[,+]| and |\s\+\s", a)) or a.startswith("hb")
            if r.matched_as == "1-to-1" and not composite:
                tier = "exact" if a == r.OL_type else "alias"
            else:
                tier = "fuzzy"
            ev = f"Sup_Table_7 row {r.row_T7}: matched as {r.matched_as}"
            if composite:
                ev += "; composite alias string (several source types)"
            if pd.notna(r.get("Notes")) and str(r.Notes).strip():
                ev += "; note: " + str(r.Notes).strip().replace("\n", " ")
            rows.append(dict(malecns_type=r.OL_type, alias=a, system=system, tier=tier,
                             flag=flag_of(r.OL_type, a, system), evidence=ev))
    # MaleCNS's own per-cell annotation columns, summarised per type (majority string, share of annotated cells)
    for col, system in [("flywireType", "malecns_flywireType"), ("hemibrainType", "malecns_hemibrainType")]:
        sub = neurons[(neurons.type != "") & neurons[col].notna() & (neurons[col].astype(str).str.strip() != "")]
        g = sub.groupby("type")[col]
        for t, s in g:
            vc = s.value_counts()
            a = str(vc.index[0]); share = vc.iloc[0] / len(s); n_ann = len(s); n_tot = int((neurons.type == t).sum())
            composite = bool(re.search(r"[,+]", a))
            if a == t and share >= 0.9 and not composite:
                tier = "exact"
            elif share >= 0.9 and not composite:
                tier = "alias"
            else:
                tier = "fuzzy"
            ev = f"MaleCNS v1.0 annotation column {col}: majority value in {vc.iloc[0]}/{n_ann} annotated cells ({n_tot} cells in type)"
            if len(vc) > 1:
                ev += "; other values: " + "; ".join(f"{k} {v}" for k, v in vc.iloc[1:4].items())
            rows.append(dict(malecns_type=t, alias=a, system=system, tier=tier, flag=flag_of(t, a, system), evidence=ev))
    df = pd.DataFrame(rows)
    print(f"\n## Sup_Table_7 FlyWire name vs MaleCNS majority flywireType: {n_compared} types compared, "
          f"{len(conflict) + len(notation)} differ as strings; {len(conflict)} name different types (flag conflict_nern7_vs_malecns), "
          f"{len(notation)} are notation / parent-subtype differences (flag conflict_nern7_vs_malecns_notation)")
    for t, (cols, mv) in conflict.items():
        print(f"  conflict  {t:12s} Sup_Table_7 {cols}  MaleCNS flywireType {mv!r}")
    print("  notation  " + ", ".join(notation))
    if fw_v31 is None:
        print(f"  (SD1 v3.1.0 not found at {SD1_V31}; absent_sd1_v3.1.0 flags not computed)")
    return df


def write_csv(df: pd.DataFrame, path: Path, header_lines: list[str]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write("# " + line + "\n")
        df.to_csv(f, index=False)


def coverage(df_map, neurons, W, raw_edges):
    tier_of = df_map.set_index("malecns_type").tier
    neurons = neurons.copy()
    neurons["tier"] = neurons.type.map(tier_of).fillna("unmatched")
    neurons.loc[neurons.type == "", "tier"] = "untyped"
    neurons["group"] = np.where(neurons.superclass.isin(OL_SUPERCLASSES), neurons.superclass, "central_or_vnc")
    # matched cells that the four-superclass filter puts in 'central_or_vnc' although they are optic-lobe cells
    # (superclass '<x>_tbc'): named here so the 'central_or_vnc class' row is not read as a central cell
    tbc = neurons[(neurons.group == "central_or_vnc") & neurons.tier.isin(["exact", "class"]) &
                  neurons.superclass.fillna("").str.endswith("_tbc")]
    if len(tbc):
        print("\n## Matched cells with a '_tbc' superclass counted under central_or_vnc: " +
              "; ".join(f"{r.type} ({r.superclass}, tier {r.tier})" for r in tbc.itertuples()))
    tiers = ["exact", "class", "unmatched", "untyped"]
    out = []
    for grp, g in list(neurons.groupby("group")) + [("ALL", neurons)]:
        tot_c, tot_w, tot_r = len(g), g.out_syn_W.sum(), g.out_syn_raw.sum()
        for tier in tiers:
            s = g[g.tier == tier]
            out.append(dict(group=grp, tier=tier, n_types=s.type.nunique(), n_cells=len(s),
                            cells_frac=len(s) / tot_c, out_syn_W=s.out_syn_W.sum(), out_syn_W_frac=s.out_syn_W.sum() / tot_w,
                            out_syn_raw=s.out_syn_raw.sum(), out_syn_raw_frac=s.out_syn_raw.sum() / tot_r if tot_r else np.nan))
    cov = pd.DataFrame(out)
    # 'unclear' Nern labels: cells and output synapses of the exact types whose nern_nt is unclear (doc section 7 recount)
    ol_mask = neurons.superclass.isin(OL_SUPERCLASSES)
    ol_raw = neurons.loc[ol_mask, "out_syn_raw"].sum()
    unc_types = set(df_map[(df_map.tier == "exact") & (df_map.nern_nt == "unclear")].malecns_type)
    unc_cells = neurons[neurons.type.isin(unc_types)]
    unc_ol = df_map[(df_map.tier == "exact") & (df_map.nern_nt == "unclear") & df_map.malecns_superclass.isin(OL_SUPERCLASSES)]
    unc_ol_cells = neurons[neurons.type.isin(set(unc_ol.malecns_type))]
    print(f"\n## Nern 'unclear' exact types: {len(unc_types)} types, {len(unc_cells):,} MaleCNS cells (all superclasses), "
          f"{int(unc_cells.out_syn_raw.sum()):,} raw out syn = {unc_cells.out_syn_raw.sum() / ol_raw:.2%} of the {int(ol_raw):,} "
          f"OL-superclass raw output synapses; the {len(unc_ol)} OL-superclass ones: {len(unc_ol_cells):,} cells, "
          f"{int(unc_ol_cells.out_syn_raw.sum()):,} raw = {unc_ol_cells.out_syn_raw.sum() / ol_raw:.2%}; "
          f"the other {len(unc_types) - len(unc_ol)} unclear types are central ({len(unc_cells) - len(unc_ol_cells):,} cells)")
    # edges with both pre and post matched (exact or class)
    matched = neurons.tier.isin(["exact", "class"]).to_numpy()
    pre_ol = neurons.superclass.isin(OL_SUPERCLASSES).to_numpy()
    Wc = W.tocoo()
    a = np.abs(Wc.data)
    both = matched[Wc.row] & matched[Wc.col]
    edge = {"W_all_both_matched": a[both].sum() / a.sum(),
            "W_olpre_both_matched": a[both & pre_ol[Wc.col]].sum() / a[pre_ol[Wc.col]].sum(),
            "W_olpre_pre_matched": a[matched[Wc.col] & pre_ol[Wc.col]].sum() / a[pre_ol[Wc.col]].sum(),
            "W_all_pre_matched": a[matched[Wc.col]].sum() / a.sum()}
    if raw_edges is not None:
        pre, post, wt = raw_edges
        both = matched[pre] & matched[post]
        edge.update({"raw_all_both_matched": wt[both].sum() / wt.sum(),
                     "raw_olpre_both_matched": wt[both & pre_ol[pre]].sum() / wt[pre_ol[pre]].sum(),
                     "raw_olpre_pre_matched": wt[matched[pre] & pre_ol[pre]].sum() / wt[pre_ol[pre]].sum(),
                     "raw_all_pre_matched": wt[matched[pre]].sum() / wt.sum()})
    return cov, edge, neurons


def crosscheck(df_map, neurons):
    m = df_map[df_map.tier.isin(["exact", "class"])].copy()
    called = lambda s: s.notna() & ~s.isin(["", "unclear", "unknown"])

    def table(a, b, label):
        both = m[called(m[a]) & called(m[b])]
        agree = both[both[a] == both[b]]
        dis = both[both[a] != both[b]]
        print(f"\n## {label}: {len(both)} types with both called; agree {len(agree)} "
              f"({agree.n_cells_malecns.sum():,} cells, {int(agree.out_syn_raw.sum()) if agree.out_syn_raw.notna().any() else agree.out_syn_W.sum():,} raw out syn); "
              f"disagree {len(dis)} ({dis.n_cells_malecns.sum():,} cells, {int(dis.out_syn_raw.sum()) if dis.out_syn_raw.notna().any() else dis.out_syn_W.sum():,} raw out syn)")
        if len(dis):
            cols = list(dict.fromkeys(["malecns_type", "malecns_superclass", "n_cells_malecns", "out_syn_raw", a, b,
                                       "validated_nt", "validated_method", "validated_in_training"]))
            print(dis.sort_values("out_syn_raw", ascending=False)[cols].to_string(index=False))
        return both, agree, dis

    r = {}
    r["nern_vs_malecns"] = table("nern_nt", "malecns_nt", "Nern 2025 prediction vs MaleCNS model label (cache nt, per-type mode)")
    r["nern_vs_fw"] = table("nern_nt", "fw_nt", "Nern 2025 prediction vs FlyWire prediction (Sup_Table_7)")
    r["nern_vs_val"] = table("nern_nt", "validated_nt", "Nern 2025 prediction vs experimental validation (Sup_Table_5)")
    r["malecns_vs_val"] = table("malecns_nt", "validated_nt", "MaleCNS model label vs experimental validation (Sup_Table_5)")
    r["fw_vs_val"] = table("fw_nt", "validated_nt", "FlyWire prediction vs experimental validation (Sup_Table_5)")
    # rescue of unknown-NT cells
    unk = m[(m.malecns_n_unknown_nt > 0)]
    resc = unk[called(unk.nern_nt)]
    print(f"\n## MaleCNS cells with model nt = unknown whose type is matched: {int(unk.malecns_n_unknown_nt.sum()):,} cells in "
          f"{len(unk)} types; Nern gives a called NT for {len(resc)} types / {int(resc.malecns_n_unknown_nt.sum()):,} cells")
    if len(resc):
        print(resc.sort_values("malecns_n_unknown_nt", ascending=False)[["malecns_type", "malecns_superclass", "n_cells_malecns", "malecns_n_unknown_nt", "malecns_nt", "nern_nt", "fw_nt", "validated_nt"]].head(40).to_string(index=False))
    # types where the model label is unknown/unclear-majority and Nern is unclear too
    watch = ["Tm5Y", "TmY14", "Mi19", "LoVCLo2", "LoVC18", "LoVC22", "LoVCLo3", "OA-AL2i1", "OA-AL2i2", "OA-ASM1", "T1", "Tm29", "Dm9", "Mi13", "TmY16", "Tm40", "Tm26", "Lat3", "Lat4"]
    print("\n## Watch-list types (audit candidates)")
    print(m[m.malecns_type.isin(watch)][["malecns_type", "n_cells_malecns", "out_syn_raw", "malecns_nt", "malecns_nt_share", "nern_nt", "fw_nt", "validated_nt", "validated_method", "validated_in_training"]].to_string(index=False))
    return r


def main():
    src, left, contradictions, t5, t7 = load_source()
    n_r = int((src.side == "R").sum())
    print(f"Sup_Table_1: {n_r + len(left)} rows = {n_r} _R + {len(left)} _L instances; {len(src)} distinct types "
          f"({n_r} with a right-side row, {len(src) - n_r} left-only); L/R NT contradictions: {len(contradictions)}")
    if len(contradictions):
        print(contradictions[["type", "nt_L", "nt_R"]].to_string(index=False))
    print("Sup_Table_1 NT:", src.nt_norm.value_counts().to_dict())
    print(f"Sup_Table_5: {len(t5)} rows, {t5.type.nunique()} distinct names; rows in Sup_Table_1: {t5.type.isin(src.index).sum()} "
          f"({t5[t5.type.isin(src.index)].type.nunique()} distinct names; Tm29 twice); "
          f"not in Sup_Table_1: {sorted(set(t5.type) - set(src.index))}")
    print("Sup_Table_5 method:", t5.method.value_counts().to_dict())
    print("Sup_Table_5 training:", t5.training.value_counts().to_dict())
    print(f"Sup_Table_7: {len(t7)} OL types; matched_as: {t7.matched_as.value_counts().to_dict()}")
    neurons, W, raw_edges = load_malecns()
    df_map = build_type_map(src, t5, t7, neurons)
    write_csv(df_map, OUT_MAP, [
        "Type map: Nern et al. 2025 optic-lobe inventory -> MaleCNS v1.0 type names. Built by scripts/build_type_map_nern2025.py.",
        "Source: " + SOURCE_CITATION + "; Supplementary Tables 1, 5, 7 (MOESM4 zip), hashes in docs/audits/receptor_sources_nern2025.md.",
        "tier: exact = identical type name in both inventories (the 684 optic-lobe-superclass rows are verified same cells: identical counts and figure bodyIds;",
        "      the 48 central rows are same-name only -- MaleCNS v1.0 has partly re-annotated those cells since the optic-lobe release, e.g. AOTU056's figure",
        "      bodyId 66210 is typed AOTU058 in MaleCNS); class = MaleCNS '<prefix>_unclear' bin whose Nern subtypes all carry one prediction;",
        "      unmatched = MaleCNS optic-lobe-superclass type with no Nern counterpart (evidence says why). No alias / fuzzy rows were needed: all 732 Nern names are MaleCNS names.",
        "nern_nt: Sup_Table_1 'predicted neurotransmitter' (EM synapse classifier trained on optic-lobe ground-truth types: 59 per the paper / gt_count.csv,",
        "      60 rows with Part_of_training_data = yes in Sup_Table_5 -- unreconciled; 'unclear' = low confidence, not independently confirmed).",
        "fw_nt: Sup_Table_7 FlyWire prediction for the matched FlyWire type (Eckstein et al. 2024 classifier, female brain). validated_*: Sup_Table_5 experimental transmitter",
        "      (FISH / bulk RNA-seq / antibody); validated_in_training = yes means the type was classifier ground truth, so it is not an independent check of nern_nt.",
        "malecns_*: from cache/neurons.parquet (model nt after the AL-LN override; mode over the type's cells, share of cells with that mode, cells with nt = unknown);",
        "out_syn_W = sum |c.W| over the type's output edges (sign-0 edges are zero); out_syn_raw = uncapped MaleCNS synapse count on the model's node set.",
        "NT vocabulary: acetylcholine gaba glutamate histamine dopamine octopamine serotonin unclear; 'a+b' = source lists two.",
    ])
    df_alias = build_aliases(t7, neurons)
    write_csv(df_alias, OUT_ALIAS, [
        "Cross-dataset type aliases for MaleCNS v1.0 type names. Built by scripts/build_type_map_nern2025.py.",
        "system flywire_matsliah2024 / flywire_schlegel2024 / hemibrain: Sup_Table_7 of " + SOURCE_CITATION + " (preliminary male optic lobe -> female FlyWire matches;",
        "      Matsliah = FlyWire optic-lobe names (Matsliah et al. 2024), Schlegel = FlyWire whole-brain names (Schlegel et al. 2024), hemibrain = Scheffer et al. 2020 names).",
        "system malecns_flywireType / malecns_hemibrainType: the MaleCNS v1.0 per-cell annotation columns (CC BY, male-cns.janelia.org), majority value per type.",
        "tier: exact = alias identical to the MaleCNS name (1-to-1 / >=90% of annotated cells); alias = different name, 1-to-1 or >=90% of annotated cells;",
        "      fuzzy = n-to-1 / 1-to-n match, composite alias string, or majority < 90% (evidence gives the cardinality or the shares).",
        "flag: conflict_nern7_vs_malecns = the type's Sup_Table_7 FlyWire name and MaleCNS majority flywireType name different types (Cm -> Sm off-by-one, Li shifts,",
        "      AOTU056, LoVP19, ...); conflict_nern7_vs_malecns_notation = same names written differently or a parent / subtype split; absent_sd1_v3.1.0 = no element",
        "      of the alias is a cell_type in flywire_annotations v3.1.0 (commit 8587524, 2026-07-21); empty = no caveat. Do not trust tier 'alias' on a flagged row.",
    ])
    print(f"\nwrote {OUT_MAP} ({len(df_map)} rows: {df_map.tier.value_counts().to_dict()})")
    print(f"wrote {OUT_ALIAS} ({len(df_alias)} rows: {df_alias.groupby(['system', 'tier']).size().to_dict()})")

    cov, edge, neurons2 = coverage(df_map, neurons, W, raw_edges)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
    print("\n## Coverage by group and tier (cells; out_syn_W = |c.W| column sums, sign-0 edges zero; out_syn_raw = uncapped counts)")
    print(cov.to_string(index=False, float_format=lambda x: f"{x:,.4f}" if abs(x) < 1.5 else f"{x:,.0f}"))
    print("\n## Synapse-level coverage (both pre and post type matched at tier exact or class)")
    for k, v in edge.items():
        print(f"{k}: {v:.4f}")
    crosscheck(df_map, neurons)


if __name__ == "__main__":
    main()
