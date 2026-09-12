"""Build the Davis et al. 2020 (GSE116969) -> MaleCNS type map, per-type expression table and coverage report.

Run from anywhere:  PYTHONIOENCODING=utf-8 python scripts/build_davis2020_tables.py [--report PATH.json]
(CPU only, ~1 min; needs cache/neurons.parquet + cache/W_post_pre.npz from connectome.load()).

Inputs (data/external/davis2020/, git-ignored; `python scripts/fetch_data.py --external davis2020`):
  GSE116969_dataTable4.genes_x_cells_TPM.coding_genes_QCpass.txt.gz   cell-type mean TPM, QC-pass samples (13,931 coding genes x 79 cells)
  GSE116969_dataTable7b.genes_x_cells_p_expression.modeled_genes.txt.gz  P(expressed) per cell type for the 12,377 modeled genes (77 cells)
  GSE116969_dataTable3.genes_x_samples_TPM.coding_genes.txt.gz        per-sample TPM (all 266 samples, incl. QC-fail) -> used only for drivers with no QC-pass sample
  elife-50901-supp1-v2.xlsx                                            sheet A (drivers, cell types) and B (samples, QC)
Outputs:
  flyverse/data/type_map_davis2020.csv
  flyverse/data/expression_davis2020.csv
  --report PATH.json (optional): coverage / cross-check / gene-range numbers quoted in docs/audits/receptor_sources_davis2020.md;
  a summary is always printed to stdout.

Round-2 changes (docs/audits/receptor_verification.md, verify:tables:davis2020 + verify:derive): repo-relative paths;
KaiR1D := CG3822 (FBgn0038837) instead of CG8916 (FBgn0030707, Lcch-14A); QC-pass-only coverage no longer drops a type that
a pass driver AND a suboptimal-only driver both reach; class rows Ilp2 -> IPC, Crz -> CRZ01/CRZ02, Dsk -> DSKMP3,
NPF -> NPFL1-I; Lat / lLNv evidence strings from supp. 1A; T4 / T5 pooled profiles flagged subtype-biased; type counts
exclude the untyped pseudo-type; gene-range statistics (Nmdar2 / VGAT / Tdc2 / DopEcR / KaiR1D) with explicit denominators.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "davis2020"
OUT = ROOT / "flyverse" / "data"
CACHE = ROOT / "cache"

# ----------------------------------------------------------------------------------------------
# 1. Gene list (panel name -> FlyBase symbol used in the Davis tables, ENSEMBL r91 / FlyBase 2017_04).
#    54 genes: 12 synthesis / transport, 21 fast ionotropic, 21 slow metabotropic.
#    KaiR1D = CG3822 (FBgn0038837; NCBI Gene 42473); CG8916 (FBgn0030707, Lcch-14A) is NOT KaiR1D (round-1 error).
#    Octalpha2R = CG18208 (FBgn0038653).
# ----------------------------------------------------------------------------------------------
GENES = [
    # synthesis / transport
    ("ChAT", "ChAT"), ("VAChT", "VAChT"), ("Gad1", "Gad1"), ("VGAT", "VGAT"), ("VGlut", "VGlut"),
    ("Hdc", "Hdc"), ("ple", "ple"), ("DAT", "DAT"), ("Tdc2", "Tdc2"), ("Tbh", "Tbh"), ("Trh", "Trh"), ("SerT", "SerT"),
    # fast receptors
    ("nAChRalpha1", "nAChRalpha1"), ("nAChRalpha2", "nAChRalpha2"), ("nAChRalpha3", "nAChRalpha3"),
    ("nAChRalpha4", "nAChRalpha4"), ("nAChRalpha5", "nAChRalpha5"), ("nAChRalpha6", "nAChRalpha6"),
    ("nAChRalpha7", "nAChRalpha7"), ("nAChRbeta1", "nAChRbeta1"), ("nAChRbeta2", "nAChRbeta2"), ("nAChRbeta3", "nAChRbeta3"),
    ("Rdl", "Rdl"), ("Lcch3", "Lcch3"), ("Grd", "Grd"), ("GluClalpha", "GluClalpha"),
    ("KaiR1D", "CG3822"), ("GluRIA", "GluRIA"), ("GluRIB", "GluRIB"), ("Nmdar1", "Nmdar1"), ("Nmdar2", "Nmdar2"),
    ("HisCl1", "HisCl1"), ("ort", "ort"),
    # slow receptors
    ("mAChR-A", "mAChR-A"), ("mAChR-B", "mAChR-B"), ("mAChR-C", "mAChR-C"),
    ("GABA-B-R1", "GABA-B-R1"), ("GABA-B-R2", "GABA-B-R2"), ("GABA-B-R3", "GABA-B-R3"), ("mGluR", "mGluR"),
    ("Dop1R1", "Dop1R1"), ("Dop1R2", "Dop1R2"), ("Dop2R", "Dop2R"), ("DopEcR", "DopEcR"),
    ("Oamb", "Oamb"), ("Octbeta1R", "Octbeta1R"), ("Octbeta2R", "Octbeta2R"), ("Octbeta3R", "Octbeta3R"), ("Octalpha2R", "CG18208"),
    ("5-HT1A", "5-HT1A"), ("5-HT1B", "5-HT1B"), ("5-HT2A", "5-HT2A"), ("5-HT2B", "5-HT2B"), ("5-HT7", "5-HT7"),
]
GENE_FBGN = {"CG3822": "FBgn0038837", "CG18208": "FBgn0038653"}   # checked against dataTable3 gene_id below
assert len(GENES) == 54, len(GENES)

NON_NEURONAL = {"Glia_Eg", "Glia_Mg", "Glia_Psg", "Muscles_App", "Muscles_Head", "lamina", "opticlobe"}
CLASS_DRIVERS = {"ChAT", "Gad1", "VGlut"}
PHOTORECEPTORS = {"R1-6", "R7", "R8_Rh5", "R8_Rh6"}
ON_P, ON_TPM = 0.5, 10.0     # 'on' rule shared with scripts/build_receptor_table.py: P(on) >= 0.5 AND TPM >= 10

TIER_ORDER = ["exact", "alias", "fuzzy", "class", "unmatched"]
OPTIC_SC = {"ol_intrinsic", "visual_projection", "ol_sensory"}


def load_sources():
    tpm_cells = pd.read_csv(EXT / "GSE116969_dataTable4.genes_x_cells_TPM.coding_genes_QCpass.txt.gz", sep="\t", index_col=0)
    p_cells = pd.read_csv(EXT / "GSE116969_dataTable7b.genes_x_cells_p_expression.modeled_genes.txt.gz", sep="\t", index_col=0)
    tpm_samples = pd.read_csv(EXT / "GSE116969_dataTable3.genes_x_samples_TPM.coding_genes.txt.gz", sep="\t")
    for sym, fbgn in GENE_FBGN.items():
        got = tpm_samples.loc[tpm_samples.gene_name == sym, "gene_id"].tolist()
        assert got == [fbgn], (sym, got)
    tpm_samples = tpm_samples.drop_duplicates("gene_name").set_index("gene_name").drop(columns=["gene_id"])
    xl = pd.ExcelFile(EXT / "elife-50901-supp1-v2.xlsx")
    drivers = xl.parse("A_all_drivers")
    samples = xl.parse("B_RNAseq_samples")
    for _, sym in GENES:
        assert sym in tpm_cells.index, sym
        assert sym in p_cells.index, sym
    assert set(tpm_cells.columns) - set(p_cells.columns) == {"lamina", "opticlobe"}, set(tpm_cells.columns) - set(p_cells.columns)
    return tpm_cells, p_cells, tpm_samples, drivers, samples


def build_expression(tpm_cells, p_cells, tpm_samples, samples):
    """TPM from table 4 (QC-pass cell means); for suboptimal-only drivers the mean of their table-3 samples (flagged);
    p_on from table 7b (NaN if not modeled / not QC-pass)."""
    cell_cols = list(tpm_cells.columns)
    samples = samples.copy()
    samples["simpleCellTypes"] = samples["simpleCellTypes"].astype(str).str.strip()
    qc_pass_n = samples[samples.qualityControl == "pass"].groupby("simpleCellTypes").size()
    qc_sub_n = samples[samples.qualityControl == "suboptimal"].groupby("simpleCellTypes").size()
    sub_only = sorted(set(qc_sub_n.index) - set(qc_pass_n.index) - {"nan"})
    rows = []
    for cell in cell_cols:
        r = {"source_name": cell, "qc": "pass", "n_samples_pass": int(qc_pass_n.get(cell, 0)), "n_samples_suboptimal": int(qc_sub_n.get(cell, 0))}
        for name, sym in GENES:
            r[f"{name}_tpm"] = float(tpm_cells.at[sym, cell])
        for name, sym in GENES:
            r[f"{name}_p_on"] = float(p_cells.at[sym, cell]) if cell in p_cells.columns else np.nan
        rows.append(r)
    for cell in sub_only:
        cols = [c for c in tpm_samples.columns if c.startswith(cell + "_d")]
        if not cols:
            continue
        r = {"source_name": cell, "qc": "suboptimal_only", "n_samples_pass": 0, "n_samples_suboptimal": len(cols)}
        for name, sym in GENES:
            r[f"{name}_tpm"] = float(tpm_samples.loc[sym, cols].mean())
        for name, _ in GENES:
            r[f"{name}_p_on"] = np.nan
        rows.append(r)
    expr = pd.DataFrame(rows)

    # transmitter call from synthesis / transport genes (data, not a decision)
    NT_MARKERS = {"acetylcholine": ("ChAT", "VAChT"), "gaba": ("Gad1", "VGAT"), "glutamate": ("VGlut", "VGlut"),
                  "histamine": ("Hdc", "Hdc"), "dopamine": ("ple", "DAT"), "octopamine": ("Tdc2", "Tbh"), "serotonin": ("Trh", "SerT")}

    def nt_call(r, thr=10.0, thr2=50.0):
        """primary = class with the largest min(TPM) over its marker pair, if >= thr; secondary = other classes >= thr2
        (thr2 is above the ~150-300 TPM VGAT / ~10 TPM Tdc2 background seen in every sample, which the pair-min already suppresses)."""
        scores = {k: min(r[f"{a}_tpm"], r[f"{b}_tpm"]) for k, (a, b) in NT_MARKERS.items()}
        ranked = sorted([(v, k) for k, v in scores.items()], reverse=True)
        primary = ranked[0][1] if ranked[0][0] >= thr else "none"
        secondary = "+".join(k for v, k in ranked[1:] if v >= thr2)
        return primary, secondary, ";".join(f"{k}={v:.0f}" for v, k in ranked[:3])

    expr["davis_nt_call"], expr["davis_nt_secondary"], expr["davis_nt_scores_top3"] = zip(*expr.apply(nt_call, axis=1))
    return expr


def gene_ranges(expr):
    """Ranges / 'on' rates for the genes whose round-1 statements were refuted, over explicit denominators:
    neuronal_69 = QC-pass cell-type rows minus glia / muscle / tissue minus the ChAT / Gad1 / VGlut class drivers;
    neuronal_72 = + the 3 class drivers; qcpass_79 = every table-4 column (glia and muscle included)."""
    qp = expr[expr.qc == "pass"].set_index("source_name")
    dens = {"neuronal_69": qp.index.difference(NON_NEURONAL | CLASS_DRIVERS),
            "neuronal_72": qp.index.difference(NON_NEURONAL), "qcpass_79": qp.index}
    out = {}
    for g in ["Nmdar2", "VGAT", "Tdc2", "DopEcR", "KaiR1D", "GluRIA", "GluRIB", "Nmdar1", "GluClalpha"]:
        d = {}
        for dn, idx in dens.items():
            t, p = qp.loc[idx, f"{g}_tpm"], qp.loc[idx, f"{g}_p_on"]
            on = (p >= ON_P) & (t >= ON_TPM)
            d[dn] = {"n": int(len(idx)), "tpm_min": round(float(t.min()), 2), "tpm_max": round(float(t.max()), 1),
                     "tpm_argmin": str(t.idxmin()), "tpm_argmax": str(t.idxmax()),
                     "p_on_mean": round(float(p.mean()), 3), "p_on_ge_0.5": int((p >= 0.5).sum()), "p_on_eq_1": int((p == 1).sum()),
                     "on_rule": int(on.sum()), "on_rule_frac": round(float(on.mean()), 3), "off_rows": sorted(idx[~on].tolist())}
        idx = dens["neuronal_69"].difference(PHOTORECEPTORS)
        t, p = qp.loc[idx, f"{g}_tpm"], qp.loc[idx, f"{g}_p_on"]
        d["neuronal_69_minus_photoreceptors"] = {"n": int(len(idx)), "tpm_min": round(float(t.min()), 2), "tpm_max": round(float(t.max()), 1),
                                                 "tpm_argmin": str(t.idxmin()), "p_on_eq_1": int((p == 1).sum()),
                                                 "on_rule": int(((p >= ON_P) & (t >= ON_TPM)).sum())}
        d["photoreceptors"] = {r: {"tpm": round(float(qp.at[r, f"{g}_tpm"]), 2), "p_on": float(qp.at[r, f"{g}_p_on"])} for r in sorted(PHOTORECEPTORS)}
        d["top10_neuronal_72"] = qp.loc[dens["neuronal_72"], f"{g}_tpm"].sort_values(ascending=False).head(10).round(1).to_dict()
        out[g] = d
    # T1 synthesis-enzyme statement
    t1 = qp.loc["T1"]
    out["T1_markers_tpm"] = {g: round(float(t1[f"{g}_tpm"]), 2) for g in ["ChAT", "VAChT", "Gad1", "VGAT", "VGlut", "Hdc", "ple", "DAT", "Tdc2", "Tbh", "Trh", "SerT"]}
    out["VGAT_L1_L2_Mi1_tpm"] = {c: round(float(qp.at[c, "VGAT_tpm"]), 1) for c in ["L1", "L2", "Mi1"]}
    return out


def build_map(expr, neurons, drv_details):
    M = []  # (source, [malecns types], tier, evidence)

    def add(src, targets, tier, ev):
        M.append((src, targets, tier, ev))

    EX = "same type name in Davis 2020 (supp. file 1A, driver %s) and MaleCNS v1.0; MaleCNS flywireType = %s"
    exact_simple = ["C2", "C3", "Dm1", "Dm4", "Dm9", "Dm10", "Dm11", "Dm12", "L1", "L2", "L3", "L4", "L5", "Lai", "Lawf1", "Lawf2",
                    "LC10a", "LC10b", "LC16", "LC4", "LC6", "LLPC1", "LPC1", "LPLC1", "LPLC2", "Mi1", "Mi15", "Mi4", "Mi9",
                    "T1", "Tm1", "Tm2", "Tm20", "Tm3", "Tm4", "Tm9", "TmY3", "TmY5a", "LC10d"]
    for t in exact_simple:
        fw = neurons.loc[neurons.type == t, "flywireType"].value_counts().index.tolist()[:1]
        add(t, [t], "exact", EX % (drv_details.loc[t, "driverIDs"] if t in drv_details.index else "?", fw[0] if fw else "n/a"))
    add("Pm3", ["Pm3"], "exact", "same name; Davis driver SS00328 identified by 'Cell type description: Nern et al 2015'; Nern et al. 2025 (same annotator) kept the Nern-2015 Pm1-Pm4 names (Pm2 split into Pm2a/b); MaleCNS flywireType = Pm09 (FlyWire renumbered Pm types, so the FlyWire name is NOT a check here)")
    add("Pm4", ["Pm4"], "exact", "same name; Davis driver SS00317 identified by 'Cell type description: Nern et al 2015'; Nern et al. 2025 kept the Nern-2015 Pm names; MaleCNS flywireType = Pm05 (FlyWire renumbered Pm types)")
    add("Tm29", ["Tm29"], "exact", "type first named in Davis 2020 supp. 1A ('Tentatively named Tm29', driver SS00308, Tm5b-like multicolumnar); MaleCNS keeps the name Tm29 (flywireType Tm5d)")
    add("LPi-34", ["LPi34"], "alias", "hyphen dropped; Davis driver SS03656 'Cell type description: Maus et al 2015' (LPi3-4 = lobula-plate intrinsic layer 3->4); MaleCNS LPi34 (flywireType LPi09)")
    add("R1-6", ["R1-R6"], "alias", "Davis 'R1-6' (ninaEfl-GAL4 = Rh1 outer photoreceptors) = MaleCNS 'R1-R6' (flywireType R1-6)")
    add("R8_Rh5", ["R8p"], "alias", "Davis R8_Rh5 (BL-7458, Rh5-GAL4) = pale R8; MaleCNS R8p (pale). Excludes R8_unclear (442 cells) and R8d (DRA R8 express Rh3, not Rh5)")
    add("R8_Rh6", ["R8y"], "alias", "Davis R8_Rh6 (BL-7464, Rh6-GAL4) = yellow R8; MaleCNS R8y (yellow). Excludes R8_unclear (442 cells)")
    add("R7", ["R7p", "R7y", "R7d"], "fuzzy", "rule: Davis R7 driver BL-8604 pools 'R7: Rh3 expressing, R7: Rh4 expressing' (+ dorsal-rim R8 per supp. 1A) -> every MaleCNS R7 subtype (pale R7p = Rh3, yellow R7y = Rh4, DRA R7d = Rh3); R7_unclear (404 cells) excluded because it may contain R8 fragments")
    add("R7_Rh3", ["R7p", "R7d"], "fuzzy", "rule: Rh3-GAL4 (BL-7457) labels pale R7 and DRA R7 (both Rh3); QC-fail samples only")
    add("T4", ["T4a", "T4b", "T4c", "T4d"], "fuzzy", "rule: Davis T4 drivers SS02344 / SS23866 label all T4 but supp. 1A records both as 'T4: strongest in T4b,T4c' -> the pooled profile is SUBTYPE-BIASED towards T4b / T4c when used per subtype; the a-d subtypes are split by lobula-plate layer in the EM; source name is the prefix of the MaleCNS subtype names")
    add("T5", ["T5a", "T5b", "T5c", "T5d"], "fuzzy", "rule: Davis T5 cell mean pools SS25175 ('T5: strongest in T5c and T5d', 'some weakly labeled T4 cells') and SS23757 ('T5: mainly T5a and T5b', 'some weakly labeled T4 cells') -> the pooled profile is SUBTYPE-BIASED (each driver favours two of the four subtypes; weak T4 contamination); prefix rule over T5a-d")
    add("T4.T5", ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"], "fuzzy", "combo driver SS00324 / SS21452 'T4,T5' (supp. 1A); prefix rule over all eight subtypes; the only Davis sample set with separate male / female replicates (T4.T5_d1_male / _female in table 6a)")
    add("Dm3", ["Dm3a", "Dm3b", "Dm3c"], "fuzzy", "rule: Davis Dm3 driver SS00974 (Nern 2015 Dm3) predates the EM split into Dm3a/b/c (flywireType Dm3p/q/v); prefix rule")
    add("Dm8", ["Dm8a", "Dm8b"], "fuzzy", "rule: Davis Dm8 driver SS00323 (Nern 2015 Dm8) predates the yellow / pale split Dm8a (flywireType yDm8) / Dm8b (pDm8); prefix rule")
    add("C2.C3", ["C2", "C3"], "fuzzy", "combo driver SS00779 'C2,C3' (supp. 1A); pooled expression of two types")
    add("L1.L2", ["L1", "L2"], "fuzzy", "combo drivers SS00806 / SS00797 'L1,L2' (supp. 1A)")
    add("LC10bc", ["LC10b", "LC10c-1", "LC10c-2"], "fuzzy", "driver SS00940 'LC10bc' (supp. 1A); MaleCNS splits LC10c into LC10c-1 / LC10c-2; QC-fail samples only")
    add("Lat", ["Lat1", "Lat2", "Lat3", "Lat4", "Lat5"], "fuzzy", "driver SS00657 'Lat' (supp. 1A: R23E12-AD x R55B04-DBD, 'Cell type description: Tuthill et al 2013, Fischbach and Dittrich 1989', driver reference 'Dionne et al in preparation'); MaleCNS has Lat1-Lat5; prefix rule; QC-fail samples only")
    add("LPTC_HS.VS", ["HSN", "HSE", "HSS", "VS"], "fuzzy", "driver SS04438 'LPTC HS,VS' (supp. 1A); MaleCNS HSN / HSE / HSS and the pooled type 'VS' (18 cells); HST, VST1/2, VSm not included (not part of the classical HS / VS sets); QC-fail samples only")
    add("lLNv", ["l-LNv"], "alias", "Davis lLNv driver SS00645 (supp. 1A: R61G12-AD x R10H10-DBD, 'Cell type description: Helfrich-Forster et al 2007', driver reference 'a gift from Heather Dionne, Rubin lab') = MaleCNS l-LNv; QC-fail samples only")
    add("Pdf", ["l-LNv", "s-LNv"], "class", "Pdf-GAL4 (BL-6900, P{Pdf-GAL4.P2.4}, Park et al 2000) labels the PDF-expressing clock neurons = l-LNv + s-LNv (Helfrich-Forster 2007); QC-fail samples only")
    # neuropeptide promoter fusions -> MaleCNS peptide types (class tier: broad driver -> a named population; the MaleCNS
    # type is the whole population the driver is defined by, but the head-dissection driver may also label cells MaleCNS
    # types differently; all five MaleCNS types have sign 0 (nt serotonin / unknown), so out_syn_malecns = 0 by convention)
    add("Ilp2", ["IPC"], "class", "Ilp2-GAL4 (BL-37516, P{Ilp2-GAL4.R}2, Rulifson et al 2002) labels the ~14 insulin-producing cells of the pars intercerebralis = MaleCNS IPC (16 cells, superclass cb_endocrine, hemibrainType 'PI1,PI2,PI3', flywireType IPC); QC-fail samples only")
    add("Crz", ["CRZ01", "CRZ02"], "class", "Crz-GAL4 (BL-51977, P{Crz-GAL4.391}4M, Tayler et al 2012) labels the Corazonin neurons (brain: dorsolateral Crz cells; also VNC / abdominal cells outside the head dissection); MaleCNS CRZ01 + CRZ02 (2 + 2 cells, hemibrainType CRZ01 / CRZ02, flywireType 'CRZ01,CRZ02') are the brain Crz types; the driver likely covers more Crz cells than these four")
    add("Dsk", ["DSKMP3"], "class", "Dsk-GAL4 (BL-51981, P{Dsk-GAL4.TH}3M, Asahina & Anderson 2013 FBrf0222772) labels the Drosulfakinin neurons (MP1 / MP3 medial protocerebrum cells, Soderberg 2012 / Wu 2020); MaleCNS DSKMP3 (4 cells, hemibrainType DSKMP3) is the only Dsk-named type (no DSKMP1 type in MaleCNS v1.0); partial; QC-fail samples only")
    add("NPF", ["NPFL1-I"], "class", "NPF-GAL4 (BL-25681, P{NPF-GAL4.1}2, Wu et al 2003) labels the NPF neurons (brain: dorsomedial, large lateral L1-l and others); MaleCNS NPFL1-I (2 cells, hemibrainType NPFL1-I = the large lateral NPF neuron) is the only NPF-named type; partial (the driver labels ~20 brain NPF cells); round-1 note that NPFL1-I is 'a different, lateral-horn type' withdrawn -- it is the NPF L1-l neuron")
    # mushroom body
    add("KC_ab_c", ["KCab-c"], "alias", "Davis 'KC a/b c' driver MB594B (Aso 2014) = MaleCNS KCab-c (hemibrainType 'KCab-m,KCab-c,KCab-s' split by MaleCNS)")
    add("KC_ab_p", ["KCab-p"], "alias", "Davis 'KC a/b p' driver MB371B = MaleCNS KCab-p (hemibrainType KCab-p)")
    add("KC_ab_s", ["KCab-s"], "alias", "Davis 'KC a/b s' driver MB185B = MaleCNS KCab-s")
    add("KC_ab_c.p.s", ["KCab-c", "KCab-p", "KCab-s"], "fuzzy", "combo driver MB008B 'KC a/b c, p, s' (supp. 1A); KCab-m (536 cells) not named by the driver, excluded")
    add("KC_gd", ["KCg-d"], "alias", "Davis 'KC gd' drivers MB419B / MB607B = MaleCNS KCg-d")
    add("KC_apbp_ap", ["KCa'b'-ap1", "KCa'b'-ap2"], "fuzzy", "driver MB463B 'KC a'/b' ap'; MaleCNS splits ap into ap1 / ap2; QC-fail samples only")
    add("KC_apbp_m", ["KCa'b'-m"], "alias", "driver MB418B 'KC a'/b' m' = MaleCNS KCa'b'-m; QC-fail samples only")
    add("MBON_bp1", ["MBON10"], "alias", "driver MB057B 'MBON-b'1'; MaleCNS instance MBON10(B'1); QC-fail samples only")
    add("MBON_g1pedc", ["MBON11"], "alias", "driver MB112C 'MBON-g1pedc>a/b'; MaleCNS instance MBON11(y1pedc>a/B); QC-fail samples only")
    PAMEV = "Davis PAM_%s driver %s = Aso-2014 compartment(s) %s; MaleCNS instance names carry the compartment: %s"
    add("PAM_1", ["PAM02"], "alias", PAMEV % ("1", "MB109B", "PAM-b'2a", "PAM02(B'2a)") + "; PAM03(B2B'2a) and PAM15(y5B'2a) also innervate b'2a but are not the MB109B population, excluded")
    add("PAM_3", ["PAM04", "PAM09", "PAM10"], "fuzzy", PAMEV % ("3", "MB213B", "PAM-b1, PAM-b2", "PAM04(B2), PAM09(B1ped), PAM10(B1)") + "; the b1 population was split into B1 / B1ped in the EM")
    add("PAM_4", ["PAM07", "PAM08"], "fuzzy", PAMEV % ("4", "MB312B", "PAM-g4, PAM-g4<g1g2", "PAM08(y4), PAM07(y4<y1y2)"))
    add("PAM_2", ["PAM02", "PAM12", "PAM08", "PAM07", "PAM01"], "fuzzy", PAMEV % ("2", "MB196B", "PAM-b'2a, g3, g4, g4<g1g2, g5", "PAM02, PAM12(y3), PAM08(y4), PAM07(y4<y1y2), PAM01(y5)") + "; QC-fail samples only")
    add("PAM_5", ["PAM01"], "alias", PAMEV % ("5", "MB315C", "PAM-g5", "PAM01(y5)") + "; QC-fail samples only")
    add("PAM_6", ["PAM06", "PAM12", "PAM08", "PAM01"], "fuzzy", PAMEV % ("6", "MB042B", "PAM-b'2m, g3, g4, g5", "PAM06(B'2m), PAM12(y3), PAM08(y4), PAM01(y5)") + "; QC-fail samples only")
    add("PAM_7", ["PAM11"], "alias", PAMEV % ("7", "MB043C", "PAM-a1", "PAM11(a1)") + "; QC-fail samples only")
    add("PAM_8", ["PAM12"], "alias", PAMEV % ("8", "MB441B", "PAM-g3", "PAM12(y3)") + "; QC-fail samples only")
    add("PAM_9", ["PAM09", "PAM10"], "fuzzy", PAMEV % ("9", "MB063B", "PAM-b1", "PAM09(B1ped), PAM10(B1)") + "; QC-fail samples only")
    # central complex (Wolff et al. 2015 names in supp. 1A)
    WR18 = "Wolff & Rubin 2018 J Comp Neurol 526:2585 Table 5"
    add("PB_1", ["Delta7"], "alias", f"Davis PB_1 driver SS00116 = Wolff-2015 'PB18.s-GxD7Gy.b / PB18.s-9i1i8c.b'; {WR18}: PB18.s-GxD7Gy.b = Delta7 (PB18.s-9i1i8c.b is the second Delta7 name, e.g. Pisokas 2020 bioRxiv 854521); MaleCNS Delta7 (hemibrainType Delta7, instance Delta7(PB15))")
    add("PB_2", ["EPG"], "alias", f"Davis PB_2 driver SS00090 = Wolff-2015 'PBG1-8.b-EBw.s-D/Vgall.b'; {WR18}: PBG1-8.b-EBw.s-D/V GA.b = E-PG, and Table 2 lists SS00090 for that type (18-21 cells/hemisphere); MaleCNS EPG (hemibrainType EPG, instance EPG(PB08)). EPGt (4 cells, PB09) excluded")
    add("PB_3", ["PEN_a(PEN1)", "PEN_b(PEN2)"], "fuzzy", f"Davis PB_3 driver SS02268 = Wolff-2015 'PBG2-9.s-EBt.b-NO1.b'; {WR18}: = P-EN1, P-EN2; MaleCNS PEN_a(PEN1) / PEN_b(PEN2); the light-level driver covers both. CAUTION supp. 1A: 'many OL cells including Mi1; also GF cells; OL cells outnumber cells in PB' -> this transcriptome is contaminated")
    add("PB_4", ["PFNp_a", "PFNp_b", "PFNp_c", "PFNp_d", "PFNp_e"], "fuzzy", f"Davis PB_4 drivers SS02204 / SS02302 = Wolff-2015 'PBG2-9.s-FBl1.b-NO3P.b'; {WR18}: = P-FNP = hemibrain / MaleCNS PFNp, split into PFNp_a-e by connectivity (Hulse 2021); prefix rule")
    add("PB_5", ["PFNa"], "alias", f"Davis PB_5 driver SS02255 = Wolff-2015 'PBG2-9.s-FBl2.b-NO3A.b'; {WR18}: = P-FNA = MaleCNS PFNa (instance PFNa(PB03))")
    add("PB_6", ["PFNd"], "alias", f"Davis PB_6 driver SS00078 = Wolff-2015 'PBG2-9.s-FBl3.b-NO2D.b'; {WR18}: = P-FND and Table 2 lists SS00078 for it (18 cells/hemisphere) = MaleCNS PFNd (instance PFNd(PB04)); QC-fail samples only")
    # broad drivers -> NT classes (whole-CNS class rows; MaleCNS label is a prediction, the Davis driver is a protein trap)
    add("ChAT", ["<nt=acetylcholine>"], "class", "ChAT-T2A-GAL4 protein trap (BL-60317): all cholinergic neurons of the head; matched to every MaleCNS cell with nt = acetylcholine (predicted label)")
    add("Gad1", ["<nt=gaba>"], "class", "Gad1-T2A-GAL4 protein trap (BL-60324): all GABAergic neurons; matched to MaleCNS nt = gaba")
    add("VGlut", ["<nt=glutamate>"], "class", "VGlut-T2A-GAL4 protein trap (BL-60312): all glutamatergic neurons; matched to MaleCNS nt = glutamate")
    add("opticlobe", ["<superclass=ol_intrinsic|ol_sensory|visual_projection|visual_centrifugal>"], "class", "dissected optic lobe tissue (supp. 1A opticlobe_d1/d2), all cells incl. glia; whole optic-lobe reference only")
    add("lamina", ["<unmatched>"], "unmatched", "dissected lamina tissue; no MaleCNS type; not used")
    for u, why in [("Kdm2", "enhancer-trap BL-30819 (P{GawB}Kdm2[36Y], Taghert et al 2001), 'putative Kdm2 expressing cells': no cell-type identity, no MaleCNS counterpart"),
                   ("CCAP", "CCAP-GAL4 (BL-25686, P{CCAP-GAL4.P}9, Park et al 2003): MaleCNS v1.0 has no CCAP-named type; QC-fail only"),
                   ("Glia_Eg", "epithelial glia: MaleCNS has no glia"), ("Glia_Mg", "marginal glia: no glia in MaleCNS"), ("Glia_Psg", "pseudocartridge glia: no glia in MaleCNS"),
                   ("Muscles_App", "muscle: not neurons"), ("Muscles_Head", "muscle: not neurons")]:
        add(u, ["<unmatched>"], "unmatched", why)

    sources_in_expr = set(expr.source_name)
    mapped_sources = {m[0] for m in M}
    assert sources_in_expr == mapped_sources, (sources_in_expr ^ mapped_sources)

    rows = []
    for src, targets, tier, ev in M:
        qc = expr.loc[expr.source_name == src, "qc"].iloc[0]
        for t in targets:
            if t.startswith("<nt="):
                sel = neurons.nt == t[4:-1]
            elif t.startswith("<superclass="):
                sel = neurons.superclass.isin(t[12:-1].split("|"))
            elif t == "<unmatched>":
                sel = pd.Series(False, index=neurons.index)
            else:
                sel = neurons.type == t
                assert sel.any(), t
            s = neurons[sel]
            rows.append({"source_name": src, "malecns_type": "" if t == "<unmatched>" else t, "tier": tier, "evidence": ev,
                         "n_cells_malecns": int(sel.sum()), "qc": qc,
                         "malecns_superclass": (s.superclass.value_counts().index[0] if len(s) and s.superclass.notna().any() else ""),
                         "malecns_nt": (s.nt.value_counts().index[0] if len(s) else ""),
                         "out_syn_malecns": int(s.out_syn.sum()) if len(s) else 0})
    tm = pd.DataFrame(rows)
    return tm.merge(expr[["source_name", "davis_nt_call", "davis_nt_secondary"]], on="source_name", how="left")


def best_tier(real):
    """Best tier per MaleCNS type. Ties (same tier from a QC-pass and a suboptimal-only driver, e.g. R7p via R7 and
    R7_Rh3; PAM08 via PAM_4 and PAM_2 / PAM_6) resolve to the QC-pass driver; the sort is stable so the remaining order
    is the map order. Round 1 used an unstable sort and a QC-pass filter applied AFTER de-duplication, which dropped
    R7p / PAM08 from the QC-pass-only coverage."""
    r = real.assign(rank=real.tier.map({t: i for i, t in enumerate(TIER_ORDER)}), qc_rank=(real.qc != "pass").astype(int))
    return r.sort_values(["rank", "qc_rank"], kind="stable").drop_duplicates("malecns_type").set_index("malecns_type")


def coverage_report(tm, expr, neurons, absW):
    real = tm[(tm.malecns_type != "") & ~tm.malecns_type.str.startswith("<")]
    best = best_tier(real)
    best_pass = best_tier(real[real.qc == "pass"])
    neurons = neurons.copy()
    neurons["tier"] = neurons.type.map(best["tier"]).fillna("unmatched")
    neurons["tier_pass"] = neurons.type.map(best_pass["tier"]).fillna("unmatched")

    def ntypes(s):   # named types only (the untyped pseudo-type "" is not a type)
        return int(s[s != ""].nunique())

    def cov(mask, col="tier"):
        g = neurons[mask]
        tot_n, tot_out, tot_in = len(g), g.out_syn.sum(), g.in_syn_abs.sum()
        res = {}
        for t in TIER_ORDER:
            h = g[g[col] == t]
            res[t] = {"types": ntypes(h.type), "cells": int(len(h)), "cells_frac": round(len(h) / tot_n, 4) if tot_n else 0,
                      "out_syn": int(h.out_syn.sum()), "out_syn_frac": round(h.out_syn.sum() / tot_out, 4) if tot_out else 0,
                      "in_syn": int(h.in_syn_abs.sum()), "in_syn_frac": round(h.in_syn_abs.sum() / tot_in, 4) if tot_in else 0}
        m = g[g[col] != "unmatched"]
        res["_matched_any"] = {"types": ntypes(m.type), "cells": int(len(m)), "cells_frac": round(len(m) / tot_n, 4) if tot_n else 0,
                               "out_syn_frac": round(m.out_syn.sum() / tot_out, 4) if tot_out else 0,
                               "in_syn_frac": round(m.in_syn_abs.sum() / tot_in, 4) if tot_in else 0}
        res["_total"] = {"types": ntypes(g.type), "untyped_cells": int((g.type == "").sum()), "cells": int(tot_n), "out_syn": int(tot_out), "in_syn": int(tot_in)}
        return res

    optic = neurons.superclass.isin(OPTIC_SC)
    central = neurons.superclass == "cb_intrinsic"
    everything = pd.Series(True, index=neurons.index)
    coverage = {"optic(ol_intrinsic+visual_projection+ol_sensory)": cov(optic), "optic_QCpass_only": cov(optic, "tier_pass"),
                "visual_centrifugal": cov(neurons.superclass == "visual_centrifugal"),
                "central(cb_intrinsic)": cov(central), "central_QCpass_only": cov(central, "tier_pass"),
                "cb_endocrine": cov(neurons.superclass == "cb_endocrine"),
                "all_CNS": cov(everything), "all_CNS_QCpass_only": cov(everything, "tier_pass")}
    matched = neurons.tier.isin(["exact", "alias", "fuzzy"]).to_numpy()
    matched_pass = neurons.tier_pass.isin(["exact", "alias", "fuzzy"]).to_numpy()
    coo = absW.tocoo()

    def both(mask_pre_post, group_mask):
        m = group_mask[coo.row] & group_mask[coo.col]
        tot = coo.data[m].sum()
        b = coo.data[m & mask_pre_post[coo.row] & mask_pre_post[coo.col]].sum()
        return {"syn_in_group": int(tot), "syn_both_matched": int(b), "frac": round(float(b / tot), 4) if tot else 0}

    coverage["edges_both_ends_matched"] = {
        "optic_internal": both(matched, optic.to_numpy()), "optic_internal_QCpass": both(matched_pass, optic.to_numpy()),
        "whole_CNS": both(matched, np.ones(len(neurons), bool)), "whole_CNS_QCpass": both(matched_pass, np.ones(len(neurons), bool))}
    coverage["class_rows"] = {f"{r.source_name}->{r.malecns_type}": {"n_cells": int(r.n_cells_malecns), "out_syn": int(r.out_syn_malecns)} for r in tm[tm.tier == "class"].itertuples()}
    # types reached by a QC-pass driver AND a suboptimal-only driver (round 1 could drop these from the QC-pass-only coverage)
    qc_mix = real[real.malecns_type.isin(best_pass.index)].groupby("malecns_type").qc.nunique()
    coverage["qcpass_and_suboptimal_types"] = sorted(qc_mix[qc_mix > 1].index.tolist())
    xc = real.merge(expr[["source_name", "davis_nt_scores_top3"]], on="source_name")
    coverage["nt_crosscheck"] = [{"source": r.source_name, "malecns_type": r.malecns_type, "tier": r.tier, "qc": r.qc, "malecns_nt": r.malecns_nt,
                                  "davis_nt_call": r.davis_nt_call, "secondary": r.davis_nt_secondary, "scores": r.davis_nt_scores_top3,
                                  "agree": bool(r.malecns_nt == r.davis_nt_call)} for r in xc.itertuples()]
    coverage["n_sources"] = {"expression_rows": int(len(expr)), "qc_pass": int((expr.qc == "pass").sum()), "suboptimal_only": int((expr.qc != "pass").sum())}
    coverage["types_per_best_tier"] = {t: int((best.tier == t).sum()) for t in TIER_ORDER}
    coverage["sources_per_tier"] = {k: int(v) for k, v in tm.drop_duplicates("source_name").tier.value_counts().items()}
    coverage["rows"] = {"type_map": int(len(tm)), "sources": int(tm.source_name.nunique()), "expression": [int(x) for x in expr.shape]}
    coverage["gene_ranges"] = gene_ranges(expr)
    return coverage


def write_tables(tm, expr):
    OUT.mkdir(parents=True, exist_ok=True)
    hdr_map = ("# Davis et al. 2020 eLife 9:e50901 (GEO GSE116969; supp. file 1A) cell-type names -> MaleCNS v1.0 types. "
               "tier: exact = same name, no rename found; alias = documented 1:1 rename; fuzzy = one source pooled over several MaleCNS "
               "subtypes or a combo driver (rule in evidence); class = broad driver / whole-tissue reference matched to an NT class, a superclass "
               "(malecns_type gives the selector) or a named peptide population (Pdf, Ilp2, Crz, Dsk, NPF); unmatched = no MaleCNS counterpart. "
               "qc: pass = Davis QC-pass samples (tables 4 / 7b); "
               "suboptimal_only = Davis marked every sample of that driver 'suboptimal' (expression from table 3 sample means, no p_on). "
               "n_cells_malecns / out_syn_malecns from cache/neurons.parquet + W_post_pre.npz (|W| column sums; sign-0 presynaptic cells count 0). "
               "malecns_nt = majority model NT label; davis_nt_call / davis_nt_secondary = transmitter class(es) from the synthesis / transport genes (rule in the expression table header). "
               "Built by scripts/build_davis2020_tables.py (round 2 of the NT-integration workflow, docs/NT_INTEGRATION.md section 7).\n")
    with open(OUT / "type_map_davis2020.csv", "w", encoding="utf-8", newline="") as f:
        f.write(hdr_map)
        tm.to_csv(f, index=False)
    hdr_expr = ("# Davis et al. 2020 eLife 9:e50901, GEO GSE116969 (Davis FP, Nern A, Picard S, Reiser MB, Rubin GM, Eddy SR, Henry GL; CC-BY 4.0). "
                "One row per Davis cell type (source_name as in GSE116969_dataTable4 / 7b column headers and supp. file 1A 'celltype'). "
                "<gene>_tpm = TPM, cell-type mean over QC-pass samples (GSE116969_dataTable4.genes_x_cells_TPM.coding_genes_QCpass; TAPIN-seq / INTACT nuclear RNA, "
                "ENSEMBL r91 / FlyBase 2017_04 gene models); for qc = suboptimal_only the mean of that driver's QC-fail samples in dataTable3 (use with caution). "
                "<gene>_p_on = probability the gene is in the 'on' component of Davis's per-gene bimodal mixture model (dataTable7b); NaN if the gene was not modeled "
                "(< 10 TPM in every sample) or the cell has no QC-pass sample. 54 panel genes; symbols follow the task list; source symbols differ for "
                "KaiR1D (= CG3822, FBgn0038837; round 1 wrongly used CG8916 = FBgn0030707 / Lcch-14A) and Octalpha2R (= CG18208, FBgn0038653). "
                "davis_nt_call: the transmitter class whose marker pair (ChAT+VAChT, Gad1+VGAT, VGlut, Hdc, ple+DAT, Tdc2+Tbh, Trh+SerT) has the largest "
                "pair-minimum TPM, if >= 10 TPM ('none' otherwise); davis_nt_secondary: further classes with pair-minimum >= 50 TPM; davis_nt_scores_top3 lists the values. "
                "This is a data summary, not a sign decision. Built by scripts/build_davis2020_tables.py.\n")
    with open(OUT / "expression_davis2020.csv", "w", encoding="utf-8", newline="") as f:
        f.write(hdr_expr)
        expr.round(3).to_csv(f, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", type=Path, default=None, help="write the full coverage / cross-check / gene-range JSON here")
    args = ap.parse_args(argv)
    tpm_cells, p_cells, tpm_samples, drivers, samples = load_sources()
    expr = build_expression(tpm_cells, p_cells, tpm_samples, samples)

    drivers = drivers.copy()
    drivers["celltype"] = drivers["celltype"].astype(str).str.strip()
    drv_details = drivers.groupby("celltype").agg(driverIDs=("driverID.external", lambda s: ";".join(str(x) for x in s)))

    neurons = pd.read_parquet(CACHE / "neurons.parquet")
    W = sp.load_npz(CACHE / "W_post_pre.npz").tocsr()
    absW = abs(W)
    neurons["out_syn"] = np.asarray(absW.sum(axis=0)).ravel()   # per presynaptic neuron (columns = pre); sign-0 pre cells contribute 0
    neurons["in_syn_abs"] = np.asarray(absW.sum(axis=1)).ravel()
    neurons["type"] = neurons["type"].fillna("")

    tm = build_map(expr, neurons, drv_details)
    coverage = coverage_report(tm, expr, neurons, absW)
    write_tables(tm, expr)

    _np = lambda o: o.item() if hasattr(o, "item") else str(o)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(coverage, indent=1, default=_np), encoding="utf-8")
    print(json.dumps({k: v for k, v in coverage.items() if k not in ("nt_crosscheck", "gene_ranges")}, indent=1, default=_np))
    print("\nNT cross-check disagreements / secondary flags:")
    for r in coverage["nt_crosscheck"]:
        if not r["agree"] or r["secondary"]:
            print(r)
    print("agree:", sum(r["agree"] for r in coverage["nt_crosscheck"]), "of", len(coverage["nt_crosscheck"]), "matched (source, type) pairs")
    print("\ngene ranges (neuronal_69 = QC-pass cell types minus glia/muscle/tissue minus ChAT/Gad1/VGlut drivers):")
    for g, d in coverage["gene_ranges"].items():
        if isinstance(d, dict) and "neuronal_69" in d:
            n69, n72, n79, nph = d["neuronal_69"], d["neuronal_72"], d["qcpass_79"], d["neuronal_69_minus_photoreceptors"]
            print(f"  {g}: TPM {n69['tpm_min']}-{n69['tpm_max']} ({n69['tpm_argmin']} .. {n69['tpm_argmax']}) over {n69['n']} neuronal rows; "
                  f"on-rule {n69['on_rule']}/{n69['n']} ({n69['on_rule_frac']:.0%}), {n72['on_rule']}/{n72['n']}, {n79['on_rule']}/{n79['n']} ({n79['on_rule_frac']:.0%}); "
                  f"P(on) mean {n69['p_on_mean']}, >=0.5 in {n69['p_on_ge_0.5']}, =1 in {n69['p_on_eq_1']}; "
                  f"minus photoreceptors: {nph['tpm_min']}-{nph['tpm_max']} ({nph['tpm_argmin']}), on {nph['on_rule']}/{nph['n']}, P(on)=1 in {nph['p_on_eq_1']}; off rows (69): {n69['off_rows']}")
            print(f"     photoreceptors: {d['photoreceptors']}; top10: {d['top10_neuronal_72']}")
        else:
            print(f"  {g}: {d}")
    print("\nrows:", len(tm), "sources:", tm.source_name.nunique(), "expr shape:", expr.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
