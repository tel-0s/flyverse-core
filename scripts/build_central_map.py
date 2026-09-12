"""Build flyverse/data/type_map_central.csv and flyverse/data/expression_central.csv from the two central-brain
single-cell atlases (Davie et al. 2018 Cell, GEO GSE107451; Fly Cell Atlas 2022 Science, head 10x stringent loom)
and report MaleCNS coverage by tier.

Inputs (scratchpad): davie_cluster_expr.parquet, fca_cluster_expr.parquet (from agg_davie.py / agg_fca.py),
malecns_types.parquet; cache/neurons.parquet; raw weights feather (for sign-0-inclusive output counts).
"""
import re, sys, json, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path

ROOT = Path(r"D:\Projects\flyverse")
SCR = Path(r"<workstation-home>\AppData\Local\Temp\claude\D--Projects-flyverse\d280c0e1-e89c-49ce-943c-279a614bcc17\scratchpad")
OUT_MAP = ROOT / "flyverse/data/type_map_central.csv"
OUT_EXPR = ROOT / "flyverse/data/expression_central.csv"
RAW_W = Path(r"D:\Datasets\male-cns-connectome-v1.0\flat-connectome\connectome-weights-male-cns-v1.0-minconf-0.5.feather")

# ---------------------------------------------------------------- MaleCNS per-type table
neurons = pd.read_parquet(ROOT / "cache/neurons.parquet")
neurons["type"] = neurons.type.fillna("")
W = sp.load_npz(ROOT / "cache/W_post_pre.npz")
neurons["out_syn_signed"] = np.asarray(abs(W).sum(axis=0)).ravel()
if RAW_W.exists():
    import pyarrow.feather as pf
    w = pf.read_table(RAW_W, columns=["body_pre", "body_post", "weight"]).to_pandas()
    node = set(neurons.bodyId.to_numpy())  # both endpoints in the model's node set (Traced + photoreceptors), as in the NT audit
    w = w[w.body_pre.isin(node) & w.body_post.isin(node)]
    raw = w.groupby("body_pre").weight.sum()
    neurons["out_syn_raw"] = neurons.bodyId.map(raw).fillna(0).astype(np.int64)
    del w
else:
    neurons["out_syn_raw"] = np.nan
T = neurons.groupby("type").agg(n_cells=("bodyId", "size"), out_syn_signed=("out_syn_signed", "sum"),
                                out_syn_raw=("out_syn_raw", "sum"),
                                superclass=("superclass", lambda s: s.value_counts(dropna=False).index[0]),
                                klass=("class", lambda s: s.value_counts(dropna=False).index[0]),
                                subclass=("subclass", lambda s: s.value_counts(dropna=False).index[0]),
                                nt=("nt", lambda s: s.value_counts().index[0]))
T = T.drop(index="", errors="ignore")
types = set(T.index)
OPTIC_SC = {"ol_intrinsic", "visual_projection", "ol_sensory"}
CENTRAL_SC = {"cb_intrinsic", "cb_sensory", "cb_endocrine", "cb_motor", "cb_efferent", "visual_centrifugal"}
BRAIN_SC = OPTIC_SC | CENTRAL_SC | {"descending_neuron"}

def sel(rule):
    """rule: 'exact:NAME' | 'list:A|B|C' | 're:REGEX' | 'query:<pandas query on T>'."""
    kind, _, arg = rule.partition(":")
    if kind == "exact":
        return [arg] if arg in types else []
    if kind == "list":
        return [a for a in arg.split("|") if a in types]
    if kind == "re":
        return sorted(t for t in types if re.search(arg, t))
    if kind == "query":
        return sorted(T.query(arg).index)
    raise ValueError(rule)

# ---------------------------------------------------------------- mapping rules
# (source, source_name, rule, tier, evidence). tier: exact | alias | fuzzy | class | unmatched.
#  exact  : the source label (or its FBbt type token) is a MaleCNS type name.
#  alias  : documented synonym of one MaleCNS type.
#  fuzzy  : the label names a family that MaleCNS splits into several named types; rule = name regex/list.
#  class  : the label is a cell class; MaleCNS side selected by `class`/`subclass`/`nt` column or a curated list.
DAV = "davie2018"; FCA = "fca2022"
CLOCK = "list:s-LNv|l-LNv|5thsLNv_LNd6|LNd_b|LNd_c|LPN_a|LPN_b|DN1a|DN1pA|DN1pB"
NONVNC = "superclass in ['cb_intrinsic','cb_sensory','cb_endocrine','cb_motor','visual_centrifugal','visual_projection','ol_intrinsic','descending_neuron']"
rules = [
    # ---- Davie 2018 (GEO metadata `annotation`; Table S2 cluster ids in evidence) — central brain
    (DAV, "G-KC", r"re:^KCg", "fuzzy", "Table S2 res2 cluster 8 'G-KC' (Awasaki 2000; Crocker 2016); sNPF-high, trio-low (mean log1p cp10k sNPF 3.57, trio 0.10); rule ^KCg -> all gamma KC types"),
    (DAV, "A/B-KC", r"re:^KCab", "fuzzy", "Table S2 res2 cluster 22 'a/b-KC' (Crocker 2016; Johard 2008); sNPF 2.93, trio 0.28; rule ^KCab"),
    (DAV, "A/B*-KC", r"re:^KCa'b'", "fuzzy", "Table S2 res2 cluster 28 \"a'/b'-KC\" (Medioni 2014); trio 1.90, DAT 1.52 (alpha'/beta' KCs express DAT); rule ^KCa'b'"),
    (DAV, "PAM", r"re:^PAM\d\d$", "fuzzy", "Table S2 subcluster of 42 'PAM (Fer2(+)) neurons' (Bou Dib 2014); ple 3.17, DAT 4.12, Fer2 1.56; rule ^PAM\\d\\d"),
    (DAV, "Dopaminergic", "query:nt == 'dopamine' and not type.str.match('^PAM') and " + NONVNC, "class", "Table S2 res2 cluster 42 'Dopaminergic' minus the PAM subcluster (= 'Fer2(-) neurons'); ple 3.35, DAT 3.33, Fer2 0.36; MaleCNS side = consensus nt dopamine, non-PAM, non-VNC (PPL1/PPL2/PPM/PAL and other DA types)"),
    (DAV, "Serotonergic", "query:nt == 'serotonin' and " + NONVNC, "class", "Table S2 res2 cluster 38 'Serotonergic' (Couch 2004); SerT 1.26, Trh 1.80; MaleCNS side = consensus nt serotonin, non-VNC (NT-defined class, not a type match)"),
    (DAV, "Octopaminergic", "query:nt == 'octopamine' and " + NONVNC, "class", "Table S2 subcluster of 64 'Octopaminergic' (Tdc2 2.45, Tbh 2.56); MaleCNS side = consensus nt octopamine, non-VNC (OA-VUMa/VPM/ASM/AL2i and others); NT-defined class"),
    (DAV, "Tyraminergic", "", "unmatched", "Table S2 subcluster of 64 'Tyraminergic' (Tdc2 2.59, Tbh 0.11); MaleCNS has no tyramine NT label and no tyraminergic type names"),
    (DAV, "Clock", CLOCK, "class", "Table S2 res2 cluster 36 'Clock neurons' (Abruzzi 2017; Klarsfeld 2004; Park 2000); 389 cells is ~2-3x the ~150 canonical clock neurons, so the cluster is broader than the list; MaleCNS side = the 10 named clock types (no DN2/DN3 names exist in MaleCNS v1.0)"),
    (DAV, "LNv", "list:s-LNv|l-LNv", "fuzzy", "Table S2 sub-clustering 'LNv' (Pdf-expressing cells; Klarsfeld 2004); Pdf 6.29; rule = the two Pdf+ LNv types (5th s-LNv is Pdf-negative and excluded)"),
    (DAV, "DN1", r"re:^DN1(a|pA|pB)$", "fuzzy", "Table S2 res2 cluster 86 'DN1' (Abruzzi 2017; Kunst 2014); VGlut 5.01, Dh31 4.80 (DN1p are glutamatergic/Dh31+); rule ^DN1(a|pA|pB)"),
    (DAV, "MBON", "query:klass == 'MBON'", "class", "Table S2 res2 cluster 57 'MBON' (mapping to Crocker 2016); MaleCNS class MBON"),
    (DAV, "Olfactory_projection_neurons", r"re:_(ad|l)PN$", "fuzzy", "Table S2 res2 cluster 27 'OPN: adPN and lPN' (Komiyama & Luo 2007; Li 2017); ChAT 1.60, acj6 0.09; rule = uniglomerular adPN/lPN types (_adPN$|_lPN$); vPN/ilPN/lvPN excluded"),
    (DAV, "adPN", r"re:_adPN$", "fuzzy", "Table S2 sub-clustering of 27, adPN lineage (Li 2017); acj6 1.41 (adPN marker); rule _adPN$; glomerulus not resolved"),
    (DAV, "adPN/C15", r"re:_adPN$", "fuzzy", "Table S2 'C15(+) adPNs' (Li 2017); acj6 2.02, C15 0.73; glomerular identity of the C15+ subset not resolved here -> whole adPN lineage"),
    (DAV, "adPN/kn", r"re:_adPN$", "fuzzy", "Table S2 'kn(+) adPNs' (Li 2017); acj6 2.43, kn 1.57; whole adPN lineage (subset unresolved)"),
    (DAV, "adPN/C15&kn", r"re:_adPN$", "fuzzy", "Table S2 'C15(+) kn(+) adPNs' (Li 2017); acj6 1.98, C15 0.97, kn 0.82; whole adPN lineage (subset unresolved)"),
    (DAV, "adPN/kn&CG31676", r"re:_adPN$", "fuzzy", "Table S2-derived label (11 cells); acj6 2.53, kn 1.74; whole adPN lineage (subset unresolved)"),
    (DAV, "lPN", r"re:_lPN$", "fuzzy", "Table S2 sub-clustering of 27, lPN lineage (Li 2017); acj6 0.00; rule _lPN$"),
    (DAV, "lPN/unpg", r"re:_lPN$", "fuzzy", "Table S2 'unpg(+) lPNs' (Li 2017); unpg 0.68; whole lPN lineage (subset unresolved)"),
    (DAV, "lPN/CG31676", r"re:_lPN$", "fuzzy", "Table S2 'CG31676(+) lPNs' (Li 2017); CG31676 1.31; whole lPN lineage (subset unresolved)"),
    (DAV, "Poxn", r"re:^ER[1-4]", "class", "Table S2 res2 cluster 80 'Poxn' (Boll & Noll 2002; Minocha 2017). Poxn+ protocerebral dorsal-cluster neurons are the EB ring neurons R1-R4 (Omoto 2019 Biol Open; Minocha 2017); cluster is Gad1-high (3.20; ring neurons GABAergic) but also contains the deutocerebral Poxn ventral cluster (~100 non-ring neurons) -> mixed; MaleCNS side ER1-ER4 types (ER5/ER6 excluded)"),
    (DAV, "dorsal_Fan-shaped_Body", "", "unmatched", "Table S2 res2 cluster 61 'dFB' (mapped to R23E10 sorted cells generated in the study); VGlut 4.48. Candidate = FB tangential neurons of dorsal layers (FB6-FB8 types) but the R23E10 -> EM-type correspondence is not established here; not mapped"),
    (DAV, "IPC", "exact:IPC", "exact", "Table S2 res8 cluster 30 'IPC' (Brogiolo 2001; Rulifson 2002); Ilp2 7.08; MaleCNS type IPC (cb_endocrine, 16 cells)"),
    (DAV, "Crz", r"re:^CRZ\d\d$", "fuzzy", "Table S2 'all cells expressing neuropeptides' -> Crz (9 cells); Crz 4.45; MaleCNS CRZ01/CRZ02 (brain Crz neurons)"),
    (DAV, "Hug", r"re:^Hugin", "fuzzy", "Table S2 'Hugin neurons' (Melcher & Pankratz 2005); Hug 7.68; MaleCNS names only Hugin-RG (4 cells) of the ~20 hugin neurons -> partial"),
    (DAV, "ITP", "", "unmatched", "label 'ITP' (25 cells) but ITP mean log1p cp10k only 0.41 (vs 2.63 in 'Mip/ITP', 1.76 in 'LNv'); the MaleCNS type ITP (7 cells) is not evidenced by this cluster's expression -> not mapped"),
    (DAV, "Mip/ITP", "", "unmatched", "9 cells; Mip 1.64, ITP 2.63, Dh31 4.54; no confident MaleCNS type (candidates ITP, DMS not distinguishable)"),
    (DAV, "Capa", "", "unmatched", "Table S2 res8 cluster 119 'Capa neurons' (Diesner 2018); Capa 4.28 but Ms 6.65 too (73 cells); MaleCNS CAPA (2 cells) / DMS (6) are far smaller endocrine types -> not mapped"),
    (DAV, "CCAP", "", "unmatched", "Table S2 res8 cluster 151 'CCAP' (Luan 2006); no CCAP-named type in MaleCNS v1.0"),
    (DAV, "FMRFa", "", "unmatched", "FMRFa 2.72; MaleCNS FMRFa_Tv is a VNC endocrine type (Davie is brain-only) -> not mapped"),
    (DAV, "AstA/NPF", "", "unmatched", "NPF 3.96, AstA 4.80, ChAT-high; no confident MaleCNS type (NPFL1-I is 2 serotonergic cells)"),
    (DAV, "AstA/Nplp1", "", "unmatched", "AstA 6.33, Gad1 3.30; MaleCNS AstA1 is a 2-cell pair; cluster (83 cells) is broader -> not mapped"),
    (DAV, "CCHa1", "", "unmatched", "no CCHa1-named MaleCNS type"),
    (DAV, "Proc/Ms", "", "unmatched", "Proc 1.46, Ms 0.55, Mip 1.07; candidate DMS (6 cells) not evidenced"),
    (DAV, "Proc/Gpb5", "", "unmatched", "Proc 5.25; no Proc-named MaleCNS type"),
    (DAV, "Mip", "", "unmatched", "Table S2 res2 cluster 17 'Mip' (Carlsson 2010; Min 2016); 814 cells, Mip 0.70 -> a broad cluster, no MaleCNS Mip type"),
    (DAV, "Mip/OCT", "", "unmatched", "26 cells; Tdc2 2.83, Mip 4.25, Dh31 4.81 (tyraminergic/octopaminergic Mip+); no MaleCNS correspondence established"),
    (DAV, "Peptidergic", "", "unmatched", "Table S2 res2 cluster 46 'Peptidergic'; heterogeneous (Proc 2.32); no MaleCNS type"),
    (DAV, "Gr43a", "", "unmatched", "Table S2 res8 cluster 129 'Gr43a neurons' (Miyamoto & Amrein 2014); Gr43a 1.17; no Gr43a-named brain type in MaleCNS"),
    (DAV, "Hsp", "", "unmatched", "heat-shock / stress signature cluster, not a cell type"),
    (DAV, "DCN", r"re:^LC14", "fuzzy", "Table S2 res8 cluster 118 'DCN' (Hassan 2000); dorsal cluster neurons = LC14 (Zschaetzsch 2014); acj6 2.29; rule ^LC14 (LC14a-1, LC14a-2, LC14b)"),
    # ---- Davie 2018 optic-lobe labels (exact / fuzzy by name)
    (DAV, "TmY14", "exact:TmY14", "exact", "Table S2 res2 cluster 11 (mapping to Konstantinides 2018); VGlut 4.04"),
    (DAV, "Mi1", "exact:Mi1", "exact", "Table S2 res2 cluster 26 (Hasegawa 2013); ChAT 2.10"),
    (DAV, "T1", "exact:T1", "exact", "Table S2 res2 cluster 37 (Hamanaka & Meinertzhagen 2010); ort 2.32"),
    (DAV, "Tm9", "exact:Tm9", "exact", "Table S2 res2 cluster 18 (mapping)"),
    (DAV, "Tm5c", "exact:Tm5c", "exact", "Table S2 res2 cluster 39 'Dm8/Tm5c' sub-annotation; VGlut 3.91"),
    (DAV, "Tm5ab", "list:Tm5a|Tm5b", "fuzzy", "Table S2 res2 cluster 41 'Tm5ab' (Konstantinides 2018); rule Tm5a|Tm5b"),
    (DAV, "Dm8/Dm11", "list:Dm8a|Dm8b|Dm11", "fuzzy", "Table S2 res2 cluster 39 sub-annotation 'Dm8/Dm11'; VGlut 4.24; rule Dm8a|Dm8b|Dm11"),
    (DAV, "Dm9", "exact:Dm9", "exact", "Table S2-derived label; VGlut 5.19"),
    (DAV, "Pm1/Pm2", "list:Pm1|Pm2a|Pm2b", "fuzzy", "Table S2 res2 cluster 12 (Erclik 2017); Gad1 3.70; rule Pm1|Pm2a|Pm2b"),
    (DAV, "Pm1/Pm2/Pm3", "list:Pm1|Pm2a|Pm2b|Pm3", "fuzzy", "Table S2 res2 cluster 48 (Erclik 2017); Gad1 3.44"),
    (DAV, "Pm3", "", "unmatched", "Table S2 res2 cluster 73 'Pm3' (Erclik 2017 markers) but the cluster is glutamatergic (VGlut 3.97, Gad1 0.20) whereas MaleCNS Pm3 is GABA 100% (Pm numbering was reorganised in the Nern 2025 optic-lobe nomenclature) -> name match not trusted"),
    (DAV, "Pm4", "exact:Pm4", "exact", "Table S2-derived label; Gad1 3.87"),
    (DAV, "T2", "exact:T2", "exact", "Table S2 res2 cluster 31 'T2/T3' sub-annotation; MaleCNS also has T2a (not included)"),
    (DAV, "T3", "exact:T3", "exact", "Table S2 res2 cluster 31 'T2/T3' sub-annotation"),
    (DAV, "T4/T5", r"re:^T[45][a-d]$", "fuzzy", "Table S2 res2 cluster 24 (Pankova & Borst 2016); rule ^T[45][a-d]"),
    (DAV, "C2", "exact:C2", "exact", "Table S2 res2 cluster 65 'C2/C3' sub-annotation; Gad1 2.58"),
    (DAV, "C3", "exact:C3", "exact", "Table S2 res2 cluster 65 'C2/C3' sub-annotation; Gad1 3.39"),
    (DAV, "Tm1/TmY8", "exact:Tm1", "fuzzy", "Table S2 res2 cluster 45 'Tm1/TmY8'; MaleCNS v1.0 has no type named TmY8 -> Tm1 only (partial)"),
    (DAV, "Lawf1", "exact:Lawf1", "exact", "Table S2 res2 cluster 58 (Chen 2016)"),
    (DAV, "Lawf2", "exact:Lawf2", "exact", "Table S2 res2 cluster 72 (Chen 2016)"),
    (DAV, "L1", "exact:L1", "exact", "Table S2 sub-clustering of 20 (Kolodziejczyk 2008; Tan 2015)"),
    (DAV, "L2", "exact:L2", "exact", "Table S2 sub-clustering of 20 (Tan 2015)"),
    (DAV, "L3", "exact:L3", "exact", "Table S2 sub-clustering of 20 (Tan 2015)"),
    (DAV, "L4/L5", "list:L4|L5", "fuzzy", "Table S2 sub-clustering of 20 (Hasegawa 2013; Tan 2015)"),
    (DAV, "Lamina_monopolar", r"re:^L[1-5]$", "fuzzy", "Table S2 res2 cluster 20 'L1/L2/L3/L4/L5' (Tan 2015); rule ^L[1-5]$"),
    (DAV, "Photoreceptors", "query:klass == 'visual'", "class", "Table S2 res2 cluster 53 (Montell & Rubin 1988); MaleCNS class visual (R1-R6, R7*, R8*)"),
    # ---- Davie non-neuronal
    (DAV, "Astrocyte-like", "", "unmatched", "glia"), (DAV, "Ensheathing_glia", "", "unmatched", "glia"),
    (DAV, "Cortex_glia", "", "unmatched", "glia"), (DAV, "Perineurial_glia", "", "unmatched", "glia"),
    (DAV, "Subperineurial_glia", "", "unmatched", "glia"), (DAV, "Chiasm_glia", "", "unmatched", "glia"),
    (DAV, "Plasmatocytes", "", "unmatched", "hemocytes"),
    # ---- FCA 2022 head (col_attrs/annotation) — central brain
    (FCA, "gamma Kenyon cell", r"re:^KCg", "fuzzy", "FCA annotation (FBbt); sNPF-high; rule ^KCg"),
    (FCA, "alpha/beta Kenyon cell", r"re:^KCab", "fuzzy", "FCA annotation (FBbt); rule ^KCab"),
    (FCA, "alpha'/beta' Kenyon cell", r"re:^KCa'b'", "fuzzy", "FCA annotation (FBbt); rule ^KCa'b'"),
    (FCA, "Kenyon cell", "query:klass == 'Kenyon_Cell'", "class", "FCA annotation (FBbt, unresolved KC); MaleCNS class Kenyon_Cell"),
    (FCA, "dopaminergic PAM neuron", r"re:^PAM\d\d$", "fuzzy", "FCA annotation (FBbt); rule ^PAM\\d\\d"),
    (FCA, "dopaminergic neuron", "query:nt == 'dopamine' and not type.str.match('^PAM') and " + NONVNC, "class", "FCA annotation (FBbt), 25 nuclei; MaleCNS side = consensus nt dopamine, non-PAM, non-VNC"),
    (FCA, "octopaminergic/tyraminergic neuron", "query:nt == 'octopamine' and " + NONVNC, "class", "FCA annotation (FBbt), Tdc2+ mixed OA/TA; MaleCNS side = consensus nt octopamine, non-VNC (tyraminergic cells have no MaleCNS label) -> mixed"),
    (FCA, "antennal lobe projection neuron", "query:klass == 'ALPN'", "class", "FCA annotation (FBbt), 23 nuclei; MaleCNS class ALPN"),
    (FCA, "Poxn neuron", r"re:^ER[1-4]", "class", "FCA annotation 'Poxn neuron' (102 nuclei); Poxn+ dorsal cluster = EB ring neurons R1-R4 (Omoto 2019; Minocha 2017); mixed with deutocerebral Poxn neurons; MaleCNS ER1-ER4"),
    (FCA, "olfactory receptor neuron", r"re:^ORN_", "class", "FCA annotation (FBbt); MaleCNS class olfactory (ORN_* types)"),
    (FCA, "adult olfactory receptor neuron Gr21a/63a", "exact:ORN_V", "alias", "Gr21a/Gr63a CO2 ORNs project to glomerulus V (Suh 2004; Jones 2007) -> ORN_V"),
    (FCA, "antennal trichoid sensillum at4", "list:ORN_VA1v|ORN_DL3|ORN_VA1d", "fuzzy", "at4 houses Or47b (VA1v), Or65a/b/c (DL3), Or88a (VA1d) (Couto 2005); 7 nuclei"),
    (FCA, "Johnston organ neuron", r"re:^JO-", "class", "FCA annotation (FBbt); MaleCNS JO-* types (class mechanosensory)"),
    (FCA, "auditory sensory neuron", "query:subclass == 'auditory'", "class", "FCA annotation (FBbt); MaleCNS subclass auditory (JO-A/JO-B/JO-CA types)"),
    # ---- FCA optic lobe (FBbt label -> type token)
    (FCA, "columnar neuron T1", "exact:T1", "exact", "label token T1"),
    (FCA, "T neuron T4/T5a-b", "list:T4a|T4b|T5a|T5b", "fuzzy", "label token T4/T5a-b -> T4a,T4b,T5a,T5b"),
    (FCA, "T neuron T4/T5c-d", "list:T4c|T4d|T5c|T5d", "fuzzy", "label token T4/T5c-d -> T4c,T4d,T5c,T5d"),
    (FCA, "T neuron T4/T5", r"re:^T[45][a-d]$", "fuzzy", "label token T4/T5 -> all eight T4/T5 subtypes"),
    (FCA, "T neuron T2", "exact:T2", "exact", "label token T2"),
    (FCA, "T neuron T2a", "exact:T2a", "exact", "label token T2a"),
    (FCA, "T neuron T3", "exact:T3", "exact", "label token T3"),
    (FCA, "lamina monopolar neuron L1", "exact:L1", "exact", "label token L1"),
    (FCA, "lamina monopolar neuron L2", "exact:L2", "exact", "label token L2"),
    (FCA, "lamina monopolar neuron L3", "exact:L3", "exact", "label token L3"),
    (FCA, "lamina monopolar neuron L4", "exact:L4", "exact", "label token L4"),
    (FCA, "lamina monopolar neuron L5", "exact:L5", "exact", "label token L5"),
    (FCA, "lamina intrinsic amacrine neuron Lai", "exact:Lai", "exact", "label token Lai"),
    (FCA, "lamina wide-field 1 neuron", "exact:Lawf1", "alias", "lamina wide-field 1 = Lawf1"),
    (FCA, "lamina wide-field 2 neuron", "exact:Lawf2", "alias", "lamina wide-field 2 = Lawf2"),
    (FCA, "medullary intrinsic neuron Mi1", "exact:Mi1", "exact", "label token Mi1"),
    (FCA, "medullary intrinsic neuron Mi4", "exact:Mi4", "exact", "label token Mi4"),
    (FCA, "medullary intrinsic neuron Mi9", "exact:Mi9", "exact", "label token Mi9"),
    (FCA, "medullary intrinsic neuron Mi15", "exact:Mi15", "exact", "label token Mi15"),
    (FCA, "transmedullary neuron Tm1", "exact:Tm1", "exact", "label token Tm1"),
    (FCA, "transmedullary neuron Tm2", "exact:Tm2", "exact", "label token Tm2"),
    (FCA, "transmedullary neuron Tm3a", "exact:Tm3", "alias", "Tm3a (Oezel 2021 subtype label) -> MaleCNS Tm3 (single type)"),
    (FCA, "transmedullary neuron Tm4", "exact:Tm4", "exact", "label token Tm4"),
    (FCA, "transmedullary neuron Tm9", "exact:Tm9", "exact", "label token Tm9"),
    (FCA, "transmedullary neuron Tm20", "exact:Tm20", "exact", "label token Tm20"),
    (FCA, "transmedullary neuron Tm5c", "exact:Tm5c", "exact", "label token Tm5c"),
    (FCA, "transmedullary neuron Tm29", "exact:Tm29", "exact", "label token Tm29"),
    (FCA, "transmedullary Y neuron TmY4", "exact:TmY4", "exact", "label token TmY4"),
    (FCA, "transmedullary Y neuron TmY5a", "exact:TmY5a", "exact", "label token TmY5a"),
    (FCA, "transmedullary Y neuron TmY8", "", "unmatched", "MaleCNS v1.0 has no type named TmY8 (renamed in the Nern 2025 optic-lobe nomenclature; correspondence not established here)"),
    (FCA, "transmedullary Y neuron TmY14", "exact:TmY14", "exact", "label token TmY14"),
    (FCA, "centrifugal neuron C2", "exact:C2", "exact", "label token C2"),
    (FCA, "centrifugal neuron C3", "exact:C3", "exact", "label token C3"),
    (FCA, "distal medullary amacrine neuron Dm3", r"re:^Dm3[abc]$", "fuzzy", "label token Dm3 -> Dm3a/b/c"),
    (FCA, "distal medullary amacrine neuron Dm8", r"re:^Dm8[ab]$", "fuzzy", "label token Dm8 -> Dm8a/b"),
    (FCA, "distal medullary amacrine neuron Dm9", "exact:Dm9", "exact", "label token Dm9"),
    (FCA, "distal medullary amacrine neuron Dm10", "exact:Dm10", "exact", "label token Dm10"),
    (FCA, "distal medullary amacrine neuron Dm11", "exact:Dm11", "exact", "label token Dm11 (4 nuclei)"),
    (FCA, "distal medullary amacrine neuron Dm12", "exact:Dm12", "exact", "label token Dm12"),
    (FCA, "proximal medullary amacrine neuron Pm2", r"re:^Pm2[ab]$", "fuzzy", "label token Pm2 -> Pm2a/b"),
    (FCA, "proximal medullary amacrine neuron Pm4", "exact:Pm4", "exact", "label token Pm4"),
    (FCA, "lobula columnar neuron LC10", r"re:^LC10", "fuzzy", "label token LC10 -> LC10a-e (+ LC10_unclear)"),
    (FCA, "lobula columnar neuron LC12", "exact:LC12", "exact", "label token LC12"),
    (FCA, "lobula columnar neuron LC17", "exact:LC17", "exact", "label token LC17"),
    (FCA, "outer photoreceptor cell", "exact:R1-R6", "alias", "outer photoreceptors R1-R6 -> MaleCNS R1-R6"),
    (FCA, "photoreceptor cell R7", r"re:^R7(y|p|d|_unclear)$", "fuzzy", "R7 -> R7y/R7p/R7d/R7_unclear"),
    (FCA, "photoreceptor cell R8", r"re:^R8(y|p|d|_unclear)$", "fuzzy", "R8 -> R8y/R8p/R8d/R8_unclear"),
    (FCA, "dorsal rim area", "list:R7d|R8d", "fuzzy", "DRA photoreceptors = R7d/R8d (2 nuclei)"),
    (FCA, "ocellus retinula cell", "", "unmatched", "ocellar photoreceptors are not in the MaleCNS node set (OCG types are ocellar-ganglion interneurons)"),
    (FCA, "photoreceptor-like", "", "unmatched", "no type"),
    # ---- FCA non-neuronal / unannotated
    (FCA, "unannotated", "", "unmatched", "unannotated nuclei"), (FCA, "artefact", "", "unmatched", "artefact"),
    (FCA, "cone cell", "", "unmatched", "non-neuronal"), (FCA, "epithelial cell", "", "unmatched", "non-neuronal"),
    (FCA, "ensheathing glial cell", "", "unmatched", "glia"), (FCA, "adult brain perineurial glial cell", "", "unmatched", "glia"),
    (FCA, "skeletal muscle of head", "", "unmatched", "non-neuronal"), (FCA, "pigment cell", "", "unmatched", "non-neuronal"),
    (FCA, "adult lamina epithelial/marginal glial cell", "", "unmatched", "glia"), (FCA, "adult reticular neuropil associated glial cell", "", "unmatched", "glia"),
    (FCA, "optic-lobe-associated cortex glial cell", "", "unmatched", "glia"), (FCA, "hemocyte", "", "unmatched", "non-neuronal"),
    (FCA, "adult optic chiasma glial cell", "", "unmatched", "glia"), (FCA, "subperineurial glial cell", "", "unmatched", "glia"),
    (FCA, "pericerebral adult fat mass", "", "unmatched", "non-neuronal"), (FCA, "adult fat body", "", "unmatched", "non-neuronal"),
    (FCA, "adult brain cell body glial cell", "", "unmatched", "glia"), (FCA, "perineurial glial sheath", "", "unmatched", "glia"),
]

# expression tables (per-cluster aggregates)
dav = pd.read_parquet(SCR / "davie_cluster_expr.parquet")
fca = pd.read_parquet(SCR / "fca_cluster_expr.parquet")
dav_labels = list(pd.unique(dav.annotation)); fca_labels = list(pd.unique(fca.annotation))
covered = {(s, n) for s, n, *_ in rules}
for lab in dav_labels:
    if (DAV, lab) not in covered:
        assert lab.isdigit(), lab
        rules.append((DAV, lab, "", "unmatched", f"Davie res2 cluster {lab}: 'Unannotated' in Table S2"))
for lab in fca_labels:
    assert (FCA, lab) in covered, lab
for s, n, *_ in rules:
    assert n in (dav_labels if s == DAV else fca_labels), (s, n)

rows = []
for src, name, rule, tier, ev in rules:
    src_n = int((dav if src == DAV else fca).query("annotation == @name and metric == 'frac_expr'" + (" and sex_subset == 'all'" if src == FCA else "")).n_cells.iloc[0])
    hits = sel(rule) if rule else []
    if not hits:
        rows.append(dict(source=src, source_name=name, malecns_type="", tier="unmatched" if not rule else tier, evidence=ev, n_cells_malecns=0, n_cells_source=src_n, rule=rule))
        continue
    for t in hits:
        rows.append(dict(source=src, source_name=name, malecns_type=t, tier=tier, evidence=ev, n_cells_malecns=int(T.loc[t, "n_cells"]), n_cells_source=src_n, rule=rule))
M = pd.DataFrame(rows)
hdr = ("# type_map_central.csv -- source cluster/annotation -> MaleCNS v1.0 type, for the two central-brain single-cell atlases.\n"
       "# sources: davie2018 = Davie et al. 2018 Cell 174:982 (GEO GSE107451, 57k-cell 10x metadata `annotation`; Table S2 = mmc2.xlsx);\n"
       "#          fca2022 = Li et al. 2022 Science 375:eabk2432, Fly Cell Atlas head 10x stringent loom col_attrs/annotation (FBbt terms).\n"
       "# tier: exact = source label (or its type token) is a MaleCNS type name; alias = documented synonym of one type; fuzzy = the label names a\n"
       "#       family that MaleCNS splits into several types, selected by the name rule in `rule` (re:regex | list:A|B); class = the label is a cell\n"
       "#       class, MaleCNS side selected by class/subclass/nt columns or a curated list (`rule`); unmatched = no evidenced correspondence.\n"
       "# One row per (source_name, malecns_type). n_cells_malecns = MaleCNS cells of that type; n_cells_source = cells/nuclei in the source cluster.\n"
       "# Sex: davie2018 is mixed-sex (male 27,854 / female 29,048), fca2022 head is mixed (male 47,409 / female 49,105 / mix 4,013); MaleCNS is male.\n"
       "# Built by scratchpad/build_central_map.py (session 10 NT integration workflow, source key central).\n")
OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MAP, "w", encoding="utf-8", newline="") as fh:
    fh.write(hdr)
    M[["source", "source_name", "malecns_type", "tier", "evidence", "n_cells_malecns", "n_cells_source", "rule"]].to_csv(fh, index=False)
print("wrote", OUT_MAP, M.shape)

# ---------------------------------------------------------------- expression table
GENES = ["ChAT", "VAChT", "Gad1", "VGAT", "VGlut", "Hdc", "ple", "DAT", "Tdc2", "Tbh", "Trh", "SerT",
         "nAChRalpha1", "nAChRalpha2", "nAChRalpha3", "nAChRalpha4", "nAChRalpha5", "nAChRalpha6", "nAChRalpha7",
         "nAChRbeta1", "nAChRbeta2", "nAChRbeta3", "Rdl", "Lcch3", "Grd", "GluClalpha", "KaiR1D", "GluRIA", "GluRIB",
         "Nmdar1", "Nmdar2", "HisCl1", "ort", "mAChR-A", "mAChR-B", "mAChR-C", "GABA-B-R1", "GABA-B-R2", "GABA-B-R3",
         "mGluR", "Dop1R1", "Dop1R2", "Dop2R", "DopEcR", "Oamb", "Octbeta1R", "Octbeta2R", "Octbeta3R", "Octalpha2R",
         "5-HT1A", "5-HT1B", "5-HT2A", "5-HT2B", "5-HT7"]
SYN = {"KaiR1D": ["KaiR1D", "CG3822"], "Octalpha2R": ["Octalpha2R", "CG18208"], "mAChR-A": ["mAChR-A", "mAcR-60C"],
       "mAChR-B": ["mAChR-B", "CG7918"], "mAChR-C": ["mAChR-C", "CG12796"], "Octbeta1R": ["Octbeta1R", "CG6919"],
       "Octbeta2R": ["Octbeta2R", "CG6989"], "Octbeta3R": ["Octbeta3R", "CG42244"], "Nmdar1": ["Nmdar1", "NMDAR1"],
       "Nmdar2": ["Nmdar2", "NMDAR2"], "GluClalpha": ["GluClalpha", "GluCl"]}
def take(df, src):
    gcols = set(df.columns)
    out = pd.DataFrame({"source": src, "source_name": df.annotation, "metric": df.metric, "n_cells": df.n_cells,
                        "sex_subset": df.sex_subset if "sex_subset" in df else "all"})
    missing = []
    for g in GENES:
        c = next((a for a in SYN.get(g, [g]) if a in gcols), None)
        out[g] = df[c].to_numpy() if c else np.nan
        if not c:
            missing.append(g)
    print(src, "genes absent from source:", missing)
    return out
E = pd.concat([take(dav, DAV), take(fca, FCA)], ignore_index=True)
E = E[E.metric.isin(["mean_log1p_cp10k", "frac_expr"])]
ehdr = ("# expression_central.csv -- per-cluster expression of transmitter-synthesis and receptor genes from the two central-brain atlases.\n"
        "# Units: raw UMI counts per cell normalised to counts-per-10k (cp10k; davie2018 over all 17,473 genes, fca2022 over the 13,056 genes\n"
        "#   retained in the stringent loom), then metric = mean_log1p_cp10k (mean over cells of log1p(cp10k)) or frac_expr (fraction of cells with >= 1 UMI).\n"
        "# davie2018: GEO GSE107451 57k-cell 10x matrix, cluster = metadata `annotation`, all ages (0-50 d) and both sexes pooled.\n"
        "# fca2022: FCA head 10x stringent loom (s_fca_biohub_head_10x.loom), cluster = col_attrs/annotation; sex_subset = all (pooled) or male.\n"
        "# NaN = gene absent from that source's gene list (fca2022 lacks mAChR-C). Davie symbols CG3822 / CG18208 read as KaiR1D / Octalpha2R.\n"
        "# Built by scratchpad/build_central_map.py.\n")
with open(OUT_EXPR, "w", encoding="utf-8", newline="") as fh:
    fh.write(ehdr)
    E.round(4).to_csv(fh, index=False)
print("wrote", OUT_EXPR, E.shape)

# ---------------------------------------------------------------- coverage
order = {"exact": 0, "alias": 1, "fuzzy": 2, "class": 3}
best = (M[M.tier != "unmatched"].assign(o=lambda d: d.tier.map(order)).sort_values("o").drop_duplicates("malecns_type").set_index("malecns_type").tier)
best_src = {}
for src in (DAV, FCA):
    best_src[src] = (M[(M.tier != "unmatched") & (M.source == src)].assign(o=lambda d: d.tier.map(order)).sort_values("o").drop_duplicates("malecns_type").set_index("malecns_type").tier)
def cov(group_mask, label, tiers):
    G = T[group_mask]
    tot = dict(types=len(G), cells=int(G.n_cells.sum()), out_signed=float(G.out_syn_signed.sum()), out_raw=float(G.out_syn_raw.sum()))
    lines = [f"### {label}: {tot['types']} types, {tot['cells']:,} cells, {tot['out_signed']:,.0f} signed output synapses (|W|), {tot['out_raw']:,.0f} raw output synapses",
             "", "| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |", "|---|---|---|---|---|---|---|---|"]
    cum = dict(types=0, cells=0, out_signed=0.0, out_raw=0.0)
    for tier in ["exact", "alias", "fuzzy", "class"]:
        idx = [t for t in G.index if tiers.get(t) == tier]
        S = G.loc[idx]
        r = dict(types=len(S), cells=int(S.n_cells.sum()), out_signed=float(S.out_syn_signed.sum()), out_raw=float(S.out_syn_raw.sum()))
        for k in cum: cum[k] += r[k]
        lines.append(f"| {tier} | {r['types']} | {r['cells']:,} | {100*r['cells']/max(tot['cells'],1):.1f}% | {r['out_signed']:,.0f} | {100*r['out_signed']/max(tot['out_signed'],1):.1f}% | {r['out_raw']:,.0f} | {100*r['out_raw']/max(tot['out_raw'],1):.1f}% |")
    lines.append(f"| any tier | {cum['types']} | {cum['cells']:,} | {100*cum['cells']/max(tot['cells'],1):.1f}% | {cum['out_signed']:,.0f} | {100*cum['out_signed']/max(tot['out_signed'],1):.1f}% | {cum['out_raw']:,.0f} | {100*cum['out_raw']/max(tot['out_raw'],1):.1f}% |")
    un = tot['cells'] - cum['cells']
    lines.append(f"| unmatched | {tot['types']-cum['types']} | {un:,} | {100*un/max(tot['cells'],1):.1f}% | {tot['out_signed']-cum['out_signed']:,.0f} | {100*(tot['out_signed']-cum['out_signed'])/max(tot['out_signed'],1):.1f}% | {tot['out_raw']-cum['out_raw']:,.0f} | {100*(tot['out_raw']-cum['out_raw'])/max(tot['out_raw'],1):.1f}% |")
    return "\n".join(lines) + "\n"
optic = T.superclass.isin(OPTIC_SC); central = T.superclass.isin(CENTRAL_SC); allb = T.superclass.isin(BRAIN_SC)
report = []
for name, tiers in [("both sources (best tier per type)", best), ("davie2018 only", best_src[DAV]), ("fca2022 only", best_src[FCA])]:
    report.append(f"## Coverage, {name}\n")
    report.append(cov(optic, "Optic lobe (ol_intrinsic + visual_projection + ol_sensory)", tiers))
    report.append(cov(central, "Central brain (cb_intrinsic + cb_sensory + cb_endocrine + cb_motor + cb_efferent + visual_centrifugal)", tiers))
    report.append(cov(allb, "All brain (optic + central + descending_neuron)", tiers))
    report.append(cov(pd.Series(True, index=T.index), "All MaleCNS typed cells", tiers))
# per-class detail for central classes
report.append("## Central-brain classes matched (both sources; MaleCNS cells / signed out / raw out)\n")
report.append("| source | source_name | tier | n MaleCNS types | MaleCNS cells | out syn (|W|) | out syn (raw) | source cells |\n|---|---|---|---|---|---|---|---|")
for (src, name), g in M[M.tier != "unmatched"].groupby(["source", "source_name"], sort=False):
    S = T.loc[g.malecns_type]
    if not S.superclass.isin(CENTRAL_SC).any():
        continue
    report.append(f"| {src} | {name} | {g.tier.iloc[0]} | {len(g)} | {int(S.n_cells.sum()):,} | {S.out_syn_signed.sum():,.0f} | {S.out_syn_raw.sum():,.0f} | {int(g.n_cells_source.iloc[0]):,} |")
report.append("\n## Unmatched source labels\n")
report.append("| source | source_name | source cells | note |\n|---|---|---|---|")
for _, r in M[M.tier == "unmatched"].iterrows():
    if r.source == DAV and r.source_name.isdigit():
        continue
    report.append(f"| {r.source} | {r.source_name} | {int(r.n_cells_source):,} | {r.evidence} |")
nd = M[(M.tier == "unmatched") & (M.source == DAV) & M.source_name.str.isdigit()]
report.append(f"| davie2018 | {len(nd)} unannotated numeric res.2 clusters | {int(nd.n_cells_source.sum()):,} | 'Unannotated' in Table S2 |")
(SCR / "coverage_central.md").write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))
summary = dict(n_map_rows=len(M), n_matched_types=int(best.shape[0]), tiers=M[M.tier != 'unmatched'].drop_duplicates('malecns_type').tier.value_counts().to_dict())
print(json.dumps(summary))
