#!/usr/bin/env python
"""Monoamine coverage of the shipped receptor table on the shipped cache, and the data-anchored magnitude bracket of the
slow term's monoamine class (docs/audits/monoamine_slow_term.md sections 1-2). CPU only; writes SCRATCH tables under
out/monoamines/ (never the shipped flyverse/data/receptors_by_type.csv) plus their md5s.

    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python scripts/build_monoamine_tables.py [--out out/monoamines] [--cache-dir DIR] [--net-rule abs]

Tables (CSV, one row per ...):
  coverage_by_nt.csv         transmitter (dopamine / octopamine / serotonin / unknown): sign-0 bodies, presynaptic bodies,
                             stored entries, raw synapses; synapses onto a profiled target (a row for that transmitter),
                             split by the row's slow sign under the net rule (+1 / -1 / 0 = tie or no group), and the
                             fallback share (unprofiled target: silenced in every model)
  coverage_by_lead.csv       transmitter x lead receptor gene of the winning slow group (slow_pos_lead when the row's
                             sign is +1, slow_neg_lead when -1): synapses, entries, target types, with the sign rule
  coverage_by_superclass.csv transmitter x postsynaptic superclass: raw synapses, profiled share, signed share (+ / -)
  top_pairs.csv              the 20 largest (presynaptic type, postsynaptic type) pairs per transmitter by raw synapses,
                             with the row's tier / slow sign / gain class / lead receptor
  target_load.csv            per postsynaptic type: cells, the monoamine slow load per cell in capped, fan-in-scaled
                             synapse-equivalents split by transmitter and sign (what W_slow carries under 'full')
  anchor_bracket.csv         the literature anchors (section 2 of the audit) and the slow_gain each implies on its named
                             model targets at the stated presynaptic rate assumptions
  md5.txt                    md5 of every table above
  summary.json               the numbers quoted in the audit + provenance (compiled connectome, receptor table md5)

The slow sign rule is the table builder's (scripts/build_receptor_table.py RECEPTOR_GROUPS): the sign of each receptor
GROUP is the direction of the postsynaptic effect on membrane potential / spiking (Gs / Gq-coupled = +1, Gi-coupled =
-1), and a (type, transmitter) row's slow sign is the larger of the two groups' expression sums with a 2-fold margin
(net rule 'abs', the shipped default) -- ties and absent groups give 0 (no slow entry). Receptor gene identity is not
carried into the model; only the sign, gain class and the two-way class are (connectome.receptor_signs).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MONO = ("dopamine", "octopamine", "serotonin")
SIGN0_NTS = MONO + ("unknown",)

# The receptor genes of each slow group and the G-protein coupling the sign rests on (scripts/build_receptor_table.py
# RECEPTOR_GROUPS; the citations are the audit's section 1).
GROUP_SIGN = {
    "dopamine": {"Dop1R1": (+1, "Gs; Sugamori et al. 1995 / Gotzes et al. 1994"), "Dop1R2": (+1, "Gs/Gq (DAMB); Han et al. 1996, Feng et al. 1996"),
                 "DopEcR": (+1, "Gs; Srivastava et al. 2005"), "Dop2R": (-1, "Gi; Hearn et al. 2002")},
    "octopamine": {"Oamb": (+1, "Gq; Han et al. 1998"), "Octbeta1R": (+1, "Gs; Maqueira et al. 2005"), "Octbeta2R": (+1, "Gs; Maqueira et al. 2005"),
                   "Octbeta3R": (+1, "Gs; Maqueira et al. 2005"), "Octalpha2R": (-1, "Gi; Qi et al. 2017")},
    "serotonin": {"5-HT1A": (-1, "Gi; Saudou et al. 1992"), "5-HT1B": (-1, "Gi; Saudou et al. 1992"), "5-HT2A": (+1, "Gq; Colas et al. 1995"),
                  "5-HT2B": (+1, "Gq; Gasque et al. 2013"), "5-HT7": (+1, "Gs; Witz et al. 1990")},
}

# Literature anchors of section 2 (effect sizes at a known target). `kind` says which slow_mode the effect maps to.
# tone_mv_lo / _hi: the g_slow (mV) that reproduces the bracket in that mode (gain mode: factor = 1 + g_slow / 7 mV,
# so a x1.5 .. x2.3 gain is 3.5 .. 9.1 mV; additive: the equivalent membrane offset by the mode normalisation).
ANCHORS = [
    dict(anchor="OA_visual_gain", transmitter="octopamine", kind="gain", effect="x1.5 .. x2.3 on visual motion responses during flight / locomotion or under the OA agonist CDM",
         targets="HSN|HSE|HSS|~^VS|Mi4|Mi1|Tm3", tone_mv_lo=3.5, tone_mv_hi=9.1,
         source="Maimon, Straw & Dickinson 2010, Nat Neurosci 13:393 (VS peak-to-peak responses double in flight); Suver, Mamiya & Dickinson 2012, Curr Biol 22:2294 "
                "(octopamine neurons necessary and sufficient for the boost; OA application mimics it in quiescent flies); Longden & Krapp 2010, Front Syst Neurosci 4:153 "
                "(CDM 2.5 uM on blowfly H2: spontaneous 7.9 -> 15.6 Hz, initial response gain +126 %, mean rate 14.3 -> 21.3 Hz = x1.49); Strother et al. 2018, PNAS 115:E102 "
                "(walking / CDM 10 uM raise Mi1, Tm3, Mi9, T4 responses at high temporal frequency; OA neuron photoactivation drives Mi4)"),
    dict(anchor="DA_KC_tone", transmitter="dopamine", kind="additive", effect="no change of gamma-KC baseline membrane voltage or evoked spiking under DAN activation; the DAN effect is a KC>MBON weight change (-80 +- 5.7 % spikes, -90 +- 3.7 % charge after pairing)",
         targets="~^KC", tone_mv_lo=0.0, tone_mv_hi=0.0,
         source="Cohn, Morantte & Ruta 2015, Cell 163:1742 (Fig. S6B: 58E02+ DAN activation has no apparent effect on gamma-KC baseline voltage or evoked spiking; DAN pairing depresses, DAN alone potentiates KC>MBON); "
                "Hige et al. 2015, Neuron 88:985 (Fig. 1F / 3D: MBON-gamma1pedc odour response 118 -> 24 spikes, charge transfer -90 %, > 40 min); Tomchik & Davis 2009, Neuron 64:510 (DA / OA raise cAMP in KCs)"),
    dict(anchor="5HT_AL_inhibition", transmitter="serotonin", kind="additive", effect="endogenous 5-HT (CSDn, 1-2 Hz spontaneous) lowers PN odour responses (fluoxetine decreases, methysergide increases DA1 PN responses) by enhancing GABAergic presynaptic inhibition of ORNs; exogenous 100 uM 5-HT raises PN responses; CSDn stimulation gives LNs a brief depolarisation then a delayed hyperpolarisation (no mV number published)",
         targets="~^ORN_|~_l2PN|_adPN|_lPN|_lvPN|_ilPN|~^(lLN|v2LN|v3LN|il3LN|l2LN|vLN)", tone_mv_lo=-0.7, tone_mv_hi=-3.5,
         source="Zhang & Gaudry 2016, eLife 5:e16836; Dacks et al. 2009, J Neurogenet 23:366 (5-HT enhances PN responses to sparse odours); Sizemore & Dacks 2016, Sci Rep 6:37119 "
                "(5-HT1A on GABAergic / peptidergic LNs, 5-HT2A/2B/7 on excitatory PNs, 5-HT2B on ORNs; no effect sizes). The bracket is the model's own gap scaled (0.1 .. 0.5 x 7 mV), not a measured mV: labelled as such"),
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/monoamines")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--net-rule", default="abs", choices=["class", "abs", "nonmda"])
    ap.add_argument("--receptor-table", default=None)
    ap.add_argument("--presyn-hz", default="5,10,20", help="presynaptic rate assumptions (Hz) for the bracket (no Drosophila OA / DAN / 5-HT rate during walking is published in Hz: Babski et al. 2024, Heliyon, report VPM1/2 tonic single spikes rising in locomotor bouts without a value)")
    args = ap.parse_args()
    from flyverse import brain as br, connectome as cn, regions
    from flyverse.interp import common

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    c = cn.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else cn.load(verbose=False)
    n = c.neurons
    table_path = Path(args.receptor_table) if args.receptor_table else cn.RECEPTOR_TABLE
    rt = cn.read_receptor_table(table_path)
    rs = cn.receptor_signs(c, table_path=str(table_path) if args.receptor_table else None, net_rule=args.net_rule, with_counts=True)
    if rs.count is None:
        raise SystemExit("cache/sign0_counts.npz is not available: the monoamine synapses have no counts")
    W = c.W.tocsr(); coo = W.tocoo(); post, pre = coo.row, coo.col
    cnt = rs.count.astype(np.float64)
    nt = n.nt.fillna("unknown").to_numpy().astype(str)
    types = n.type.fillna("").to_numpy().astype(str)
    sc = n.superclass.fillna("").to_numpy().astype(str)
    mod = regions.labels(c)
    nt_pre = nt[pre]
    ssfx = "_abs" if args.net_rule == "abs" else ""
    # the row of every matched entry: (post type, pre nt) -> table row
    rows = rt[~rt.malecns_type.astype(str).str.startswith("<")].reset_index(drop=True)
    key = {(t, x): i for i, (t, x) in enumerate(zip(rows.malecns_type, rows.transmitter))}
    lead_pos = rows.slow_pos_lead.fillna("none").to_numpy().astype(str); lead_neg = rows.slow_neg_lead.fillna("none").to_numpy().astype(str)
    tier_row = rows.tier.to_numpy().astype(str); gain_row = rows["slow_gain_class" + ssfx].fillna("none").to_numpy().astype(str)
    net_row = rows["slow_net" + ssfx].fillna("none").to_numpy().astype(str)
    summary = {"connectome": common.connectome_fingerprint(c, args.cache_dir), "receptor_table": {"path": str(table_path), "md5": md5(Path(table_path))},
               "net_rule": args.net_rule, "n_neurons": int(c.n), "entries": int(W.nnz), "raw_synapses_incl_sign0": float(cnt.sum()),
               "syn_W": float(np.abs(W.data).sum())}
    print(f"cache {summary['connectome'].get('cache_dir')} n {c.n:,} entries {W.nnz:,} raw syn {cnt.sum():,.0f}; table {table_path} md5 {summary['receptor_table']['md5']}; net rule {args.net_rule}")

    # ---------------------------------------------------------------- 1. coverage by transmitter
    cov = []
    lead_rows = []
    for t in SIGN0_NTS:
        m = nt_pre == t
        bodies = int((nt == t).sum()); presyn = int(len(np.unique(pre[m])))
        matched = m & rs.matched
        pos = matched & (rs.slow_sign > 0); neg = matched & (rs.slow_sign < 0); zero = matched & (rs.slow_sign == 0)
        d = {"transmitter": t, "bodies": bodies, "presyn_bodies_with_output": presyn, "entries": int(m.sum()), "raw_syn": float(cnt[m].sum()),
             "matched_syn": float(cnt[matched].sum()), "matched_entries": int(matched.sum()), "matched_post_types": int(len(np.unique(types[post[matched]]))),
             "slow_pos_syn": float(cnt[pos].sum()), "slow_neg_syn": float(cnt[neg].sum()), "slow_zero_syn": float(cnt[zero].sum()),
             "fallback_syn": float(cnt[m & ~rs.matched].sum()), "fallback_entries": int((m & ~rs.matched).sum()),
             "fallback_post_types": int(len(np.unique(types[post[m & ~rs.matched]])))}
        d["matched_frac"] = d["matched_syn"] / max(d["raw_syn"], 1); d["signed_frac"] = (d["slow_pos_syn"] + d["slow_neg_syn"]) / max(d["raw_syn"], 1)
        d["silenced_frac_all_models"] = 1.0 - d["signed_frac"]         # fallback + tie/none: zero on the fast path, absent from W_slow
        cov.append(d)
        if t in MONO:
            idx = np.array([key.get((types[q], t), -1) for q in post[matched]]) if matched.any() else np.array([], int)
            sgn = rs.slow_sign[matched]; cm = cnt[matched]; pt = types[post[matched]]
            lead = np.where(sgn > 0, lead_pos[idx], np.where(sgn < 0, lead_neg[idx], "none (tie / no group)"))
            df = pd.DataFrame({"lead": lead, "sign": sgn, "syn": cm, "post_type": pt, "net": net_row[idx]})
            for (ld, s), g in df.groupby(["lead", "sign"]):
                gs, cite = GROUP_SIGN[t].get(ld, (0, "no group wins (tie under the 2-fold margin, or neither group expressed)"))
                lead_rows.append({"transmitter": t, "lead_receptor": ld, "slow_sign": int(s), "group_sign_rule": gs, "coupling_citation": cite,
                                  "syn": float(g.syn.sum()), "entries": int(len(g)), "post_types": int(g.post_type.nunique()),
                                  "net_classes": ";".join(f"{k}:{v}" for k, v in g.net.value_counts().items())})
    cov_df = pd.DataFrame(cov); lead_df = pd.DataFrame(lead_rows).sort_values(["transmitter", "syn"], ascending=[True, False])
    cov_df.to_csv(out / "coverage_by_nt.csv", index=False); lead_df.to_csv(out / "coverage_by_lead.csv", index=False)
    print(cov_df.to_string(index=False)); print(lead_df.to_string(index=False))

    # ---------------------------------------------------------------- 2. by postsynaptic superclass
    sup = []
    for t in SIGN0_NTS:
        m = nt_pre == t
        df = pd.DataFrame({"sc": sc[post[m]], "syn": cnt[m], "matched": rs.matched[m], "sign": rs.slow_sign[m]})
        g = df.groupby("sc").apply(lambda x: pd.Series({"raw_syn": x.syn.sum(), "matched_syn": x.syn[x.matched].sum(),
                                                        "pos_syn": x.syn[x.sign > 0].sum(), "neg_syn": x.syn[x.sign < 0].sum()}), include_groups=False).reset_index()
        g.insert(0, "transmitter", t); sup.append(g)
    sup_df = pd.concat(sup, ignore_index=True)
    sup_df["matched_frac"] = sup_df.matched_syn / sup_df.raw_syn.clip(lower=1); sup_df["signed_frac"] = (sup_df.pos_syn + sup_df.neg_syn) / sup_df.raw_syn.clip(lower=1)
    sup_df = sup_df.sort_values(["transmitter", "raw_syn"], ascending=[True, False])
    sup_df.to_csv(out / "coverage_by_superclass.csv", index=False)

    # ---------------------------------------------------------------- 3. top pairs
    pairs = []
    for t in MONO:
        m = nt_pre == t
        df = pd.DataFrame({"pre_type": types[pre[m]], "post_type": types[post[m]], "syn": cnt[m], "matched": rs.matched[m], "sign": rs.slow_sign[m],
                           "gain": rs.slow_gain[m], "tier": rs.tier[m], "post_module": mod[post[m]]})
        g = df.groupby(["pre_type", "post_type"]).agg(syn=("syn", "sum"), entries=("syn", "size"), matched=("matched", "first"), sign=("sign", "first"),
                                                       tier=("tier", "first"), post_module=("post_module", "first")).reset_index().sort_values("syn", ascending=False).head(20)
        g["tier"] = [cn.RECEPTOR_TIERS[int(x)] for x in g.tier]
        g["lead"] = [(lead_pos if s > 0 else lead_neg)[key[(pt, t)]] if (pt, t) in key and s != 0 else ("tie/none" if (pt, t) in key else "no row") for pt, s in zip(g.post_type, g.sign)]
        g["gain_class"] = [gain_row[key[(pt, t)]] if (pt, t) in key else "" for pt in g.post_type]
        g.insert(0, "transmitter", t); g.insert(1, "rank", range(1, len(g) + 1)); pairs.append(g)
    pairs_df = pd.concat(pairs, ignore_index=True)
    pairs_df.to_csv(out / "top_pairs.csv", index=False)
    print(pairs_df.to_string(index=False))

    # ---------------------------------------------------------------- 4. per-target slow load (what W_slow carries) and the bracket
    p = br.LIFParams(receptor_model="full", receptor_gain={"none": 1.0, "low": 1.0, "mid": 1.0, "high": 1.0})   # the shipped fast weights under 'full'
    gain = br._receptor_gain(p)
    Wsh = br._shaped_weights(c, p, rs)
    tot = np.asarray(abs(Wsh).sum(axis=1)).ravel()
    scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0)
    is_pr = n.type.isin(cn.PHOTORECEPTOR_TYPES).to_numpy(); rate_unit = (sc == "ol_intrinsic") & ~is_pr
    load = {}
    sf = rs.slow_factor(gain, slow_class="monoamine")
    capped = np.sign(sf) * np.minimum(np.abs(cnt * sf), float(p.conn_cap))
    for t in MONO:
        for sgn, name in ((1, "pos"), (-1, "neg")):
            m = (nt_pre == t) & (np.sign(capped) == sgn)
            v = np.bincount(post[m], weights=np.abs(capped[m]), minlength=c.n) * scale
            load[f"{t}_{name}"] = v
    df = pd.DataFrame(load); df["type"] = types; df["rate_unit"] = rate_unit; df["module"] = mod
    tl = df.groupby("type").agg(cells=("module", "size"), module=("module", "first"), rate_unit=("rate_unit", "mean"),
                                **{k: (k, "mean") for k in load}).reset_index()
    tl["net_syn_eq_per_cell"] = sum(tl[f"{t}_pos"] - tl[f"{t}_neg"] for t in MONO)
    tl["abs_syn_eq_per_cell"] = sum(tl[f"{t}_pos"] + tl[f"{t}_neg"] for t in MONO)
    tl = tl[tl.abs_syn_eq_per_cell > 0].sort_values("abs_syn_eq_per_cell", ascending=False)
    tl.to_csv(out / "target_load.csv", index=False)
    summary["cells_with_monoamine_slow_input"] = int(((df[[k for k in load]].sum(axis=1)) > 0).sum())
    summary["spiking_cells_with_monoamine_slow_input"] = int((((df[[k for k in load]].sum(axis=1)) > 0) & ~rate_unit).sum())
    summary["monoamine_slow_entries"] = int((sf != 0).sum()); summary["monoamine_slow_syn_eq"] = float(np.abs(capped[sf != 0]).sum())
    # the bracket: tone = slow_gain x w_syn x load(syn-eq, capped, fan-in scaled) x R(Hz) x tau(s)
    hz = [float(x) for x in args.presyn_hz.split(",")]
    tau_s = p.slow_tau_ms / 1000.0
    br_rows = []
    for a in ANCHORS:
        specs = a["targets"].split("|") if not a["targets"].startswith("~^ORN_") else ["~^ORN_", "~_l2PN|_adPN|_lPN|_lvPN|_ilPN", "~^(lLN|v2LN|v3LN|il3LN|l2LN|vLN)"]
        for spec in specs:
            idx = common.resolve(c, spec)
            if len(idx) == 0:
                continue
            t = a["transmitter"]
            lp = load[f"{t}_pos"][idx]; ln = load[f"{t}_neg"][idx]
            row = {"anchor": a["anchor"], "transmitter": t, "slow_mode": a["kind"], "target_spec": spec, "cells": int(len(idx)),
                   "rate_units": int(rate_unit[idx].sum()), "pos_syn_eq_per_cell": float(lp.mean()), "neg_syn_eq_per_cell": float(ln.mean()),
                   "cells_with_load": int(((lp + ln) > 0).sum()), "tone_mv_lo": a["tone_mv_lo"], "tone_mv_hi": a["tone_mv_hi"], "effect": a["effect"], "source": a["source"]}
            net = float(lp.mean() - ln.mean())
            for R in hz:
                per_gain = p.w_syn * abs(net) * R * tau_s          # mV of steady tone per unit slow_gain, all inputs at R Hz
                row[f"tone_mv_per_unit_gain_at_{R:g}Hz"] = per_gain
                row[f"slow_gain_for_lo_at_{R:g}Hz"] = (abs(a["tone_mv_lo"]) / per_gain) if per_gain > 0 else np.nan
                row[f"slow_gain_for_hi_at_{R:g}Hz"] = (abs(a["tone_mv_hi"]) / per_gain) if per_gain > 0 else np.nan
            br_rows.append(row)
    br_df = pd.DataFrame(br_rows)
    br_df.to_csv(out / "anchor_bracket.csv", index=False)
    pd.set_option("display.width", 250)
    print(br_df.drop(columns=["effect", "source"]).round(3).to_string(index=False))
    # presynaptic monoamine populations: cells, types, output synapses (the tone is R-dependent: these must fire)
    presyn_pop = []
    for t in MONO:
        m = nt_pre == t
        g = pd.DataFrame({"pre_type": types[pre[m]], "syn": cnt[m], "signed": rs.slow_sign[m] != 0}).groupby("pre_type").agg(syn=("syn", "sum"), signed_syn=("syn", lambda x: 0.0), cells=("syn", "size"))
        g2 = pd.DataFrame({"pre_type": types[pre[m]], "syn": np.where(rs.slow_sign[m] != 0, cnt[m], 0.0)}).groupby("pre_type").syn.sum()
        g["signed_syn"] = g2; g["cells"] = pd.Series(types[nt == t]).value_counts()
        g = g.sort_values("syn", ascending=False).head(15).reset_index(); g.insert(0, "transmitter", t); presyn_pop.append(g)
    pre_df = pd.concat(presyn_pop, ignore_index=True); pre_df.to_csv(out / "presyn_types.csv", index=False)
    summary["coverage_by_nt"] = cov_df.to_dict("records"); summary["anchor_bracket"] = br_df.drop(columns=["effect", "source"]).to_dict("records")
    summary["anchors"] = ANCHORS; summary["lif_params"] = common.to_jsonable(dataclasses.asdict(p)); summary["presyn_hz_assumptions"] = hz
    summary["generator"] = "scripts/build_monoamine_tables.py " + " ".join(sys.argv[1:])
    files = ["coverage_by_nt.csv", "coverage_by_lead.csv", "coverage_by_superclass.csv", "top_pairs.csv", "target_load.csv", "anchor_bracket.csv", "presyn_types.csv"]
    lines = [f"{md5(out / f)}  {f}" for f in files]
    (out / "md5.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["md5"] = {f: md5(out / f) for f in files}
    (out / "summary.json").write_text(json.dumps(common.to_jsonable(summary), indent=1), encoding="utf-8")
    print("\n".join(lines)); print(f"wrote {out}/summary.json")


if __name__ == "__main__":
    main()
