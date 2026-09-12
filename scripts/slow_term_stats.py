"""Slow-term statistics for docs/audits/slow_term.md, on the adopted cache and the round-2 receptor table (class rule).
Per slow class: entries, synapse-equivalents (count x |slow sign|, uncapped), gain-weighted syn-eq, by target module;
monoamine entries onto rate units (optic) split by presynaptic superclass; rate-cell monoamine outputs onto spiking
targets (outside both models); per-target-cell monoamine slow load and the implied steady tone at the sweep gains;
dopamine-lead variant counts. Writes JSON to out/slow_term_stats.json."""
import json, sys
from pathlib import Path
ROOT = Path(r"D:\Projects\flyverse")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from flyverse import connectome as cn
from flyverse import brain as br
from flyverse import regions

c = cn.load(verbose=False)
n = c.neurons
rs = cn.receptor_signs(c, with_counts=True)
W = c.W.tocsr(); coo = W.tocoo()
post, pre = coo.row, coo.col
nt_pre = n.nt.to_numpy()[pre]
sc = n.superclass.fillna("").to_numpy()
types = n.type.fillna("").to_numpy()
lab = regions.labels(c)
cnt = rs.count
wabs_total = float(np.abs(W.data).sum())
raw_total = float(cnt.sum())
out = {"neurons": int(c.n), "entries": int(W.nnz), "syn_W": wabs_total, "syn_raw_incl_sign0": raw_total,
       "nt_counts": {k: int(v) for k, v in n.nt.value_counts().items()}}
print("neurons", c.n, "entries", W.nnz, "|W| syn", wabs_total, "raw incl sign0", raw_total)

gain = br.DEFAULT_RECEPTOR_GAIN
sf = rs.slow_factor(gain)
classes = {}
for i, name in enumerate(cn.SLOW_CLASSES):
    m = rs.slow_class == i
    classes[name] = {"entries": int(m.sum()), "entries_frac": float(m.mean()), "syn_eq": float(cnt[m].sum()),
                     "syn_eq_frac_of_raw": float(cnt[m].sum() / raw_total),
                     "syn_eq_pos": float(cnt[m & (rs.slow_sign > 0)].sum()), "syn_eq_neg": float(cnt[m & (rs.slow_sign < 0)].sum()),
                     "gain_weighted_pos": float((cnt * sf)[m & (rs.slow_sign > 0)].sum()), "gain_weighted_neg": float(-(cnt * sf)[m & (rs.slow_sign < 0)].sum())}
    if i:
        by_nt = pd.DataFrame({"nt": nt_pre[m], "syn": cnt[m], "sign": rs.slow_sign[m]}).groupby("nt").agg(entries=("syn", "size"), syn_eq=("syn", "sum"), pos=("sign", lambda x: int((x > 0).sum())), neg=("sign", lambda x: int((x < 0).sum())))
        classes[name]["by_pre_nt"] = by_nt.to_dict("index")
        by_mod = pd.DataFrame({"mod": lab[post[m]], "syn": cnt[m]}).groupby("mod").syn.agg(["size", "sum"]).rename(columns={"size": "entries", "sum": "syn_eq"})
        classes[name]["by_post_module"] = by_mod.to_dict("index")
out["classes"] = classes
print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in classes.items()}, indent=1))

# monoamine synapses overall vs those carrying a slow sign
mono = np.isin(nt_pre, ["dopamine", "octopamine", "serotonin"])
per_nt = {}
for t in ["dopamine", "octopamine", "serotonin"]:
    s = nt_pre == t
    per_nt[t] = {"entries": int(s.sum()), "syn": float(cnt[s].sum()), "matched_entries": int((s & rs.matched).sum()), "matched_syn": float(cnt[s & rs.matched].sum()),
                 "slow_pos_syn": float(cnt[s & (rs.slow_sign > 0)].sum()), "slow_neg_syn": float(cnt[s & (rs.slow_sign < 0)].sum()),
                 "mixed_syn": float(cnt[s & rs.matched & (rs.slow_sign == 0) & (rs.slow_gain > 0)].sum()),
                 "matched_none_syn": float(cnt[s & rs.matched & (rs.slow_sign == 0) & (rs.slow_gain == 0)].sum())}
out["monoamine_by_nt"] = per_nt
print(json.dumps(per_nt, indent=1))

# where the monoamine slow entries land: rate (ol_intrinsic non-PR) targets vs spiking, and who sends them
is_pr = n.type.isin(cn.PHOTORECEPTOR_TYPES).to_numpy()
rate = (sc == "ol_intrinsic") & ~is_pr
mz = mono & (rs.slow_sign != 0)
r_post, r_pre = rate[post], rate[pre]
land = {"onto_rate_from_spiking": {"entries": int((mz & r_post & ~r_pre).sum()), "syn_eq": float(cnt[mz & r_post & ~r_pre].sum())},
        "onto_rate_from_rate": {"entries": int((mz & r_post & r_pre).sum()), "syn_eq": float(cnt[mz & r_post & r_pre].sum())},
        "onto_spiking_from_spiking": {"entries": int((mz & ~r_post & ~r_pre).sum()), "syn_eq": float(cnt[mz & ~r_post & ~r_pre].sum())},
        "onto_spiking_from_rate_DROPPED": {"entries": int((mz & ~r_post & r_pre).sum()), "syn_eq": float(cnt[mz & ~r_post & r_pre].sum()),
                                           "pre_types": pd.Series(types[pre[mz & ~r_post & r_pre]]).value_counts().head(5).to_dict()}}
out["monoamine_slow_landing"] = land
print(json.dumps(land, indent=1))
src = pd.DataFrame({"pre_sc": sc[pre[mz & r_post]], "syn": cnt[mz & r_post]}).groupby("pre_sc").syn.agg(["size", "sum"])
out["monoamine_slow_onto_rate_by_pre_superclass"] = src.rename(columns={"size": "entries", "sum": "syn_eq"}).to_dict("index")
print(src)
# the octopaminergic visual centrifugal output: matched fraction vs silenced (fast 0 and no slow sign)
oa_vc = (nt_pre == "octopamine") & (sc[pre] == "visual_centrifugal")
out["octopamine_visual_centrifugal"] = {"entries": int(oa_vc.sum()), "syn": float(cnt[oa_vc].sum()),
                                       "slow_signed_syn": float(cnt[oa_vc & (rs.slow_sign != 0)].sum()),
                                       "onto_rate_slow_signed_syn": float(cnt[oa_vc & (rs.slow_sign != 0) & r_post].sum()),
                                       "matched_syn": float(cnt[oa_vc & rs.matched].sum())}
print(out["octopamine_visual_centrifugal"])

# per-target monoamine slow load (capped like the model: min(count, 60) x |sign| x gain factor), fan-in scaled
S = W.copy(); S.data = cnt * rs.slow_factor(gain, slow_class="monoamine")
S.data = np.sign(S.data) * np.minimum(np.abs(S.data), 60.0); S.eliminate_zeros()
p = br.LIFParams(receptor_model="full")
Wsh = br._shaped_weights(c, p, rs)
tot = np.asarray(abs(Wsh).sum(axis=1)).ravel()
scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0)
pos = np.asarray(S.maximum(0).sum(axis=1)).ravel() * scale
neg = -np.asarray(S.minimum(0).sum(axis=1)).ravel() * scale
df = pd.DataFrame({"type": types, "module": lab, "pos": pos, "neg": neg, "spiking": ~rate})
top = df[df.spiking].groupby("type").agg(cells=("pos", "size"), pos=("pos", "mean"), neg=("neg", "mean"), module=("module", "first")).assign(net=lambda d: d.pos - d.neg)
top = top[(top.pos + top.neg) > 0].sort_values("net", ascending=False)
out["top_spiking_targets_by_net_monoamine_slow_syn_eq"] = top.head(20).round(2).reset_index().to_dict("records")
out["bottom_spiking_targets"] = top.tail(10).round(2).reset_index().to_dict("records")
print(top.head(20).round(2).to_string()); print(top.tail(10).round(2).to_string())
for t in ["KCg-m", "KCab-m", "KCa'b'-m", "MBON01", "DNp01", "MN9", "DNp18", "GNG175", "LC4", "LPLC2", "PAM08", "PPL101"]:
    if t in top.index:
        print(t, top.loc[t].to_dict())
out["named_targets"] = {t: {k: (round(float(v), 3) if k != "module" else v) for k, v in top.loc[t].to_dict().items()}
                        for t in ["KCg-m", "KCab-m", "KCa'b'-m", "MBON01", "DNp01", "MN9", "DNp18", "GNG175", "LC4", "LPLC2", "PAM08", "PPL101", "OA-VPM3", "ExR3", "EL"] if t in top.index}
# octopaminergic cells receiving octopamine (autoreceptors): same-type damping is not applied to the slow matrix
oa_cells = np.flatnonzero(n.nt.to_numpy() == "octopamine")
oa_self = mz & np.isin(pre, oa_cells) & np.isin(post, oa_cells)
out["octopamine_onto_octopaminergic_cells"] = {"entries": int(oa_self.sum()), "syn_eq": float(cnt[oa_self].sum()),
                                               "pos_entries": int((oa_self & (rs.slow_sign > 0)).sum()), "same_type_entries": int((oa_self & (types[pre] == types[post])).sum())}
print(out["octopamine_onto_octopaminergic_cells"])
# implied steady tone (mV) of the strongest target at the sweep gains, presyn at 50 Hz: g x 0.275 x syneq x 0.05/ms x tau
for g in (0.01, 0.02, 0.03):
    print("gain", g, "tone at 50 Hz all + inputs active, top net cell:", g * 0.275 * float(top.net.max()) * 0.05 * 200, "mV")
out["implied_tone_mV_top_cell_50Hz"] = {str(g): g * 0.275 * float(top.net.max()) * 0.05 * 200 for g in (0.01, 0.02, 0.03)}
out["fraction_spiking_cells_with_monoamine_slow_input"] = float(((pos + neg) > 0)[~rate].mean())
out["cells_with_monoamine_slow_input"] = int(((pos + neg) > 0).sum())
json.dump(out, open(ROOT / "out" / "slow_term_stats.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("wrote out/slow_term_stats.json")
