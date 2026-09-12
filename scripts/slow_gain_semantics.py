"""'gain' mode semantics check for slow_term.md section 7 (round 3), CPU only.

(A) Two-neuron test graph (tests/test_receptor_model.two_neuron_graph): the documented factor f = clamp(1 + g_slow/gap, 0, 4)
    multiplies the NET fast input g.  Show, with the real Brain step, what a NEGATIVE tone does to a net-INHIBITED
    target (g < 0) vs a net-excited one (g > 0).
(B) Structural quadrant count on the full graph under the settings the cluster batch runs (abs fast weights,
    --receptor-gain 1,1,1, dop1r1 table, monoamine class only): per cell, the sign of the monoamine slow load
    (row sum of the fan-in-scaled slow matrix, syn-eq) against the sign of the net fast input (row sum of the fan-in-scaled
    fast matrix, syn-eq).  Cells in the (tone < 0, net fast < 0) quadrant are the ones on which 'gain' mode acts as
    disinhibition.  This is a structural proxy: the run-time g is the net input of the presynaptic cells that actually
    fire, which the JSONs do not record.
Writes out/r3_gain_semantics.json.
"""
import json, os, sys, math
import numpy as np, pandas as pd, scipy.sparse as sp
ROOT = r"D:\Projects\flyverse"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import torch
from flyverse import connectome as cn
from flyverse import brain as br
from flyverse.brain import Brain, LIFParams, _shaped_weights, _slow_weights, _slow_spec, DEFAULT_RECEPTOR_GAIN
from flyverse.connectome import receptor_signs
import test_receptor_model as trm

out = {}
# ---------------- (A) two-neuron graph -------------------------------------------------------------------------
c2 = trm.two_neuron_graph()                       # dopamine (TC) -> TA, 20 synapses (explicit zero in W, count 20)
import tempfile
tmp = tempfile.mkdtemp()
table = os.path.join(tmp, "receptors.csv")
with open(table, "w", encoding="utf-8") as f:
    f.write("# test table\n"); trm.small_table().to_csv(f, index=False)
rs2 = receptor_signs(c2, table_path=table, counts=np.array([20.0], np.float32))
p2 = LIFParams(event_driven=False, receptor_model="full", receptor_table=table, input_norm_alpha=0.0, path_gain=[],
               type_path_gain=[], adapt_jump=0.0, same_type_gain=1.0, slow_mode="gain", slow_gain=0.1, slow_tau_ms=1e9)
res_a = {}
for tone in (+3.5, -3.5, -7.0, -100.0):
    for g in (+2.0, -2.0):
        b = Brain(c2, p2, device="cpu", receptor=rs2)
        b0 = Brain(c2, p2, device="cpu", receptor=rs2)            # same, tone kept at 0 -> the no-tone reference
        b.g_slow_cls[0, 0, 1] = tone; b.g_slow.copy_(b.g_slow_cls.sum(0))
        b.g[0, 1] = g; b0.g[0, 1] = g
        b.step(1); b0.step(1)
        gap = p2.v_th - p2.v_rest
        f = min(max(1.0 + tone / gap, 0.0), 4.0)
        key = f"tone{tone:+g}_g{g:+g}"
        res_a[key] = {"factor": f, "v_with_tone_minus_rest": float(b.v[0, 1]) - p2.v_rest,
                      "v_no_tone_minus_rest": float(b0.v[0, 1]) - p2.v_rest,
                      "tone_depolarises_relative_to_no_tone": bool(float(b.v[0, 1]) > float(b0.v[0, 1]) + 1e-9)}
        print(key, res_a[key])
out["two_neuron"] = res_a

# ---------------- (B) structural quadrants on the full graph -----------------------------------------------------
c = cn.load(verbose=False)
n = c.neurons
tbl = os.path.join(ROOT, "out", "receptors_r3_dop1r1.csv")
p = LIFParams(receptor_model="full", receptor_net_rule="abs", receptor_table=tbl,
              receptor_gain={"none": 1.0, "low": 1.0, "mid": 1.0, "high": 1.0}, slow_mode="gain",
              slow_gain_by_class={"monoamine": 0.02})
spec = _slow_spec(p)
rs = receptor_signs(c, table_path=tbl, net_rule="abs", with_counts=True)
W = _shaped_weights(c, p, rs)                                             # fast, signed, capped, path gains (syn-eq)
S = _slow_weights(c, p, rs, spec)["monoamine"]                             # slow monoamine, signed, capped (syn-eq)
tot = np.asarray(abs(W).sum(axis=1)).ravel()
scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0).astype(np.float32)
Ws = (sp.diags(scale) @ W).tocsr(); Ss = (sp.diags(scale) @ S).tocsr()
net_fast = np.asarray(Ws.sum(axis=1)).ravel()                             # signed syn-eq, fan-in scaled
exc_fast = np.asarray(Ws.maximum(0).sum(axis=1)).ravel(); inh_fast = -np.asarray(Ws.minimum(0).sum(axis=1)).ravel()
tone = np.asarray(Ss.sum(axis=1)).ravel()                                 # signed monoamine slow load (syn-eq, scaled)
tone_pos = np.asarray(Ss.maximum(0).sum(axis=1)).ravel(); tone_neg = -np.asarray(Ss.minimum(0).sum(axis=1)).ravel()
has = np.asarray((Ss != 0).sum(axis=1)).ravel() > 0
from flyverse import optic
rate_mask = np.zeros(c.n, bool)
try:
    ol = optic.OpticLobe(c) if False else None
except Exception:
    ol = None
sc = n.superclass.fillna("").to_numpy(); ty = n.type.fillna("").to_numpy(); nt = n.nt.fillna("").to_numpy()
is_pr = np.array([t.startswith("R") and t[1:].replace("-", "").isdigit() or t in ("R7", "R8", "R1-R6") for t in ty])
print("cells with any monoamine slow input (dop1r1, abs, 1,1,1):", int(has.sum()), "of", c.n)
print("monoamine slow matrix: entries", S.nnz, "syn-eq +", float(S.maximum(0).sum()), "-", float(-S.minimum(0).sum()))

def quadrants(mask, label):
    m = mask & has
    q = {"cells": int(m.sum()),
         "tone_neg_net_inhibited (gain mode = disinhibition)": int((m & (tone < 0) & (net_fast < 0)).sum()),
         "tone_neg_net_excited (gain mode = silencing)": int((m & (tone < 0) & (net_fast > 0)).sum()),
         "tone_pos_net_inhibited (gain mode = more inhibition)": int((m & (tone > 0) & (net_fast < 0)).sum()),
         "tone_pos_net_excited (gain mode = more excitation)": int((m & (tone > 0) & (net_fast > 0)).sum()),
         "tone_zero_or_net_zero": int((m & ((tone == 0) | (net_fast == 0))).sum()),
         "syn_eq_tone_neg_net_inhibited": float(-tone[m & (tone < 0) & (net_fast < 0)].sum()),
         "syn_eq_tone_neg_net_excited": float(-tone[m & (tone < 0) & (net_fast > 0)].sum()),
         "syn_eq_tone_pos_net_inhibited": float(tone[m & (tone > 0) & (net_fast < 0)].sum()),
         "syn_eq_tone_pos_net_excited": float(tone[m & (tone > 0) & (net_fast > 0)].sum())}
    print(label, json.dumps(q))
    return q

out["quadrants_all"] = quadrants(np.ones(c.n, bool), "ALL")
out["quadrants_by_superclass"] = {s: quadrants(sc == s, s) for s in sorted(set(sc[has]))}
# net-inhibited cells overall (context): fraction of cells whose fan-in-scaled row sum is negative
out["cells_net_inhibited_all"] = int((net_fast < 0).sum()); out["cells_net_excited_all"] = int((net_fast > 0).sum())
print("all cells: net-inhibited", int((net_fast < 0).sum()), "net-excited", int((net_fast > 0).sum()))

# the populations behind the suite checks
groups = {"KC (all Kenyon cells)": np.array([t.startswith("KC") for t in ty]),
          "MN9": ty == "MN9", "GF (DNp01)": np.isin(ty, ["GF", "DNp01"]), "DNp18": ty == "DNp18",
          "DNp20": ty == "DNp20", "HSN": ty == "HSN", "HSE": ty == "HSE", "MBON (all)": np.array([t.startswith("MBON") for t in ty]),
          "PAM (all)": np.array([t.startswith("PAM") for t in ty]), "PPL1 (all)": np.array([t.startswith("PPL1") for t in ty]),
          "DPM": ty == "DPM", "APL": ty == "APL", "GNG (superclass gng?)": np.array([t.startswith("GNG") for t in ty]),
          "sugar GRN (Gr64f / sweet)": np.array([("Gr64" in t) or ("sweet" in t.lower()) for t in ty]),
          "OA cells (nt octopamine)": nt == "octopamine", "DA cells (nt dopamine)": nt == "dopamine", "5-HT cells": nt == "serotonin"}
pergroup = {}
for k, m in groups.items():
    if m.sum() == 0:
        continue
    mm = m & has
    pergroup[k] = {"cells": int(m.sum()), "with_tone": int(mm.sum()),
                   "tone_neg_net_inhibited": int((mm & (tone < 0) & (net_fast < 0)).sum()),
                   "tone_neg_net_excited": int((mm & (tone < 0) & (net_fast > 0)).sum()),
                   "tone_pos_net_inhibited": int((mm & (tone > 0) & (net_fast < 0)).sum()),
                   "tone_pos_net_excited": int((mm & (tone > 0) & (net_fast > 0)).sum()),
                   "mean_tone_syneq_scaled": float(tone[mm].mean()) if mm.sum() else 0.0,
                   "mean_net_fast_syneq_scaled": float(net_fast[m].mean()),
                   "mean_exc_fast": float(exc_fast[m].mean()), "mean_inh_fast": float(inh_fast[m].mean())}
    print(k, json.dumps(pergroup[k]))
out["groups"] = pergroup

# per-type table of the strongest tones in each quadrant (top 12 by |tone| x cells)
df = pd.DataFrame({"type": ty, "superclass": sc, "tone": tone, "net_fast": net_fast, "has": has})
df = df[df.has]
agg = df.groupby("type").agg(cells=("tone", "size"), tone_mean=("tone", "mean"), net_fast_mean=("net_fast", "mean"),
                             n_disinh=("tone", lambda x: int(((x < 0) & (df.loc[x.index, "net_fast"] < 0)).sum())))
agg["abs_load"] = agg.tone_mean.abs() * agg.cells
top_disinh = agg[(agg.tone_mean < 0) & (agg.net_fast_mean < 0)].sort_values("abs_load", ascending=False).head(15)
top_neg_exc = agg[(agg.tone_mean < 0) & (agg.net_fast_mean > 0)].sort_values("abs_load", ascending=False).head(10)
top_pos = agg[(agg.tone_mean > 0)].sort_values("abs_load", ascending=False).head(10)
print("TOP disinhibition-quadrant types:\n", top_disinh.round(2).to_string())
print("TOP negative-tone net-excited types:\n", top_neg_exc.round(2).to_string())
print("TOP positive-tone types:\n", top_pos.round(2).to_string())
out["top_disinhibition_types"] = top_disinh.round(3).reset_index().to_dict("records")
out["top_negative_tone_net_excited_types"] = top_neg_exc.round(3).reset_index().to_dict("records")
out["top_positive_tone_types"] = top_pos.round(3).reset_index().to_dict("records")
# steady-tone magnitude at the two gains (all monoamine inputs at 50 Hz, tau 200 ms): g_slow = w_syn x gain x load x 0.05/ms x 200 ms
for g in (0.01, 0.02):
    out[f"steady_tone_mV_at_50Hz_gain{g}"] = {"min": float(p.w_syn * g * tone.min() * 0.05 * 200), "max": float(p.w_syn * g * tone.max() * 0.05 * 200),
                                             "median_abs_over_cells_with_tone": float(np.median(np.abs(p.w_syn * g * tone[has] * 0.05 * 200)))}
    print("gain", g, out[f"steady_tone_mV_at_50Hz_gain{g}"])
out["inputs"] = {"table": tbl, "net_rule": "abs", "receptor_gain": p.receptor_gain, "w_syn": p.w_syn, "conn_cap": p.conn_cap,
                 "input_norm_alpha": p.input_norm_alpha, "input_norm_ref": p.input_norm_ref, "slow_entries": int(S.nnz),
                 "note": "net_fast / tone = row sums of the fan-in-scaled fast / monoamine-slow matrices in synapse-equivalents (path gains applied to fast only, as in Brain); structural proxy for the run-time net input"}
json.dump(out, open(os.path.join(ROOT, "out", "r3_gain_semantics.json"), "w"), indent=1)
print("wrote out/r3_gain_semantics.json")
