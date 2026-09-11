"""Skeptic's cross-check for docs/audits/cx_wedge.md (scripts/cx_wedge.py).

Same protocol as `cx_wedge.py --sim` (FlyBrain, full connectome, compass adaptation 0, Poisson background on every
EPG, `--width` wedges driven at +pulse for `--pulse-s`, then `--seconds` free) but with the three Delta7 gains
separately settable (--gD-epg / --gD-pen / --gD-peg; 0 = cut) and, during the last second of the pulse, an exact
spike-count rate per neuron and a decomposition of the mean synaptic input (tau_syn * W @ r, mV) of every PEN and
EPG by presynaptic type, using the matrix the Brain actually installed (brain._W_cpu). Output: one JSON row.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyverse import brain, connectome  # noqa: E402
from cx_wedge import COMPASS_RE, RING16, RING_RE, compass_cells, glomerulus  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gE", type=float, default=1.0)
    ap.add_argument("--gD-epg", type=float, default=1.0)
    ap.add_argument("--gD-pen", type=float, default=1.0)
    ap.add_argument("--gD-peg", type=float, default=1.0)
    ap.add_argument("--ring-gain", type=float, default=1.0)
    ap.add_argument("--background", type=float, default=10.0)
    ap.add_argument("--pulse-hz", type=float, default=40.0)
    ap.add_argument("--pulse-s", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--start-wedge", type=int, default=0)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from flyverse.fly import FlyBrain
    c = connectome.load(verbose=False)
    cells = compass_cells(c)
    ty = c.neurons.type.fillna("").to_numpy()
    inst = c.neurons.instance.fillna("").to_numpy()
    epg = cells["EPG"]; idx_epg = epg["idx"]
    wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    driven = [(a.start_wedge + j) % 16 for j in range(a.width)]
    inside = np.isin(wedge_of, driven)
    pen_idx, d7_idx, peg_idx, ring_idx = cells["PEN"]["idx"], cells["Delta7"]["idx"], cells["PEG"]["idx"], cells["Ring"]["idx"]
    compass_all = np.concatenate([idx_epg, pen_idx, d7_idx, peg_idx, cells["EPGt"]["idx"]])
    others = np.setdiff1d(np.arange(c.n), compass_all)

    tpg = list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", a.gE), (r"^PEN_", r"^EPG$", a.gE),
                                                (r"^EPG$", r"^PEG$", a.gE), (r"^PEG$", r"^EPG$", a.gE),
                                                (r"^Delta7$", r"^EPG$", a.gD_epg), (r"^Delta7$", r"^PEN_", a.gD_pen),
                                                (r"^Delta7$", r"^PEG$", a.gD_peg),
                                                (RING_RE, r"^(EPG$|PEN_|PEG$)", a.ring_gain)]
    params = brain.LIFParams(adapt_by_type={COMPASS_RE: 0.0}, type_path_gain=tpg)
    t0 = time.time()
    fb = FlyBrain(c, lif_params=params, seed=a.seed, cuda_graphs=not a.no_graphs)
    W = fb.brain._W_cpu.tocsr()          # [post, pre] mV per presynaptic spike, as installed
    tau = fb.brain.p.tau_syn / 1000.0
    total_ms = (1.0 + a.pulse_s + a.seconds) * 1000
    fb.stimulate(idx_epg, a.background, total_ms + 100)

    def sample(tag):
        r = fb.brain.rates(idx_epg).copy()
        d = {f"{tag}_in_mean": float(r[inside].mean()), f"{tag}_out_mean": float(r[~inside].mean()),
             f"{tag}_in_above": int((r[inside] > 22).sum()), f"{tag}_out_above": int((r[~inside] > 22).sum()),
             f"{tag}_pen": float(fb.brain.mean_rate(pen_idx)), f"{tag}_delta7": float(fb.brain.mean_rate(d7_idx)),
             f"{tag}_peg": float(fb.brain.mean_rate(peg_idx)), f"{tag}_rest": float(fb.brain.mean_rate(others)),
             f"{tag}_ring": float(fb.brain.mean_rate(ring_idx)),
             f"{tag}_wedge_profile": [float(r[wedge_of == w].mean()) for w in range(16)]}
        ang = 2 * np.pi * wedge_of / 16
        z = np.sum(r * np.exp(1j * ang)) / max(r.sum(), 1e-9)
        d[f"{tag}_vector_strength"] = float(np.abs(z))
        return d

    row = dict(vars(a), n_in=int(inside.sum()), n_out=int((~inside).sum()))
    fb.step(1000.0)
    row.update(sample("pre"))
    fb.stimulate(idx_epg[inside], a.background + a.pulse_hz, a.pulse_s * 1000)
    fb.step(max(a.pulse_s - 1.0, 0.0) * 1000)
    c0 = fb.brain.spike_counts[0].detach().cpu().numpy().copy()
    fb.step(min(a.pulse_s, 1.0) * 1000)
    c1 = fb.brain.spike_counts[0].detach().cpu().numpy().copy()
    r_exact = (c1 - c0) / min(a.pulse_s, 1.0)
    row.update(sample("during"))

    # exact rates by type during the last second of the pulse
    def mean_of(mask):
        return float(r_exact[mask].mean()) if mask.any() else 0.0
    row["during_exact"] = {"EPG_in": mean_of(np.isin(np.arange(c.n), idx_epg[inside])),
                           "EPG_out": mean_of(np.isin(np.arange(c.n), idx_epg[~inside])),
                           "PEN": mean_of(np.isin(np.arange(c.n), pen_idx)), "PEG": mean_of(np.isin(np.arange(c.n), peg_idx)),
                           "Delta7": mean_of(np.isin(np.arange(c.n), d7_idx)),
                           "rest": mean_of(np.isin(np.arange(c.n), others))}
    for t in ("ExR6", "ExR4", "ER6", "ER4m", "ExR5", "ER2_c", "ER4d", "ExR1", "ExR7"):
        row["during_exact"][t] = mean_of(ty == t)
    # input decomposition: tau_syn * W[post, pre-group] @ r_exact[pre-group]
    groups = {"EPG": ty == "EPG", "EPGt": ty == "EPGt", "PEN": np.array([t.startswith("PEN_") for t in ty]),
              "PEG": ty == "PEG", "Delta7": ty == "Delta7", "ExR6": ty == "ExR6", "ExR4": ty == "ExR4", "ER6": ty == "ER6",
              "ER4m": ty == "ER4m"}
    ring_mask = np.array([bool(re.match(RING_RE, t)) for t in ty])
    groups["other_ER_ExR"] = ring_mask & ~(groups["ExR6"] | groups["ExR4"] | groups["ER6"] | groups["ER4m"])
    groups["rest_of_brain"] = ~(ring_mask | groups["EPG"] | groups["EPGt"] | groups["PEN"] | groups["PEG"] | groups["Delta7"])
    u = {}
    for g, m in groups.items():
        cols = np.flatnonzero(m)
        u[g] = tau * (W[:, cols] @ r_exact[cols])
    u_total = sum(u.values())
    pen_glom = [glomerulus(inst[i]) for i in pen_idx]
    pen_pos = np.asarray(np.round(cells["PEN"]["pos"]), int) % 16
    near = np.isin(pen_pos, driven)
    pen_rows = []
    for j, i in enumerate(pen_idx):
        pen_rows.append(dict(glom=pen_glom[j], wedge=int(pen_pos[j]), near=bool(near[j]), rate=float(r_exact[i]),
                             **{g: round(float(u[g][i]), 2) for g in groups}, total=round(float(u_total[i]), 2)))
    row["pen_inputs_mV"] = pen_rows
    def summ(mask_idx):
        return {g: round(float(u[g][mask_idx].mean()), 2) for g in groups} | {"total": round(float(u_total[mask_idx].mean()), 2)}
    row["pen_near_mean_mV"] = summ(pen_idx[near])
    row["pen_all_mean_mV"] = summ(pen_idx)
    jmax = int(np.argmax(u["EPG"][pen_idx]))
    row["pen_max_epg_input"] = pen_rows[jmax]
    row["epg_in_mean_mV"] = summ(idx_epg[inside])
    row["epg_out_mean_mV"] = summ(idx_epg[~inside])
    row["ring_feedback_on_pen_near_mV"] = round(float(sum(u[g][pen_idx[near]].mean() for g in ("ExR6", "ExR4", "ER6", "ER4m", "other_ER_ExR"))), 2)

    t = 0.0
    for mark in (0.5, 1.0, 2.0, 3.0, 5.0):
        if mark > a.seconds:
            break
        fb.step((mark - t) * 1000); t = mark
        row.update(sample(f"t{mark}"))
    row["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in row.items() if k != "pen_inputs_mV"}, indent=1))
    print("PEN inputs (mV, last second of the pulse):")
    for r in pen_rows:
        print("  ", r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(row, f, indent=1)


if __name__ == "__main__":
    main()
