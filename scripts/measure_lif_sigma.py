"""Thread 6B (docs/audits/compass_local_recurrence.md): MEASURE the spiking LIF's effective input noise on the ring.

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python scripts/measure_lif_sigma.py --label S --out out/cx7/sigma
    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python scripts/measure_lif_sigma.py --label H3 --out out/cx7/sigma \
        --hold-edges '^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)'

Every slope bound of threads 5A / 6A is a property of `cx_ring_structure.SIGMA_MV = 2.0`, the Gaussian input-noise width
over which `cx_wedge.lif_fi` smooths the deterministic LIF f-I -- an ASSUMPTION the project had never measured. This
script runs the cx_wedge protocol itself (FlyBrain on the FULL connectome, no world, compass adaptation 0, 10 Hz Poisson
background on all 46 EPG, then wedges 0-3 at +40 Hz) on the CPU, and records at every LIF step (0.5 ms) the membrane
potential `v`, the synaptic input `g` (mV above rest; `Brain._membrane_target` = v_rest + g + drive - adapt) and the
refractory state of every EPG / PEN / PEG / Delta7 / EPGt / GLNO / ExR6 / ER6 / ER4m cell, over the 1 s settle and the
2 s pulse. From those it reports, per cell and per group:

  * sigma_g   -- the SD over time of the synaptic input g (the quantity the rate model's u is the mean of);
  * sigma_v   -- the SD over time of the FREE membrane potential (samples outside the refractory period), i.e. the
                 input noise after the membrane's tau_m = 20 ms low-pass, the number the diffusion picture uses;
  * sigma_v on the cells that never spiked in the window (the cleanest estimate: no reset transients at all);
  * mean g and mean v, and the 200 ms transient is dropped from every window.

Which of these the rate model "should assume" is stated in the audit: the smoothed f-I treats the noise as quasi-static
over an interspike interval, so its sigma is bounded below by sigma_v (fast noise, filtered) and above by sigma_g
(frozen noise), and the tool is re-run at both. No brain code is touched: `v`, `g` and `refrac` are the Brain's own
state tensors read after each step (docs/INTERP.md 2.3). Nothing here changes a default.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyverse import brain  # noqa: E402
import cx_wedge  # noqa: E402

GROUPS = ("EPG", "PEN", "PEG", "Delta7", "EPGt", "GLNO", "ExR6", "ER6", "ER4m")


def group_indices(c, cells) -> dict:
    ty = c.neurons.type.fillna("").to_numpy()
    out = {g: np.asarray(cells[g]["idx"]) for g in ("EPG", "PEN", "PEG", "Delta7", "EPGt")}
    for t in ("GLNO", "ExR6", "ER6", "ER4m"):
        out[t] = np.flatnonzero(ty == t)
    return out


def window_stats(v, g, refrac, spikes, transient_steps: int) -> dict:
    """Per-cell statistics over one window of (steps, cells) records; the first `transient_steps` are dropped."""
    v, g, refrac, spikes = (x[transient_steps:] for x in (v, g, refrac, spikes))
    free = refrac <= 0.0
    n_free = free.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_g = g.mean(axis=0); sd_g = g.std(axis=0)
        vf = np.where(free, v, np.nan)
        mean_v = np.nanmean(vf, axis=0); sd_v = np.nanstd(vf, axis=0)
    spiked = spikes.sum(axis=0) > 0
    rate = spikes.sum(axis=0) / (v.shape[0] * 0.5e-3)
    return dict(n_cells=int(v.shape[1]), n_steps=int(v.shape[0]),
                mean_g_mV=mean_g.tolist(), sigma_g_mV=sd_g.tolist(), mean_v_free_mV=mean_v.tolist(), sigma_v_free_mV=sd_v.tolist(),
                free_fraction=(n_free / v.shape[0]).tolist(), rate_hz=rate.tolist(), spiked=spiked.tolist())


def summarise(st: dict) -> dict:
    """Group summary: median / mean / min / max over cells of each per-cell statistic, plus the never-spiked subset."""
    out = {}
    spiked = np.asarray(st["spiked"], bool)
    for key in ("sigma_g_mV", "sigma_v_free_mV", "mean_g_mV", "mean_v_free_mV", "rate_hz"):
        x = np.asarray(st[key], float)
        ok = np.isfinite(x)
        out[key] = dict(median=float(np.median(x[ok])) if ok.any() else float("nan"), mean=float(x[ok].mean()) if ok.any() else float("nan"),
                        min=float(x[ok].min()) if ok.any() else float("nan"), max=float(x[ok].max()) if ok.any() else float("nan"), n=int(ok.sum()))
        q = ok & ~spiked
        out[key + "_never_spiked"] = dict(median=float(np.median(x[q])) if q.any() else float("nan"), mean=float(x[q].mean()) if q.any() else float("nan"),
                                          min=float(x[q].min()) if q.any() else float("nan"), max=float(x[q].max()) if q.any() else float("nan"), n=int(q.sum()))
    out["n_never_spiked"] = int((~spiked).sum()); out["n_cells"] = int(len(spiked))
    return out


def summarise_dir(out: Path) -> dict:
    """Pool every sigma_<label>_s<seed>.json in `out` into sigma_summary.json: the headline is the median free-membrane
    SD of the sub-threshold relay cells (PEN / PEG / EPGt that never spiked in the settle window) over the SHIPPED arm's
    seeds -- the cleanest estimate the protocol offers (no reset transient) -- with the EPG's own value, the input-side
    SD of g (the frozen-noise upper bound) and every per-arm / per-group median beside it."""
    files = sorted(out.glob("sigma_*_s*.json"))
    docs = [json.load(open(f, encoding="utf-8")) for f in files]
    per = []
    for d in docs:
        for w, groups in d["windows"].items():
            for g, v in groups.items():
                s = v["summary"]
                per.append(dict(label=d["label"], seed=d["seed"], window=w, group=g, n=s["n_cells"], never_spiked=s["n_never_spiked"],
                                sigma_v_median=s["sigma_v_free_mV"]["median"], sigma_v_min=s["sigma_v_free_mV"]["min"], sigma_v_max=s["sigma_v_free_mV"]["max"],
                                sigma_v_never_spiked_median=s["sigma_v_free_mV_never_spiked"]["median"],
                                sigma_g_median=s["sigma_g_mV"]["median"], sigma_g_min=s["sigma_g_mV"]["min"], sigma_g_max=s["sigma_g_mV"]["max"],
                                mean_g_median=s["mean_g_mV"]["median"], rate_mean=s["rate_hz"]["mean"]))
    pooled_v, pooled_g, pooled_epg_v, pooled_epg_g = [], [], [], []
    for d in docs:
        if d["label"] != "S":
            continue
        for g in ("PEN", "PEG", "EPGt"):
            pc = d["windows"]["settle"][g]["per_cell"]
            for sv, sp in zip(pc["sigma_v_free_mV"], pc["spiked"]):
                if not sp and np.isfinite(sv):
                    pooled_v.append(float(sv))
            pooled_g += [float(x) for x in pc["sigma_g_mV"]]
        pe = d["windows"]["settle"]["EPG"]["per_cell"]
        pooled_epg_v += [float(x) for x in pe["sigma_v_free_mV"] if np.isfinite(x)]
        pooled_epg_g += [float(x) for x in pe["sigma_g_mV"]]
    head = dict(sigma_v_relays_subthreshold_mV=float(np.median(pooled_v)) if pooled_v else float("nan"),
                sigma_v_relays_subthreshold_range=[float(np.min(pooled_v)), float(np.max(pooled_v))] if pooled_v else None, n_relay_cells=len(pooled_v),
                sigma_v_EPG_mV=float(np.median(pooled_epg_v)) if pooled_epg_v else float("nan"), n_EPG_cells=len(pooled_epg_v),
                sigma_g_relays_mV=float(np.median(pooled_g)) if pooled_g else float("nan"), sigma_g_EPG_mV=float(np.median(pooled_epg_g)) if pooled_epg_g else float("nan"),
                seeds=sorted({d["seed"] for d in docs if d["label"] == "S"}), window="settle (1 s on the 10 Hz background, first 0.2 s dropped), arm S")
    sig_m = round(head["sigma_v_relays_subthreshold_mV"], 1)
    sig_g = round(head["sigma_g_relays_mV"], 1)
    doc = dict(schema="flyverse.lif_sigma_summary/1", generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), generator="python " + " ".join(sys.argv),
               files=[f.name for f in files], assumed_sigma_mV=2.0, headline=head, measured_sigma_mV=sig_m, measured_sigma_g_mV=sig_g,
               consequence=(f"The rate model assumed sigma = 2.0 mV. The free membrane of the sub-threshold relays fluctuates with SD {sig_m} mV at rest "
                            f"(median over {len(pooled_v)} never-spiked PEN / PEG / EPGt cells, S seeds {head['seeds']}); the synaptic input itself with SD {sig_g} mV. "
                            "The smoothed f-I's sigma is the width of a quasi-static input distribution, so it lies between the membrane-filtered value (fast noise) "
                            f"and the input value (frozen noise): the tool is re-run at sigma = {sig_m} (the membrane reading, the lower bound) and validated at both "
                            f"{sig_m} and {sig_g} against the cx5 batch. A spiking cell's free-membrane SD is clipped by reset-to-threshold (every firing ring cell "
                            "reads 2.1-2.4 mV whatever its rate), so only sub-threshold cells give an unbiased membrane reading."),
               per_group=per)
    (out / "sigma_summary.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    # the per-seed table the audit pastes from (docs/INTERP.md 10.4 rule 28): one row per (arm, seed, window, group)
    cols = ["label", "seed", "window", "group", "n", "never_spiked", "sigma_v_median", "sigma_v_min", "sigma_v_max",
            "sigma_v_never_spiked_median", "sigma_g_median", "sigma_g_min", "sigma_g_max", "mean_g_median", "rate_mean"]
    lines = [",".join(cols)]
    for r in per:
        lines.append(",".join(("" if r[c] is None else (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))) for c in cols))
    (out / "sigma_table.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(head, indent=1)); print("measured sigma (membrane, relays):", sig_m, "mV; input-side:", sig_g, "mV")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory (never a default: the 6A --out incident)")
    ap.add_argument("--summarise", action="store_true", help="pool the sigma_*.json files in --out into sigma_summary.json (no simulation)")
    ap.add_argument("--label", default=None, help="arm label recorded in the JSON and used as the file stem (required unless --summarise)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--pulse-s", type=float, default=2.0)
    ap.add_argument("--transient-s", type=float, default=0.2, help="dropped from the start of every window")
    ap.add_argument("--background", type=float, default=10.0)
    ap.add_argument("--pulse-hz", type=float, default=40.0)
    ap.add_argument("--start-wedge", type=int, default=0)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--hold-edges", action="append", default=None, metavar="PRE_REGEX:POST_REGEX")
    ap.add_argument("--edge-gain", action="append", default=None, metavar="PRE_REGEX:POST_REGEX:FACTOR")
    ap.add_argument("--lif", action="append", default=None, metavar="KEY=VALUE")
    ap.add_argument("--nt-override", action="append", default=None, metavar="TYPE=nt")
    ap.add_argument("--receptor-model", default="shipped", choices=["off", "sign", "sign+gain", "full", "shipped"])
    ap.add_argument("--save-traces", action="store_true", help="also save the per-step v / g records (npz; ~30 MB)")
    a = ap.parse_args()
    if a.summarise:
        summarise_dir(Path(a.out)); return
    if not a.label:
        raise SystemExit("--label is required")
    from flyverse.fly import FlyBrain
    from flyverse.interp import common
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nt_override = cx_wedge.parse_nt_override(a.nt_override)
    holds = cx_wedge.parse_hold_edges(a.hold_edges)
    gains = cx_wedge.parse_edge_gains(a.edge_gain)
    c, cache_dir, table = cx_wedge.load_connectome(nt_override)
    cells = cx_wedge.compass_cells(c)
    receptor_model, rule = (None, "class") if a.receptor_model == "off" else (a.receptor_model, "class")
    if a.receptor_model == "shipped":
        receptor_model, rule = brain.LIFParams().receptor_model, brain.LIFParams().receptor_net_rule
    lif_overrides = common.parse_kv(a.lif) if a.lif else {}
    # exactly cx_wedge.simulate's parameter construction at gE = gD = gR = 1, Delta7 -> PEN x gD (the shipped path)
    tpg = list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", 1.0), (r"^PEN_", r"^EPG$", 1.0), (r"^EPG$", r"^PEG$", 1.0),
                                                (r"^PEG$", r"^EPG$", 1.0), (r"^Delta7$", r"^(EPG$|PEN_)", 1.0), (cx_wedge.RING_RE, r"^(EPG$|PEN_|PEG$)", 1.0)]
    tpg += [(p, q, float(f)) for p, q, f in holds] + [(p, q, float(f)) for p, q, f in gains]
    params = brain.LIFParams(adapt_by_type={cx_wedge.COMPASS_RE: 0.0}, type_path_gain=tpg, receptor_model=receptor_model,
                             receptor_net_rule=rule, **lif_overrides)
    fb = FlyBrain(c, lif_params=params, seed=a.seed, cuda_graphs=False, device="cpu")
    b = fb.brain
    dt = b.p.dt
    epg = cells["EPG"]; idx_epg = epg["idx"]
    wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    inside = np.isin(wedge_of, [(a.start_wedge + j) % 16 for j in range(a.width)])
    gi = group_indices(c, cells)
    order = np.concatenate([gi[g] for g in GROUPS])
    sl = {}; s0 = 0
    for g in GROUPS:
        sl[g] = slice(s0, s0 + len(gi[g])); s0 += len(gi[g])
    total_ms = (a.settle_s + a.pulse_s) * 1000
    fb.stimulate(idx_epg, a.background, total_ms + 100)
    idx_t = b._idx(order)

    def record(ms: float):
        n = int(round(ms / dt))
        V = np.zeros((n, len(order)), np.float32); G = np.zeros_like(V); R = np.zeros_like(V); S = np.zeros_like(V)
        for k in range(n):
            fb.step(dt)
            V[k] = b.v[0, idx_t].cpu().numpy(); G[k] = b.g[0, idx_t].cpu().numpy()
            R[k] = b.refrac[0, idx_t].cpu().numpy(); S[k] = b.spikes[0, idx_t].cpu().numpy()
        return V, G, R, S

    rec = {}
    rec["settle"] = record(a.settle_s * 1000)
    fb.stimulate(idx_epg[inside], a.background + a.pulse_hz, a.pulse_s * 1000)
    rec["pulse"] = record(a.pulse_s * 1000)
    trans = int(round(a.transient_s * 1000 / dt))
    res = dict(schema="flyverse.lif_sigma/1", generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               generator="python " + " ".join(sys.argv), label=a.label, seed=a.seed, device=str(b.device), dt_ms=dt,
               settle_s=a.settle_s, pulse_s=a.pulse_s, transient_s=a.transient_s, background_hz=a.background, pulse_hz=a.pulse_hz,
               hold_edges=[[p, q, f] for p, q, f in holds], hold_edges_resolved=cx_wedge.hold_edge_counts(c, holds) if holds else [],
               edge_gains=[[p, q, f] for p, q, f in gains], edge_gains_resolved=cx_wedge.hold_edge_counts(c, gains, same_type_gain=params.same_type_gain) if gains else [],
               lif_overrides=lif_overrides, nt_override=nt_override, receptor_model=receptor_model, receptor_net_rule=rule,
               assumed_sigma_mV=2.0, note=("sigma_g = SD over time of the synaptic input g (mV); sigma_v_free = SD of the membrane potential "
                                           "over non-refractory samples; the smoothed f-I's sigma lies between them (see the module docstring)"),
               groups={g: dict(n=int(len(gi[g]))) for g in GROUPS}, windows={})
    for w, (V, G, R, S) in rec.items():
        res["windows"][w] = {}
        for g in GROUPS:
            if len(gi[g]) == 0:
                continue
            st = window_stats(V[:, sl[g]], G[:, sl[g]], R[:, sl[g]], S[:, sl[g]], trans)
            res["windows"][w][g] = dict(summary=summarise(st), per_cell=st)
        # the driven / undriven EPG split during the pulse
        if w == "pulse":
            for name, m in (("EPG_in", inside), ("EPG_out", ~inside)):
                e = sl["EPG"]; cols = np.flatnonzero(m) + e.start
                st = window_stats(V[:, cols], G[:, cols], R[:, cols], S[:, cols], trans)
                res["windows"][w][name] = dict(summary=summarise(st), per_cell=st)
    res["provenance"] = common.to_jsonable(common.provenance(c, b.p, None, fb=fb, device="cpu", seeds=[a.seed],
                                                             stimulus={"protocol": "measure_lif_sigma (cx_wedge protocol, per-step v/g record)",
                                                                       "params": {k: res[k] for k in ("settle_s", "pulse_s", "background_hz", "pulse_hz", "hold_edges", "edge_gains", "lif_overrides", "nt_override")},
                                                                       "control": "arm S (the shipped path)"}))
    res["wall_s"] = round(time.time() - t0, 1)
    path = out / f"sigma_{a.label}_s{a.seed}.json"
    path.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    if a.save_traces:
        np.savez_compressed(out / f"sigma_{a.label}_s{a.seed}_traces.npz", order=order, **{f"{w}_{k}": arr for w, arrs in rec.items() for k, arr in zip(("v", "g", "refrac", "spikes"), arrs)},
                            **{f"slice_{g}": np.array([sl[g].start, sl[g].stop]) for g in GROUPS})
    print(f"[{a.label} seed {a.seed}] device {b.device}, {res['wall_s']} s; hold {res['hold_edges']} gains {res['edge_gains']}")
    for w in ("settle", "pulse"):
        print(f"  {w}:")
        for g, d in res["windows"][w].items():
            s = d["summary"]
            print(f"    {g:>7} (n {s['n_cells']:3d}, never spiked {s['n_never_spiked']:3d}): sigma_g median {s['sigma_g_mV']['median']:.3f} "
                  f"[{s['sigma_g_mV']['min']:.3f}-{s['sigma_g_mV']['max']:.3f}] mV; sigma_v(free) median {s['sigma_v_free_mV']['median']:.3f} "
                  f"[{s['sigma_v_free_mV']['min']:.3f}-{s['sigma_v_free_mV']['max']:.3f}]"
                  + (f"; never-spiked sigma_v {s['sigma_v_free_mV_never_spiked']['median']:.3f}" if s['n_never_spiked'] else "")
                  + f"; mean g {s['mean_g_mV']['median']:+.2f} mV; rate {s['rate_hz']['mean']:.2f} Hz")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
