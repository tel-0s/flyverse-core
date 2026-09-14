"""Per-transmitter unitary strength (thread:unitary; docs/audits/unitary_strength.md).

LIFParams.w_syn = 0.275 mV per synapse is one scalar for every transmitter (Shiu et al. 2024). This probe puts the
measured per-transmitter brackets (BRACKETS below; the ledger rows `unitary.*` in flyverse/data/expected_responses.csv)
through the opt-in field LIFParams.w_syn_by_nt and asks whether a data-anchored per-transmitter scale makes the ring a
working attractor at the SHIPPED compass gains, and what it costs elsewhere. Nothing here changes a default.

    python scripts/probe_unitary.py brackets                                   # CPU: the brackets and the EM anchors
    python scripts/probe_unitary.py structure --arm mid --json out/unitary/structure_mid.json      # CPU: ring E / I
    python scripts/probe_unitary.py compass --arm mid --seed 0 --out out/unitary/fam_compass_mid_s0.json   # GPU
    python scripts/probe_unitary.py room --arm mid --seed 0 --out out/unitary/fam_room_mid_s0.json         # GPU
    python scripts/probe_unitary.py plan --out-dir out/unitary                 # writes the cluster batch scripts
    python scripts/probe_unitary.py report --dir out/unitary                   # CPU: tables from the landed JSONs

`compass` is scripts/cx_wedge.py's simulate protocol (10 Hz Poisson background on the 46 EPG, one 4-wedge block driven
+40 Hz for 2 s, then free) at gE = gD = gR = 1 (no compass gains) under the shipped receptor rule (sign / abs), with the
per-frame EPG record scored by scripts/probe_compass_room.py's confinement rule (bump rate / width / survival; the
ledger rows compass.EPG.*). `room` is scripts/probe_walk_straightness.py's plain-fly rollout (16 flies, no program, no
fence, 60 s) under the arm. The suite (rest / taste / smell / walk incl. loom / motion) runs through
scripts/interp_lesion.py with a `lif` lesion per arm (the manifest is embedded in the job line; out/ is not shipped).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flyverse import brain, connectome  # noqa: E402
from flyverse.interp import common  # noqa: E402

# ------------------------------------------------------------------------------------------------ the brackets
# Multipliers on LIFParams.w_syn (0.275 mV) per PRESYNAPTIC transmitter, from the measured unitary strengths
# (docs/audits/unitary_strength.md section 1; ledger rows unitary.*). Two data-anchored quantities: the cholinergic
# per-synapse EPSP relative to 0.275 (ORN -> PN 5 mV / 23 synapses = 0.22 mV = x0.79, Kazama & Wilson 2008 + Tobin et
# al. 2017; PN -> LHN ~1 mV per spike over a median 3 / mean 8 synapses per pair in MaleCNS = x0.45-1.2, Jeanne & Wilson
# 2015; PN -> KC 0.59-1.76 mV per claw over a mean 17.8 synapses per pair in MaleCNS = x0.12-0.36, Gruntman & Turner
# 2013) and the fast-inhibitory (GABA-A / GluCl chloride) per-synapse IPSP relative to the cholinergic EPSP (the
# chloride driving force at v_rest -52 mV against E_Cl of about -60 to -70 mV is 8-18 mV against ~50 mV for a cation
# channel, so per unit conductance an IPSP is 0.15-0.35 of an EPSP; no Drosophila central unitary IPSP per synapse is on
# record, so the bracket runs to 1.0 = the shipped equality). Histamine stays x1 in every arm: the photoreceptor ->
# lamina synapse is graded, not convertible to mV per spike, and lives in the optic-lobe rate model (not w_syn).
BRACKETS = {
    "default": {},
    "low":  {"acetylcholine": 0.5, "gaba": 0.125, "glutamate": 0.125},      # ACh x0.5,  I/E 0.25
    "mid":  {"acetylcholine": 0.8, "gaba": 0.4,   "glutamate": 0.4},        # ACh x0.8,  I/E 0.5
    "high": {"acetylcholine": 1.0, "gaba": 0.75,  "glutamate": 0.75},       # ACh x1.0,  I/E 0.75
}
ARMS = list(BRACKETS)
SUITE_SECTIONS = "rest,taste,smell,walk,motion"
DEFAULT_OUT = "out/unitary"


def lif_for(arm: str, **kw) -> brain.LIFParams:
    by_nt = BRACKETS[arm]
    return brain.LIFParams(w_syn_by_nt=dict(by_nt) if by_nt else None, **kw)


def patch_lif(arm: str, extra: dict | None = None):
    """Every brain.LIFParams built from here on (BatchSim's, the demo's) carries the arm (probe_walk_straightness'
    pattern). Returns the original class for restoring."""
    L = brain.LIFParams
    by_nt = BRACKETS[arm]

    def make(**kw):
        p = L(**kw)
        if by_nt:
            p.w_syn_by_nt = dict(by_nt)
        for k, v in (extra or {}).items():
            setattr(p, k, v)
        return p
    brain.LIFParams = make
    return L


# ------------------------------------------------------------------------------------------------ brackets / anchors
def em_anchors(c) -> dict:
    """Synapses per unitary pair in the compiled MaleCNS graph for the three cholinergic anchors (the EM counts the
    literature EPSPs are divided by), plus entries / synapses per presynaptic transmitter."""
    import re
    n = c.neurons; t = n.type.fillna("").to_numpy()
    coo = c.W.tocoo(); d = np.abs(coo.data)

    def pairs(pre_re, post_re):
        pm = np.array([bool(re.match(pre_re, x)) for x in t]); qm = np.array([bool(re.match(post_re, x)) for x in t])
        m = pm[coo.col] & qm[coo.row]; v = d[m]
        return {"pairs": int(m.sum()), "mean": float(v.mean()), "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    out = {"ORN->uPN": pairs(r"^ORN_", r"^(D|V|DA|DC|DL|DM|DP|VA|VC|VL|VM)\w*(_adPN|_lPN|_ilPN|_vPN|_l2PN)"),
           "uPN->KC": pairs(r".*_(adPN|lPN)$", r"^KC"), "uPN->LH": pairs(r".*_(adPN|lPN)$", r"^LH")}
    nt = n.nt.fillna("unknown").to_numpy()[coo.col]
    by = {}
    for k in sorted(set(nt)):
        m = nt == k
        by[k] = {"entries": int(m.sum()), "synapses": float(d[m].sum()), "explicit_zeros": int((coo.data[m] == 0).sum())}
    out["entries_by_pre_nt"] = by
    return out


def cmd_brackets(a):
    c = connectome.load(verbose=False)
    anchors = em_anchors(c)
    rows = []
    for arm, by in BRACKETS.items():
        rows.append({"arm": arm, **{k: by.get(k, 1.0) for k in ("acetylcholine", "gaba", "glutamate", "histamine")},
                     "I_over_E": (by.get("gaba", 1.0) / by.get("acetylcholine", 1.0)) if by else 1.0})
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nEM synapses per unitary pair (compiled MaleCNS graph):")
    for k in ("ORN->uPN", "uPN->KC", "uPN->LH"):
        print(f"  {k:9s} {anchors[k]}")
    print("\nper-synapse mV implied by the literature EPSPs over these counts (relative to 0.275):")
    print(f"  ORN->uPN: 5 mV / 23 (Tobin 2017) = {5/23:.3f} mV = x{5/23/0.275:.2f}; / {anchors['ORN->uPN']['mean']:.1f} (MaleCNS mean) = x{5/anchors['ORN->uPN']['mean']/0.275:.2f}")
    print(f"  uPN->KC : 0.59-1.76 mV / {anchors['uPN->KC']['mean']:.1f} = x{0.59/anchors['uPN->KC']['mean']/0.275:.2f}-x{1.76/anchors['uPN->KC']['mean']/0.275:.2f}")
    print(f"  uPN->LH : ~1 mV / median {anchors['uPN->LH']['median']:.0f} = x{1/anchors['uPN->LH']['median']/0.275:.2f}; / mean {anchors['uPN->LH']['mean']:.1f} = x{1/anchors['uPN->LH']['mean']/0.275:.2f}")
    print("\nentries / synapses by presynaptic transmitter:")
    for k, v in anchors["entries_by_pre_nt"].items():
        print(f"  {k:14s} {v}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"brackets": BRACKETS, "anchors": anchors, "w_syn": brain.LIFParams().w_syn,
                   "connectome": common.connectome_fingerprint(c)}, open(a.json, "w"), indent=1, default=common.to_jsonable)
        print("wrote", a.json)


# ------------------------------------------------------------------------------------------------ structure (CPU)
RING_GROUPS = {"EPG": r"^EPG$", "PEN": r"^PEN_", "PEG": r"^PEG$", "Delta7": r"^Delta7$", "Ring": r"^(ER|ExR)"}


def ring_balance(c, ew) -> dict:
    """The one-step effective weights between the compass groups (mV per post cell per presynaptic-group volley,
    the paths tool's type_matrix quantity) and the net E / I per volley onto EPG and PEN from the whole graph."""
    import re
    t = c.neurons.type.fillna("").to_numpy()
    idx = {k: np.flatnonzero([bool(re.match(p, x)) for x in t]) for k, p in RING_GROUPS.items()}
    A = ew.A
    M = {}
    for post, pi in idx.items():
        for pre, qi in idx.items():
            M[f"{pre}->{post}"] = float(A[pi][:, qi].sum(axis=1).mean())
    tot = {}
    for post, pi in idx.items():
        rows = A[pi]
        pos = rows.multiply(rows > 0).sum(axis=1); neg = rows.multiply(rows < 0).sum(axis=1)
        tot[post] = {"E_per_volley_mv": float(np.mean(pos)), "I_per_volley_mv": float(np.mean(neg)),
                     "scale_mean": float(ew.scale[pi].mean()), "n": int(len(pi))}
    return {"links": M, "totals": tot}


def cmd_structure(a):
    from flyverse.interp.common import effective_weights, Result, provenance
    c = connectome.load(verbose=False)
    out = {}
    base = effective_weights(c, brain.LIFParams())
    for arm in (a.arms.split(",") if a.arms else ARMS):
        p = lif_for(arm)
        ew = effective_weights(c, p)
        rb = ring_balance(c, ew)
        changed = ew.scale != base.scale
        rb["fan_in"] = {"cells_scale_changed": int(changed.sum()),
                        "scale_ratio_mean_changed": float((ew.scale[changed] / base.scale[changed]).mean()) if changed.any() else 1.0,
                        "sum_abs_A": float(np.abs(ew.A.data).sum()), "sum_abs_A_over_default": float(np.abs(ew.A.data).sum() / np.abs(base.A.data).sum())}
        rb["effective_weights_md5"] = ew.md5
        rb["w_syn_by_nt"] = BRACKETS[arm]
        out[arm] = rb
        L = rb["links"]; T = rb["totals"]
        print(f"{arm:8s} EPG->PEN {L['EPG->PEN']:+7.1f} PEN->EPG {L['PEN->EPG']:+7.1f} D7->EPG {L['Delta7->EPG']:+7.1f} D7->PEN {L['Delta7->PEN']:+7.1f} "
              f"Ring->EPG {L['Ring->EPG']:+7.1f} Ring->PEN {L['Ring->PEN']:+7.1f} EPG->Ring {L['EPG->Ring']:+6.1f} | EPG E {T['EPG']['E_per_volley_mv']:+7.0f} I {T['EPG']['I_per_volley_mv']:+7.0f} "
              f"PEN E {T['PEN']['E_per_volley_mv']:+7.0f} I {T['PEN']['I_per_volley_mv']:+7.0f} | cells rescaled {rb['fan_in']['cells_scale_changed']} sum|A| x{rb['fan_in']['sum_abs_A_over_default']:.3f} md5 {ew.md5[:8]}")
    if a.json:
        res = Result.new("paths", provenance(c, brain.LIFParams(), device="cpu", stimulus={"protocol": "structure", "params": {"arms": list(out)}, "control": "default"}))
        res.summary = {"arms": out, "brackets": BRACKETS}
        res.files = {"generator": "python " + " ".join(shlex.quote(x) for x in sys.argv)}
        res.save(a.json)
        print("wrote", a.json, "problems:", res.check() or "none")


# ------------------------------------------------------------------------------------------------ compass (GPU)
def cmd_compass(a):
    import torch
    import cx_wedge
    import probe_compass_room as pcr
    from flyverse.fly import FlyBrain
    c = connectome.load(verbose=False)
    cells = cx_wedge.compass_cells(c)
    epg = cells["EPG"]; idx_epg = epg["idx"]; wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    inside = np.isin(wedge_of, [(a.start_wedge + j) % 16 for j in range(a.width)])
    groups = {k: cells[k]["idx"] for k in ("PEN", "Delta7", "PEG", "Ring")}
    t = c.neurons.type.fillna("").to_numpy()
    groups["GLNO"] = np.flatnonzero(t == "GLNO")
    others = np.setdiff1d(np.arange(c.n), np.concatenate([idx_epg, groups["PEN"], groups["Delta7"], groups["PEG"], cells["EPGt"]["idx"]]))
    extra = {}
    if a.adapt == "off":
        extra["adapt_by_type"] = {cx_wedge.COMPASS_RE: 0.0}          # cx_wedge / compass_room's experiment convention
    p = lif_for(a.arm, **extra)
    t0 = time.time()
    fb = FlyBrain(c, lif_params=p, seed=a.seed, cuda_graphs=not a.no_graphs, device=a.device)
    dev = str(fb.brain.device)
    print(f"arm {a.arm} seed {a.seed}: w_syn_by_nt {p.w_syn_by_nt}, adapt {a.adapt}, receptor {p.receptor_model}/{p.receptor_net_rule}, device {dev}, build {time.time() - t0:.0f} s", flush=True)
    settle_s, pulse_at = 1.0, 1.0
    total_ms = (settle_s + a.pulse_s + a.seconds) * 1000
    if a.background > 0:
        fb.stimulate(idx_epg, a.background, total_ms + 100)
    frames = int(round(total_ms / 10.0))
    rec_epg = np.zeros((frames, len(idx_epg)), np.float32)
    rec_grp = {k: np.zeros(frames, np.float32) for k in list(groups) + ["rest"]}
    tt = np.arange(frames) * 0.01
    pulsed = False
    for k in range(frames):
        if not pulsed and tt[k] >= pulse_at - 1e-9:
            fb.stimulate(idx_epg[inside], a.background + a.pulse_hz, a.pulse_s * 1000)
            pulsed = True
        fb.step(10.0)
        rec_epg[k] = fb.brain.rates(idx_epg)
        for g, gi in groups.items():
            rec_grp[g][k] = fb.brain.mean_rate(gi)
        rec_grp["rest"][k] = fb.brain.mean_rate(others)
        if k % 100 == 99:
            print(f"  t {tt[k] + 0.01:4.1f} s EPG in {rec_epg[k][inside].mean():6.1f} out {rec_epg[k][~inside].mean():6.1f} PEN {rec_grp['PEN'][k]:5.1f} D7 {rec_grp['Delta7'][k]:5.1f} Ring {rec_grp['Ring'][k]:5.2f} rest {rec_grp['rest'][k]:.3f}", flush=True)
    wall = time.time() - t0
    b = pcr.bump_frames(rec_epg, wedge_of)
    pre = tt < pulse_at; during = (tt >= pulse_at) & (tt < pulse_at + a.pulse_s); post = tt >= pulse_at + a.pulse_s
    conf = b["confined"]

    def mean_if(x, m):
        mm = m & np.isfinite(x)
        return float(np.mean(x[mm])) if mm.any() else float("nan")
    last = np.flatnonzero(conf & post)
    survival = float(tt[last[-1]] - (pulse_at + a.pulse_s)) if len(last) else 0.0
    # the ledger's rows (compass.EPG.*): rate and width over the confined post frames, survival after the pulse
    m = {"frac_confined_pre": float(conf[pre].mean()), "frac_confined_during": float(conf[during].mean()),
         "frac_confined_post": float(conf[post].mean()), "survival_s": survival,
         "bump_hz_post": mean_if(b["bump_hz"], post & conf), "out_hz_post": mean_if(b["out_hz"], post & conf),
         "width_half_post": mean_if(b["width_half"], post & conf), "vs_post_all": mean_if(b["vs"], post),
         "epg_in_mean_post": float(rec_epg[post][:, inside].mean()), "epg_out_mean_post": float(rec_epg[post][:, ~inside].mean()),
         "epg_in_mean_during": float(rec_epg[during][:, inside].mean()), "epg_out_mean_during": float(rec_epg[during][:, ~inside].mean()),
         "epg_max_post": float(rec_epg[post].max()), "epg_mean_pre": float(rec_epg[pre].mean()),
         **{f"{g}_mean_post": float(v[post].mean()) for g, v in rec_grp.items()},
         **{f"{g}_mean_during": float(v[during].mean()) for g, v in rec_grp.items()},
         "in_above_end": int((rec_epg[-1][inside] > pcr.THRESH_HZ).sum()), "out_above_end": int((rec_epg[-1][~inside] > pcr.THRESH_HZ).sum()),
         "n_in": int(inside.sum()), "n_out": int((~inside).sum())}
    # cx_wedge's marks after the pulse
    for mark in (0.5, 1.0, 2.0, 3.0, 5.0):
        k = int(round((pulse_at + a.pulse_s + mark) / 0.01)) - 1
        if 0 <= k < frames:
            m[f"t{mark}_in_mean"] = float(rec_epg[k][inside].mean()); m[f"t{mark}_out_mean"] = float(rec_epg[k][~inside].mean())
            m[f"t{mark}_in_above"] = int((rec_epg[k][inside] > pcr.THRESH_HZ).sum()); m[f"t{mark}_out_above"] = int((rec_epg[k][~inside] > pcr.THRESH_HZ).sum())
    ledger = {"compass.EPG.bump_survival_s": {"value": survival, "op": ">=", "bound": 5.0, "status": "PASS" if survival >= 5.0 else "FAIL"}}
    for key, val, lo, hi in (("compass.EPG.bump_rate_hz", m["bump_hz_post"], 5.0, 60.0), ("compass.EPG.bump_width_wedges", m["width_half_post"], 2.5, 5.0)):
        st = "NOT_APPLICABLE" if survival < 5.0 else ("PASS" if (np.isfinite(val) and lo <= val <= hi) else "FAIL")
        ledger[key] = {"value": val, "op": "range", "bound": [lo, hi], "status": st, "requires": "compass.EPG.bump_survival_s"}
    prov = common.provenance(c, fb.brain.p, getattr(fb.optic, "p", None), fb=fb, device=a.device, seeds=[a.seed],
                             stimulus={"protocol": "cx_wedge.simulate at gE=gD=gR=1", "params": {"background_hz": a.background, "pulse_hz": a.pulse_hz, "pulse_s": a.pulse_s,
                                       "seconds_after": a.seconds, "width": a.width, "start_wedge": a.start_wedge, "adapt": a.adapt, "settle_s": settle_s},
                                       "control": "arm default"})
    out = {"schema": "flyverse.probe_unitary.compass/1", "arm": a.arm, "w_syn_by_nt": BRACKETS[a.arm], "seed": a.seed, "metrics": m, "ledger": ledger,
           "wedge_profile_post": [float(rec_epg[post][:, wedge_of == w].mean()) for w in range(16)],
           "wedge_profile_end": [float(rec_epg[-1][wedge_of == w].mean()) for w in range(16)],
           "wall_s": wall, "device": dev, "provenance": prov, "generator": "python " + " ".join(shlex.quote(x) for x in sys.argv)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(common.to_jsonable(out), open(a.out, "w"), indent=1)
    npz = Path(a.out).with_suffix(".npz")
    np.savez_compressed(npz, t=tt, epg=rec_epg, wedge_of=wedge_of, inside=inside, **{f"g__{k}": v for k, v in rec_grp.items()})
    print(f"\narm {a.arm} seed {a.seed}: confined pre {m['frac_confined_pre']:.2f} post {m['frac_confined_post']:.2f}, survival {survival:.2f} s, bump {m['bump_hz_post']:.1f} Hz, "
          f"width {m['width_half_post']:.1f}, in/out post {m['epg_in_mean_post']:.1f}/{m['epg_out_mean_post']:.1f} Hz, PEN {m['PEN_mean_post']:.1f} D7 {m['Delta7_mean_post']:.1f} "
          f"Ring {m['Ring_mean_post']:.2f} rest {m['rest_mean_post']:.3f}; ledger {[(k, v['status']) for k, v in ledger.items()]}; device {dev}; {wall:.0f} s")
    print("wrote", a.out)


# ------------------------------------------------------------------------------------------------ room (GPU)
def cmd_room(a):
    import torch
    from flyverse import world
    from flyverse.batch_sim import BatchSim
    L0 = patch_lif(a.arm)
    try:
        seeds = list(range(a.seed * 100, a.seed * 100 + a.batch))
        info0 = world.make_room(0, "all")[1]
        start = (-0.15, 0.15, float(info0["table_top_z"]))     # batch_sustain / compass_room's start, on the table top
        sim = BatchSim(a.batch, seed=a.seed, seeds=seeds, start=start, program=a.program, fruit_set="all", fence=a.fence, device=a.device,
                       cuda_graphs=torch.cuda.is_available(), cuda_kernels=torch.cuda.is_available() or None,
                       event_driven=True if torch.cuda.is_available() else None, cuda_sparse=a.cuda_sparse)
        lp = sim.fb.brain.p
        dev = str(sim.fb.brain.device)
        print(f"arm {a.arm}: w_syn_by_nt {lp.w_syn_by_nt}, receptor {lp.receptor_model}/{lp.receptor_net_rule}, device {dev}, B {a.batch}", flush=True)
        info = world.make_room(0, "all")[1]; top_z = float(info["table_top_z"]); ext = info["table_extent"]
        B = a.batch; n = int(a.seconds * 100)
        heading = np.zeros((n, B)); pos = np.zeros((n, B, 3)); on_top = np.zeros((n, B), bool); airborne = np.zeros((n, B), bool)
        dna02 = np.full((n, B), np.nan); legasym = np.full((n, B), np.nan); dist_fruit = np.zeros((n, B))
        t0 = time.time()
        for k in range(n):
            sim.step()
            for i, f in enumerate(sim.flies):
                heading[k, i] = float(f.heading); pos[k, i] = f.pos; airborne[k, i] = bool(f.airborne)
                # table_extent = (xmin, xmax, ymin, ymax); probe_walk_straightness.py's |x| <= xmin rule never holds
                on_top[k, i] = (ext[0] - 1e-3 <= f.pos[0] <= ext[1] + 1e-3) and (ext[2] - 1e-3 <= f.pos[1] <= ext[3] + 1e-3) and abs(f.pos[2] - top_z) < 0.01
                cmd = sim.commands[i] if sim.commands else {}
                r = cmd.get("rates", {}) if isinstance(cmd, dict) else {}
                if r and "DNa02_R" in r:                    # batch_body.readout's keys: DNa02_L / DNa02_R, legMN_L / legMN_R (Hz)
                    dna02[k, i] = float(r["DNa02_R"]) - float(r["DNa02_L"])
                    legasym[k, i] = float(r["legMN_L"]) - float(r["legMN_R"])
            _names, d = sim.nearest_fruit()
            dist_fruit[k] = np.asarray(d, dtype=float).ravel()[:B]
            if k % 1000 == 999:
                print(f"  t {(k + 1) / 100:5.0f} s  on table {on_top[k].mean():.2f}  airborne {airborne[k].mean():.2f}  min fruit dist {dist_fruit[k].min() * 100:.1f} cm", flush=True)
        dh = np.diff(np.unwrap(heading, axis=0), axis=0) * 100.0
        walking = ~airborne[1:]
        rows = []
        for i in range(B):
            w = walking[:, i]
            path = float(np.linalg.norm(np.diff(pos[:, i, :2], axis=0), axis=1).sum())
            net = float(np.linalg.norm(pos[-1, i, :2] - pos[0, i, :2]))
            left = np.flatnonzero(~on_top[:, i])
            rows.append({"row": i, "seed": seeds[i], "yaw_sd_deg_s": float(np.degrees(np.std(dh[w, i]))) if w.any() else None,
                         "yaw_mean_abs_deg_s": float(np.degrees(np.mean(np.abs(dh[w, i])))) if w.any() else None,
                         "path_m": path, "net_m": net, "straightness": net / path if path > 0 else None,
                         "left_table_s": float(left[0] / 100.0) if len(left) else None, "frac_on_table": float(on_top[:, i].mean()),
                         "min_fruit_cm": float(dist_fruit[:, i].min() * 100), "frames_within_2cm": int((dist_fruit[:, i] < 0.02).sum()),
                         "hops": int(np.sum(np.diff(airborne[:, i].astype(int)) == 1)), "frac_airborne": float(airborne[:, i].mean()),
                         "dna02_abs_mean": float(np.nanmean(np.abs(dna02[:, i]))) if np.isfinite(dna02[:, i]).any() else None,
                         "dna02_mean": float(np.nanmean(dna02[:, i])) if np.isfinite(dna02[:, i]).any() else None,
                         "leg_abs_mean": float(np.nanmean(np.abs(legasym[:, i]))) if np.isfinite(legasym[:, i]).any() else None,
                         "leg_mean": float(np.nanmean(legasym[:, i])) if np.isfinite(legasym[:, i]).any() else None})

        def agg(key):
            v = [r[key] for r in rows if r[key] is not None]
            return {"mean": float(np.mean(v)), "median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)} if v else None
        keys = ("yaw_sd_deg_s", "yaw_mean_abs_deg_s", "straightness", "path_m", "left_table_s", "frac_on_table", "min_fruit_cm", "frames_within_2cm",
                "hops", "frac_airborne", "dna02_abs_mean", "dna02_mean", "leg_abs_mean", "leg_mean")
        summary = {k: agg(k) for k in keys}
        summary["n_left_table"] = int(sum(r["left_table_s"] is not None for r in rows))
        summary["hops_per_fly_min"] = float(np.mean([r["hops"] for r in rows]) / (a.seconds / 60.0))
        prov = common.provenance(sim.fb.c, lp, getattr(sim.fb.optic, "p", None), fb=sim.fb, device=a.device, seeds=[a.seed], env_seeds=seeds, batch=B,
                                 stimulus={"protocol": "probe_walk_straightness plain-fly room rollout", "params": {"seconds": a.seconds, "program": a.program, "fence": a.fence, "fruit": "all"},
                                           "control": "arm default"})
        out = {"schema": "flyverse.probe_unitary.room/1", "arm": a.arm, "w_syn_by_nt": BRACKETS[a.arm], "seed": a.seed, "seconds": a.seconds, "program": a.program,
               "fence": a.fence, "device": dev, "wall_s": time.time() - t0, "summary": summary, "rows": rows, "provenance": prov,
               "generator": "python " + " ".join(shlex.quote(x) for x in sys.argv)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(common.to_jsonable(out), open(a.out, "w"), indent=1)
        s = summary
        print(f"\narm {a.arm} seed {a.seed}: yaw SD {s['yaw_sd_deg_s']['median']:.2f} deg/s (median over flies), straightness {s['straightness']['median']:.2f}, "
              f"left the table {s['n_left_table']}/{B}, hops {s['hops']['mean']:.2f}/fly ({s['hops_per_fly_min']:.2f}/min), airborne {s['frac_airborne']['mean']:.3f}, "
              f"DNa02 |R-L| {s['dna02_abs_mean']['mean'] if s['dna02_abs_mean'] else float('nan'):.3f} Hz, device {dev}, wall {out['wall_s']:.0f} s")
        print("wrote", a.out)
    finally:
        brain.LIFParams = L0


# ------------------------------------------------------------------------------------------------ plan
def suite_manifest() -> dict:
    return {"name": "unitary", "sections": SUITE_SECTIONS, "brain_seeds": [0, 1, 2],
            "baseline": {"id": "fam_suite_default", "kind": "none", "note": "the shipped model"},
            "lesions": [{"id": f"fam_suite_{arm}", "kind": "lif", "spec": {"w_syn_by_nt": dict(by)},
                         "note": f"per-transmitter unitary bracket {arm} (docs/audits/unitary_strength.md)"}
                        for arm, by in BRACKETS.items() if by]}


def job(line: str, stem: str, out_dir: str) -> str:
    return (f"mkdir -p {out_dir} && source .venv/bin/activate && {line} > {stem}.txt 2>&1; st=$?; tail -4 {stem}.txt; exit $st")


def cmd_plan(a):
    out_dir = a.out_dir.rstrip("/")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    man = suite_manifest()
    man_arg = shlex.quote(json.dumps(man, separators=(",", ":")))
    (Path(out_dir) / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    jobs1 = []
    for arm in ARMS:                                                     # compass: 4 arms x seeds, short
        for s in range(a.seeds):
            stem = f"{out_dir}/fam_compass_{arm}_s{s}"
            jobs1.append(job(f"python scripts/probe_unitary.py compass --arm {arm} --seed {s} --adapt {a.adapt} --seconds {a.seconds} --out {stem}.json", stem, out_dir))
    for arm in ARMS:                                                     # suite: 4 arms x draws, through the lesion tool
        for k in range(a.draws):
            stem = f"{out_dir}/fam_suite_{arm}_r{k}"
            jobs1.append(job(f"python scripts/interp_lesion.py run --manifest {man_arg} --one fam_suite_{arm} --replicate {k} --out {out_dir} "
                             f"--sections {SUITE_SECTIONS} --seeds 0,1,2", stem, out_dir))
    jobs2 = []
    for arm in (a.room_arms.split(",") if a.room_arms else ["default", "mid"]):
        for s in range(a.room_runs):
            stem = f"{out_dir}/fam_room_{arm}_s{s}"
            jobs2.append(job(f"python scripts/probe_unitary.py room --arm {arm} --seed {s} --out {stem}.json", stem, out_dir))

    def script(name, jobs, minutes, log, block):
        # ONE block for the whole family through an explicit --arm-block-map: `--arm-block fam` resolves to the token
        # after `fam_` in the command (the file name), i.e. one block per JOB, which is what batch 1 got (its 16 compass
        # jobs were dealt round-robin over r3-h200a / b, both H200s; the suite's 12 landed on one box).
        bmap = Path(out_dir) / f"{block}.blocks.json"
        bmap.write_text(json.dumps([block] * len(jobs)), encoding="utf-8")
        call = (f"python scripts/cluster_run.py --name {name} --minutes {minutes} --arm-block-map {bmap.as_posix()} \\\n  "
                + " \\\n  ".join(shlex.quote(j) for j in jobs) + f" \\\n  --fetch {out_dir}/")
        return (f"#!/bin/sh\n# generated by scripts/probe_unitary.py plan on {time.strftime('%Y-%m-%d %H:%M')}: {len(jobs)} jobs\n"
                f"mkdir -p {out_dir} out\n" f'if [ -f {log} ]; then mv {log} "{log[:-4]}.$(date +%Y%m%dT%H%M%S).log"; fi\n' f"{call} 2>&1 | tee {log}\n")
    p1 = Path(out_dir) / "batch1_compass_suite.sh"; p2 = Path(out_dir) / "batch2_room.sh"
    if a.only in (None, "1"):
        p1.write_text(script("unit1", jobs1, a.minutes, "out/unitary_batch1_cluster.log", "unit1"), encoding="utf-8", newline="\n")
        print(f"{len(jobs1)} jobs -> {p1}; manifest {out_dir}/manifest.json (embedded in the suite job lines)")
    if a.only in (None, "2"):
        p2.write_text(script("unit2", jobs2, a.minutes, "out/unitary_batch2_cluster.log", "unit2"), encoding="utf-8", newline="\n")
        print(f"{len(jobs2)} jobs -> {p2}")


# ------------------------------------------------------------------------------------------------ report (CPU)
def _load(glob_pat):
    import glob
    out = []
    for f in sorted(glob.glob(glob_pat)):
        try:
            out.append((f, json.load(open(f, encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {f}: {e!r}")
    return out


def _arm_of(name: str, fam: str) -> str:
    for arm in ARMS:
        if f"{fam}_{arm}_" in name:
            return arm
    return "?"


def _stats(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return {"n": len(v), "mean": float(np.mean(v)) if v else float("nan"), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
            "min": float(np.min(v)) if v else float("nan"), "max": float(np.max(v)) if v else float("nan"), "values": [round(float(x), 4) for x in v]}


def cmd_report(a):
    import pandas as pd
    d = a.dir.rstrip("/")
    lines = [f"# probe_unitary report ({time.strftime('%Y-%m-%d %H:%M')}; {d})", ""]
    rep = {"brackets": BRACKETS, "compass": {}, "suite": {}, "room": {}}
    # ---- compass
    comp = _load(f"{d}/fam_compass_*_s*.json")
    if comp:
        by_arm = {}
        for f, j in comp:
            by_arm.setdefault(j["arm"], []).append((f, j))
        keys = ("survival_s", "frac_confined_post", "frac_confined_pre", "bump_hz_post", "width_half_post", "vs_post_all", "epg_in_mean_post", "epg_out_mean_post",
                "epg_in_mean_during", "epg_out_mean_during", "PEN_mean_post", "Delta7_mean_post", "Ring_mean_post", "rest_mean_post", "GLNO_mean_post", "epg_max_post")
        lines += ["## Compass (cx_wedge protocol at the shipped gains; per-arm runs = jobs)", "",
                  "| arm | runs | device | " + " | ".join(keys) + " | ledger survival / rate / width |", "|" + "---|" * (len(keys) + 4)]
        arm_stats = {}
        for arm in ARMS:
            if arm not in by_arm:
                continue
            runs = by_arm[arm]
            st = {k: _stats([j["metrics"].get(k) for _, j in runs]) for k in keys}
            devs = sorted(set(j.get("device", "?") + "/" + str(j.get("provenance", {}).get("execution", {}).get("device_name")) for _, j in runs))
            led = {k: [j["ledger"][k]["status"] for _, j in runs] for k in ("compass.EPG.bump_survival_s", "compass.EPG.bump_rate_hz", "compass.EPG.bump_width_wedges")}
            arm_stats[arm] = {"stats": st, "devices": devs, "ledger": led, "files": [f for f, _ in runs]}
            cells = [f"{st[k]['mean']:.2f} [{st[k]['min']:.2f}-{st[k]['max']:.2f}]" for k in keys]
            lines.append(f"| {arm} | {len(runs)} | {';'.join(devs)} | " + " | ".join(cells) + " | " + " / ".join(",".join(v) for v in led.values()) + " |")
        if "default" in arm_stats:
            lines += ["", "compare vs default (common.compare; z = diff / SD(null), exact U):", ""]
            for arm in ARMS:
                if arm == "default" or arm not in arm_stats:
                    continue
                for k in ("survival_s", "frac_confined_post", "bump_hz_post", "epg_in_mean_post", "epg_out_mean_post", "PEN_mean_post", "Ring_mean_post", "rest_mean_post"):
                    cmp = common.compare(arm_stats[arm]["stats"][k]["values"], arm_stats["default"]["stats"][k]["values"])
                    arm_stats[arm].setdefault("compare", {})[k] = cmp
                    lines.append(f"- {arm} {k}: diff {cmp['diff']:+.3f} z {cmp['z']:+.2f} p {cmp['p']} verdict **{cmp['verdict']}**" if cmp.get("z") is not None and np.isfinite(cmp["z"])
                                 else f"- {arm} {k}: diff {cmp['diff']:+.3f} p {cmp['p']} verdict **{cmp['verdict']}** (null sd zero: {cmp.get('null_sd_zero')})")
        rep["compass"] = arm_stats
        lines.append("")
    # ---- suite
    suite = _load(f"{d}/fam_suite_*_r*.json")
    if suite:
        by_arm = {}
        for f, j in suite:
            by_arm.setdefault(_arm_of(os.path.basename(f), "fam_suite"), []).append((f, j))
        checks = {}
        for arm, runs in by_arm.items():
            for f, j in runs:
                for ch in j.get("checks", []):
                    checks.setdefault(ch["key"], {}).setdefault(arm, []).append((ch.get("value"), ch.get("status")))
        devs = {arm: sorted(set(str(j.get("provenance", {}).get("execution", {}).get("device")) + "/" + str(j.get("provenance", {}).get("execution", {}).get("device_name")) for _, j in runs)) for arm, runs in by_arm.items()}
        lines += [f"## Suite ({SUITE_SECTIONS}; draws = jobs; devices {devs})", "", "| check | " + " | ".join(f"{arm} (n)" for arm in ARMS if arm in by_arm) + " |", "|---|" + "---|" * len([x for x in ARMS if x in by_arm])]
        rows = []
        for key in sorted(checks):
            cells = []
            for arm in ARMS:
                if arm not in by_arm:
                    continue
                vs = checks[key].get(arm, [])
                vals = [v for v, _ in vs if v is not None]
                sts = [s for _, s in vs]
                stat = ",".join(sorted(set(sts))) if sts else "-"
                cells.append(f"{np.mean(vals):.3f} [{np.min(vals):.3f}-{np.max(vals):.3f}] {stat} ({len(vs)})" if vals else f"{stat} ({len(vs)})")
                rows.append({"check": key, "arm": arm, "values": vals, "statuses": sts})
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        rep["suite"] = {"checks": rows, "devices": devs, "files": {arm: [f for f, _ in runs] for arm, runs in by_arm.items()}}
        # status counts per arm
        lines.append("")
        for arm in ARMS:
            if arm not in by_arm:
                continue
            for f, j in by_arm[arm]:
                cs = j.get("checks", [])
                n_pass = sum(c["status"].startswith("PASS") for c in cs); n_fail = sum(c["status"] == "FAIL" for c in cs)
                n_gap = sum(c["status"] == "KNOWN GAP" for c in cs); n_miss = sum(c["status"] == "MISSING" for c in cs)
                lines.append(f"- {os.path.basename(f)}: {n_pass} pass, {n_fail} fail, {n_gap} known gap, {n_miss} missing; FAIL: {[c['key'] for c in cs if c['status'] == 'FAIL']}")
        lines.append("")
    # ---- room
    room = _load(f"{d}/fam_room_*_s*.json")
    if room:
        by_arm = {}
        for f, j in room:
            by_arm.setdefault(j["arm"], []).append((f, j))
        keys = ("yaw_sd_deg_s", "straightness", "hops", "frac_airborne", "left_table_s", "min_fruit_cm", "dna02_abs_mean", "dna02_mean", "leg_abs_mean", "leg_mean")
        lines += ["## Room (plain fly, 16 x 60 s per run; per-run medians over flies unless noted)", "",
                  "| arm | runs | device | n_left_table | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 4)]
        arm_stats = {}
        for arm in ARMS:
            if arm not in by_arm:
                continue
            runs = by_arm[arm]
            st = {}
            for k in keys:
                st[k] = _stats([(j["summary"][k] or {}).get("median" if k not in ("hops", "frac_airborne", "dna02_mean", "leg_mean") else "mean") if j["summary"].get(k) else None for _, j in runs])
            st["n_left_table"] = _stats([j["summary"]["n_left_table"] for _, j in runs])
            devs = sorted(set(j.get("device", "?") + "/" + str(j.get("provenance", {}).get("execution", {}).get("device_name")) for _, j in runs))
            arm_stats[arm] = {"stats": st, "devices": devs, "files": [f for f, _ in runs]}
            lines.append(f"| {arm} | {len(runs)} | {';'.join(devs)} | {st['n_left_table']['mean']:.1f} [{st['n_left_table']['min']:.0f}-{st['n_left_table']['max']:.0f}] | "
                         + " | ".join(f"{st[k]['mean']:.3f} [{st[k]['min']:.3f}-{st[k]['max']:.3f}]" for k in keys) + " |")
        if "default" in arm_stats:
            lines.append("")
            for arm in ARMS:
                if arm == "default" or arm not in arm_stats:
                    continue
                for k in keys + ("n_left_table",):
                    cmp = common.compare(arm_stats[arm]["stats"][k]["values"], arm_stats["default"]["stats"][k]["values"])
                    arm_stats[arm].setdefault("compare", {})[k] = cmp
                    z = cmp.get("z")
                    lines.append(f"- {arm} {k}: diff {cmp['diff']:+.3f} z {z if z is None else round(z, 2)} p {cmp['p']} verdict **{cmp['verdict']}**")
        rep["room"] = arm_stats
        lines.append("")
    text = "\n".join(lines)
    print(text)
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / "report.md").write_text(text, encoding="utf-8")
    json.dump(common.to_jsonable(rep), open(Path(d) / "report.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {d}/report.md, {d}/report.json")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("brackets"); s.add_argument("--json", default=None); s.set_defaults(fn=cmd_brackets)
    s = sub.add_parser("structure"); s.add_argument("--arms", default=None, help="comma list (default all)"); s.add_argument("--json", default=None); s.set_defaults(fn=cmd_structure)
    s = sub.add_parser("compass")
    s.add_argument("--arm", choices=ARMS, required=True); s.add_argument("--seed", type=int, default=0); s.add_argument("--out", required=True)
    s.add_argument("--seconds", type=float, default=5.0, help="free-running seconds after the pulse (cx_wedge: 5)")
    s.add_argument("--pulse-s", type=float, default=2.0); s.add_argument("--pulse-hz", type=float, default=40.0)
    s.add_argument("--background", type=float, default=10.0); s.add_argument("--width", type=int, default=4); s.add_argument("--start-wedge", type=int, default=0)
    s.add_argument("--adapt", choices=("off", "shipped"), default="off", help="compass adaptation: 'off' = cx_wedge / compass_room's experiment convention (adapt_by_type compass 0), 'shipped' = uniform 1.5 mV")
    s.add_argument("--device", default=None); s.add_argument("--no-graphs", action="store_true"); s.set_defaults(fn=cmd_compass)
    s = sub.add_parser("room")
    s.add_argument("--arm", choices=ARMS, required=True); s.add_argument("--seed", type=int, default=0); s.add_argument("--out", required=True)
    s.add_argument("--batch", type=int, default=16); s.add_argument("--seconds", type=float, default=60.0); s.add_argument("--program", default="none")
    s.add_argument("--fence", action="store_true"); s.add_argument("--device", default=None); s.add_argument("--cuda-sparse", default="torch"); s.set_defaults(fn=cmd_room)
    s = sub.add_parser("plan")
    s.add_argument("--out-dir", default=DEFAULT_OUT); s.add_argument("--seeds", type=int, default=4, help="compass runs per arm")
    s.add_argument("--draws", type=int, default=3, help="suite draws per arm"); s.add_argument("--room-runs", type=int, default=4); s.add_argument("--room-arms", default=None)
    s.add_argument("--seconds", type=float, default=5.0); s.add_argument("--adapt", default="off"); s.add_argument("--minutes", type=int, default=60)
    s.add_argument("--only", choices=("1", "2"), default=None, help="write only batch 1 (compass + suite) or only batch 2 (room)"); s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("report"); s.add_argument("--dir", default=DEFAULT_OUT); s.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
