"""Can the compass be moved?  (compass-shift thread, first dynamics round; docs/audits/cx_shift.md)

Two experiments on the LIF ring attractor of docs/audits/cx_wedge.md / cx_glno.md (full connectome, no room, compass
adaptation 0, EPG <-> PEN / PEG x gE, Delta7 -> EPG x gD with Delta7 -> PEN x1, 10 Hz EPG background) at the two
GLNO-sign-robust operating points gE 2 / gD 15 and gE 2.5 / gD 25, GLNO silent (sign 0, the default) and GLNO = gaba
(cx_wedge --nt-override scratch cache), 3 seeds each, plus a structural / room-side look at the rotation input.

  --structure (CPU, local)   PEN's input types ranked by raw synapses (sign-0 entries counted from cache/sign0_counts.npz),
                             marked silent where the presynaptic sign is 0; the same for GLNO (fed by PS196_b); for the
                             candidate rotation carriers (GLNO, LNO1, LNO2, LNOa, PS196_b, SpsP, IbSpsP) their L / R -> PEN
                             L / R wiring, their own inputs by type and superclass, their input from the populations that
                             flip under imposed rotation (out/screen_rotation.csv, |d'| >= 2), and the transmitter
                             evidence per type: MaleCNS body call (c.neurons.nt / sign), MaleCNS T-bar prediction
                             (audit_nt.tbar_lean over the raw file, if present), FlyWire top_nt (Schlegel 2024 file 1).
                             -> out/cx_shift_structure.json, out/cx_shift_structure.md
  --shift (GPU)              Experiment 1, the PEN L / R asymmetry: 1 s settle, 2 s pulse (wedges 0-3 at +40 Hz) that
                             establishes the bump, 2 s free, then fb.stimulate({'type': '~^PEN_', 'somaSide': SIDE},
                             PEN_HZ, PEN_MS) for SIDE in L / R / none, then 2 s free; the EPG rate vector and the EPG spike
                             counts are sampled every 100 ms and the bump centre (circular mean over the 16 wedges, ring
                             order L1 R8 L2 R7 ... L8 R1) is tracked before / during / after. Reported per run: the
                             centre shift over the stimulus window (wedges), its rate (wedges / s), the rate per Hz of PEN
                             drive, the drift in the free windows, and whether the bump survived. Runs are deterministic
                             given (connectome, gains, seed) so the L / R / none runs of one seed share their first 5 s.
  --rotation (GPU)           Experiment 2, the rotation input in the room: scripts/screen_rotation.py's protocol (pinned
                             fly in room_demo.Sim, no wind, rest / +RATE deg/s / rest / -RATE deg/s yaw for SECONDS each)
                             recording every candidate by side plus PEN / EPG / Delta7 / the optomotor references (HSN,
                             HSE, VS, DNp20, LPT26 / 50, Nod1 / 4), at the shipped defaults (GLNO silent) and with
                             GLNO = gaba; optionally (--gains gE:gD) with the compass gains applied and a bump established
                             by a 2 s pulse in the first rest phase, so that a rotation-driven bump shift in the room is
                             measured directly. The L - R flip (ccw - cw) per type in Hz and d' is the screen's statistic.
  --report (CPU)             tables from the JSONs -> out/cx_shift_shift.md / out/cx_shift_rotation.md

Every run records the connectome it used (cache dir, TYPE_NT_OVERRIDE table, GLNO's nt / sign), the device, and md5s of
flyverse/brain.py and flyverse/fly.py. Experiments apply gains (LIFParams type_path_gain / adapt_by_type overrides) to ask
a question; nothing here changes a default.

Cluster (one batch; each job asserts CUDA):
  python scripts/cluster_run.py --name cx-shift --minutes 25 \
    "python -c 'import torch; assert torch.cuda.is_available()' && python scripts/cx_shift.py --shift --gains 2:15 --seeds 0,1,2 --sides L,R,none --pen-hz 20 --no-scratch --out out/cx_shift_base_2_15.json > out/cx_shift_base_2_15.txt; cat out/cx_shift_base_2_15.txt" \
    "... --shift --gains 2:15 ... --nt-override GLNO=gaba --out out/cx_shift_gaba_2_15.json ..." (and 2.5:25, and --pen-hz 10,40) \
    "python -c '...' && python scripts/cx_shift.py --rotation --condition default --seeds 0,1,2 --out out/cx_shift_rot_default.json > ...; cat ..." \
    "... --rotation --condition gaba ..." "... --rotation --condition default --gains 2:15 ..." "... --condition gaba --gains 2:15 ..." --fetch out/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyverse import brain, connectome  # noqa: E402
import cx_wedge  # noqa: E402

OUT = ROOT / "out"
CANDIDATES = ["GLNO", "LNO1", "LNO2", "LNOa", "PS196_b", "SpsP", "IbSpsP", "LPsP"]   # LPsP added: 8th PEN input by synapses (structure run)
FLYWIRE_NAME = {"PS196_b": "PS196b"}
FLYWIRE_FILE = ROOT / "data" / "external" / "typing" / "schlegel2024_Supplemental_file1_neuron_annotations.tsv"
ROT_PATTERN = r"^(PEN_|GLNO|LNO|LPsP|PS196|SpsP|IbSpsP|EPG$|EPGt|Delta7|PEG|PFNd|PFNv|PFNa|HSN|HSE|VS$|DNp20|LPT26|LPT50|Nod1|Nod4|ExR|ER)"
ROT_REFERENCE = ["HSN", "HSE", "VS", "DNp20", "LPT26", "LPT50", "Nod1", "Nod4"]
ROT_SCREEN_CSV = OUT / "screen_rotation.csv"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


def provenance() -> dict:
    d = {"brain_py_md5": md5(ROOT / "flyverse" / "brain.py"), "fly_py_md5": md5(ROOT / "flyverse" / "fly.py"),
         "connectome_py_md5": md5(ROOT / "flyverse" / "connectome.py"), "body_py_md5": md5(ROOT / "flyverse" / "body.py")}
    try:
        import torch
        d["torch"] = torch.__version__
        d["device"] = torch.cuda.get_device_name(0) if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    except Exception:  # pragma: no cover
        pass
    return d


def circ_centre(profile16: np.ndarray):
    """Circular mean of a 16-wedge rate profile -> (centre wedge in [0, 16), vector strength)."""
    ang = 2 * np.pi * np.arange(16) / 16
    tot = float(profile16.sum())
    if tot <= 1e-9:
        return float("nan"), 0.0
    z = np.sum(profile16 * np.exp(1j * ang)) / tot
    return float((np.angle(z) % (2 * np.pi)) / (2 * np.pi) * 16), float(np.abs(z))


def unwrap_wedges(centres):
    """Unwrap a sequence of centre wedges (period 16) into a continuous track; NaNs are carried forward."""
    c = np.asarray(centres, float).copy()
    if np.isnan(c).all():
        return c
    first = np.flatnonzero(~np.isnan(c))[0]
    c[:first] = c[first]
    for i in range(1, len(c)):
        if np.isnan(c[i]):
            c[i] = c[i - 1]
    ang = np.unwrap(c / 16 * 2 * np.pi)
    return ang / (2 * np.pi) * 16


# ================================================================================================== structure (CPU)
def ranked_inputs(c, post_idx, raw, coo, top=None) -> pd.DataFrame:
    n = c.neurons
    ty = n.type.fillna("(untyped)").to_numpy()
    sel = np.isin(coo.row, post_idx)
    df = pd.DataFrame({"pre_type": ty[coo.col[sel]], "raw": raw[sel], "W": coo.data[sel],
                       "pre_sign": n.sign.to_numpy()[coo.col[sel]], "pre_nt": n.nt.fillna("?").to_numpy()[coo.col[sel]],
                       "pre_sc": n.superclass.fillna("?").to_numpy()[coo.col[sel]], "pre": coo.col[sel]})
    g = df.groupby("pre_type").agg(raw_syn=("raw", "sum"), W_sum=("W", "sum"), entries=("raw", "size"),
                                   cells=("pre", "nunique"),
                                   nt=("pre_nt", lambda s: "/".join(f"{k}" for k in s.value_counts().index[:2])),
                                   sign=("pre_sign", lambda s: "/".join(f"{v:+.0f}" for v in sorted(set(s)))),
                                   superclass=("pre_sc", lambda s: s.value_counts().index[0]))
    g["share"] = g.raw_syn / max(g.raw_syn.sum(), 1)
    g["silent"] = g.sign == "+0"
    g = g.sort_values("raw_syn", ascending=False)
    return g.head(top) if top else g


def side_matrix(c, pre_idx, post_idx, raw, coo) -> dict:
    """Raw synapses pre side x post side (somaSide L / R)."""
    side = c.neurons.somaSide.fillna("?").to_numpy()
    sel = np.isin(coo.row, post_idx) & np.isin(coo.col, pre_idx)
    m = {}
    for ps in ("L", "R"):
        for qs in ("L", "R"):
            mm = sel & (side[coo.col] == ps) & (side[coo.row] == qs)
            m[f"{ps}->{qs}"] = float(raw[mm].sum())
    return m


def transmitter_evidence(c, types, log=print) -> dict:
    n = c.neurons
    ev = {}
    fw = None
    if FLYWIRE_FILE.exists():
        fw = pd.read_csv(FLYWIRE_FILE, sep="\t", low_memory=False, usecols=["cell_type", "top_nt", "top_nt_conf", "known_nt", "known_nt_source"])
    tb = None
    try:
        import audit_nt
        m = n.type.fillna("").isin(types)
        if (connectome.DATA_DIR / audit_nt.TBAR_FILE).exists():
            tb = audit_nt.tbar_lean(n.bodyId.to_numpy()[m.to_numpy()], log)
    except Exception as e:  # pragma: no cover
        log(f"T-bar lean unavailable: {e}")
    for t in types:
        m = (n.type.fillna("") == t).to_numpy()
        d = {"cells": int(m.sum()), "malecns_nt": n.nt[m].value_counts().to_dict(), "sign": sorted(set(n.sign[m].tolist())),
             "silent_in_model": bool((n.sign[m] == 0).all()) if m.any() else None}
        if tb is not None:
            sub = tb.reindex(n.bodyId.to_numpy()[m]).dropna()
            if len(sub):
                shares = {k: float(v) for k, v in (sub[audit_nt.TBAR_NTS].multiply(sub.n_tbars, axis=0).sum() / sub.n_tbars.sum()).round(3).items()}
                d["tbar_argmax_share"] = dict(sorted(shares.items(), key=lambda kv: -kv[1])[:3])
                d["tbars"] = int(sub.n_tbars.sum())
        if fw is not None:
            name = FLYWIRE_NAME.get(t, t)
            f = fw[fw.cell_type.astype(str) == name]
            d["flywire_cells"] = int(len(f))
            d["flywire_top_nt"] = f.top_nt.value_counts().to_dict()
            d["flywire_conf"] = [float(x) for x in f.top_nt_conf.round(2)] if len(f) else []
            d["flywire_known_nt"] = f.known_nt.dropna().value_counts().to_dict() if "known_nt" in f else {}
        ev[t] = d
    return ev


def structure(log=print) -> dict:
    c = connectome.load(verbose=False)
    n = c.neurons
    ty = n.type.fillna("").to_numpy()
    W = c.W.tocsr(); coo = W.tocoo()
    cnt = connectome.sign0_counts(c, W=W, build=False)
    raw = np.abs(coo.data).astype(np.float64)
    if cnt is not None:
        raw = np.where(coo.data == 0, cnt, raw)
    res = {"provenance": provenance(), "sign0_counts_available": cnt is not None, "n_cells": int(c.n), "nnz": int(W.nnz)}
    pen = np.flatnonzero(np.char.startswith(ty.astype(str), "PEN_"))
    glno = np.flatnonzero(ty == "GLNO")
    res["pen_inputs"] = ranked_inputs(c, pen, raw, coo, top=30).reset_index().to_dict("records")
    res["pen_raw_input"] = float(raw[np.isin(coo.row, pen)].sum())
    res["pen_silent_share"] = float(raw[np.isin(coo.row, pen) & (n.sign.to_numpy()[coo.col] == 0)].sum() / res["pen_raw_input"])
    res["glno_inputs"] = ranked_inputs(c, glno, raw, coo, top=20).reset_index().to_dict("records")
    log(f"PEN (42 cells) raw input {res['pen_raw_input']:.0f} syn, silent (pre sign 0) share {100 * res['pen_silent_share']:.1f} %")
    log("PEN input types by raw synapses (top 30):")
    for r in res["pen_inputs"]:
        log(f"  {r['pre_type']:>12} {r['raw_syn']:8.0f} ({100 * r['share']:5.1f} %) {r['entries']:4d} entries {r['cells']:3d} cells  nt {r['nt']:<22} sign {r['sign']:>5}  {r['superclass']:<18}{' SILENT' if r['silent'] else ''}")
    log("GLNO (4 cells) input types by raw synapses (top 20):")
    for r in res["glno_inputs"]:
        log(f"  {r['pre_type']:>12} {r['raw_syn']:8.0f} ({100 * r['share']:5.1f} %) {r['entries']:4d} entries  nt {r['nt']:<22} sign {r['sign']:>5}  {r['superclass']:<18}{' SILENT' if r['silent'] else ''}")
    # rotation-flip populations from the screen (if on file)
    rot_types = []
    if ROT_SCREEN_CSV.exists():
        sr = pd.read_csv(ROT_SCREEN_CSV)
        rot_types = sr[sr.flip_d.abs() >= 2.0].type.tolist()
        res["rotation_flip_types"] = sr[sr.flip_d.abs() >= 2.0][["type", "cells", "flip_d", "flip_hz", "rate_hz"]].to_dict("records")
        log(f"rotation-flip populations (|d'| >= 2 in {ROT_SCREEN_CSV.name}): {rot_types}")
    rot_idx = np.flatnonzero(np.isin(ty, rot_types))
    optic_sc = ["ol_intrinsic", "visual_projection", "visual_centrifugal", "optic_lobe"]
    cand = {}
    for t in CANDIDATES:
        idx = np.flatnonzero(ty == t)
        if not len(idx):
            cand[t] = {"cells": 0}
            continue
        sel_in = np.isin(coo.row, idx)
        tot_in = float(raw[sel_in].sum())
        sc = n.superclass.fillna("?").to_numpy()[coo.col]
        by_sc = pd.Series(raw[sel_in]).groupby(sc[sel_in]).sum().sort_values(ascending=False)
        d = {"cells": int(len(idx)), "nt": n.nt[idx].value_counts().to_dict(), "sign": sorted(set(n.sign[idx].tolist())),
             "somaSide": n.somaSide[idx].value_counts().to_dict(),
             "to_pen_raw": float(raw[np.isin(coo.row, pen) & np.isin(coo.col, idx)].sum()),
             "to_pen_share_of_pen_input": float(raw[np.isin(coo.row, pen) & np.isin(coo.col, idx)].sum() / res["pen_raw_input"]),
             "to_pen_entries": int((np.isin(coo.row, pen) & np.isin(coo.col, idx)).sum()),
             "to_pen_side_matrix": side_matrix(c, idx, pen, raw, coo),
             "to_glno_raw": float(raw[np.isin(coo.row, glno) & np.isin(coo.col, idx)].sum()),
             "raw_input": tot_in,
             "input_by_superclass": {k: float(v) for k, v in by_sc.items()},
             "input_from_optic_superclasses": float(sum(v for k, v in by_sc.items() if k in optic_sc)),
             "input_from_rotation_flip_types": float(raw[sel_in & np.isin(coo.col, rot_idx)].sum()),
             "input_from_rotation_flip_by_type": {},
             "top_inputs": ranked_inputs(c, idx, raw, coo, top=12).reset_index().to_dict("records")}
        if len(rot_idx):
            rr = pd.Series(raw[sel_in & np.isin(coo.col, rot_idx)]).groupby(ty[coo.col[sel_in & np.isin(coo.col, rot_idx)]]).sum()
            d["input_from_rotation_flip_by_type"] = {k: float(v) for k, v in rr.sort_values(ascending=False).items()}
        # second step: what feeds the candidate's top-6 input types (share optic / rotation-flip)
        second = {}
        for r in d["top_inputs"][:6]:
            pidx = np.flatnonzero(ty == r["pre_type"])
            s2 = np.isin(coo.row, pidx)
            t2 = float(raw[s2].sum())
            second[r["pre_type"]] = {"raw_input": t2,
                                     "optic_share": float(raw[s2 & np.isin(n.superclass.fillna("?").to_numpy()[coo.col], optic_sc)].sum() / max(t2, 1)),
                                     "rotation_flip_share": float(raw[s2 & np.isin(coo.col, rot_idx)].sum() / max(t2, 1)),
                                     "top3": ranked_inputs(c, pidx, raw, coo, top=3).reset_index()[["pre_type", "raw_syn", "nt", "sign"]].to_dict("records")}
        d["second_step"] = second
        cand[t] = d
        log(f"\n{t}: {d['cells']} cells nt {d['nt']} sign {d['sign']} sides {d['somaSide']}; -> PEN {d['to_pen_raw']:.0f} syn "
            f"({100 * d['to_pen_share_of_pen_input']:.1f} % of PEN input; sides {d['to_pen_side_matrix']}); -> GLNO {d['to_glno_raw']:.0f}; "
            f"own input {tot_in:.0f} syn: optic superclasses {d['input_from_optic_superclasses']:.0f}, rotation-flip types {d['input_from_rotation_flip_types']:.0f} "
            f"{d['input_from_rotation_flip_by_type']}")
        log("   by superclass: " + ", ".join(f"{k} {100 * v / max(tot_in, 1):.0f} %" for k, v in list(by_sc.items())[:6]))
        for r in d["top_inputs"]:
            log(f"   {r['pre_type']:>12} {r['raw_syn']:7.0f} ({100 * r['share']:5.1f} %) nt {r['nt']:<22} sign {r['sign']:>5} {r['superclass']:<18}{' SILENT' if r['silent'] else ''}")
        for k, v in second.items():
            log(f"   2nd step {k:>12}: input {v['raw_input']:.0f}, optic {100 * v['optic_share']:.1f} %, rotation-flip {100 * v['rotation_flip_share']:.1f} %; top3 " +
                ", ".join(f"{x['pre_type']} {x['raw_syn']:.0f} ({x['nt']} {x['sign']})" for x in v["top3"]))
    res["candidates"] = cand
    res["transmitter_evidence"] = transmitter_evidence(c, CANDIDATES + ["PEN_a(PEN1)", "PEN_b(PEN2)"], log)
    log("\ntransmitter evidence:")
    for t, d in res["transmitter_evidence"].items():
        log(f"  {t:>8}: MaleCNS {d['malecns_nt']} sign {d['sign']}; T-bars {d.get('tbars', '-')} argmax {d.get('tbar_argmax_share', '-')}; "
            f"FlyWire {d.get('flywire_cells', '-')} cells top_nt {d.get('flywire_top_nt', '-')} conf {d.get('flywire_conf', '-')} known {d.get('flywire_known_nt', '-')}")
    OUT.mkdir(exist_ok=True)
    with open(OUT / "cx_shift_structure.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_structure_md(res, OUT / "cx_shift_structure.md")
    log(f"-> {OUT / 'cx_shift_structure.json'}, {OUT / 'cx_shift_structure.md'}")
    return res


def write_structure_md(res: dict, path: Path):
    L = ["## PEN input types by raw synapses (42 PEN cells; sign-0 entries counted from sign0_counts)", "",
         f"PEN raw input {res['pen_raw_input']:.0f} synapses; silent (presynaptic sign 0) share {100 * res['pen_silent_share']:.1f} %.", "",
         "| rank | pre type | raw syn | share | entries | cells | nt | sign | superclass | silent |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(res["pen_inputs"], 1):
        L.append(f"| {i} | {r['pre_type']} | {r['raw_syn']:.0f} | {100 * r['share']:.1f} % | {r['entries']} | {r['cells']} | {r['nt']} | {r['sign']} | {r['superclass']} | {'yes' if r['silent'] else ''} |")
    L += ["", "## GLNO input types (4 cells)", "", "| rank | pre type | raw syn | share | entries | nt | sign | superclass | silent |", "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(res["glno_inputs"], 1):
        L.append(f"| {i} | {r['pre_type']} | {r['raw_syn']:.0f} | {100 * r['share']:.1f} % | {r['entries']} | {r['nt']} | {r['sign']} | {r['superclass']} | {'yes' if r['silent'] else ''} |")
    L += ["", "## Candidate rotation carriers", "",
          "| type | cells | MaleCNS nt | sign | -> PEN raw | % of PEN input | side matrix pre->post (L->L, L->R, R->L, R->R) | -> GLNO | own input | optic-superclass input | rotation-flip input (types) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, d in res["candidates"].items():
        if not d.get("cells"):
            L.append(f"| {t} | 0 | | | | | | | | | |"); continue
        sm = d["to_pen_side_matrix"]
        L.append(f"| {t} | {d['cells']} | {d['nt']} | {d['sign']} | {d['to_pen_raw']:.0f} | {100 * d['to_pen_share_of_pen_input']:.1f} % | "
                 f"{sm['L->L']:.0f}, {sm['L->R']:.0f}, {sm['R->L']:.0f}, {sm['R->R']:.0f} | {d['to_glno_raw']:.0f} | {d['raw_input']:.0f} | "
                 f"{d['input_from_optic_superclasses']:.0f} | {d['input_from_rotation_flip_types']:.0f} {d['input_from_rotation_flip_by_type']} |")
    for t, d in res["candidates"].items():
        if not d.get("cells"):
            continue
        L += ["", f"### {t}: inputs by type (top 12) and by superclass", "",
              "by superclass: " + ", ".join(f"{k} {100 * v / max(d['raw_input'], 1):.1f} %" for k, v in list(d["input_by_superclass"].items())[:6]), "",
              "| pre type | raw syn | share | nt | sign | superclass | silent | 2nd step: optic share / rotation-flip share of that type's input |", "|---|---|---|---|---|---|---|---|"]
        for r in d["top_inputs"]:
            s2 = d["second_step"].get(r["pre_type"])
            s2s = f"{100 * s2['optic_share']:.1f} % / {100 * s2['rotation_flip_share']:.1f} %" if s2 else ""
            L.append(f"| {r['pre_type']} | {r['raw_syn']:.0f} | {100 * r['share']:.1f} % | {r['nt']} | {r['sign']} | {r['superclass']} | {'yes' if r['silent'] else ''} | {s2s} |")
    L += ["", "## Transmitter evidence per type", "",
          "| type | cells | MaleCNS body call | sign in model | T-bars | T-bar argmax shares | FlyWire cells | FlyWire top_nt | FlyWire conf | FlyWire known_nt |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t, d in res["transmitter_evidence"].items():
        L.append(f"| {t} | {d['cells']} | {d['malecns_nt']} | {d['sign']} | {d.get('tbars', '-')} | {d.get('tbar_argmax_share', '-')} | {d.get('flywire_cells', '-')} | "
                 f"{d.get('flywire_top_nt', '-')} | {d.get('flywire_conf', '-')} | {d.get('flywire_known_nt', '-')} |")
    if res.get("rotation_flip_types"):
        L += ["", "## Rotation-flip populations used (out/screen_rotation.csv, |d'| >= 2)", "", "| type | cells | flip d' | flip Hz | rate Hz |", "|---|---|---|---|---|"]
        for r in res["rotation_flip_types"]:
            L.append(f"| {r['type']} | {r['cells']} | {r['flip_d']:.2f} | {r['flip_hz']:.2f} | {r['rate_hz']:.2f} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ================================================================================================== experiment 1 (GPU)
def shift_run(c, cells, gE, gD, seed, side, pen_hz, pen_ms=1000.0, settle_s=1.0, pulse_s=2.0, free_s=2.0, after_s=2.0,
              bin_ms=100.0, background_hz=10.0, pulse_hz=40.0, start_wedge=0, width=4, cuda_graphs=True, nt_override=None,
              cache_dir=None, log=print) -> dict:
    from flyverse.fly import FlyBrain
    import torch
    epg = cells["EPG"]
    wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    inside = np.isin(wedge_of, [(start_wedge + j) % 16 for j in range(width)])
    idx_epg = epg["idx"]
    n = c.neurons
    ty = n.type.fillna("").to_numpy()
    pen_side = {s: c.select(type="~^PEN_", somaSide=s) for s in ("L", "R")}
    glno_side = {s: c.select(type="GLNO", somaSide=s) for s in ("L", "R")}
    d7 = cells["Delta7"]["idx"]
    glom = n.instance.fillna("").str.extract(r"_([LR]\d)")[0].to_numpy()
    tpg = list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", gE), (r"^PEN_", r"^EPG$", gE),
                                                (r"^EPG$", r"^PEG$", gE), (r"^PEG$", r"^EPG$", gE),
                                                (r"^Delta7$", r"^EPG$", gD), (cx_wedge.RING_RE, r"^(EPG$|PEN_|PEG$)", 1.0)]
    params = brain.LIFParams(adapt_by_type={cx_wedge.COMPASS_RE: 0.0}, type_path_gain=tpg, receptor_model=None)
    t0 = time.time()
    fb = FlyBrain(c, lif_params=params, seed=seed, cuda_graphs=cuda_graphs)
    pen_s = pen_ms / 1000.0
    t_pulse0, t_pulse1 = settle_s, settle_s + pulse_s
    t_pen0 = t_pulse1 + free_s
    t_pen1 = t_pen0 + pen_s
    t_end = t_pen1 + after_s
    fb.stimulate(idx_epg, background_hz, t_end * 1000 + 100)
    rec_idx = np.concatenate([idx_epg, pen_side["L"], pen_side["R"], glno_side["L"], glno_side["R"], d7])
    sl = {}
    o = 0
    for name, arr in (("epg", idx_epg), ("penL", pen_side["L"]), ("penR", pen_side["R"]), ("glnoL", glno_side["L"]), ("glnoR", glno_side["R"]), ("d7", d7)):
        sl[name] = slice(o, o + len(arr)); o += len(arr)
    rec_t = fb.brain._idx(rec_idx)
    n_wedge = np.bincount(wedge_of, minlength=16).astype(float)

    def counts():
        return fb.brain.spike_counts[0][rec_t].cpu().numpy().astype(np.float64)

    prev = counts()
    bin_s = bin_ms / 1000.0
    n_bins = int(round(t_end / bin_s))
    samples = []
    eps = 1e-6
    stim_side = side if side in ("L", "R") and pen_hz > 0 else None
    for k in range(n_bins):
        t = k * bin_s
        if abs(t - t_pulse0) < eps:
            fb.stimulate(idx_epg[inside], background_hz + pulse_hz, pulse_s * 1000)
        if abs(t - t_pen0) < eps and stim_side:
            fb.stimulate(pen_side[stim_side], pen_hz, pen_ms)
        fb.step(bin_ms)
        cur = counts(); dc = cur - prev; prev = cur
        r = fb.brain.rates(idx_epg)
        prof_rate = np.array([float(r[wedge_of == w].mean()) for w in range(16)])
        prof_cnt = np.bincount(wedge_of, weights=dc[sl["epg"]], minlength=16) / n_wedge / bin_s
        c_rate, vs_rate = circ_centre(prof_rate)
        c_cnt, vs_cnt = circ_centre(prof_cnt)
        samples.append(dict(t=round(t + bin_s, 3), centre_rate=c_rate, vs_rate=vs_rate, centre_cnt=c_cnt, vs_cnt=vs_cnt,
                            peak_rate=float(prof_rate.max()), in_mean=float(r[inside].mean()), out_mean=float(r[~inside].mean()),
                            in_above=int((r[inside] > 22).sum()), out_above=int((r[~inside] > 22).sum()),
                            penL=float(dc[sl["penL"]].mean() / bin_s), penR=float(dc[sl["penR"]].mean() / bin_s),
                            glnoL=float(dc[sl["glnoL"]].mean() / bin_s), glnoR=float(dc[sl["glnoR"]].mean() / bin_s),
                            d7=float(dc[sl["d7"]].mean() / bin_s), profile_cnt=[round(float(x), 2) for x in prof_cnt]))
    wall = time.time() - t0
    row = dict(gE=gE, gD=gD, seed=seed, side=side, pen_hz=pen_hz, pen_ms=pen_ms, settle_s=settle_s, pulse_s=pulse_s, free_s=free_s,
               after_s=after_s, bin_ms=bin_ms, background_hz=background_hz, pulse_hz=pulse_hz, start_wedge=start_wedge, width=width,
               t_pulse=[t_pulse0, t_pulse1], t_pen=[t_pen0, t_pen1], t_end=t_end,
               n_pen_side={s: int(len(v)) for s, v in pen_side.items()},
               pen_side_glomeruli={s: sorted(set(glom[v].tolist())) for s, v in pen_side.items()},
               nt_override=dict(nt_override or {}), cache_dir=str(cache_dir) if cache_dir else None,
               glno_nt=sorted(set(n.nt.to_numpy()[ty == "GLNO"].tolist())), glno_sign=sorted(set(n.sign.to_numpy()[ty == "GLNO"].tolist())),
               receptor_model=None, ring16=cx_wedge.RING16, samples=samples, wall_s=round(wall, 1), **provenance())
    row.update(shift_metrics(row))
    m = row
    log(f"gE {gE} gD {gD} seed {seed} side {side} PEN {pen_hz} Hz x {pen_ms:.0f} ms GLNO {row['glno_nt']}: bump at stim onset {m['bump_alive_at_onset']} "
        f"(peak {m['peak_at_onset']:.0f} Hz, vs {m['vs_at_onset']:.2f}), centre {m['centre_at_onset']:.2f} -> {m['centre_at_offset']:.2f} -> end {m['centre_at_end']:.2f}; "
        f"shift during {m['shift_during_wedges']:+.2f} wedges ({m['shift_rate_wedges_per_s']:+.2f} /s, slope {m['slope_during_wedges_per_s']:+.2f} /s; "
        f"per Hz {m['shift_rate_per_hz']:+.4f}), drift free {m['drift_free_wedges_per_s']:+.2f} /s, after {m['drift_after_wedges_per_s']:+.2f} /s; "
        f"PEN L/R during {m['penL_during']:.1f}/{m['penR_during']:.1f} Hz (free {m['penL_free']:.1f}/{m['penR_free']:.1f}); GLNO L/R during {m['glnoL_during']:.1f}/{m['glnoR_during']:.1f}; "
        f"alive at end {m['bump_alive_at_end']}; {wall:.0f} s")
    # Drop the last reference before empty_cache(), as `del fb` did. Not `del`: `counts()` above closes over
    # `fb`, and ruff reads a `del` of a closed-over name as unbinding it for the whole enclosing scope, so the
    # CI lint (E9,F63,F7,F82) reported the closure's `fb` as F821. Rebinding frees the FlyBrain identically.
    fb = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def shift_metrics(row: dict, key="centre_cnt") -> dict:
    """Shift statistics from the 100-ms samples: centre from the spike counts (key) unwrapped over the run."""
    S = row["samples"]
    t = np.array([s["t"] for s in S])
    cen = unwrap_wedges([s[key] for s in S])
    vs = np.array([s["vs_cnt"] for s in S]); peak = np.array([s["peak_rate"] for s in S])
    t_pen0, t_pen1 = row["t_pen"]; t_pulse1 = row["t_pulse"][1]; t_end = row["t_end"]
    pen_s = t_pen1 - t_pen0

    def mean_at(t_ref, before=True, n=2):
        # mean of n bins ending at t_ref (before) or starting after it (after)
        m = (t <= t_ref + 1e-6) & (t > t_ref - n * (t[1] - t[0]) - 1e-6) if before else (t > t_ref + 1e-6) & (t <= t_ref + n * (t[1] - t[0]) + 1e-6)
        return float(np.nanmean(cen[m])) if m.any() else float("nan")

    def slope(t0, t1):
        m = (t > t0 + 1e-6) & (t <= t1 + 1e-6)
        if m.sum() < 3:
            return float("nan")
        return float(np.polyfit(t[m], cen[m], 1)[0])

    c_on, c_off, c_end = mean_at(t_pen0), mean_at(t_pen1), mean_at(t_end)
    c_free0 = mean_at(t_pulse1 + 0.5, before=False)
    shift = c_off - c_on
    i_on = int(np.argmin(np.abs(t - t_pen0))); i_end = len(t) - 1
    d = dict(centre_at_onset=float(cen[i_on] % 16), centre_at_offset=float(c_off % 16), centre_at_end=float(c_end % 16),
             shift_during_wedges=float(shift), shift_rate_wedges_per_s=float(shift / pen_s) if pen_s > 0 else float("nan"),
             shift_rate_per_hz=float(shift / pen_s / row["pen_hz"]) if (pen_s > 0 and row["pen_hz"] > 0) else float("nan"),
             slope_during_wedges_per_s=slope(t_pen0, t_pen1),
             drift_free_wedges_per_s=float((c_on - c_free0) / max(t_pen0 - (t_pulse1 + 0.5), 1e-9)),
             slope_free_wedges_per_s=slope(t_pulse1 + 0.5, t_pen0),
             drift_after_wedges_per_s=float((c_end - c_off) / max(t_end - t_pen1, 1e-9)),
             slope_after_wedges_per_s=slope(t_pen1, t_end),
             total_shift_onset_to_end=float(c_end - c_on),
             peak_at_onset=float(peak[i_on]), vs_at_onset=float(vs[i_on]), peak_at_end=float(peak[i_end]), vs_at_end=float(vs[i_end]),
             bump_alive_at_onset=bool(peak[i_on] > 50 and vs[i_on] > 0.4), bump_alive_at_end=bool(peak[i_end] > 50 and vs[i_end] > 0.4))
    for name in ("penL", "penR", "glnoL", "glnoR", "d7"):
        v = np.array([s[name] for s in S])
        d[f"{name}_free"] = float(v[(t > t_pulse1 + 0.5) & (t <= t_pen0 + 1e-6)].mean())
        d[f"{name}_during"] = float(v[(t > t_pen0 + 1e-6) & (t <= t_pen1 + 1e-6)].mean())
        d[f"{name}_after"] = float(v[(t > t_pen1 + 1e-6)].mean())
    return d


def run_shift(a):
    import torch
    assert (torch.cuda.is_available() and torch.cuda.device_count() > 0) or a.allow_cpu, "CUDA is not available on this node (resubmit the job)"
    print(f"device {torch.cuda.get_device_name(0) if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'}; torch {torch.__version__}")
    nt_override = cx_wedge.parse_nt_override(a.nt_override)
    c, cache_dir, table = cx_wedge.load_connectome(nt_override, scratch=(not a.no_scratch) and bool(nt_override), verbose=False)
    print(f"connectome: {c.n} cells, nnz {c.W.nnz}, sum|W| {float(abs(c.W).sum()):.0f}; cache {cache_dir}; TYPE_NT_OVERRIDE {table}")
    ty = c.neurons.type.fillna("")
    print(f"GLNO: nt {c.neurons.nt[ty == 'GLNO'].value_counts().to_dict()} sign {c.neurons.sign[ty == 'GLNO'].value_counts().to_dict()}")
    cells = cx_wedge.compass_cells(c)
    gains = [tuple(float(x) for x in g.split(":")) for g in a.gains.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    sides = a.sides.split(",")
    hzs = [float(x) for x in a.pen_hz.split(",")]
    kw = dict(pen_ms=a.pen_ms, settle_s=a.settle, pulse_s=a.pulse, free_s=a.free, after_s=a.after, bin_ms=a.bin_ms,
              cuda_graphs=not a.no_graphs, nt_override=nt_override, cache_dir=cache_dir)
    rows = []
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for gE, gD in gains:
        for seed in seeds:
            for hz in hzs:
                for side in sides:
                    if side == "none" and hz != hzs[0]:
                        continue                                  # one control per (gain, seed)
                    rows.append(shift_run(c, cells, gE, gD, seed, side, hz if side != "none" else 0.0, **kw))
                    with open(out_path, "w") as f:
                        json.dump(rows, f, indent=1)
    print(f"{len(rows)} rows -> {out_path}; {time.time() - t0:.0f} s")


# ================================================================================================== experiment 2 (GPU, room)
def rotation_run(seed, condition, gains, seconds, rate, skip_s=3.0, pulse=True, cuda_graphs=False, log=print, smoke=False) -> dict:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    pygame.init(); pygame.display.set_mode((64, 64))
    import room_demo as rd
    from flyverse import screen, fly as flymod
    nt_override = {"GLNO": "gaba"} if condition == "gaba" else {}
    c, cache_dir, table = cx_wedge.load_connectome(nt_override, scratch=bool(nt_override), verbose=False)
    orig_load = connectome.load
    L = brain.LIFParams
    tpg = None
    if gains:
        gE, gD = gains
        tpg = list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", gE), (r"^PEN_", r"^EPG$", gE), (r"^EPG$", r"^PEG$", gE),
                                                    (r"^PEG$", r"^EPG$", gE), (r"^Delta7$", r"^EPG$", gD)]

        def make(**kw):
            p = L(**kw); p.adapt_by_type = {cx_wedge.COMPASS_RE: 0.0}; p.type_path_gain = tpg
            return p
        brain.LIFParams = make
    try:
        connectome.load = lambda *args, **kw: c                       # room_demo.Sim -> FlyBrain(c=None) -> connectome.load()
        t0 = time.time()
        sim = rd.Sim(seed, start=(0.0, 0.0, 0.75), trail_seconds=0.0, wind_speed=0.0, cuda_graphs=cuda_graphs)
    finally:
        connectome.load = orig_load
        brain.LIFParams = L
    assert sim.c is c, "the room Sim did not pick up the requested connectome"
    p = sim.fb.brain.p
    log(f"Sim built in {time.time() - t0:.0f} s: receptor {p.receptor_model}/{p.receptor_net_rule}, adapt_by_type {p.adapt_by_type}, "
        f"compass gains {[g for g in (p.type_path_gain or []) if 'EPG' in g[0] or 'Delta7' in g[0]]}; GLNO nt {sorted(set(c.neurons.nt[c.neurons.type == 'GLNO']))}")
    cells = cx_wedge.compass_cells(c)
    epg = cells["EPG"]; idx_epg = epg["idx"]
    wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    inside = np.isin(wedge_of, [0, 1, 2, 3])
    rec = screen.TypeRecorder.build(c, pattern=ROT_PATTERN, by_side=True)
    state = {"h": 0.0}
    centres = {}
    if gains:
        total_ms = 4 * seconds * 1000 + 1000
        sim.fb.stimulate(idx_epg, 10.0, total_ms)
        if pulse:
            sim.fb.stimulate(idx_epg[inside], 50.0, 2000.0 if not smoke else 100.0)

    def stepper(rate_dps, name):
        centres[name] = []

        def step():
            state["h"] += np.deg2rad(rate_dps) * 0.01
            sim.fly.place(0.0, 0.0, 0.75, heading=state["h"])
            sim.step()
            r = sim.fb.brain.rate_np()
            re_ = r[idx_epg]
            prof = np.array([float(re_[wedge_of == w].mean()) for w in range(16)])
            cc, vs = circ_centre(prof)
            centres[name].append((cc, vs, float(prof.max())))
            return r
        return step

    runs = {}
    for name, r_dps in [("rest", 0.0), ("ccw", rate), ("rest2", 0.0), ("cw", -rate)]:
        t1 = time.time()
        runs[name] = screen.record(stepper(r_dps, name), rec, seconds)
        log(f"  {name} ({r_dps:+.0f} deg/s, {seconds} s): {time.time() - t1:.0f} s wall; EPG mean {runs[name][:, [i for i, k in enumerate(rec.keys) if k.startswith('EPG_')]].mean():.2f} Hz")
    skip = int(skip_s * 100)
    S = {k: screen._smooth(v, 1.0)[skip:] for k, v in runs.items()}
    types = []
    for t, iL, iR in screen.lateral_pairs(rec):
        asym = {k: S[k][:, iL] - S[k][:, iR] for k in S}
        m = {k: float(v.mean()) for k, v in asym.items()}
        sd = float(np.sqrt(np.mean([v.std() ** 2 for v in asym.values()]))) + 1.0
        flip = m["ccw"] - m["cw"]
        rest = 0.5 * (m["rest"] + m["rest2"])
        types.append({"type": t, "cells": int(rec.n_cells[iL] + rec.n_cells[iR]), "flip_d": flip / sd, "flip_hz": flip,
                      "ccw_LR": m["ccw"], "cw_LR": m["cw"], "rest_LR": rest,
                      "rate_hz": {k: float(S[k][:, [iL, iR]].mean()) for k in S},
                      "rate_L": {k: float(S[k][:, iL].mean()) for k in S}, "rate_R": {k: float(S[k][:, iR].mean()) for k in S}})
    # the EPG bump per phase: centre drift (wedges / s over the scored window), vs, peak
    bump = {}
    for name, arr in centres.items():
        A = np.array(arr)[skip:] if len(arr) > skip else np.array(arr)
        cen = unwrap_wedges(A[:, 0]); tt = np.arange(len(cen)) * 0.01
        bump[name] = dict(vs=float(np.nanmean(A[:, 1])), peak=float(A[:, 2].mean()), centre_start=float(cen[0] % 16) if len(cen) else float("nan"),
                          centre_end=float(cen[-1] % 16) if len(cen) else float("nan"),
                          drift_wedges_per_s=float(np.polyfit(tt, cen, 1)[0]) if len(cen) > 10 and not np.isnan(cen).any() else float("nan"),
                          net_wedges=float(cen[-1] - cen[0]) if len(cen) else float("nan"))
    row = dict(seed=seed, condition=condition, gains=list(gains) if gains else None, seconds=seconds, rate_dps=rate, skip_s=skip_s,
               nt_override=nt_override, cache_dir=str(cache_dir) if cache_dir else None, receptor_model=p.receptor_model,
               receptor_net_rule=p.receptor_net_rule, glno_nt=sorted(set(c.neurons.nt[c.neurons.type == "GLNO"])),
               types=types, bump=bump, wall_s=round(time.time() - t0, 1), **provenance())
    df = pd.DataFrame(types)
    df["absd"] = df.flip_d.abs()
    show = df[df.type.isin(CANDIDATES + ["PEN_a", "PEN_b", "EPG", "Delta7"] + ROT_REFERENCE)].sort_values("absd", ascending=False)
    log(f"seed {seed} {condition} gains {gains}: bump per phase " + "; ".join(f"{k} vs {v['vs']:.2f} peak {v['peak']:.0f} Hz drift {v['drift_wedges_per_s']:+.2f} w/s" for k, v in bump.items()))
    for _, r in show.iterrows():
        log(f"   {r['type']:>8} ({r['cells']:2d} cells): L-R rest {r['rest_LR']:+6.2f} ccw {r['ccw_LR']:+6.2f} cw {r['cw_LR']:+6.2f} Hz; flip {r['flip_hz']:+6.2f} Hz d' {r['flip_d']:+5.2f}; "
            f"rate rest/ccw/cw {r['rate_hz']['rest']:.2f}/{r['rate_hz']['ccw']:.2f}/{r['rate_hz']['cw']:.2f} Hz")
    # Same as above: step() closes over `sim`, so `del sim` made the CI lint call that closure's `sim` undefined.
    sim = None
    return row


def run_rotation(a):
    import torch
    assert (torch.cuda.is_available() and torch.cuda.device_count() > 0) or a.allow_cpu, "CUDA is not available on this node (resubmit the job)"
    print(f"device {torch.cuda.get_device_name(0) if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'}; torch {torch.__version__}")
    gains = tuple(float(x) for x in a.gains.split(":")) if a.gains else None
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = []
    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        rows.append(rotation_run(seed, a.condition, gains, a.seconds, a.rate, skip_s=a.skip, cuda_graphs=a.graphs, smoke=a.smoke))
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=1, default=float)
    print(f"{len(rows)} rows -> {out_path}")


# ================================================================================================== report (CPU)
def load_rows(patterns):
    rows = []
    for pat in patterns:
        for p in (sorted(Path(q) for q in __import__("glob").glob(pat)) if any(ch in pat for ch in "*?[") else [Path(pat)]):   # glob.glob accepts absolute patterns
            if not p.exists():
                print(f"missing {p}"); continue
            for r in json.load(open(p)):
                r["_src"] = str(p); rows.append(r)
    return rows


def report_shift(files, table="cx_shift_shift"):
    rows = load_rows(files)
    if not rows:
        print("no shift rows"); return
    for r in rows:
        r.update(shift_metrics(r))                          # recompute from the samples (the shipped definition)
        r["glno"] = "gaba" if r.get("nt_override") else "silent"
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "samples"} for r in rows])
    df = df.sort_values(["gE", "gD", "glno", "pen_hz", "seed", "side"]).reset_index(drop=True)
    cols = ["gE", "gD", "glno", "seed", "side", "pen_hz", "bump_alive_at_onset", "centre_at_onset", "shift_during_wedges", "shift_rate_wedges_per_s",
            "slope_during_wedges_per_s", "shift_rate_per_hz", "drift_free_wedges_per_s", "drift_after_wedges_per_s", "total_shift_onset_to_end",
            "penL_during", "penR_during", "glnoL_during", "glnoR_during", "vs_at_onset", "vs_at_end", "peak_at_end", "bump_alive_at_end", "wall_s"]
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    hdr = ["gE", "gD", "GLNO", "seed", "side", "PEN Hz", "bump at onset", "centre at onset", "shift during (wedges)", "rate (w/s)", "slope fit (w/s)",
           "rate per Hz (w/s/Hz)", "drift free (w/s)", "drift after (w/s)", "onset->end (w)", "PEN L / R during (Hz)", "GLNO L / R during", "vs onset / end", "peak end Hz", "alive at end"]
    L = ["## Per run (centre = circular mean of the 16-wedge EPG spike-count profile per 100 ms, unwrapped; ring order L1 R8 L2 R7 L3 R6 L4 R5 L5 R4 L6 R3 L7 R2 L8 R1; + = towards increasing wedge index)", "",
         "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for _, r in df.iterrows():
        L.append(f"| {r.gE} | {r.gD:.0f} | {r.glno} | {r.seed} | {r.side} | {r.pen_hz:.0f} | {'yes' if r.bump_alive_at_onset else 'NO'} | {r.centre_at_onset:.2f} | {r.shift_during_wedges:+.2f} | "
                 f"{r.shift_rate_wedges_per_s:+.2f} | {r.slope_during_wedges_per_s:+.2f} | {r.shift_rate_per_hz:+.4f} | {r.drift_free_wedges_per_s:+.2f} | {r.drift_after_wedges_per_s:+.2f} | "
                 f"{r.total_shift_onset_to_end:+.2f} | {r.penL_during:.1f} / {r.penR_during:.1f} | {r.glnoL_during:.1f} / {r.glnoR_during:.1f} | {r.vs_at_onset:.2f} / {r.vs_at_end:.2f} | {r.peak_at_end:.0f} | {'yes' if r.bump_alive_at_end else 'NO'} |")
    # summary per (gE, gD, GLNO, pen_hz, side): mean +- sd over seeds, and the L - R contrast per seed
    def agg(s):
        return f"{s.mean():+.2f} +- {s.std(ddof=0):.2f} ({', '.join(f'{x:+.2f}' for x in s)})"

    summ = df.groupby(["gE", "gD", "glno", "pen_hz", "side"], sort=True).agg(
        n=("seed", "size"), alive_onset=("bump_alive_at_onset", "sum"), alive_end=("bump_alive_at_end", "sum"),
        shift=("shift_during_wedges", agg), rate=("shift_rate_wedges_per_s", agg), slope=("slope_during_wedges_per_s", agg),
        per_hz=("shift_rate_per_hz", lambda s: f"{s.mean():+.4f} +- {s.std(ddof=0):.4f}"),
        drift_free=("drift_free_wedges_per_s", agg), drift_after=("drift_after_wedges_per_s", agg),
        penL=("penL_during", "mean"), penR=("penR_during", "mean")).reset_index()
    print("\nsummary:"); print(summ.to_string(index=False))
    hdr2 = ["gE", "gD", "GLNO", "PEN Hz", "side", "n", "alive onset / end", "shift during (w): mean +- sd (per seed)", "rate (w/s)", "slope fit (w/s)", "rate per Hz", "drift free (w/s)", "drift after (w/s)", "PEN L / R during"]
    L += ["", "## Summary over seeds", "", "| " + " | ".join(hdr2) + " |", "|" + "---|" * len(hdr2)]
    for _, r in summ.iterrows():
        L.append(f"| {r.gE} | {r.gD:.0f} | {r.glno} | {r.pen_hz:.0f} | {r.side} | {r.n} | {r.alive_onset} / {r.alive_end} | {r['shift']} | {r['rate']} | {r['slope']} | {r['per_hz']} | {r['drift_free']} | {r['drift_after']} | {r.penL:.1f} / {r.penR:.1f} |")
    # seed-matched L - R contrast
    L += ["", "## Seed-matched contrast: shift(L) - shift(R) and each side minus the no-stimulus control (wedges over the stimulus window)", "",
          "| gE | gD | GLNO | PEN Hz | seed | L | R | none | L - R | L - none | R - none |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (gE, gD, gl, hz), g in df[df.side != "none"].groupby(["gE", "gD", "glno", "pen_hz"]):
        for seed in sorted(g.seed.unique()):
            def val(side, hz_=hz):
                q = df[(df.gE == gE) & (df.gD == gD) & (df.glno == gl) & (df.seed == seed) & (df.side == side) & ((df.pen_hz == hz_) | (side == "none"))]
                return float(q.shift_during_wedges.iloc[0]) if len(q) else float("nan")
            l, rr, nn = val("L"), val("R"), val("none")
            L.append(f"| {gE} | {gD:.0f} | {gl} | {hz:.0f} | {seed} | {l:+.2f} | {rr:+.2f} | {nn:+.2f} | {l - rr:+.2f} | {l - nn:+.2f} | {rr - nn:+.2f} |")
    (OUT / f"{table}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    df[cols].to_csv(OUT / f"{table}.csv", index=False)
    print(f"-> {OUT / (table + '.md')}, {OUT / (table + '.csv')}")
    return df


def report_rotation(files, table="cx_shift_rotation"):
    rows = load_rows(files)
    if not rows:
        print("no rotation rows"); return
    recs = []
    for r in rows:
        cond = r["condition"] + (f"+gains {r['gains'][0]:g}/{r['gains'][1]:g}" if r.get("gains") else "")
        for t in r["types"]:
            recs.append(dict(condition=cond, seed=r["seed"], type=t["type"], cells=t["cells"], flip_d=t["flip_d"], flip_hz=t["flip_hz"],
                             rest_LR=t["rest_LR"], ccw_LR=t["ccw_LR"], cw_LR=t["cw_LR"],
                             rate_rest=t["rate_hz"]["rest"], rate_ccw=t["rate_hz"]["ccw"], rate_cw=t["rate_hz"]["cw"]))
    df = pd.DataFrame(recs)
    keep = CANDIDATES + ["PEN_a", "PEN_b", "EPG", "Delta7", "PEG"] + ROT_REFERENCE
    d = df[df.type.isin(keep)].copy()
    d["type"] = pd.Categorical(d.type, keep, ordered=True)
    d = d.sort_values(["condition", "type", "seed"])
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    L = ["## Imposed rotation in the room (pinned fly, no wind; rest / +90 / rest / -90 deg/s, 10 s each, 1 s smoothing, first 3 s of each phase skipped)", "",
         "flip = (L - R at ccw) - (L - R at cw) in Hz; d' = flip / (pooled sd + 1). Rates are the L+R mean per phase.", ""]
    for cond, g in d.groupby("condition", sort=False):
        L += [f"### {cond}", "", "| type | cells | seed | L-R rest | L-R ccw | L-R cw | flip Hz | d' | rate rest / ccw / cw Hz |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| {r['type']} | {r['cells']} | {r['seed']} | {r['rest_LR']:+.2f} | {r['ccw_LR']:+.2f} | {r['cw_LR']:+.2f} | {r['flip_hz']:+.2f} | {r['flip_d']:+.2f} | {r['rate_rest']:.2f} / {r['rate_ccw']:.2f} / {r['rate_cw']:.2f} |")
        L.append("")
    # summary: mean flip over seeds per type x condition
    summ = d.groupby(["condition", "type"], sort=False, observed=True).agg(n=("seed", "size"), flip_hz=("flip_hz", lambda s: f"{s.mean():+.2f} ({', '.join(f'{x:+.2f}' for x in s)})"),
                                                                          flip_d=("flip_d", lambda s: f"{s.mean():+.2f} ({', '.join(f'{x:+.2f}' for x in s)})"),
                                                                          rate=("rate_rest", "mean")).reset_index()
    L += ["## Summary (mean over seeds; per-seed values in brackets)", "", "| condition | type | n | flip Hz | flip d' | rate at rest Hz |", "|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        L.append(f"| {r.condition} | {r['type']} | {r.n} | {r.flip_hz} | {r.flip_d} | {r.rate:.2f} |")
    # bump per phase (gains runs)
    L += ["", "## EPG bump per phase (gains runs: 10 Hz background + a 2 s pulse on wedges 0-3 at the start of the first rest phase)", "",
          "| condition | seed | phase | vs | peak Hz | centre start -> end | drift (wedges / s) | net (wedges) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cond = r["condition"] + (f"+gains {r['gains'][0]:g}/{r['gains'][1]:g}" if r.get("gains") else "")
        for ph, b in r["bump"].items():
            L.append(f"| {cond} | {r['seed']} | {ph} | {b['vs']:.2f} | {b['peak']:.0f} | {b['centre_start']:.2f} -> {b['centre_end']:.2f} | {b['drift_wedges_per_s']:+.3f} | {b['net_wedges']:+.2f} |")
    # the full ranking per condition (top 12 by |d'| over all recorded types, seed-mean)
    full = df.groupby(["condition", "type"]).agg(n=("seed", "size"), flip_d=("flip_d", "mean"), flip_hz=("flip_hz", "mean"), rate=("rate_rest", "mean"), cells=("cells", "first")).reset_index()
    L += ["", "## Strongest sustained rotation signals among every recorded type (seed-mean |d'|, top 12 per condition)", "", "| condition | type | cells | n | flip d' | flip Hz | rate Hz |", "|---|---|---|---|---|---|---|"]
    for cond, g in full.groupby("condition", sort=False):
        for _, r in g.reindex(g.flip_d.abs().sort_values(ascending=False).index).head(12).iterrows():
            L.append(f"| {cond} | {r['type']} | {r.cells} | {r.n} | {r.flip_d:+.2f} | {r.flip_hz:+.2f} | {r.rate:.2f} |")
    (OUT / f"{table}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    d.to_csv(OUT / f"{table}.csv", index=False)
    print(f"-> {OUT / (table + '.md')}, {OUT / (table + '.csv')}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--shift", action="store_true")
    ap.add_argument("--rotation", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--files", nargs="*", default=None, help="--report: JSON paths / globs (default out/cx_shift_*.json)")
    ap.add_argument("--gains", default="2:15", help="--shift: gE:gD list; --rotation: one gE:gD to apply the compass gains (default none)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sides", default="L,R,none")
    ap.add_argument("--pen-hz", default="20")
    ap.add_argument("--pen-ms", type=float, default=1000.0)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--pulse", type=float, default=2.0)
    ap.add_argument("--free", type=float, default=2.0)
    ap.add_argument("--after", type=float, default=2.0)
    ap.add_argument("--bin-ms", type=float, default=100.0)
    ap.add_argument("--nt-override", action="append", default=None, metavar="TYPE=nt")
    ap.add_argument("--no-scratch", action="store_true", help="--shift without an override: read the shared cache")
    ap.add_argument("--condition", default="default", choices=["default", "gaba"])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--rate", type=float, default=90.0)
    ap.add_argument("--skip", type=float, default=3.0)
    ap.add_argument("--no-graphs", action="store_true", help="--shift: eager path (cuda graphs are the default, as in cx_wedge)")
    ap.add_argument("--graphs", action="store_true", help="--rotation: CUDA graphs in the room Sim (default off, as scripts/screen_rotation.py)")
    ap.add_argument("--allow-cpu", action="store_true", help="smoke tests only")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.rotation and a.gains == "2:15" and not any(x in " ".join(sys.argv) for x in ("--gains",)):
        a.gains = None
    if a.structure:
        structure()
    if a.shift:
        a.out = a.out or str(OUT / "cx_shift_shift.json")
        run_shift(a)
    if a.rotation:
        a.out = a.out or str(OUT / "cx_shift_rotation.json")
        run_rotation(a)
    if a.report:
        files = a.files or [str(OUT / "cx_shift_base_*.json"), str(OUT / "cx_shift_gaba_*.json")]
        sh = [f for f in files if "rot" not in Path(f).name]
        ro = [f for f in files if "rot" in Path(f).name] or [str(OUT / "cx_shift_rot_*.json")]
        report_shift(sh)
        report_rotation(ro)


if __name__ == "__main__":
    main()
