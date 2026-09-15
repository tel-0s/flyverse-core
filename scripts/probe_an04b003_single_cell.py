"""AN04B003 alone: is the C-over-L excess of the leg-afferent ascending neuron the per-phase MODULATION of its
chordotonal input, or the hair-plate LEVEL acting through the sign-negative route SNpp45 -> IN13B001 -| AN04B003?
(docs/audits/level_matched_control.md 7 item 6 as corrected by its skeptic pass; docs/audits/level_controls.md.)

CPU, a subset brain (flyverse.Connectome.subset; docs/INTERP.md 2): the three leg-afferent channels of
senses.Proprioception (chordotonal 615, hair plate 113, leg campaniform 12 cells) + AN04B003 (6) + IN13B001 -- nothing
else, so AN04B003 receives ONLY its direct afferent input and the disynaptic hair-plate route through IN13B001; its
absolute rate is therefore not the room's, and every statement below is a within-protocol contrast. The afferents are
driven as the room drives them (Poisson at a commanded rate per cell per 10 ms frame, FlyBrain.stimulate) with
    modulated  the cycle arm's own per-leg / per-phase law: body.LegCycle at 9 mm/s (the room's realised speed), yaw 0,
               through Proprioception('chordotonal,hair_plate,campaniform+leg_cycle').rates(legs=...) -- the tripod
               alternation, the stance sweep 1 -> 0 and the swing burst, per cell, at the cycle's 7.8 Hz
    steady     each cell at its own time-mean of that sequence (the SAME per-cell mean by construction)
and IN13B001 either FREE (its afferent input intact) or CLAMPED (its incoming synapses zeroed in the subset W and the
cell driven by Poisson at its rate in the room's cycle arm, 65.2 Hz -- so its inhibition of AN04B003 is the same
Poisson train in every row it is clamped in). Conditions are rows of one FlyBrain batch per clamp state; runs
(brain seeds) are the replicate unit; verdicts are flyverse.interp.common.compare over runs.

    (i)  steady vs modulated chordotonal at the same per-cell mean, IN13B001 clamped   -> the modulation term alone
         (and the same contrast with IN13B001 free, and with the hair plate / campaniform modulated too)
    (ii) the hair-plate level alone (the cycle law's own per-cell mean on this subset vs 56.8 Hz = the level control's
         realised mean) at a fixed steady chordotonal, IN13B001 free; the campaniform alone (the cycle law's mean vs
         49.7 = the level control's); both together. The reference is the cycle law's per-cell mean, not the room's
         47.1 / 24.9 (the room mean carries its own airborne / standing frames), so the hair-plate step here is
         smaller than the room's L-vs-C mismatch and the audit quotes the contrast PER Hz of hair plate.

    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python scripts/probe_an04b003_single_cell.py --runs 5 --out out/vncd5/single_cell
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")            # CPU only: this desktop's GPU is never used (the cluster rule)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from flyverse import body, connectome
from flyverse.connectome import Connectome
from flyverse.fly import FlyBrain
from flyverse.interp import common
from flyverse.senses import Proprioception

SPEED_M_S = 0.009                # the room's realised walking speed under the cycle (~9 mm/s; body_sided_state.md 4.1)
IN13B001_CLAMP_HZ = 65.2         # IN13B001's window rate in the cycle arm C (level_matched_control.md skeptic R1 table)
HAIR_C, HAIR_L = 47.1, 56.8      # realised hair-plate means, cycle arm C / level control L (level_matched_control.md 4.2)
CAMP_C, CAMP_L = 24.9, 49.7      # realised leg-campaniform means, C / L
ROWS_CLAMP = ["steady_clamp", "mod_clamp"]
ROWS_FREE = ["steady_free", "mod_free", "modall_free", "hairL_free", "campL_free", "hairL_campL_free"]
ROW_DOC = {"steady_clamp": "chordotonal steady at the per-cell cycle mean; hair plate / campaniform steady at theirs; IN13B001 clamped at 65.2 Hz",
           "mod_clamp": "chordotonal MODULATED (the cycle law); hair plate / campaniform steady; IN13B001 clamped",
           "steady_free": "as steady_clamp with IN13B001 free (the reference of series (ii): every channel at the cycle law's own per-cell mean)",
           "mod_free": "as mod_clamp with IN13B001 free",
           "modall_free": "chordotonal, hair plate and campaniform all MODULATED (the full cycle pattern); IN13B001 free",
           "hairL_free": "chordotonal steady; hair plate raised to the LEVEL CONTROL's realised 56.8 Hz on every cell (from the cycle law's per-cell mean); campaniform at the cycle mean; IN13B001 free",
           "campL_free": "chordotonal steady; hair plate at the cycle mean; campaniform raised to the level control's 49.7 Hz on every cell; IN13B001 free",
           "hairL_campL_free": "chordotonal steady; hair plate 56.8 and campaniform 49.7 on every cell (the level control's two channels); IN13B001 free"}
CONTRASTS = [("mod_clamp", "steady_clamp", "(i) modulation alone: IN13B001 clamped, same per-cell chordotonal mean"),
             ("mod_free", "steady_free", "(i') modulation with IN13B001 free"),
             ("modall_free", "mod_free", "the hair plate / campaniform modulation on top of the chordotonal modulation"),
             ("hairL_free", "steady_free", "(ii) the hair-plate LEVEL alone: 56.8 Hz against the cycle law's own per-cell mean, chordotonal fixed (the cmd_hair_plate columns give the exact step)"),
             ("campL_free", "steady_free", "(ii) the campaniform level alone: 49.7 Hz against the cycle law's mean, chordotonal fixed"),
             ("hairL_campL_free", "steady_free", "(ii) both unmatched channels at the level control's realised values")]


def log(msg):
    print(msg, flush=True)


def build_subset(c):
    """The subset: leg afferents + AN04B003 + IN13B001, and the same subset with IN13B001's incoming synapses zeroed."""
    sense = Proprioception(c, "chordotonal,hair_plate,campaniform")
    an04 = c.select(type="AN04B003"); in13 = c.select(type="IN13B001")
    idx = np.unique(np.concatenate([sense.idx[ch] for ch in sense.channels] + [an04, in13]))
    sub = c.subset(idx)
    in13_sub = sub.select(type="IN13B001")
    W2 = sub.W.tolil(); W2[in13_sub, :] = 0.0
    clamp = Connectome(sub.neurons.copy(), W2.tocsr(), sub.body_to_index, sub.reference, _cache_dir=sub._cache_dir,
                       dataset=sub.dataset, release=sub.release, _manifest=sub._manifest)
    return sub, clamp, idx


def cycle_sequence(sub, n_frames, dt_s=0.01):
    """(T, n) commanded Hz per leg-afferent channel under the cycle law at SPEED_M_S, yaw 0, plus the sense on the subset."""
    s = Proprioception(sub, "chordotonal,hair_plate,campaniform+leg_cycle")
    cyc = body.LegCycle()
    phase = cyc.initial_phase(1)
    seq = {ch: np.zeros((n_frames, len(s.idx[ch])), np.float32) for ch in s.channels}
    for k in range(n_frames):
        st = cyc.advance(phase, [SPEED_M_S], [0.0], [False], dt_s); phase = st["phase"]
        legs = dict(phase=st["phase"], stance=st["stance"], amp=st["amp"], beta=st["beta"])
        for ch, _, hz in s.rates(0.0, 0.0, 0.0, False, 0.0, 1, legs=legs):
            seq[ch][k] = hz[0]
    return s, seq, cyc


def run_one(sub, clamp, seq, sense, seed, n_settle, n_win, device="cpu"):
    """One run: two FlyBrains (free / clamped), rows = conditions. Returns per-row rates and the realised commands."""
    T = n_settle + n_win
    idx = {ch: sense.idx[ch] for ch in sense.channels}
    means = {ch: seq[ch].mean(0) for ch in idx}                                  # per-cell time-mean over the WINDOW-length sequence
    an04 = sub.select(type="AN04B003"); in13 = sub.select(type="IN13B001")
    side = sub.neurons.somaSide.fillna("?").to_numpy()
    out = {}
    for name, graph, rows in (("free", sub, ROWS_FREE), ("clamp", clamp, ROWS_CLAMP)):
        fb = FlyBrain(graph, batch=len(rows), device=device, seed=seed)
        B = fb.B
        cmd_sum = {ch: np.zeros((B, len(idx[ch])), np.float64) for ch in idx}
        counts0 = None
        for k in range(T):
            kk = k % seq["chordotonal"].shape[0]
            mats = {}
            for ch in idx:
                m = np.tile(means[ch][None], (B, 1)).astype(np.float32)
                for r, row in enumerate(rows):
                    if row in ("mod_clamp", "mod_free", "modall_free") and ch == "chordotonal":
                        m[r] = seq[ch][kk]
                    elif row == "modall_free" and ch in ("hair_plate", "campaniform"):
                        m[r] = seq[ch][kk]
                    elif row in ("hairL_free", "hairL_campL_free") and ch == "hair_plate":
                        m[r] = HAIR_L
                    elif row in ("campL_free", "hairL_campL_free") and ch == "campaniform":
                        m[r] = CAMP_L
                mats[ch] = m
                fb.stimulate(idx[ch], m, 10.0)
            if name == "clamp":
                fb.stimulate(in13, np.full((B, len(in13)), IN13B001_CLAMP_HZ, np.float32), 10.0)
            fb.step(10.0)
            if k == n_settle - 1:
                counts0 = fb.brain.spike_counts.detach().clone()
            if k >= n_settle:
                for ch in idx:
                    cmd_sum[ch] += mats[ch]
        rate = ((fb.brain.spike_counts - counts0).double() / (n_win * 0.01)).cpu().numpy()      # (B, n) Hz over the window
        for r, row in enumerate(rows):
            d = {"AN04B003_hz": float(rate[r, an04].mean()), "AN04B003_L_hz": float(rate[r, an04[side[an04] == "L"]].mean()),
                 "AN04B003_R_hz": float(rate[r, an04[side[an04] == "R"]].mean()), "IN13B001_hz": float(rate[r, in13].mean()),
                 "AN04B003_cells_hz": [float(x) for x in rate[r, an04]]}
            for ch in idx:
                d[f"cmd_{ch}_hz"] = float(cmd_sum[ch][r].mean() / n_win)
                d[f"realised_{ch}_hz"] = float(rate[r, idx[ch]].mean())
            out[row] = d
        out[f"_{name}_provenance"] = common.provenance(graph, fb=fb, device=device, seeds=[seed], batch=B,
                                                         stimulus={"protocol": "an04b003_single_cell", "params": {"rows": rows, "speed_m_s": SPEED_M_S, "in13b001_clamp_hz": IN13B001_CLAMP_HZ if name == "clamp" else None,
                                                                    "hair_L": HAIR_L, "camp_L": CAMP_L, "n_settle": n_settle, "n_win": n_win, "frame_ms": 10.0}, "control": "the steady row of the same brain"})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--settle-s", type=float, default=2.0); ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--out", default="out/vncd5/single_cell"); ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    sub, clamp, idx = build_subset(c)
    n_settle, n_win = int(round(args.settle_s * 100)), int(round(args.window_s * 100))
    sense, seq, cyc = cycle_sequence(sub, n_win)
    tau, f, beta = cyc.timing([SPEED_M_S])
    log(f"subset {sub.n} cells (afferents {sum(len(v) for v in sense.idx.values())}: " + ", ".join(f"{ch} {len(v)}" for ch, v in sense.idx.items())
        + f"; AN04B003 {len(sub.select(type='AN04B003'))}, IN13B001 {len(sub.select(type='IN13B001'))}); clamp graph zeroes {int((sub.W[sub.select(type='IN13B001')] != 0).sum())} incoming entries of IN13B001")
    log(f"cycle at {SPEED_M_S * 1000:.1f} mm/s: step {f[0]:.2f} Hz, stance fraction {beta[0]:.3f}, amplitude {SPEED_M_S * tau[0] / cyc.step_ref_m:.3f}; "
        + "; ".join(f"{ch} time-mean over cells {seq[ch].mean():.2f} Hz (per-cell SD over time {seq[ch].std(0).mean():.1f})" for ch in seq))
    runs = []
    for k in range(args.runs):
        seed = args.seed0 + k; t1 = time.time()
        r = run_one(sub, clamp, seq, sense, seed, n_settle, n_win)
        r["seed"] = seed; r["wall_s"] = round(time.time() - t1, 1); runs.append(r)
        log(f"  run {k} (seed {seed}, {r['wall_s']} s): " + "; ".join(f"{row} AN04B003 {r[row]['AN04B003_hz']:.2f} (IN13B001 {r[row]['IN13B001_hz']:.1f})" for row in ROWS_CLAMP + ROWS_FREE))
    rows_all = ROWS_CLAMP + ROWS_FREE
    table = []
    for row in rows_all:
        d = {"row": row, "doc": ROW_DOC[row]}
        for key in ("AN04B003_hz", "AN04B003_L_hz", "AN04B003_R_hz", "IN13B001_hz", "cmd_chordotonal_hz", "cmd_hair_plate_hz", "cmd_campaniform_hz",
                    "realised_chordotonal_hz", "realised_hair_plate_hz", "realised_campaniform_hz"):
            v = np.array([r[row][key] for r in runs], float)
            d[key] = float(v.mean()); d[key + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan; d[key + "_runs"] = [round(float(x), 4) for x in v]
        table.append(d)
    tdf = pd.DataFrame(table)
    contrasts = []
    for a, b, doc in CONTRASTS:
        rec = {"treatment": a, "reference": b, "doc": doc}
        for key in ("AN04B003_hz", "AN04B003_L_hz", "AN04B003_R_hz", "IN13B001_hz"):
            va = np.array([r[a][key] for r in runs], float); vb = np.array([r[b][key] for r in runs], float)
            cmp_ = common.compare(va, vb)
            rec.update({f"{key}_a": float(va.mean()), f"{key}_b": float(vb.mean()), f"{key}_diff": cmp_["diff"], f"{key}_z": cmp_["z"], f"{key}_p": cmp_["p"], f"{key}_verdict": cmp_["verdict"]})
        # the precondition of a same-mean contrast: the commanded chordotonal means of the two rows agree
        for ch in ("chordotonal", "hair_plate", "campaniform"):
            rec[f"cmd_{ch}_a"] = float(np.mean([r[a][f"cmd_{ch}_hz"] for r in runs])); rec[f"cmd_{ch}_b"] = float(np.mean([r[b][f"cmd_{ch}_hz"] for r in runs]))
        contrasts.append(rec)
    cdf = pd.DataFrame(contrasts)
    tdf.to_csv(out / "rows.csv", index=False); cdf.to_csv(out / "contrasts.csv", index=False)
    print("\n== AN04B003 per condition (run mean +- SD over runs; per-run values in rows.csv)")
    for _, d in tdf.iterrows():
        print(f"  {d.row:18s} AN04B003 {d.AN04B003_hz:7.3f} +- {d.AN04B003_hz_sd:.3f} [{' '.join(f'{x:.3f}' for x in d.AN04B003_hz_runs)}]  L {d.AN04B003_L_hz:6.3f} R {d.AN04B003_R_hz:6.3f}  IN13B001 {d.IN13B001_hz:6.2f} +- {d.IN13B001_hz_sd:.2f}  "
              f"cmd chord {d.cmd_chordotonal_hz:6.2f} hair {d.cmd_hair_plate_hz:6.2f} camp {d.cmd_campaniform_hz:6.2f} (realised {d.realised_chordotonal_hz:.2f} / {d.realised_hair_plate_hz:.2f} / {d.realised_campaniform_hz:.2f})")
    print("\n== contrasts (common.compare over runs)")
    for _, r in cdf.iterrows():
        print(f"  {r.treatment:18s} v {r.reference:14s} AN04B003 {r.AN04B003_hz_b:.3f} -> {r.AN04B003_hz_a:.3f}  diff {r.AN04B003_hz_diff:+.3f} z {r.AN04B003_hz_z:+.1f} p {r.AN04B003_hz_p:.4f} {r.AN04B003_hz_verdict:12s} "
              f"| IN13B001 {r.IN13B001_hz_b:.2f} -> {r.IN13B001_hz_a:.2f} ({r.IN13B001_hz_verdict}) | cmd chord {r.cmd_chordotonal_b:.2f} / {r.cmd_chordotonal_a:.2f}, hair {r.cmd_hair_plate_b:.1f} / {r.cmd_hair_plate_a:.1f}, camp {r.cmd_campaniform_b:.1f} / {r.cmd_campaniform_a:.1f}   {r.doc}")
    res = common.Result.new("atlas", runs[0]["_free_provenance"])
    res.validation = {"target": "a mechanism check on a subset brain (not the atlas validation target)", "measured": None, "status": "n/a"}
    res.replicates = {"n": len(runs), "unit": "runs", "runs": [f"seed {r['seed']}" for r in runs], "null": "the steady row of the same brain (a within-run contrast; verdicts over runs)"}
    res.add_table("rows", tdf); res.add_table("contrasts", cdf)
    res.summary = {"question": "AN04B003 under steady vs modulated chordotonal input at the same per-cell mean (IN13B001 clamped / free), and under the hair-plate / campaniform level alone",
                   "subset": {"n_cells": int(sub.n), "afferents": {ch: int(len(v)) for ch, v in sense.idx.items()}, "AN04B003": int(len(sub.select(type="AN04B003"))), "IN13B001": int(len(sub.select(type="IN13B001"))),
                              "note": "afferents + AN04B003 + IN13B001 only: AN04B003's rate is not the room's; every statement is a within-protocol contrast"},
                   "cycle": {"speed_m_s": SPEED_M_S, "step_hz": float(f[0]), "stance_fraction": float(beta[0]), "amplitude": float(SPEED_M_S * tau[0] / cyc.step_ref_m),
                             "per_cell_sd_over_time_hz": {ch: float(seq[ch].std(0).mean()) for ch in seq}, "sequence_means_hz": {ch: float(seq[ch].mean()) for ch in seq}},
                   "clamp": {"IN13B001_hz": IN13B001_CLAMP_HZ, "method": "incoming synapses of IN13B001 zeroed in the subset W (Connectome(neurons, W2, ...) on sub.reference); the cell driven by FlyBrain.stimulate Poisson at the clamp rate every frame"},
                   "levels": {"hair_plate_C": HAIR_C, "hair_plate_L": HAIR_L, "campaniform_C": CAMP_C, "campaniform_L": CAMP_L},
                   "rows": ROW_DOC, "contrasts": common.to_jsonable(cdf.to_dict("records")), "window": {"settle_s": args.settle_s, "window_s": args.window_s},
                   "clamp_provenance_md5": hashlib.md5(json.dumps(common.to_jsonable(runs[0]["_clamp_provenance"]["compiled_connectome"]), sort_keys=True).encode()).hexdigest(),
                   "wall_s": round(time.time() - t0, 1)}
    res.files = {"generator": "scripts/probe_an04b003_single_cell.py " + " ".join(sys.argv[1:]), "rows_csv": str(out / "rows.csv"), "contrasts_csv": str(out / "contrasts.csv"),
                 "runs_json": str(out / "runs.json"), "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    res.save(out / "an04b003_single_cell.json")
    Path(out / "runs.json").write_text(json.dumps(common.to_jsonable([{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]), indent=1), encoding="utf-8")
    print(f"\nwrote {out}/an04b003_single_cell.json ({time.time() - t0:.0f} s); check {res.check()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
