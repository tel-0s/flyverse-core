"""apply:rotation -- where does an imposed yaw rotation die on its way to PEN?  (docs/audits/deficit_rotation.md)

The toolkit applied to one deficit: the ring bump does not follow the fly's rotation (docs/audits/cx_shift.md: 0.00
wedges/s against 4.0 ideal at 90 deg/s; cx_glno.md: GLNO, 19.4 % of PEN's input, is sign 0). Three validated tools are
composed here -- nothing in flyverse/ is edited, nothing is tuned:

    paths     (CPU)  every population that carries yaw in the model -> PEN_a / PEN_b and -> GLNO, k <= 3, the top
                     signed and silent walks, the strongest silent link per k            (flyverse.interp.paths)
    record    (GPU)  ONE run of the rotation protocol (scripts/screen_rotation.py / cx_shift.py --rotation: pinned fly
                     at (0, 0, 0.75), wind off, rest / +90 / rest / -90 deg/s, 10 s each) under one condition
                     (GLNO silent = the shipped default | GLNO=gaba = cx_wedge's --nt-override scratch cache) and one
                     mode (visual = the fly is re-placed every 10 ms, the world turns; efferent = the fly turns ITSELF:
                     DNa02 of one side is stimulated and the body integrates the turn, position pinned) with the compass
                     gains gE 2 / gD 15 (a bump in the ring) -> per phase a trace recording of every cell (per-cell
                     time-means + pooled series: flyverse.interp.trace.ArmAccumulator), a per-frame recording of PEN's and
                     GLNO's presynaptic cells (the decompose input), the bump centre track and the realised heading
    analyse   (CPU)  >= 3 runs of one (condition, mode) -> trace (stimulus = ccw, control = rest, null = rest again)
                     from the photoreceptors and from the yaw cells, the sided L - R flip per type against its null,
                     decompose of PEN and GLNO during ccw / cw / rest against rest2, the bump drift per phase; one
                     Result JSON per tool call plus a summary Result (+ .md) per (condition, mode)
    report    (CPU)  the summary JSONs of several (condition, mode) side by side (the audit's tables)
    chain     (CPU)  per-phase L / R rates of the efference-copy chain cells over the runs (chain_rates.csv)
    verify-batch (CPU)  a fetched batch directory is from ONE job per run (run.json / recording meta / pen meta /
                     console agree per phase) -- run it before analysing; two clients fetching one directory
                     interleave their files (docs/audits/deficit_rotation.md 2.1)
    selftest  (CPU)  the flip / bump statistics on synthetic data

Cluster batch (20 jobs in ONE call: 2 conditions x 2 modes x 5 runs; --fetch a NAMED subdirectory that no other
client fetches; the audit ran two same-code batches, rot-cf0c43 and rot-7de91e, and pools them as 10 runs):
    cmds=(); for cond in default gaba; do for mode in visual efferent; do for s in 0 1 2 3 4; do
      cmds+=("python -c 'import torch; assert torch.cuda.is_available()' && mkdir -p out/rot && python scripts/interp_apply_rotation.py record --condition $cond --mode $mode --seed $s --out out/rot/${cond}_${mode}_r$s > out/rot/${cond}_${mode}_r$s.txt 2>&1; tail -12 out/rot/${cond}_${mode}_r$s.txt"); done; done; done
    python scripts/cluster_run.py --name rot --minutes 25 "${cmds[@]}" --fetch out/rot/ > out/rot_cluster.log 2>&1
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_rotation.py verify-batch out/rot_cf0c43 out/rot_7de91e --out out/interp/apply_rotation/verify
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_rotation.py analyse --runs "out/rot_cf0c43/default_visual_r*" "out/rot_7de91e/default_visual_r*" --out out/interp/apply_rotation/default_visual
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_rotation.py paths --out out/interp/apply_rotation/paths
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_rotation.py paths --recording out/rot_cf0c43/default_efferent_r0_ccw.npz --out out/interp/apply_rotation/paths_nf
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_rotation.py report --summaries "out/interp/apply_rotation/*/summary.json" --out out/interp/apply_rotation
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_rotation.py chain --runs "out/rot_cf0c43/*_r*" "out/rot_7de91e/*_r*"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if "record" in sys.argv[1:2] and any(sys.argv[i] == "--device" and sys.argv[i + 1].startswith("cpu") for i in range(len(sys.argv) - 1)):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # a CPU smoke test never touches this machine's GPU (the cluster rule): before torch is imported ('' is ignored on Windows)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flyverse.interp import common  # noqa: E402
from flyverse.interp.common import Recording, Result  # noqa: E402

# ---------------------------------------------------------------------------------------------- protocol constants
POS = (0.0, 0.0, 0.75)                 # screen_rotation.py / cx_shift.py --rotation: the pinned spot in the room
RATE_DPS = 90.0
SECONDS = 10.0
SKIP_S = 3.0
PHASES = [("rest", 0.0), ("ccw", +1.0), ("rest2", 0.0), ("cw", -1.0)]     # name, sign of the rotation
GAINS_DEFAULT = "2:15"                 # gE (EPG <-> PEN, EPG <-> PEG), gD (Delta7 -> EPG): the cx_shift / cx_glno operating point
BACKGROUND_HZ = 10.0                   # Poisson background on all 46 EPG for the whole run (cx_shift.rotation_run)
PULSE_HZ, PULSE_S, PULSE_WEDGES = 50.0, 2.0, (0, 1, 2, 3)   # the bump-forming pulse in the first rest phase
DNA02_HZ = 20.0                        # efferent mode: DNa02 of one side; body.Locomotion.k_turn = 200 deg/s per 40 Hz of DNa02 L - R
PEN = "~^PEN_"
PEN_TYPES = ["PEN_a(PEN1)", "PEN_b(PEN2)"]
PHOTORECEPTORS = "type:R1-R6|R7y|R7p|R7d|R7_unclear|R8y|R8p|R8_unclear"
# the populations that carry yaw in the model (docs/audits/cx_shift.md 3b, out/screen_rotation.csv |d'| >= 2; the HS / VS
# cells; Johnston's organ; the descending / premotor efference candidates that feed GLNO)
YAW_SOURCES = {
    "optic_yaw": "HSN|HSE|HSS|VS|H2|LPT26|LPT50|Nod1|Nod4",
    "descending_yaw": "DNp20|DNp15",
    "jo": "~^JO-",
    "efference": "DNa02|DNa01|DNa03|PS196_b|PS196_a|LAL139|LAL184|WED040_a",
}
YAW_CELLS = "HSN|HSE|HSS|VS|H2|LPT26|LPT50|Nod1|Nod4|DNp20|DNp15"
NAMED = ["HSN", "HSE", "HSS", "VS", "H2", "LPT26", "LPT50", "Nod1", "Nod4", "DNp20", "DNp15", "DNa02", "DNa01", "PS196_b",
         "LAL139", "LAL184", "WED040_a", "PS047_b", "PS048_a", "PS099_a", "PS099_b", "PS262", "AN07B037_a", "WED153", "CB2037", "LAL104",
         "GLNO", "PEN_a(PEN1)", "PEN_b(PEN2)", "EPG", "Delta7", "PEG", "LPsP", "IbSpsP", "LNO1", "LNO2", "LNOa", "SpsP", "PFNd", "PFNv"]
DECOMPOSE_AT = ["PEN_a(PEN1)", "PEN_b(PEN2)", "GLNO", "PS196_b"]


# ---------------------------------------------------------------------------------------------- shared helpers
def parse_gains(s):
    if not s or s.lower() in ("none", "off", "0"):
        return None
    gE, gD = (float(x) for x in s.split(":"))
    return (gE, gD)


def load_condition_connectome(condition: str, cache_dir=None, verbose=False):
    """The connectome of a condition: 'default' = the shared cache (GLNO nt unknown, sign 0); 'gaba' = cx_wedge's
    scratch cache compiled with TYPE_NT_OVERRIDE + {GLNO: gaba} (out/cache_<hash>/, built if missing)."""
    from flyverse import connectome as cn
    import cx_wedge
    if cache_dir:
        return cn.load(cache_dir=Path(cache_dir), verbose=verbose), Path(cache_dir)
    override = {"GLNO": "gaba"} if condition == "gaba" else {}
    c, cdir, _table = cx_wedge.load_connectome(override, scratch=bool(override), verbose=verbose)
    return c, cdir


def circ_centre(profile16: np.ndarray):
    """Circular mean of a 16-wedge rate profile -> (centre wedge in [0, 16), vector strength) (cx_shift.circ_centre)."""
    ang = 2 * np.pi * np.arange(16) / 16
    tot = float(profile16.sum())
    if tot <= 1e-9:
        return float("nan"), 0.0
    z = np.sum(profile16 * np.exp(1j * ang)) / tot
    return float((np.angle(z) % (2 * np.pi)) / (2 * np.pi) * 16), float(np.abs(z))


def unwrap_wedges(centres):
    c = np.asarray(centres, float).copy()
    if np.isnan(c).all():
        return c
    first = np.flatnonzero(~np.isnan(c))[0]
    c[:first] = c[first]
    for i in range(1, len(c)):
        if np.isnan(c[i]):
            c[i] = c[i - 1]
    return np.unwrap(c / 16 * 2 * np.pi) / (2 * np.pi) * 16


def bump_metrics(track, skip: int, frame_s: float = 0.01) -> dict:
    """From a per-frame (centre, vs, peak) track of one phase: drift (wedges / s, linear fit over the scored window),
    net wedges, mean vs and peak, the centre at start / end."""
    A = np.asarray(track, float)
    if len(A) == 0:
        return dict(vs=float("nan"), peak=float("nan"), centre_start=float("nan"), centre_end=float("nan"), drift_wedges_per_s=float("nan"), net_wedges=float("nan"))
    A = A[skip:] if len(A) > skip else A
    cen = unwrap_wedges(A[:, 0]); tt = np.arange(len(cen)) * frame_s
    ok = len(cen) > 10 and not np.isnan(cen).any()
    return dict(vs=float(np.nanmean(A[:, 1])), peak=float(A[:, 2].mean()),
                centre_start=float(cen[0] % 16) if ok else float("nan"), centre_end=float(cen[-1] % 16) if ok else float("nan"),
                drift_wedges_per_s=float(np.polyfit(tt, cen, 1)[0]) if ok else float("nan"),
                net_wedges=float(cen[-1] - cen[0]) if ok else float("nan"))


def heading_metrics(headings, skip: int, frame_s: float = 0.01) -> dict:
    """Realised yaw of the body over the scored window: deg/s (linear fit of the unwrapped heading) and the net turn."""
    h = np.unwrap(np.asarray(headings, float))
    h = h[skip:] if len(h) > skip else h
    if len(h) < 10:
        return dict(rate_dps=float("nan"), net_deg=float("nan"))
    tt = np.arange(len(h)) * frame_s
    return dict(rate_dps=float(np.degrees(np.polyfit(tt, h, 1)[0])), net_deg=float(np.degrees(h[-1] - h[0])))


def flip_table(c, runs: dict, quantity_spiking: str = "rate_hz", min_cells_side: int = 1) -> pd.DataFrame:
    """The sided rotation statistic of scripts/screen_rotation.py on the per-cell time-means of trace recordings,
    replicated over runs and referenced to its own null.

    `runs` = {phase: [Recording (per-cell means, all cells), ...]} with the same run order in every phase. Per type
    with cells on both sides and per run: LR(phase) = mean over L cells - mean over R cells of the phase's time-mean
    (rate_hz on spiking cells, optic_dr on rate units); flip = LR(ccw) - LR(cw); the null draw of the same run is
    LR(rest2) - LR(rest) (the two rest phases, the control-again contrast). common.compare(flips, nulls) gives z /
    Welch / U / p / verdict per type. Also per phase the mean L - R and the mean rate over both sides."""
    n = c.neurons
    ty = n.type.fillna("").to_numpy(); side = n.somaSide.fillna("?").to_numpy()
    kinds = common.unit_kinds(c)
    ph = list(runs)
    n_runs = min(len(v) for v in runs.values())
    idx0 = np.asarray(runs[ph[0]][0].idx)
    ty_r, side_r, kind_r = ty[idx0], side[idx0], kinds[idx0]
    keys, inv = np.unique(ty_r, return_inverse=True)
    nT = len(keys)
    L = side_r == "L"; R = side_r == "R"
    nL = np.bincount(inv[L], minlength=nT); nR = np.bincount(inv[R], minlength=nT)
    both = (nL >= min_cells_side) & (nR >= min_cells_side) & (keys != "")

    def per_type_LR(rec: Recording):
        q = rec.quantities
        x = np.asarray(q[quantity_spiking][0], float)
        if "optic_dr" in q:
            graded = kind_r != "spiking"
            x = np.where(graded, np.asarray(q["optic_dr"][0], float), x)
        okL = L & np.isfinite(x); okR = R & np.isfinite(x)
        mL = np.bincount(inv[okL], weights=x[okL], minlength=nT) / np.maximum(np.bincount(inv[okL], minlength=nT), 1)
        mR = np.bincount(inv[okR], weights=x[okR], minlength=nT) / np.maximum(np.bincount(inv[okR], minlength=nT), 1)
        ok = np.isfinite(x)
        m = np.bincount(inv[ok], weights=x[ok], minlength=nT) / np.maximum(np.bincount(inv[ok], minlength=nT), 1)
        return mL - mR, m

    LR = {p: [] for p in ph}; M = {p: [] for p in ph}
    for p in ph:
        for r in range(n_runs):
            lr, m = per_type_LR(runs[p][r]); LR[p].append(lr); M[p].append(m)
    LR = {p: np.stack(v) for p, v in LR.items()}; M = {p: np.stack(v) for p, v in M.items()}
    rows = []
    for i in np.flatnonzero(both):
        flips = LR["ccw"][:, i] - LR["cw"][:, i]
        nulls = LR["rest2"][:, i] - LR["rest"][:, i]
        cmp = common.compare(flips, nulls)
        rows.append({"type": keys[i], "unit_kind": str(kind_r[inv == i][0]), "n_L": int(nL[i]), "n_R": int(nR[i]),
                     "flip": cmp["diff"] + float(np.mean(nulls)), "flip_mean": float(np.mean(flips)), "flip_sd": float(np.std(flips, ddof=1)) if n_runs > 1 else float("nan"),
                     "null_mean": float(np.mean(nulls)), "null_sd": float(np.std(nulls, ddof=1)) if n_runs > 1 else float("nan"),
                     "flip_runs": [float(v) for v in flips], "null_runs": [float(v) for v in nulls],
                     "z": cmp["z"], "welch": cmp["welch"], "U": cmp["U"], "p": cmp["p"], "verdict": cmp["verdict"],
                     **{f"LR_{p}": float(LR[p][:, i].mean()) for p in ph}, **{f"rate_{p}": float(M[p][:, i].mean()) for p in ph},
                     "n_runs": int(n_runs)})
    df = pd.DataFrame(rows)
    if len(df):
        df["abs_z"] = df.z.abs().fillna(0.0)
        df = df.sort_values(["abs_z", "flip_mean"], ascending=[False, False]).drop(columns="abs_z").reset_index(drop=True)
    return df


def phase_recordings(prefix, phases=None) -> tuple[list, dict, dict]:
    """The runs of one (condition, mode): stems matching '<prefix>_rest.json' (a prefix / glob, or a list of them --
    several batches of the same protocol pool into one set of runs); per phase the trace recording (`<stem>_<phase>`)
    and the decompose recording (`<stem>_<phase>_pen`)."""
    from flyverse.interp import trace as tr
    phases = phases or [p for p, _ in PHASES]
    stems = []
    for pre in ([prefix] if isinstance(prefix, str) else list(prefix)):
        pat = pre if any(ch in pre for ch in "*?[") else pre + "*"
        stems += sorted({f[: -len("_rest.json")] for f in glob.glob(pat + "_rest.json")})
    cells = {p: [] for p in phases}; pen = {p: [] for p in phases}
    for s in stems:
        for p in phases:
            cells[p].append(tr.load_run(f"{s}_{p}"))
            pp = Path(f"{s}_{p}_pen.npz")
            if pp.exists():
                r = Recording.load(pp.with_suffix("")); r.meta.setdefault("file", str(pp)); pen[p].append(r)
    return stems, cells, pen


# ---------------------------------------------------------------------------------------------- paths (CPU)
def cmd_paths(args) -> int:
    from flyverse.interp import paths as P
    lif, _ = common.params_from_args(args)
    c, cdir = load_condition_connectome(args.condition, args.cache_dir, verbose=not args.quiet)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ew = common.effective_weights(c, lif)
    counts, _sign0_ok = P.raw_counts(c)
    targets = {"PEN": PEN, "GLNO": "GLNO"}
    sources = dict(YAW_SOURCES)
    if args.source:
        sources = {f"src{i}": s for i, s in enumerate(args.source)}
    rows = []; files = {}
    for sname, spec in sources.items():
        for tname, tspec in targets.items():
            if sname == "efference" and tname == "GLNO" and False:
                continue
            t0 = time.time()
            kw = dict(params=lif, k_max=args.k, top=args.top, frozen="static", ew=ew, cache_dir=str(cdir) if cdir else None)
            if args.recording:                      # the never_firing flag judged on one named rollout (a batch recording)
                kw["recording"] = args.recording
            if counts is not None:
                kw["counts"] = counts
            try:
                res = P.paths(c, P.spec_from_cli(spec) if hasattr(P, "spec_from_cli") else spec, tspec, **kw)
            except ValueError as e:
                print(f"{sname} -> {tname}: {e}"); continue
            res.files["generator"] = "scripts/interp_apply_rotation.py paths " + " ".join(sys.argv[2:])
            res.summary["apply"] = {"deficit": "rotation", "source_group": sname, "source_spec": spec, "target": tname, "condition": args.condition}
            path = out / f"paths_{sname}_to_{tname}.json"
            res.save(path); files[f"{sname}->{tname}"] = str(path)
            s = res.summary
            pt = res.table("paths")
            direct = s.get("direct") or {}
            top_signed = {k: v for k, v in (s.get("top_walk_per_k") or s.get("top_signed_walk_per_k") or {}).items()}
            top_silent = {k: v for k, v in (s.get("top_silent_walk_per_k") or {}).items()}
            ssl = s.get("strongest_silent_link_per_k") or {}
            dom = s.get("dominant_silent_input_of_b") or {}
            row = {"source": sname, "target": tname, "a_cells": s.get("a_cells"), "b_cells": s.get("b_cells"),
                   "direct_raw_syn": direct.get("raw_count"), "direct_mv_per_post_volley": direct.get("mv_per_post_volley"),
                   "b_raw_input": s.get("b_raw_input_total"), "b_sign0_share": s.get("b_sign0_input_share"),
                   "dominant_silent_input_of_b": dom.get("pre_type"), "dominant_silent_share": dom.get("share_of_post_input") or dom.get("share_of_b_input"),
                   "wall_s": round(time.time() - t0, 1)}
            for k in range(1, args.k + 1):
                ts_ = top_signed.get(str(k)) or top_signed.get(k) or {}
                tsi = top_silent.get(str(k)) or top_silent.get(k) or {}
                sl = ssl.get(str(k)) or ssl.get(k) or {}
                row[f"k{k}_top_signed"] = ts_.get("path"); row[f"k{k}_top_signed_gain"] = ts_.get("gain")
                row[f"k{k}_top_silent"] = tsi.get("path"); row[f"k{k}_top_silent_gain_if_signed"] = tsi.get("gain_if_signed")
                row[f"k{k}_silent_link"] = (f"{sl.get('pre')} -> {sl.get('post')} [{sl.get('silent')}] {sl.get('mv_per_post_volley_if_signed'):+.1f} mV/volley if signed"
                                            if sl else None)
            # every yaw source type's direct link onto b (raw synapses), from the links / b_inputs tables
            bi = res.table("b_inputs")
            src_types = set(c.neurons.type.fillna("").to_numpy()[common.resolve(c, P.spec_from_cli(spec) if hasattr(P, "spec_from_cli") else spec)])
            if len(bi):
                hit = bi[bi.pre_type.isin(src_types)]
                row["source_types_direct_onto_b"] = {r.pre_type: float(r.raw_count) for r in hit.itertuples()} if len(hit) else {}
            rows.append(row)
            if not args.quiet:
                print(f"\n== {sname} ({spec}) -> {tname}: {s.get('a_cells')} -> {s.get('b_cells')} cells, {s.get('n_links')} links, {row['wall_s']} s")
                if len(pt):
                    common.print_table(pt[["k", "kind", "rank", "path", "gain", "gain_if_signed", "silent_links", "raw_counts"]].head(args.max_rows), max_rows=args.max_rows)
                for k in range(1, args.k + 1):
                    print(f"  strongest silent link k={k}: {row[f'k{k}_silent_link']}")
    df = pd.DataFrame(rows)
    summary = {"deficit": "rotation", "condition": args.condition, "k": args.k, "sources": sources, "targets": targets, "files": files,
               "table": common.to_jsonable(df.to_dict("records")), "generator": "scripts/interp_apply_rotation.py " + " ".join(sys.argv[1:]),
               "connectome": common.connectome_fingerprint(c, cdir), "lif": common.model_record(lif, None)["lif"]}
    with open(out / "paths_summary.json", "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable(summary), f, indent=1)
    if not args.quiet and len(df):
        print("\n== summary (per source group and target)")
        cols = ["source", "target", "a_cells", "direct_raw_syn", "direct_mv_per_post_volley", "b_sign0_share", "dominant_silent_input_of_b",
                "k2_top_signed", "k2_top_signed_gain", "k2_silent_link", "k3_top_signed_gain", "k3_silent_link"]
        common.print_table(df[[x for x in cols if x in df]], max_rows=50)
    print(f"written {out / 'paths_summary.json'}")
    return 0


# ---------------------------------------------------------------------------------------------- record (GPU)
def build_sim(c, gains, seed, device, allow_cpu, sparse="warp"):
    """room_demo.Sim on the given connectome with the compass gains installed (cx_shift.rotation_run's patch pattern:
    LIFParams built by the Sim carry adapt_by_type {compass: 0} and the extended type_path_gain; both restored after)."""
    import torch
    from flyverse import brain, connectome
    import cx_wedge
    import room_demo as rd
    L = brain.LIFParams; orig_load = connectome.load
    tpg = None
    if gains:
        gE, gD = gains
        tpg = list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", gE), (r"^PEN_", r"^EPG$", gE), (r"^EPG$", r"^PEG$", gE),
                                                    (r"^PEG$", r"^EPG$", gE), (r"^Delta7$", r"^EPG$", gD)]

        def make(**kw):
            p = L(**kw); p.adapt_by_type = {cx_wedge.COMPASS_RE: 0.0}; p.type_path_gain = tpg
            return p
        brain.LIFParams = make
    use_cuda = torch.cuda.is_available() and not allow_cpu
    flags = dict(cuda_kernels=True, cuda_graphs=False, event_driven=True, cuda_sparse=sparse) if use_cuda else {}
    t0 = time.time()
    try:
        connectome.load = lambda *a, **k: c
        sim = rd.Sim(seed, start=POS, trail_seconds=0.0, wind_speed=0.0, **flags)
    finally:
        connectome.load = orig_load; brain.LIFParams = L
    assert sim.c is c, "the room Sim did not pick up the requested connectome"
    dev = sim.fb.brain.device
    print(f"sim ready in {time.time() - t0:.0f} s; device {dev} (requested {device}); cuda available {torch.cuda.is_available()}; "
          f"receptor {sim.fb.brain.p.receptor_model}/{sim.fb.brain.p.receptor_net_rule}; GLNO nt {sorted(set(c.neurons.nt[c.neurons.type == 'GLNO']))}; "
          f"gains {gains}", flush=True)
    if not allow_cpu:
        assert dev.type == "cuda", f"device {dev}: not CUDA (node race; resubmit)"
    return sim


def cmd_record(args) -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if args.device and str(args.device).startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # (already set at import when the flag is on the command line)
        args.allow_cpu = True
    import torch
    if not args.allow_cpu:
        assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    import pygame
    pygame.init(); pygame.display.set_mode((64, 64))
    from flyverse.interp import trace as tr
    import cx_wedge
    t0 = time.time()
    seed = int(args.seed)
    gains = parse_gains(args.gains)
    c, cdir = load_condition_connectome(args.condition, args.cache_dir, verbose=False)
    print(f"connectome: condition {args.condition}, cache {cdir or 'shared'}, GLNO nt {sorted(set(c.neurons.nt[c.neurons.type == 'GLNO']))}, "
          f"{c.n} neurons, {c.W.nnz} entries ({time.time() - t0:.0f} s)", flush=True)
    sim = build_sim(c, gains, seed, args.device, args.allow_cpu, sparse=args.sparse)
    fb = sim.fb
    seconds = 2.0 if args.quick else float(args.seconds)
    skip = int(round((0.5 if args.quick else float(args.skip)) * 100))
    n_frames = int(round(seconds * 100))
    rate_dps = float(args.rate)
    # the compass state: 10 Hz background on every EPG for the whole run; the bump-forming pulse in the first rest phase
    cells = cx_wedge.compass_cells(c); epg = cells["EPG"]; idx_epg = epg["idx"]
    wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    inside = np.isin(wedge_of, list(PULSE_WEDGES))
    if gains:
        fb.stimulate(idx_epg, BACKGROUND_HZ, 4 * seconds * 1000 + 1000)
        fb.stimulate(idx_epg[inside], PULSE_HZ, PULSE_S * 1000 if not args.quick else 200.0)
    # the decompose recording: PEN and GLNO with every presynaptic cell that has a stored entry onto them
    Wc = c.W.tocsr()
    pen_idx = common.resolve(c, PEN); glno_idx = common.resolve(c, "GLNO")
    pre_idx = np.unique(np.concatenate([Wc[pen_idx].tocoo().col, Wc[glno_idx].tocoo().col]))
    sel = np.union1d(np.union1d(pen_idx, glno_idx), pre_idx)
    print(f"decompose recording: PEN {len(pen_idx)} + GLNO {len(glno_idx)} cells, {len(pre_idx)} presynaptic, {len(sel)} recorded per frame", flush=True)
    dna02 = {s: fb.c.select(type="DNa02", somaSide=s) for s in ("L", "R")}
    ty = c.neurons.type.fillna("").to_numpy(); side = c.neurons.somaSide.fillna("?").to_numpy()
    named_idx = {f"{t}_{s}": np.flatnonzero((ty == t) & (side == s)) for t in ("HSN", "DNa02", "PEN_a(PEN1)", "PEN_b(PEN2)", "GLNO", "PS196_b", "EPG", "Delta7") for s in ("L", "R")}
    state = {"h": 0.0}
    stim_common = {"protocol": "rotation", "mode": args.mode, "condition": args.condition, "gains": list(gains) if gains else None,
                   "params": {"pos": list(POS), "rate_dps": rate_dps, "seconds": seconds, "skip_s": skip / 100.0, "phases": [p for p, _ in PHASES],
                              "wind_speed": 0.0, "background_hz": BACKGROUND_HZ if gains else 0.0, "pulse_hz": PULSE_HZ if gains else 0.0,
                              "pulse_s": PULSE_S, "pulse_wedges": list(PULSE_WEDGES), "dna02_hz": float(args.dna02_hz) if args.mode == "efferent" else 0.0,
                              "compass_adaptation": 0.0 if gains else None, "sparse": args.sparse},
                   "control": "the rest phases of the same run (rest = control, rest2 = control again = the null)"}
    lif_p = fb.brain.p; optic_p = fb.optic.p if fb.optic is not None else None
    retina_rec = tr.retina_record(fb.retina, c)
    prov = common.provenance(c, lif_p, optic_p, fb=fb, device=args.device, seeds=[seed], env_seeds=[seed], batch=1, stimulus=stim_common,
                             retina=retina_rec, cache_dir=str(cdir) if cdir else args.cache_dir)
    prov["compiled_connectome"]["glno_nt"] = sorted(set(c.neurons.nt[c.neurons.type == "GLNO"]))
    prov["compiled_connectome"]["glno_to_pen_W_sum"] = float(Wc[pen_idx][:, glno_idx].sum())
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    summary_phases = {}
    written = []
    for name, sgn in PHASES:
        t1 = time.time()
        acc = tr.ArmAccumulator(fb, keep_series=not args.no_series, series_every=args.series_every)
        rec = common.Recorder(c, sel, quantities=("rate_hz",))
        track, headings, airborne = [], [], 0
        if args.mode == "efferent" and sgn != 0.0:
            fb.stimulate(dna02["L" if sgn > 0 else "R"], float(args.dna02_hz), seconds * 1000)
        for k in range(n_frames):
            if args.mode == "visual":
                state["h"] += np.deg2rad(sgn * rate_dps) * 0.01
                sim.fly.place(*POS, heading=state["h"])
            else:                                          # efferent: position pinned, heading integrated by the body from DNa02 / leg MNs
                sim.fly.place(*POS, heading=sim.fly.heading)
            sim.step()
            acc.add(fb, k, skip)
            rec.capture(fb, t_ms=10.0 * k)
            r = fb.brain.rate_np(); re_ = r[idx_epg]
            prof = np.array([float(re_[wedge_of == w].mean()) for w in range(16)])
            cc, vs = circ_centre(prof)
            track.append((cc, vs, float(prof.max())))
            headings.append(float(sim.fly.heading)); airborne += int(bool(sim.fly.airborne))
            state["h"] = float(sim.fly.heading) if args.mode == "efferent" else state["h"]
        bump = bump_metrics(track, skip); head = heading_metrics(headings, skip)
        rates = {}
        rate_mean = np.asarray(fb.brain.rate_np()) * 0 + (acc.counts_last - acc.counts0) / max(acc.n * 0.01, 1e-9)
        for k_, ii in named_idx.items():
            if len(ii):
                rates[k_] = float(rate_mean[ii].mean())
        meta = {"protocol": "rotation", "arm": name, "phase": name, "rotation_sign": sgn, "rate_dps": sgn * rate_dps if args.mode == "visual" else None,
                "condition": args.condition, "mode": args.mode, "gains": list(gains) if gains else None, "seed": seed,
                "window_s": [skip / 100.0, n_frames / 100.0], "stimulus": dict(stim_common, arm=name, phase=name, rotation_sign=sgn),
                "default_quantity": {"spiking": "rate_hz"}, "provenance": prov, "run_id": f"rotation-{args.condition}-{args.mode}-{name}-r{seed}",
                "bump": bump, "heading": head, "airborne_frames": airborne, "named_rates_hz": rates,
                "generator": " ".join(sys.argv)}
        cells_rec, series = acc.finish(meta)
        stem = Path(f"{out}_{name}")
        tr.save_recording(cells_rec, stem); written.append(str(stem.with_suffix(".npz")))
        if series is not None:
            tr.save_recording(series, Path(str(stem) + "_series"))
        pen_meta = {"protocol": "rotation", "arm": "as-given", "phase": name, "condition": args.condition, "mode": args.mode, "seed": seed,
                    "target": f"{PEN}|GLNO", "target_idx": np.union1d(pen_idx, glno_idx).tolist(), "n_presynaptic": int(len(pre_idx)),
                    "stimulus": meta["stimulus"], "provenance": prov, "window_s": meta["window_s"], "bump": bump, "heading": head,
                    "t_convention": "t_ms = 10 k is the frame's start within the phase; state at its end",
                    "generator": " ".join(sys.argv)}
        R = rec.finish(pen_meta); R.meta["file"] = str(Path(f"{out}_{name}_pen").with_suffix(".npz"))
        tr.save_recording(R, Path(f"{out}_{name}_pen")); written.append(R.meta["file"])
        summary_phases[name] = {"bump": bump, "heading": head, "airborne_frames": airborne, "rates": rates, "wall_s": round(time.time() - t1, 1)}
        print(f"  {name} ({'%+.0f deg/s' % (sgn * rate_dps) if args.mode == 'visual' else ('DNa02_%s %.0f Hz' % ('L' if sgn > 0 else 'R', args.dna02_hz) if sgn else 'no stimulus')}, {seconds} s): "
              f"{time.time() - t1:.0f} s wall; bump vs {bump['vs']:.2f} peak {bump['peak']:.0f} Hz drift {bump['drift_wedges_per_s']:+.3f} w/s net {bump['net_wedges']:+.2f}; "
              f"heading {head['rate_dps']:+.1f} deg/s (net {head['net_deg']:+.0f}); airborne {airborne}/{n_frames}; "
              + "; ".join(f"{k} {v:.1f}" for k, v in rates.items() if k.startswith(("HSN", "DNa02", "PEN_a", "GLNO"))), flush=True)
    dev = prov["execution"]["device"]
    with open(f"{out}_run.json", "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable({"condition": args.condition, "mode": args.mode, "seed": seed, "gains": gains, "phases": summary_phases,
                                      "device": dev, "device_name": prov["execution"].get("device_name"), "cache_dir": str(cdir) if cdir else None,
                                      "files": written, "wall_s": round(time.time() - t0, 1), "generator": " ".join(sys.argv)}), f, indent=1)
    print(f"[rotation {args.condition} {args.mode} seed {seed}] device {dev} ({prov['execution'].get('device_name')}), {time.time() - t0:.0f} s wall; "
          f"written {out}_<phase>[.npz|_series.npz|_pen.npz] and {out}_run.json")
    if dev is None or "cpu" in str(dev):
        print("WARNING: device cpu -- resubmit (the JSON records the realised device)")
    return 0


# ---------------------------------------------------------------------------------------------- analyse (CPU)
def readout_rows(c, cells: dict, types, quantity="rate_hz") -> pd.DataFrame:
    """Neurome readout_per_body rows: per body of the named types the phase means over runs (stimulus = ccw, control = rest)."""
    n = c.neurons; ty = n.type.fillna("").to_numpy(); kinds = common.unit_kinds(c)
    idx0 = np.asarray(cells["rest"][0].idx)
    rows = []
    for t in types:
        m = np.flatnonzero(ty[idx0] == t)
        for j in m:
            i = int(idx0[j])
            def vals(ph):
                return np.array([float(r.quantities[quantity][0, j]) for r in cells[ph]])
            v_ccw, v_cw, v_rest, v_rest2 = vals("ccw"), vals("cw"), vals("rest"), vals("rest2")
            rows.append({"bodyId": str(int(n.bodyId.iloc[i])), "model_index": i, "type": t, "unit_kind": str(kinds[i]), "quantity": "output_Hz",
                         "window_start_s": float(cells["rest"][0].meta.get("window_s", [SKIP_S, SECONDS])[0]), "window_end_s": float(cells["rest"][0].meta.get("window_s", [SKIP_S, SECONDS])[1]),
                         "stimulus_value": float(np.nanmean(v_ccw)), "control_value": float(np.nanmean(v_rest)), "stimulus_minus_control": float(np.nanmean(v_ccw - v_rest)),
                         "cw_value": float(np.nanmean(v_cw)), "rest2_value": float(np.nanmean(v_rest2)), "unit": "Hz", "n_trials": int(len(v_ccw)),
                         "trial_sd": float(np.nanstd(v_ccw - v_rest, ddof=1)) if len(v_ccw) > 1 else float("nan"),
                         "control_ids": [r.meta.get("run_id") for r in cells["rest"]], "somaSide": str(n.somaSide.iloc[i])})
    return pd.DataFrame(rows)


def cmd_analyse(args) -> int:
    from flyverse import brain
    from flyverse.interp import trace as tr
    from flyverse.interp import decompose as dec
    t0 = time.time()
    stems, cells, pen = phase_recordings(args.runs)
    if not stems:
        raise SystemExit(f"no runs match {args.runs}_rest.json")
    meta0 = cells["rest"][0].meta
    condition, mode = meta0.get("condition", args.condition), meta0.get("mode", "visual")
    devices = sorted({str(((r.meta.get("provenance") or {}).get("execution") or {}).get("device")) for ph in cells.values() for r in ph})
    print(f"{len(stems)} runs of condition {condition} mode {mode} (gains {meta0.get('gains')}); devices {devices}; seeds {[r.meta.get('seed') for r in cells['rest']]}")
    if any("cpu" in d for d in devices) and not args.allow_cpu_runs:
        raise SystemExit("a run realised device cpu: resubmit it (or pass --allow-cpu-runs for a smoke analysis)")
    c, cdir = load_condition_connectome(condition, args.cache_dir, verbose=False)
    lif = brain.LIFParams()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = {}
    n_runs = len(stems)
    pf = tr.p_floor(n_runs, n_runs)

    # ---- 1. the sided flip statistic (screen_rotation's signature) against its own null, every type
    flip = flip_table(c, cells)
    files["flip"] = str(out / "flip.csv"); flip.to_csv(files["flip"], index=False)
    named = flip[flip.type.isin(NAMED)]
    if not args.quiet:
        print(f"\n== sided flip (L - R at ccw) - (L - R at cw) vs its null (L - R at rest2) - (L - R at rest); {n_runs} runs, p floor {pf:.4f}")
        cols = ["type", "unit_kind", "n_L", "n_R", "flip_mean", "flip_sd", "null_mean", "null_sd", "z", "U", "p", "verdict", "LR_rest", "LR_ccw", "LR_cw", "rate_rest", "rate_ccw"]
        common.print_table(flip[cols].head(args.max_rows), max_rows=args.max_rows)
        print("-- the named populations --")
        common.print_table(named[cols], max_rows=60)

    # ---- 2. the bump and the body
    brows = []
    for ph in cells:
        for r in cells[ph]:
            b = r.meta.get("bump", {}); h = r.meta.get("heading", {})
            brows.append({"phase": ph, "seed": r.meta.get("seed"), "vs": b.get("vs"), "peak_hz": b.get("peak"), "drift_wedges_per_s": b.get("drift_wedges_per_s"),
                          "net_wedges": b.get("net_wedges"), "centre_start": b.get("centre_start"), "centre_end": b.get("centre_end"),
                          "heading_rate_dps": h.get("rate_dps"), "heading_net_deg": h.get("net_deg"), "airborne_frames": r.meta.get("airborne_frames"),
                          **{f"rate_{k}": v for k, v in (r.meta.get("named_rates_hz") or {}).items()}})
    bump = pd.DataFrame(brows)
    files["bump"] = str(out / "bump.csv"); bump.to_csv(files["bump"], index=False)
    ideal = RATE_DPS / 22.5
    bsum = {}
    for ph in cells:
        d = bump[bump.phase == ph]
        bsum[ph] = {k: {"mean": float(d[k].mean()), "sd": float(d[k].std(ddof=1)) if len(d) > 1 else float("nan"), "values": [float(x) for x in d[k]]}
                    for k in ("drift_wedges_per_s", "net_wedges", "vs", "peak_hz", "heading_rate_dps")}
    drift_vs_null = {ph: common.compare(bump[bump.phase == ph].drift_wedges_per_s.to_numpy(), bump[bump.phase.isin(["rest", "rest2"])].drift_wedges_per_s.to_numpy())
                     for ph in ("ccw", "cw")}
    if not args.quiet:
        print(f"\n== bump drift per phase (wedges / s; a heading-anchored bump owes {ideal:+.3f} at {RATE_DPS:.0f} deg/s) and the realised heading")
        common.print_table(bump[["phase", "seed", "vs", "peak_hz", "drift_wedges_per_s", "net_wedges", "heading_rate_dps", "heading_net_deg", "airborne_frames"]], max_rows=60)
        for ph in ("ccw", "cw"):
            v = drift_vs_null[ph]
            print(f"  {ph}: drift {v['stim']['mean']:+.4f} +- {v['stim']['sd']:.4f} w/s vs rest {v['null']['mean']:+.4f} +- {v['null']['sd']:.4f}; z {v['z']:+.2f} U {v['U']:.0f} p {v['p']:.3f} {v['verdict']}; "
                  f"heading {bsum[ph]['heading_rate_dps']['mean']:+.1f} +- {bsum[ph]['heading_rate_dps']['sd']:.1f} deg/s")

    # ---- 3. trace: where along the depth is the rotation lost (ccw vs rest, null rest2 vs rest), two sources
    traces = {}
    for label, source in (("photoreceptors", PHOTORECEPTORS), ("yaw_cells", YAW_CELLS)):
        for stim_ph in ("ccw", "cw"):
            if args.skip_trace:
                break
            t1 = time.time()
            res = tr.trace(c, source, stimulus=cells[stim_ph], control=cells["rest"], null=cells["rest2"], params=lif, stat=args.stat,
                           quantity="rate_hz", min_cells=2, stage_table=None, decompose_at=DECOMPOSE_AT, min_share=args.min_share,
                           depth_max=args.depth_max, per_body="lost")
            res.files["generator"] = "scripts/interp_apply_rotation.py analyse " + " ".join(sys.argv[2:])
            res.summary["apply"] = {"deficit": "rotation", "condition": condition, "mode": mode, "source": label, "stimulus_phase": stim_ph}
            path = out / f"trace_{label}_{stim_ph}.json"; res.save(path); files[f"trace_{label}_{stim_ph}"] = str(path)
            traces[(label, stim_ph)] = res
            pt = res.table("per_type")
            if not args.quiet:
                carriers = pt[pt.verdict == "result"] if len(pt) else pt
                print(f"\n== trace from {label} ({source}), stimulus {stim_ph} vs rest, null rest2 vs rest; stat {args.stat}; {len(pt)} types scored, "
                      f"{len(carriers)} carriers; first lost depth {res.summary.get('first_lost_depth')}; {time.time() - t1:.0f} s")
                if len(pt):
                    cols = [x for x in ("type", "depth", "n_cells", "stim_mean", "ctrl_level", "null_mean", "null_sd", "diff", "z", "U", "p", "verdict") if x in pt]
                    print("-- carriers by depth (top by |z|) --")
                    common.print_table(carriers.reindex(carriers.z.abs().sort_values(ascending=False).index)[cols].head(args.max_rows), max_rows=args.max_rows)
                    print("-- the named populations --")
                    common.print_table(pt[pt.type.isin(NAMED)].sort_values("depth")[cols], max_rows=60)
                    cd = res.summary.get("carriers_by_depth") or res.summary.get("n_carriers_by_depth")
                    if cd:
                        print(f"  carriers by depth: {cd}")
                li = res.table("lost_inputs")
                if len(li):
                    print("-- lost-stage inputs (the decompose_at types) --")
                    cols = [x for x in ("target_type", "target_depth", "target_z", "target_verdict", "pre_type", "pre_depth", "share", "sign", "mv_per_volley",
                                        "pre_z", "pre_verdict", "pre_diff") if x in li]
                    common.print_table(li[cols].head(60), max_rows=60)

    # ---- 4. decompose PEN and GLNO during the phases (rest2 = the null arm)
    decs = {}
    if pen["rest"] and not args.skip_decompose:
        for tname, tspec, by in (("PEN", PEN, ("type",)), ("PEN_by_side", PEN, ("type", "side")), ("GLNO", "GLNO", ("type",))):
            t1 = time.time()
            res = dec.decompose(c, tspec, recording={ph: pen[ph] for ph in ("ccw", "cw", "rest")}, null_recording={"rest2": pen["rest2"]}, params=lif,
                                by=by, tiers=False, window=(float(args.window.split(",")[0]), float(args.window.split(",")[1])), top=args.top, keep_links=False)
            res.files["generator"] = "scripts/interp_apply_rotation.py analyse " + " ".join(sys.argv[2:])
            res.summary["apply"] = {"deficit": "rotation", "condition": condition, "mode": mode, "target": tname}
            path = out / f"decompose_{tname}.json"; res.save(path); files[f"decompose_{tname}"] = str(path)
            decs[tname] = res
            if not args.quiet:
                pt = res.table("per_type")
                print(f"\n== decompose {tname} ({tspec}) by {by}: window {args.window} s, arms ccw / cw / rest vs null rest2 (mV/s per post cell); {time.time() - t1:.0f} s")
                cols = [x for x in ("post_type", "pre_group", "weight_mv_per_volley", "raw_count", "sign_rule", "rest_mean", "ccw_mean", "cw_mean", "rest2_mean",
                                    "ccw_z", "ccw_verdict", "cw_z", "cw_verdict", "rest_z", "ccw_rate_hz", "cw_rate_hz", "rest_rate_hz") if x in pt]
                common.print_table(pt[cols].head(args.max_rows), max_rows=args.max_rows)
                dyn = res.summary.get("dynamic", {})
                for arm, v in dyn.items():
                    for t, s in v.items():
                        print(f"  {arm} {t}: E {s['E_total']:+.1f} I {s['I_total']:+.1f} net {s['net']:+.1f} cancelling {s.get('cancelling_pair')}")

    # ---- 5. the summary Result (+ .md): the flip and bump tables, the readout per body, the per-tool files
    prov = dict(meta0.get("provenance") or common.provenance(c, lif, None, device="cpu"))
    prov["analysis"] = {"flyverse_commit": common.git_state(), "runs": stems, "devices": devices, "seeds": [r.meta.get("seed") for r in cells["rest"]],
                        "condition": condition, "mode": mode, "cache_dir": str(cdir) if cdir else None}
    res = Result.new("trace", prov)
    res.validation = {"name": "rotation deficit", "reference": {"bump_drift_wedges_per_s": [-0.010, 0.010], "ideal": ideal, "flip_types": ["HSN", "HSE", "Nod1", "DNp20", "LPT26", "LPT50"]},
                      "source": "docs/audits/cx_shift.md 3b (visual mode, GLNO silent / gaba, gains 2/15)", "measured": None, "status": "not run"}
    ref_types = ["HSN", "HSE", "Nod1", "DNp20", "LPT26", "LPT50"]
    ft = {r.type: r for r in flip.itertuples()}
    measured = {"bump_drift_ccw": bsum["ccw"]["drift_wedges_per_s"], "bump_drift_cw": bsum["cw"]["drift_wedges_per_s"],
                "flip_named": {t: {"flip_mean": ft[t].flip_mean, "z": ft[t].z, "verdict": ft[t].verdict} for t in ref_types if t in ft},
                "glno_flip": {"flip_mean": ft["GLNO"].flip_mean, "z": ft["GLNO"].z, "verdict": ft["GLNO"].verdict} if "GLNO" in ft else None}
    if mode == "visual" and condition in ("default", "gaba"):
        ok_drift = all(abs(v) <= 0.05 for v in bsum["ccw"]["drift_wedges_per_s"]["values"] + bsum["cw"]["drift_wedges_per_s"]["values"])
        ok_flip = all(t in ft and ft[t].flip_mean != 0 and np.sign(ft[t].flip_mean) == (1 if t == "LPT50" else -1) for t in ref_types)
        res.validation["status"] = "reproduced" if (ok_drift and ok_flip) else "not reproduced"
    else:
        res.validation["status"] = "not applicable (efferent mode has no reference; this is the open question)"
    res.validation["measured"] = common.to_jsonable(measured)
    res.add_population(common.population(c, PEN, "PEN"), unit_kind="spiking")
    res.add_population(common.population(c, "GLNO", "GLNO"), unit_kind="spiking")
    res.add_population(common.population(c, YAW_CELLS, "yaw_cells"), unit_kind="spiking")
    res.replicates = {"n": n_runs, "unit": "runs", "runs": [{"run_index": i, "seed": r.meta.get("seed"), "file": s, "device": str(((r.meta.get("provenance") or {}).get("execution") or {}).get("device"))}
                                                            for i, (s, r) in enumerate(zip(stems, cells["rest"]))],
                      "null": {"arm": "rest2 - rest (the second rest phase of the same run against the first)", "n": n_runs}, "p_floor": pf}
    res.add_table("flip", flip)
    res.add_table("bump", bump)
    res.add_table("readout_per_body", readout_rows(c, cells, ["GLNO", "PS196_b", "HSN", "HSE", "DNp20", "DNa02", "Nod1", "LPT26", "LPT50"] + PEN_TYPES))
    res.summary = {"apply": "rotation", "condition": condition, "mode": mode, "gains": meta0.get("gains"), "n_runs": n_runs, "p_floor": pf,
                   "protocol": meta0.get("stimulus"), "bump": bsum, "bump_drift_vs_rest": drift_vs_null, "ideal_wedges_per_s": ideal,
                   "flip_named": {t: {k: getattr(ft[t], k) for k in ("flip_mean", "flip_sd", "null_mean", "null_sd", "z", "U", "p", "verdict", "LR_rest", "LR_ccw", "LR_cw", "rate_rest")} for t in NAMED if t in ft},
                   "flip_results": flip[flip.verdict == "result"].type.tolist(), "flip_top": flip.head(15).type.tolist(),
                   "trace": {f"{k[0]}_{k[1]}": {"first_lost_depth": v.summary.get("first_lost_depth"), "n_carriers": int((v.table("per_type").verdict == "result").sum()) if len(v.table("per_type")) else 0,
                                                "carriers_by_depth": v.summary.get("carriers_by_depth"), "lost_types": v.summary.get("lost_types"),
                                                "named": {t: {kk: r[kk] for kk in ("depth", "z", "verdict", "stim_mean", "ctrl_level", "diff") if kk in r}
                                                          for t, r in ((row["type"], row) for row in v.tables.get("per_type", []) if row["type"] in NAMED)}}
                             for k, v in traces.items()},
                   "decompose": {k: {"arms": v.summary.get("arms"), "dynamic": v.summary.get("dynamic"), "arm_weights": v.summary.get("arm_weights"),
                                     "top_groups": v.table("per_type").head(12)[["post_type", "pre_group", "rest_mean", "ccw_mean", "cw_mean", "ccw_z", "cw_z"]].to_dict("records") if len(v.table("per_type")) else []}
                                 for k, v in decs.items()},
                   "files": files, "wall_s": round(time.time() - t0, 1)}
    res.files = {"generator": "scripts/interp_apply_rotation.py analyse " + " ".join(sys.argv[2:]), "recordings": stems, **files}
    path = out / "summary.json"; res.save(path)
    problems = res.check()
    if problems:
        print("CHECK: " + "; ".join(problems))
    (out / "summary.md").write_text(summary_markdown(res), encoding="utf-8")
    print(f"written {path} (+ summary.md, {len(files)} tool files) in {time.time() - t0:.0f} s")
    return 0


def _f(v, spec: str = "+.2f") -> str:
    """Format a number that may be None / NaN (JSON round trips turn NaN into null)."""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "nan"
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def summary_markdown(res: Result) -> str:
    s = res.summary
    L = [f"## rotation: condition `{s['condition']}`, mode `{s['mode']}`, gains {s['gains']}, {s['n_runs']} runs (p floor {s['p_floor']:.4f})", ""]
    L += ["| phase | bump drift w/s (mean +- sd; per run) | net wedges | vs | peak Hz | heading deg/s |", "|---|---|---|---|---|---|"]
    for ph, b in s["bump"].items():
        d = b["drift_wedges_per_s"]; h = b["heading_rate_dps"]
        L.append(f"| {ph} | {_f(d['mean'], '+.4f')} +- {_f(d['sd'], '.4f')} ({', '.join(_f(v, '+.3f') for v in d['values'])}) | {_f(b['net_wedges']['mean'], '+.3f')} | {_f(b['vs']['mean'])} | "
                 f"{_f(b['peak_hz']['mean'], '.0f')} | {_f(h['mean'], '+.1f')} +- {_f(h['sd'], '.1f')} |")
    bd = s['bump_drift_vs_rest']
    L += ["", f"ideal drift {s['ideal_wedges_per_s']:+.3f} w/s; ccw vs rest: z {_f(bd['ccw']['z'])} p {_f(bd['ccw']['p'], '.3f')} {bd['ccw']['verdict']}; "
          f"cw: z {_f(bd['cw']['z'])} p {_f(bd['cw']['p'], '.3f')} {bd['cw']['verdict']}", ""]
    L += ["| type | flip Hz (mean +- sd) | null (mean +- sd) | z | U | p | verdict | L-R rest / ccw / cw | rate rest Hz |", "|---|---|---|---|---|---|---|---|---|"]
    for t, f in s["flip_named"].items():
        L.append(f"| {t} | {_f(f['flip_mean'], '+.3f')} +- {_f(f['flip_sd'], '.3f')} | {_f(f['null_mean'], '+.3f')} +- {_f(f['null_sd'], '.3f')} | {_f(f['z'])} | {_f(f['U'], '.0f')} | {_f(f['p'], '.3f')} | {f['verdict']} | "
                 f"{_f(f['LR_rest'])} / {_f(f['LR_ccw'])} / {_f(f['LR_cw'])} | {_f(f['rate_rest'])} |")
    L += ["", f"flip results (|z| >= 3, p <= 0.05): {s['flip_results']}", f"flip top 15 by |z|: {s['flip_top']}", ""]
    for k, v in s.get("trace", {}).items():
        L.append(f"trace {k}: carriers {v['n_carriers']}, by depth {v['carriers_by_depth']}, first lost depth {v['first_lost_depth']}; named: " +
                 "; ".join(f"{t} d{r.get('depth')} z {_f(r.get('z'), '+.1f')} {r.get('verdict')}" for t, r in v["named"].items()))
    L.append("")
    for k, v in s.get("decompose", {}).items():
        L.append(f"decompose {k}: arm weights {v.get('arm_weights')}")
        L += ["| post | pre group | rest mV/s | ccw | cw | ccw z | cw z |", "|---|---|---|---|---|---|---|"]
        for r in v["top_groups"]:
            L.append(f"| {r['post_type']} | {r['pre_group']} | {_f(r['rest_mean'], '+.1f')} | {_f(r['ccw_mean'], '+.1f')} | {_f(r['cw_mean'], '+.1f')} | {_f(r['ccw_z'])} | {_f(r['cw_z'])} |")
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------- chain (CPU)
CHAIN_TYPES = ["PS196_b", "LAL184", "WED040_a", "CB2037", "LPsP", "AN07B037_a", "PS099_a", "PS262", "PS047_b", "LAL139", "WED153",
               "GLNO", "DNa02", "HSN", "HSE", "Nod1", "DNp20", "LPT26", "PEN_a(PEN1)", "PEN_b(PEN2)", "EPG", "Delta7"]


def chain_rates(c, cells: dict, types=CHAIN_TYPES) -> pd.DataFrame:
    """Per (phase, type): the L and R mean rate over runs (Hz; the per-cell time-means of the trace recordings pooled
    per side, then mean +- sd over runs) and the maximum single-cell rate over all cells and runs -- the table that
    says whether a chain cell fires at all in a phase (docs/audits/deficit_rotation.md 2.2)."""
    n = c.neurons; ty = n.type.fillna("").to_numpy(); side = n.somaSide.fillna("?").to_numpy()
    idx0 = np.asarray(cells["rest"][0].idx)
    rows = []
    for ph, recs in cells.items():
        X = np.stack([np.asarray(r.quantities["rate_hz"][0], float) for r in recs])       # (runs, cells)
        for t in types:
            m = ty[idx0] == t
            if not m.any():
                continue
            row = {"phase": ph, "type": t, "n_runs": len(recs), "n_cells": int(m.sum())}
            for s in ("L", "R"):
                ms = m & (side[idx0] == s)
                v = X[:, ms].mean(axis=1) if ms.any() else np.full(len(recs), np.nan)
                row[s] = float(np.nanmean(v)); row[f"{s}_sd"] = float(np.nanstd(v, ddof=1)) if len(recs) > 1 else float("nan")
                row[f"{s}_runs"] = [float(x) for x in v]
            row["max_cell"] = float(np.nanmax(X[:, m]))
            rows.append(row)
    return pd.DataFrame(rows)


def cmd_chain(args) -> int:
    """The per-phase L / R rates of the chain cells over the runs of every (condition, mode) matched by --runs."""
    frames = []
    groups = {}                                      # (condition, mode) -> [stems]: every stem is keyed by its own meta, so patterns may span arms and batches
    for pat in args.runs:
        p = pat if any(ch in pat for ch in "*?[") else pat + "*"
        stems = sorted({f[: -len("_rest.json")] for f in glob.glob(p + "_rest.json")})
        if not stems:
            print(f"no runs match {pat}"); continue
        for s in stems:
            with open(f"{s}_rest.json", encoding="utf-8") as f:
                m = json.load(f).get("meta", {})
            groups.setdefault((m.get("condition", "default"), m.get("mode", "visual")), []).append(s)
    conns = {}
    for (cond, mode), stems in groups.items():
        _, cells, _pen = phase_recordings(stems)
        if cond not in conns:
            conns[cond] = load_condition_connectome(cond, args.cache_dir, verbose=False)[0]
        df = chain_rates(conns[cond], cells, types=args.types.split(",") if args.types else CHAIN_TYPES)
        df.insert(0, "mode", mode); df.insert(0, "cond", cond); df["runs"] = ",".join(s.replace("\\", "/") for s in stems)
        frames.append(df)
    if not frames:
        raise SystemExit("no runs")
    out = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    common.print_table(out[["cond", "mode", "phase", "type", "n_cells", "L", "L_sd", "R", "R_sd", "max_cell"]], max_rows=400)
    print(f"written {args.out}")
    return 0


# ---------------------------------------------------------------------------------------------- verify-batch (CPU)
def verify_batch(directory: str) -> pd.DataFrame:
    """Per (run, phase) of a fetched batch directory: do the `_run.json`, the recording meta, the `_pen` meta and the
    console `.txt` agree on the phase's bump drift and heading (they must: all four are written by ONE job)? A
    mismatch means the directory holds files of two different jobs (two fetches of same-code batches interleaving,
    docs/audits/deficit_rotation.md 2.1). Also records the run directory each console names."""
    rows = []
    for rj in sorted(glob.glob(os.path.join(directory, "*_run.json"))):
        stem = rj[: -len("_run.json")]
        d = json.load(open(rj, encoding="utf-8"))
        txt = Path(stem + ".txt").read_text(encoding="utf-8", errors="replace") if Path(stem + ".txt").exists() else ""
        rdirs = sorted(set(re.findall(r"runs/(rot-[a-z0-9]+)", txt)))
        for ph in (p for p, _ in PHASES):
            m = json.load(open(f"{stem}_{ph}.json", encoding="utf-8"))["meta"]
            mp = json.load(open(f"{stem}_{ph}_pen.json", encoding="utf-8"))["meta"]
            a = d["phases"][ph]["bump"]["drift_wedges_per_s"]; h = d["phases"][ph]["heading"]["rate_dps"]
            mt = re.search(rf"\n  {ph} \(.*?drift ([+-]\d+\.\d+) w/s.*?heading ([+-]\d+\.\d+) deg/s", txt)
            ok = (abs(a - m["bump"]["drift_wedges_per_s"]) < 1e-9 and abs(a - mp["bump"]["drift_wedges_per_s"]) < 1e-9
                  and mt is not None and abs(float(mt.group(1)) - a) < 0.0015 and abs(float(mt.group(2)) - h) < 0.06)
            rows.append({"run": Path(stem).name, "phase": ph, "device": d.get("device"), "run_dirs_in_console": ",".join(rdirs),
                         "drift_run_json": a, "drift_recording": m["bump"]["drift_wedges_per_s"], "drift_pen": mp["bump"]["drift_wedges_per_s"],
                         "drift_console": float(mt.group(1)) if mt else float("nan"), "heading_run_json": h,
                         "heading_console": float(mt.group(2)) if mt else float("nan"), "consistent": bool(ok)})
    return pd.DataFrame(rows)


def cmd_verify_batch(args) -> int:
    for d in args.dirs:
        df = verify_batch(d)
        n_bad = int((~df.consistent).sum()) if len(df) else 0
        print(f"{d}: {len(df)} (run, phase) files checked, {n_bad} inconsistent; devices {sorted(set(df.device))}; "
              f"run dirs named by the consoles {sorted(set(df.run_dirs_in_console))}")
        if n_bad:
            common.print_table(df[~df.consistent], max_rows=100)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(args.out) / f"verify_{Path(d.rstrip('/')).name}.csv", index=False)
    return 0


# ---------------------------------------------------------------------------------------------- report (CPU)
def cmd_report(args) -> int:
    files = sorted(f for pat in args.summaries for f in glob.glob(pat))
    if not files:
        raise SystemExit("no summary JSONs")
    rows = []; flips = []
    for f in files:
        r = Result.load(f); s = r.summary
        for ph, b in s["bump"].items():
            rows.append({"condition": s["condition"], "mode": s["mode"], "phase": ph, "n": s["n_runs"], "drift_mean": b["drift_wedges_per_s"]["mean"], "drift_sd": b["drift_wedges_per_s"]["sd"],
                         "drift_runs": ", ".join(f"{v:+.3f}" for v in b["drift_wedges_per_s"]["values"]), "vs": b["vs"]["mean"], "peak_hz": b["peak_hz"]["mean"],
                         "heading_dps": b["heading_rate_dps"]["mean"], "heading_sd": b["heading_rate_dps"]["sd"]})
        for t, v in s["flip_named"].items():
            flips.append({"condition": s["condition"], "mode": s["mode"], "type": t, **{k: v[k] for k in ("flip_mean", "flip_sd", "null_sd", "z", "p", "verdict", "rate_rest")}})
    b = pd.DataFrame(rows); fl = pd.DataFrame(flips)
    print("== bump drift (wedges / s) and realised heading per (condition, mode, phase)")
    common.print_table(b, max_rows=80)
    print("\n== named flips per (condition, mode)")
    fl["arm"] = fl.condition + "/" + fl["mode"]
    piv = fl.pivot(index="type", columns="arm", values="z")
    piv = piv.reindex([t for t in NAMED if t in piv.index])
    print(piv.to_string(float_format=lambda v: f"{v:+.2f}"))
    piv2 = fl.pivot(index="type", columns="arm", values="flip_mean").reindex([t for t in NAMED if t in piv.index])
    print("\n(flip Hz)")
    print(piv2.to_string(float_format=lambda v: f"{v:+.3f}"))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        b.to_csv(Path(args.out) / "report_bump.csv", index=False); fl.to_csv(Path(args.out) / "report_flip.csv", index=False)
        print(f"written {args.out}/report_bump.csv, report_flip.csv")
    return 0


# ---------------------------------------------------------------------------------------------- selftest (CPU)
def cmd_selftest(args) -> int:
    """The flip and bump statistics on synthetic data (no connectome, no simulation): a planted L - R flip on one type
    comes out with verdict 'result' over 5 runs and its null-only neighbour 'null'; a bump moving at 4 wedges / s is read
    back as +4.000; a bump crossing the wrap is unwrapped."""
    rng = np.random.default_rng(0)
    # a fake connectome with the attributes flip_table reads
    class N:  # noqa: D401
        pass
    types = np.array(["HSN", "HSN", "GLNO", "GLNO", "PEN", "PEN", "T4", "T4"])
    sides = np.array(["L", "R", "L", "R", "L", "R", "L", "R"])
    n = pd.DataFrame({"type": types, "somaSide": sides, "bodyId": np.arange(8) + 100, "superclass": ["visual_projection"] * 6 + ["ol_intrinsic"] * 2})
    c = N(); c.n = 8; c.neurons = n
    orig = common.unit_kinds
    common.unit_kinds = lambda cc, fb=None: np.array(["spiking"] * 6 + ["graded"] * 2, dtype=object)
    try:
        def rec(phase, k):
            x = rng.normal(5.0, 0.3, 8); x[6:] = np.nan
            dr = np.full(8, np.nan); dr[6:] = rng.normal(0.0, 0.01, 2)
            if phase == "ccw":
                x[0] += 3.0; dr[6] += 0.05
            if phase == "cw":
                x[1] += 3.0; dr[7] += 0.05
            return Recording(np.array([0.0]), np.arange(8), n.bodyId.to_numpy(), types, {"rate_hz": x[None].astype(np.float32), "optic_dr": dr[None].astype(np.float32)}, {}, {"seed": k, "arm": phase})
        runs = {ph: [rec(ph, k) for k in range(5)] for ph in ("rest", "ccw", "rest2", "cw")}
        df = flip_table(c, runs).set_index("type")
        assert df.loc["HSN", "verdict"] == "result" and df.loc["HSN", "flip_mean"] > 5, df.loc["HSN"]
        assert df.loc["GLNO", "verdict"] == "null", df.loc["GLNO"]
        assert df.loc["T4", "verdict"] == "result" and df.loc["T4", "unit_kind"] == "graded", df.loc["T4"]
        print(f"flip_table: HSN flip {df.loc['HSN', 'flip_mean']:+.2f} z {df.loc['HSN', 'z']:+.1f} {df.loc['HSN', 'verdict']}; GLNO {df.loc['GLNO', 'flip_mean']:+.2f} {df.loc['GLNO', 'verdict']}; "
              f"T4 (graded) {df.loc['T4', 'flip_mean']:+.3f} {df.loc['T4', 'verdict']}  ok")
    finally:
        common.unit_kinds = orig
    # bump metrics: a bump at 4 wedges / s across the wrap
    track = []
    for k in range(1000):
        cen = (13.0 + 4.0 * k * 0.01) % 16
        prof = np.exp(-0.5 * ((np.arange(16) - cen + 8) % 16 - 8) ** 2 / 1.5) * 200
        cc, vs = circ_centre(prof); track.append((cc, vs, prof.max()))
    b = bump_metrics(track, 300)
    assert abs(b["drift_wedges_per_s"] - 4.0) < 0.01, b
    assert abs(b["net_wedges"] - 4.0 * 6.99) < 0.1, b
    h = heading_metrics(np.deg2rad(90.0) * np.arange(1000) * 0.01, 300)
    assert abs(h["rate_dps"] - 90.0) < 0.01, h
    print(f"bump_metrics: drift {b['drift_wedges_per_s']:+.3f} w/s (planted +4.000), net {b['net_wedges']:+.2f}, vs {b['vs']:.2f}; heading {h['rate_dps']:+.1f} deg/s  ok")
    print("selftest ok")
    return 0


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paths", help="CPU: the yaw sources -> PEN / GLNO walks")
    p.add_argument("--condition", default="default", choices=["default", "gaba"])
    p.add_argument("--source", action="append", default=[], help="a source spec (repeatable; default the four YAW_SOURCES groups)")
    p.add_argument("--k", type=int, default=3); p.add_argument("--top", type=int, default=12); p.add_argument("--max-rows", type=int, default=30)
    p.add_argument("--recording", default=None, help="a batch recording (npz/json) whose max rates give paths' never_firing flag")
    p.add_argument("--out", default="out/interp/apply_rotation/paths")
    common.add_common_args(p)
    r = sub.add_parser("record", help="GPU: one run of the rotation protocol")
    r.add_argument("--condition", default="default", choices=["default", "gaba"])
    r.add_argument("--mode", default="visual", choices=["visual", "efferent"])
    r.add_argument("--gains", default=GAINS_DEFAULT, help="gE:gD compass gains (default 2:15); 'none' = the shipped ring (no bump)")
    r.add_argument("--seconds", type=float, default=SECONDS); r.add_argument("--skip", type=float, default=SKIP_S); r.add_argument("--rate", type=float, default=RATE_DPS)
    r.add_argument("--dna02-hz", type=float, default=DNA02_HZ, help="efferent mode: DNa02 pulse rate of the turning side")
    r.add_argument("--sparse", default="warp", choices=["warp", "torch"])
    r.add_argument("--out", required=True, help="output stem: <out>_<phase>.npz / _series.npz / _pen.npz and <out>_run.json")
    r.add_argument("--quick", action="store_true"); r.add_argument("--allow-cpu", action="store_true")
    r.add_argument("--no-series", action="store_true"); r.add_argument("--series-every", type=int, default=5)
    common.add_common_args(r)
    a = sub.add_parser("analyse", help="CPU: runs of one (condition, mode) -> trace / flip / decompose / bump")
    a.add_argument("--runs", required=True, nargs="+", help="stem prefix(es) / glob(s) of the runs, e.g. out/rot_cf0c43/default_visual_r* out/rot_7de91e/default_visual_r* (several batches pool)")
    a.add_argument("--out", required=True, help="output directory")
    a.add_argument("--condition", default="default")
    a.add_argument("--stat", default="best_cell", choices=["best_cell", "mean", "dprime"])
    a.add_argument("--min-share", type=float, default=0.02); a.add_argument("--depth-max", type=int, default=8)
    a.add_argument("--window", default=f"{SKIP_S},{SECONDS}"); a.add_argument("--top", type=int, default=40); a.add_argument("--max-rows", type=int, default=40)
    a.add_argument("--skip-trace", action="store_true"); a.add_argument("--skip-decompose", action="store_true")
    a.add_argument("--allow-cpu-runs", action="store_true", help="analyse runs that realised device cpu (smoke tests only)")
    common.add_common_args(a)
    q = sub.add_parser("report", help="CPU: summaries side by side")
    q.add_argument("--summaries", nargs="+", required=True); q.add_argument("--out", default=None)
    ch = sub.add_parser("chain", help="CPU: per-phase L / R rates of the chain cells over the runs (chain_rates.csv)")
    ch.add_argument("--runs", nargs="+", required=True, help="stem prefixes / globs, one per (condition, mode)")
    ch.add_argument("--types", default=None, help="comma-separated types (default CHAIN_TYPES)")
    ch.add_argument("--out", default="out/interp/apply_rotation/chain_rates.csv")
    common.add_common_args(ch)
    vb = sub.add_parser("verify-batch", help="CPU: are a fetched batch directory's run.json / recordings / consoles from one job per run?")
    vb.add_argument("dirs", nargs="+"); vb.add_argument("--out", default=None)
    sub.add_parser("selftest", help="CPU: the statistics on synthetic data")
    args = ap.parse_args(argv)
    return {"paths": cmd_paths, "record": cmd_record, "analyse": cmd_analyse, "report": cmd_report, "chain": cmd_chain,
            "verify-batch": cmd_verify_batch, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
