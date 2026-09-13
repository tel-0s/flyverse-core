"""apply:object -- where between the medulla and the lobula is the moving ball lost, and what kind of fact is the loss?

The interpretability toolkit (docs/INTERP.md) applied to the object deficit of docs/audits/object_sweep.md /
optic_measures.md: the medulla carries a 1 cm black ball (Mi1 / Mi4 / Tm3 at z +7 to +29 above the none-vs-none
null), the small-field inputs of LC11 / LC10a (T2, T3, Tm5Y, TmY21, ...) and the LC cells themselves sit at the null.
This script composes the validated tools -- trace (the depth table with the null and the lost-stage decomposition),
decompose (what is in force at that stage), lesion-style counterfactuals over the OpticParams and over per-type input
classes (recorded through the trace's own object protocol), and export (the Neurome run directories) -- and never
edits the model: every counterfactual is an OpticParams override or an in-process edit of the loaded graph /
receptor signs, recorded in the run's meta. Report: docs/audits/deficit_object.md.

    # CPU: the object trace with the null, LC11 / LC10a and their inputs decomposed (out/trv recordings of interp_trace.md)
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py trace --dir out/trv --prefix obj --json out/interp/apply_object/trace_base.json
    # CPU: what is in force at the lost stage (ON / OFF channels, signs, tiers, pair gains, taus, baselines, normalisation)
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py stage --trace out/interp/apply_object/trace_base.json --json out/interp/apply_object/stage.json
    # CPU -> GPU (one cluster batch) -> CPU: the lesion arms
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py plan --out out/apply_object/les --runs 4 --name apobj-les
    sh out/apply_object/les/batch.sh
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py analyse --dir out/apply_object/les --json out/interp/apply_object/lesions.json
    # the size ladder for Neurome (docs/NEUROME_INTERFACE.md section 3): plan -> batch -> one export per size + the summary
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py ladder-plan --out out/apply_object/ladder --runs 5 --name apobj-lad
    sh out/apply_object/ladder/batch.sh
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py ladder --dir out/apply_object/ladder --json out/interp/apply_object/ladder.json
    # CPU: replicate scatter -- the cancellation fractions per run, and every arm's per-cell figure regressed on the deterministic fb0 arm
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py perrun --dir out/apply_object/les --json out/interp/apply_object/perrun.json
    # CPU: the pieces on the synthetic graph
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py selftest

`record` is the only GPU subcommand (one arm of the object protocol under one named lesion) and runs on the cluster
through the batch `plan` writes; the ladder's GPU half is `scripts/interp_export.py record` verbatim.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from flyverse.interp import common                         # noqa: E402
from flyverse.interp import trace as tr                    # noqa: E402
from flyverse.interp.common import Result, compare, to_jsonable   # noqa: E402

OUT_JSON = ROOT / "out" / "interp" / "apply_object"
PHOTORECEPTORS = "type:R1-R6|R7y|R7p|R7d|R7_unclear|R8y|R8p|R8_unclear"
STAT = "best_cell"
# the small-object stage and the two LC types (docs/NEUROME_INTERFACE.md: their per-type input tables)
STAGE_TYPES = ["T3", "T2", "Tm5Y", "TmY21", "LC11", "LC10a"]
LC11_INPUTS = ["T3", "T2", "Tm6", "T2a", "Tm12", "TmY18", "LC11", "Li15"]
LC10A_INPUTS = ["LC10a", "TuTuA_2", "AOTU042", "Tm5Y", "LC9", "LC10c-1", "TmY21", "LC10c-2"]
CHECK_TYPES = ["L1", "L2", "Mi1", "Mi4", "Mi9", "Tm3", "Tm1", "Tm2", "Tm4", "Tm20", "C3", "L5",
               "T3", "T2", "T2a", "Tm6", "Tm12", "TmY18", "Li15", "Tm5Y", "TmY21", "TmY13", "TmY5a", "LC9", "LC10c-1", "LC10c-2",
               "T4c", "T4d", "T5a", "LPLC2", "LC10b", "LC16", "LC4", "LC11", "LC10a"]
# the canonical pathway of each medulla input (Nern 2025 / Shinomiya 2019 families): what a dark object does to each
# is the MEASURED signed figure in the trace table; the label only says which motion channel the type belongs to
PATHWAY = {"L1": "ON (lamina)", "Mi1": "ON", "Tm3": "ON", "Mi4": "ON (T4 delay)", "Mi9": "ON (T4 delay)", "C3": "ON (T4 delay)", "CT1": "ON/OFF (T4/T5 delay)",
           "L2": "OFF (lamina)", "L3": "OFF (lamina)", "Tm1": "OFF", "Tm2": "OFF", "Tm4": "OFF", "Tm9": "OFF (T5 delay)", "L5": "ON (lamina)",
           "Tm20": "L3-fed (OFF-ish)", "Tm5a": "colour (R7/R8)", "Tm5b": "colour", "Tm5c": "colour", "Dm8a": "colour (R7)", "Tm5Y": "colour / small-field",
           "TmY5a": "small-field", "TmY13": "small-field", "TmY21": "small-field", "T2": "small-field", "T2a": "small-field", "T3": "small-field",
           "Tm6": "small-field", "Tm12": "small-field", "TmY18": "small-field", "Li15": "lobula inhibitory", "Pm1": "medulla inhibitory (Pm)",
           "Pm5": "medulla inhibitory (Pm)", "Mi2": "medulla inhibitory", "Li26": "lobula inhibitory", "Li19": "lobula inhibitory",
           "Y3": "Y (medulla-lobula)", "TmY3": "TmY", "Tm32": "Tm", "Tm5Y": "colour / small-field"}
T4T5_ZERO = {t: 0.0 for t in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")}

# ----------------------------------------------------------------------------------------------- the lesion manifest
# Every arm is a diagnosis, not a proposal: an OpticParams override (kinds 'optic'), or an in-process edit of the loaded
# graph and of the receptor signs for ONE class of edges (kind 'edges': pre spec -> post spec x factor; 0 silences the
# class, -1 flips its sign, the normalisation denominators of the rate lobe come from the neuron table and are untouched).
LESIONS = {
    "base": {"kind": "none", "note": "the shipped model (receptor sign/abs, OpticParams defaults)"},
    "fb0": {"kind": "optic", "spec": {"gain_fb": 0.0},
            "note": "spiking -> rate feedback off: the deterministic rate lobe, optic_measures.md 6 (dynamics: is the object masked by the feedback noise?)"},
    "inl1": {"kind": "optic", "spec": {"norm": "l1"},
             "note": "rate-lobe input normalisation l1 instead of l2 (optic_measures.md #8; rate-model artefact: normalisation)"},
    "outl2": {"kind": "optic", "spec": {"out_norm": "l2"},
              "note": "optic -> spiking output normalisation l2 instead of l1 (#13; the LC pooling stage)"},
    "rect": {"kind": "optic", "spec": {"baseline_by_type": dict(T4T5_ZERO, T2=0.0, T3=0.0)},
             "note": "T2 / T3 at operating point 0 (ReLU units), object_sweep.md --rectify-t2t3 now with a null (rate-model artefact: rectification)"},
    "t3_off_held": {"kind": "edges", "pre": "Tm1|Tm4", "post": "T3", "factor": 0.0,
                    "note": "the OFF-channel carriers onto T3 silenced (wiring: does the ON figure then pass T3?)"},
    "t3_on_held": {"kind": "edges", "pre": "Mi1|Tm3", "post": "T3", "factor": 0.0,
                   "note": "the ON-channel carriers onto T3 silenced (wiring: does the OFF figure then pass T3?)"},
    "t3_off_flip": {"kind": "edges", "pre": "Tm1|Tm4", "post": "T3", "factor": -1.0,
                    "note": "COUNTERFACTUAL sign: Tm1 / Tm4 -> T3 made inhibitory. The data say + (exact tier, ACh / nAChR on T3, 0 receptor "
                            "changes on T3's inputs): this arm asks whether the loss is a sign fact, it does not propose a sign"},
}
ARMS = ("stim", "ctrl", "null")

# ----------------------------------------------------------------------------------------------- the size ladder
AHEAD, HALF_SWEEP, EYE_ABOVE_TABLE_M = 0.05, 0.06, 0.0012          # probe_object_sweep's geometry; the eye height it prints
SIZES_DEG = [4.5, 11.4, 20.0, 30.0]                                 # docs/NEUROME_INTERFACE.md section 3


def size_id(deg: float) -> str:
    return f"d{int(round(deg * 10)):03d}"


def ball_radius_for(deg: float, ahead: float = AHEAD) -> float:
    """probe_object_sweep's convention: angular diameter = 2 atan(r / ahead)."""
    return float(round(ahead * np.tan(np.radians(deg) / 2), 6))


def angular_size_from_eye(r: float, ahead: float = AHEAD, eye_above_table: float = EYE_ABOVE_TABLE_M) -> dict:
    """The ball rests on the table (centre r above it) and the eye is `eye_above_table` above it: the exact angular
    diameter from the eye is 2 asin(r / distance) with the distance eye -> centre, plus the centre's elevation."""
    dz = r - eye_above_table
    dist = float(np.sqrt(ahead ** 2 + dz ** 2))
    return {"ball_radius_m": r, "ahead_m": ahead, "centre_above_eye_m": dz, "distance_eye_to_centre_m": dist,
            "angular_diameter_deg_probe": float(2 * np.degrees(np.arctan(r / ahead))),
            "angular_diameter_deg_from_eye": float(2 * np.degrees(np.arcsin(min(r / dist, 1.0)))),
            "centre_elevation_deg": float(np.degrees(np.arctan2(dz, ahead))),
            "azimuth_sweep_deg": float(np.degrees(np.arctan(HALF_SWEEP / ahead)))}


# ----------------------------------------------------------------------------------------------- shared helpers
def load_c(cache_dir=None):
    from flyverse import connectome as cn
    return cn.load(cache_dir=Path(cache_dir), verbose=False) if cache_dir else cn.load(verbose=False)


def edge_selection(c, pre_spec, post_spec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean mask over the stored entries of c.W (CSR order = the order every ReceptorSigns array uses) of the
    block pre_spec -> post_spec, plus the two index sets."""
    pre = common.resolve(c, pre_spec); post = common.resolve(c, post_spec)
    pre_m = np.zeros(c.n, bool); pre_m[pre] = True
    post_m = np.zeros(c.n, bool); post_m[post] = True
    coo = c.W.tocsr().tocoo()
    return pre_m[coo.col] & post_m[coo.row], pre, post


def apply_edge_lesion(c, pre_spec, post_spec, factor: float) -> dict:
    """Multiply the stored entries of the block pre -> post by `factor` in the loaded graph (explicit zeros kept, so the
    sparsity pattern -- and every receptor array aligned with it -- is unchanged). Returns the record of what changed."""
    sel, pre, post = edge_selection(c, pre_spec, post_spec)
    W = c.W.tocsr().copy(); W.data = W.data.copy()
    syn = float(np.abs(W.data[sel]).sum())
    W.data[sel] = W.data[sel] * np.float32(factor)
    c.W = W
    bid = c.neurons.bodyId.to_numpy()
    return {"pre_spec": pre_spec, "post_spec": post_spec, "factor": float(factor), "n_entries": int(sel.sum()),
            "synapses": syn, "n_pre_cells": int(len(pre)), "n_post_cells": int(len(post)),
            "pre_bodies": [str(int(b)) for b in bid[pre][:5000]], "post_bodies": [str(int(b)) for b in bid[post][:5000]],
            "applies_to": "c.W (the rate optic lobe's W_rr / W_rs / W_sr and brain._shaped_weights both read it) and, when a receptor "
                          "model is in force, the fast sign of the same entries (brain._receptor wrapped: a matched entry's table "
                          "sign would otherwise override a flip); the rate lobe's normalisation denominators (in_syn / in_syn_l2 "
                          "of the neuron table) are untouched"}


def install_edge_lesion(pre_spec, post_spec, factor: float, cache_dir=None) -> dict:
    """Load the connectome, edit the block, and make every later `connectome.load()` (room_demo.Sim -> FlyBrain) return
    the edited graph; wrap brain._receptor so the receptor signs of the block follow the factor's sign."""
    from flyverse import connectome as cn, brain
    c = load_c(cache_dir)
    rec = apply_edge_lesion(c, pre_spec, post_spec, factor)
    sel, _, _ = edge_selection(c, pre_spec, post_spec)

    def load(*a, **k):
        return c
    cn.load = load
    orig = brain._receptor

    def _receptor(cc, p, receptor=None, with_counts=False):
        r = orig(cc, p, receptor, with_counts)
        if r is not None and factor <= 0:
            m = sel if cc is c else edge_selection(cc, pre_spec, post_spec)[0]
            fs = np.array(r.fast_sign, copy=True)
            fs[m] = fs[m] * np.float32(0.0 if factor == 0 else -1.0)
            try:
                r.fast_sign = fs
            except Exception:  # noqa: BLE001 -- a frozen dataclass
                object.__setattr__(r, "fast_sign", fs)
            rec["receptor_entries_edited"] = int(m.sum())
        return r
    brain._receptor = _receptor
    return rec


def lesion_record(lid: str) -> dict:
    les = dict(LESIONS[lid]); les["id"] = lid
    return les


# ----------------------------------------------------------------------------------------------- record (GPU)
def cmd_record(args) -> int:
    """One arm (stim / ctrl / null) of the object protocol under one named lesion: scripts/interp_trace.py record's
    code path (the same recordings: per-cell time-means + the pooled series), with the lesion installed first."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    les = lesion_record(args.lesion)
    import interp_trace as it
    optic_kv = [f"{k}={json.dumps(v)}" for k, v in (les.get("spec") or {}).items()] if les["kind"] == "optic" else []
    lif_kv = [f"{k}={json.dumps(v)}" for k, v in (les.get("spec") or {}).items()] if les["kind"] == "lif" else []
    edge_rec = None
    if les["kind"] == "edges":
        if args.device and str(args.device).startswith("cpu"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        it.patch_cache(args.cache_dir)
        edge_rec = install_edge_lesion(les["pre"], les["post"], float(les["factor"]), args.cache_dir)
        les.update(edge_rec)
        print(f"edge lesion {args.lesion}: {les['n_entries']} entries / {les['synapses']:.0f} synapses of {les['pre']} -> {les['post']} x {les['factor']}", flush=True)
    ns = SimpleNamespace(protocol="object", arm=args.arm, out=args.out, seconds=args.seconds, settle=args.settle, quick=args.quick,
                         allow_cpu=args.allow_cpu, no_series=False, series_every=args.series_every, seed=args.seed, device=args.device,
                         cache_dir=args.cache_dir, receptor_model=args.receptor_model, receptor_net_rule=args.receptor_net_rule,
                         receptor_table=args.receptor_table, lif=list(args.lif) + lif_kv, optic=list(args.optic) + optic_kv)
    print(f"interp_apply_object record: lesion {args.lesion} ({les['kind']}: {les.get('spec') or les.get('pre', '')}) arm {args.arm} seed {args.seed}", flush=True)
    it.cmd_record(ns)
    if edge_rec is not None:                                   # written by the brain._receptor wrapper during the run
        les["receptor_entries_edited"] = edge_rec.get("receptor_entries_edited")
    p = Path(args.out).with_suffix(".json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    d["meta"]["lesion"] = to_jsonable(les)
    d["meta"]["run_id"] = f"object-{args.lesion}-{args.arm}-r{args.seed}"
    d["meta"]["generator"] = " ".join(sys.argv)
    prov = d["meta"].get("provenance") or {}
    prov.setdefault("stimulus", {})["lesion"] = to_jsonable({k: v for k, v in les.items() if k not in ("pre_bodies", "post_bodies")})
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    dev = (prov.get("execution") or {}).get("device")
    print(f"lesion {args.lesion} stamped into {p}; realised device {dev}", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------- plan (CPU)
def job_line(lid: str, arm: str, seed: int, out_dir: str, series_every: int = 5) -> str:
    stem = f"{out_dir}/{lid}_{arm}_r{seed}"
    return (f"mkdir -p {out_dir} && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"python scripts/interp_apply_object.py record --lesion {lid} --arm {arm} --seed {seed} --series-every {series_every} "
            f"--out {stem} > {stem}.txt 2>&1; tail -3 {stem}.txt")


def write_batch(path: Path, name: str, minutes: int, jobs: list, fetch: str, header: str) -> None:
    log = f"out/{name}_cluster.log"
    call = (f"python scripts/cluster_run.py --name {name} --minutes {int(minutes)} \\\n  "
            + " \\\n  ".join(shlex.quote(j) for j in jobs) + f" \\\n  --fetch {fetch}/")
    path.write_text("#!/bin/sh\n" + header + f"set -e\nmkdir -p {fetch} out\n"
                    f'if [ -f {log} ]; then mv {log} "{log[:-4]}.$(date +%Y%m%dT%H%M%S).log"; fi\n'
                    f"{call} 2>&1 | tee {log}\n", encoding="utf-8", newline="\n")


def cmd_plan(args) -> int:
    """Resolve the lesions on the connectome and write ONE cluster batch: lesions x (stim, ctrl, null) x runs."""
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rel = out_dir.relative_to(ROOT).as_posix() if out_dir.is_absolute() and str(out_dir).startswith(str(ROOT)) else args.out.replace("\\", "/").rstrip("/")
    lids = [x for x in args.lesions.split(",") if x] if args.lesions else list(LESIONS)
    c = load_c(args.cache_dir)
    resolved = []
    for lid in lids:
        les = lesion_record(lid)
        if les["kind"] == "edges":
            sel, pre, post = edge_selection(c, les["pre"], les["post"])
            W = c.W.tocsr()
            les.update({"n_entries": int(sel.sum()), "synapses": float(np.abs(W.data[sel]).sum()), "n_pre_cells": int(len(pre)),
                        "n_post_cells": int(len(post)), "share_of_post_raw_input": float(np.abs(W.data[sel]).sum() / max(np.abs(W[post].data).sum(), 1))})
        resolved.append(les)
    jobs = [job_line(lid, arm, s, rel, args.series_every) for lid in lids for arm in ARMS for s in range(args.runs)]
    header = (f"# generated by scripts/interp_apply_object.py plan on {time.strftime('%Y-%m-%d %H:%M')}: {len(jobs)} jobs = "
              f"{len(lids)} lesions x {len(ARMS)} arms x {args.runs} runs (the replicate unit is the job)\n")
    write_batch(out_dir / "batch.sh", args.name, args.minutes, jobs, rel, header)
    man = {"lesions": resolved, "arms": list(ARMS), "runs": int(args.runs), "jobs": jobs, "out_dir": rel, "cluster_log": f"out/{args.name}_cluster.log",
           "connectome": common.connectome_fingerprint(c), "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(out_dir / "manifest.resolved.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(to_jsonable(man), f, indent=1)
    print(f"{len(jobs)} jobs -> {out_dir / 'batch.sh'} (fetch {rel}/, log out/{args.name}_cluster.log)")
    for les in resolved:
        if les["kind"] == "edges":
            print(f"  {les['id']:12s} {les['pre']} -> {les['post']} x {les['factor']}: {les['n_entries']} entries, {les['synapses']:.0f} synapses, "
                  f"{100 * les['share_of_post_raw_input']:.1f} % of the post cells' raw input")
        else:
            print(f"  {les['id']:12s} {les['kind']}: {les.get('spec', '')}")
    return 0


# ----------------------------------------------------------------------------------------------- trace (CPU)
def run_trace(c, stem_dir, prefix, *, decompose_at, ew=None, min_cells=2, quantity=None, per_body="lost", max_lost=12, top_inputs=16):
    stim = tr.load_runs(f"{stem_dir}/{prefix}_stim_r*"); ctrl = tr.load_runs(f"{stem_dir}/{prefix}_ctrl_r*"); nul = tr.load_runs(f"{stem_dir}/{prefix}_null_r*")
    if not stim or not ctrl:
        raise SystemExit(f"no recordings for {prefix} in {stem_dir}: stim {len(stim)} ctrl {len(ctrl)} null {len(nul)}")
    devs = sorted({str(((r.meta.get('provenance') or {}).get('execution') or {}).get('device')) for r in stim + ctrl + nul})
    print(f"[{prefix}] {len(stim)} stimulus, {len(ctrl)} control, {len(nul)} null recordings; devices {devs}", flush=True)
    t0 = time.time()
    res = tr.trace(c, PHOTORECEPTORS, stimulus=stim, control=ctrl, null=nul or None, stat=STAT, stage_table="family",
                   decompose_at=decompose_at, min_cells=min_cells, quantity=quantity, ew=ew, per_body=per_body, max_lost=max_lost, top_inputs=top_inputs)
    res.summary["seconds_cpu"] = round(time.time() - t0, 1)
    return res, stim


def cmd_trace(args) -> int:
    """The object trace with the null on an existing set of recordings (default: docs/audits/interp_trace.md's
    out/trv), decomposed at the small-object stage AND at LC11 / LC10a, plus a second scoring of the spiking types on
    their firing rate (the central-brain inputs of LC10a receive no optic drive: TuTuA_2, AOTU042)."""
    c = load_c(args.cache_dir)
    targets = [t for t in args.decompose_at.split(",") if t]
    res, stim = run_trace(c, args.dir, args.prefix, decompose_at=targets)
    res.files["generator"] = " ".join(sys.argv)
    path = Path(args.json) if args.json else OUT_JSON / f"trace_{args.prefix}.json"
    res.save(path)
    tr.print_trace(res, max_rows=60)
    print(f"written {path}  (check: {res.check() or 'ok'})")
    if not args.no_rate:
        res2, _ = run_trace(c, args.dir, args.prefix, decompose_at=None, quantity="rate_hz", per_body="none")
        res2.files["generator"] = " ".join(sys.argv) + " [spiking scored on rate_hz]"
        p2 = path.with_name(path.stem + "_rate.json")
        res2.save(p2)
        print(f"written {p2} (spiking types scored on rate_hz)")
    return 0


# ----------------------------------------------------------------------------------------------- stage (CPU)
def pair_gain_factor(pair_gain, pre_t: str, post_t: str) -> float:
    f = 1.0
    for a, b, g in (pair_gain or []):
        if re.match(a, pre_t) and re.match(b, post_t):
            f *= float(g)
    return f


def optic_input_shares(c, op: dict, post_types: list, receptor=None, receptor_gain=None) -> pd.DataFrame:
    """The rate lobe's OWN input table (optic.OpticLobe's construction on the CPU, torch-free): per (post type, pre
    type) the mean over post cells of the summed normalised, pair-gained input -- l2 (norm) for rate targets, l1 /
    l2 (out_norm) for spiking targets -- with its sign, its share of the post cells' |input|, the pair-gain factor,
    the presynaptic tau / baseline, and the receptor tier that decided the entries' sign."""
    import scipy.sparse as sp
    from flyverse import connectome as cn
    n = c.neurons
    types = n.type.fillna("").to_numpy()
    is_pr = n.type.isin(cn.PHOTORECEPTOR_TYPES).to_numpy()
    is_rate = (n.superclass == "ol_intrinsic").to_numpy() & ~is_pr
    W = c.W.tocsr()
    if receptor is not None:
        W = W.copy(); W.data = np.abs(W.data) * receptor.fast_factor(receptor_gain)
    tot = n.in_syn.to_numpy(); l2 = n.in_syn_l2.to_numpy()
    norm = op.get("norm", "l2"); out_norm = op.get("out_norm", "l1")
    pair_gain = op.get("pair_gain") if op.get("pair_gain") is not None else __import__("flyverse.optic", fromlist=["x"]).DEFAULT_PAIR_GAIN
    tau_map = op.get("tau_by_type") if op.get("tau_by_type") is not None else __import__("flyverse.optic", fromlist=["x"]).DEFAULT_TAU_BY_TYPE
    bl_map = op.get("baseline_by_type") if op.get("baseline_by_type") is not None else __import__("flyverse.optic", fromlist=["x"]).DEFAULT_BASELINE_BY_TYPE
    tier_names = list(cn.RECEPTOR_TIERS)
    coo_all = W.tocoo()
    rows = []
    for pt in post_types:
        post = np.flatnonzero(types == pt)
        if not len(post):
            continue
        graded = bool(is_rate[post].all())
        denom = (l2 if norm == "l2" else tot) if graded else (l2 if out_norm == "l2" else tot)
        sub = (sp.diags(1.0 / np.maximum(denom[post], 1.0)) @ W[post]).tocoo()
        pre_ty = types[sub.col]
        keep = is_rate[sub.col] | (is_pr[sub.col] & graded)                   # a rate target takes rate + photoreceptor input; a spiking one rate input only
        sub_r, sub_c, sub_d = sub.row[keep], sub.col[keep], sub.data[keep].astype(np.float64)
        pre_ty = pre_ty[keep]
        gains = np.array([pair_gain_factor(pair_gain, a, pt) for a in pd.unique(pre_ty)])
        gmap = dict(zip(pd.unique(pre_ty), gains))
        g = np.array([gmap[a] for a in pre_ty]); sub_d = sub_d * g
        df = pd.DataFrame({"pre_type": pre_ty, "post": sub_r, "v": sub_d, "av": np.abs(sub_d)})
        per_pre = df.groupby("pre_type").agg(v=("v", "sum"), av=("av", "sum"), n_entries=("v", "size")).reset_index()
        per_pre["v"] /= len(post); per_pre["av"] /= len(post)
        total_abs = float(per_pre.av.sum())
        # the receptor tier of the block's entries (the entries of c.W's CSR order that fall in the block)
        tier_of = {}
        if receptor is not None:
            post_m = np.zeros(c.n, bool); post_m[post] = True
            in_post = post_m[coo_all.row]
            pre_all = types[coo_all.col[in_post]]; t_codes = receptor.tier[in_post]; fs = receptor.fast_sign[in_post]
            nt_sign = np.sign(c.W.tocsr().tocoo().data[in_post])
            for a in per_pre.pre_type:
                m = pre_all == a
                if m.any():
                    tc = np.bincount(t_codes[m].astype(np.int64), minlength=len(tier_names))
                    tier_of[a] = {"tier": tier_names[int(np.argmax(tc))], "tier_frac": float(tc.max() / tc.sum()),
                                  "changed_vs_nt_sign": int((fs[m] != nt_sign[m]).sum())}
        for _, r in per_pre.sort_values("av", ascending=False).iterrows():
            a = r.pre_type
            rows.append({"post_type": pt, "post_kind": "graded" if graded else "spiking", "pre_type": a, "pathway": PATHWAY.get(a, ""),
                         "mean_input_per_post": float(r.v), "abs_input_per_post": float(r.av), "share_abs": float(r.av / max(total_abs, 1e-12)),
                         "sign": int(np.sign(r.v)) if r.v != 0 else 0, "n_entries": int(r.n_entries), "pair_gain": float(gmap[a]),
                         "normalisation": (f"norm {norm}" if graded else f"out_norm {out_norm}") + " (denominator from the neuron table)",
                         "pre_tau_ms": float(tau_map.get(a, op.get("tau_ms", 10.0))), "pre_baseline": float(bl_map.get(a, op.get("baseline", 0.5))),
                         "pre_kind": "photoreceptor" if (is_pr[types == a].any() and not is_rate[types == a].any()) else "graded",
                         "receptor_tier": tier_of.get(a, {}).get("tier", "nt_sign" if receptor is None else "nt_sign"),
                         "tier_frac": tier_of.get(a, {}).get("tier_frac", float("nan")),
                         "changed_vs_nt_sign": tier_of.get(a, {}).get("changed_vs_nt_sign", 0),
                         "post_denominator_mean": float(denom[post].mean()), "post_tau_ms": float(tau_map.get(pt, op.get("tau_ms", 10.0))),
                         "post_baseline": float(bl_map.get(pt, op.get("baseline", 0.5))), "n_post": int(len(post))})
    return pd.DataFrame(rows)


def lc_pooling(c, lc_types, op: dict) -> pd.DataFrame:
    """How a spiking LC cell pools the optic lobe: per cell the number of retinal columns with non-zero rate-unit input
    and the |input| share of its best three columns (the untraced 'LC11 pools 94 columns / 30 % from its best three' of
    optic_measures.md 6, generated). Columns from trace.column_of_cells on a CPU retina."""
    import scipy.sparse as sp
    from flyverse import retina as rt, connectome as cn
    n = c.neurons; types = n.type.fillna("").to_numpy()
    is_pr = n.type.isin(cn.PHOTORECEPTOR_TYPES).to_numpy()
    rate_idx = np.flatnonzero((n.superclass == "ol_intrinsic").to_numpy() & ~is_pr)
    r = rt.build_retina(c)
    col, _ = tr.column_of_cells(c, r, rate_idx)
    W = sp.csr_matrix(abs(c.W))
    rows = []
    for t in lc_types:
        cells = np.flatnonzero(types == t)
        ncol, best3, ninp = [], [], []
        for i in cells:
            s0, s1 = W.indptr[i], W.indptr[i + 1]
            pre = W.indices[s0:s1]; w = W.data[s0:s1]
            m = (col[pre] >= 0) & np.isin(pre, rate_idx)
            if not m.any():
                continue
            cw = np.bincount(col[pre[m]], weights=w[m])
            cw = cw[cw > 0]
            ncol.append(len(cw)); best3.append(float(np.sort(cw)[-3:].sum() / cw.sum())); ninp.append(int(m.sum()))
        rows.append({"type": t, "n_cells": int(len(cells)), "cells_with_columns": len(ncol), "columns_median": float(np.median(ncol)) if ncol else float("nan"),
                     "columns_p10": float(np.percentile(ncol, 10)) if ncol else float("nan"), "columns_p90": float(np.percentile(ncol, 90)) if ncol else float("nan"),
                     "best3_share_median": float(np.median(best3)) if best3 else float("nan"), "rate_inputs_median": float(np.median(ninp)) if ninp else float("nan"),
                     "out_norm": op.get("out_norm", "l1"), "gain_out_mv": op.get("gain_out_mv", 100.0), "drive_clip_mv": op.get("drive_clip_mv", 35.0),
                     "in_syn_mean": float(n.in_syn.to_numpy()[cells].mean())})
    return pd.DataFrame(rows)


def cmd_stage(args) -> int:
    """What is in force at the lost stage: the trace's lost_inputs (LIF shares, verdicts, signed figures) joined with the
    rate lobe's own input table (l2 shares, pair gains, taus, baselines, receptor tiers), the E / I and ON / OFF sums,
    and the LC pooling arithmetic. Reads the trace JSON of `trace`; CPU only."""
    from flyverse import brain
    res = Result.load(args.trace)
    c = load_c(args.cache_dir)
    prov = res.provenance
    op = dict((prov.get("model") or {}).get("optic") or {})
    lif = tr.params_from_provenance(prov)
    receptor = brain._receptor(c, lif)
    targets = [t for t in args.targets.split(",") if t]
    per_type = res.table("per_type"); row = {t: r for t, r in zip(per_type.type, per_type.to_dict("records"))} if len(per_type) else {}
    rate_row = {}
    p_rate = Path(args.trace).with_name(Path(args.trace).stem + "_rate.json")
    if p_rate.exists():
        pr = Result.load(p_rate).table("per_type"); rate_row = {t: r for t, r in zip(pr.type, pr.to_dict("records"))}
    lost = res.table("lost_inputs")
    fig_of = {}
    if len(lost):
        fig_of = {(r.target_type, r.pre_type): r for r in lost.itertuples()}
    tab = optic_input_shares(c, op, targets, receptor=receptor, receptor_gain=brain._receptor_gain(lif))
    out_rows = []
    for r in tab.to_dict("records"):
        pre = r["pre_type"]; pt = r["post_type"]
        v = row.get(pre) or {}
        if (not v or v.get("quantity") == "drive_mv" and (v.get("stim_mean") or 0) == 0) and pre in rate_row:
            v = rate_row[pre]                                                    # a central-brain input: its firing rate
        li = fig_of.get((pt, pre))
        r.update({"pre_verdict": v.get("verdict", "unscored"), "pre_z": v.get("z", float("nan")), "pre_stim_mean": v.get("stim_mean", float("nan")),
                  "pre_null_mean": v.get("null_mean", float("nan")), "pre_null_sd": v.get("null_sd", float("nan")), "pre_quantity": v.get("quantity"),
                  "pre_signed_figure": getattr(li, "pre_signed_figure", float("nan")) if li is not None else float("nan"),
                  "share_lif": getattr(li, "share", float("nan")) if li is not None else float("nan")})
        r["term_optic"] = float(r["sign"] * r["share_abs"] * r["pre_signed_figure"]) if np.isfinite(r["pre_signed_figure"]) else float("nan")
        out_rows.append(r)
    table = pd.DataFrame(out_rows)
    # per target: E / I totals, carrier terms by measured sign, the parameters in force
    summary = {}
    for pt in targets:
        g = table[table.post_type == pt]
        if not len(g):
            continue
        gc = g[g.pre_verdict == "result"]
        raising = gc[gc.term_optic > 0]; lowering = gc[gc.term_optic < 0]
        own = row.get(pt) or {}
        summary[pt] = {"verdict": own.get("verdict"), "z": own.get("z"), "stim_mean": own.get("stim_mean"), "null_mean": own.get("null_mean"), "null_sd": own.get("null_sd"),
                       "depth": own.get("depth"), "n_inputs": int(len(g)), "E_share": float(g.share_abs[g.sign > 0].sum()), "I_share": float(g.share_abs[g.sign < 0].sum()),
                       "carrier_share": float(gc.share_abs.sum()), "carriers_raising": list(raising.pre_type), "carriers_lowering": list(lowering.pre_type),
                       "sum_raising": float(raising.term_optic.sum()), "sum_lowering": float(lowering.term_optic.sum()),
                       "cancellation_fraction": float(1 - abs(raising.term_optic.sum() + lowering.term_optic.sum()) /
                                                      max(abs(raising.term_optic.sum()) + abs(lowering.term_optic.sum()), 1e-15)) if len(gc) else float("nan"),
                       "carriers_excitatory_share": float(gc.share_abs[gc.sign > 0].sum()),
                       "pair_gains_on_inputs": {a: f for a, f in zip(g.pre_type, g.pair_gain) if f != 1.0},
                       "receptor_tiers": g.groupby("receptor_tier").share_abs.sum().to_dict(), "entries_changed_vs_nt_sign": int(g.changed_vs_nt_sign.sum()),
                       "post_tau_ms": float(g.post_tau_ms.iloc[0]), "post_baseline": float(g.post_baseline.iloc[0]), "post_denominator_mean": float(g.post_denominator_mean.iloc[0]),
                       "normalisation": g.normalisation.iloc[0], "post_kind": g.post_kind.iloc[0],
                       "pathways_raising": sorted({PATHWAY.get(a, "") for a in raising.pre_type}), "pathways_lowering": sorted({PATHWAY.get(a, "") for a in lowering.pre_type})}
    lc = [t for t in targets if t in ("LC11", "LC10a", "LC10b", "LPLC2", "LC4", "LC16")]
    pool = lc_pooling(c, lc, op) if lc else pd.DataFrame()
    # the LIF-side input table of the spiking targets (the trace's lost_inputs: shaped weights A, every presynaptic
    # kind, so the central-brain inputs of LC10a -- TuTuA_2, AOTU042 -- appear with their firing-rate verdicts)
    lif_rows = []
    if len(lost):
        for r in lost[lost.target_type.isin(lc)].itertuples():
            v = row.get(r.pre_type) or {}
            if (not v or (v.get("quantity") == "drive_mv" and not v.get("stim_mean"))) and r.pre_type in rate_row:
                v = rate_row[r.pre_type]
            lif_rows.append({"post_type": r.target_type, "pre_type": r.pre_type, "pathway": PATHWAY.get(r.pre_type, ""), "share_lif": r.share, "sign": r.sign,
                             "mv_per_volley": r.mv_per_volley, "raw_synapses_per_post": r.raw_synapses_per_post, "pre_depth": r.pre_depth,
                             "pre_quantity": v.get("quantity"), "pre_n_cells": v.get("n_cells"), "pre_stim_level": v.get("stim_level"), "pre_ctrl_level": v.get("ctrl_level"),
                             "pre_stim_mean": v.get("stim_mean"), "pre_null_mean": v.get("null_mean"), "pre_null_sd": v.get("null_sd"), "pre_z": v.get("z"),
                             "pre_verdict": v.get("verdict", "unscored"), "pre_signed_figure": r.pre_signed_figure})
    lif_table = pd.DataFrame(lif_rows)
    out = Result.new("decompose", prov)
    out.add_table("stage_inputs", table)
    out.add_table("lif_inputs", lif_table)
    out.add_table("lc_pooling", pool)
    out.summary = {"targets": targets, "per_target": summary, "optic_params_in_force": op, "receptor_model": lif.receptor_model, "receptor_net_rule": lif.receptor_net_rule,
                   "trace": str(args.trace), "rate_trace": str(p_rate) if p_rate.exists() else None}
    out.files = {"generator": " ".join(sys.argv), "trace": str(args.trace)}
    out.validation = dict(out.validation, status="not run", measured={"note": "a composition of trace + the rate lobe's own input table; the decompose validation targets are in interp_decompose.md"})
    path = Path(args.json) if args.json else OUT_JSON / "stage.json"
    out.save(path)
    for pt in targets:
        g = table[table.post_type == pt].head(args.top)
        s = summary.get(pt, {})
        print(f"\n== {pt}: verdict {s.get('verdict')} z {s.get('z')}; E share {s.get('E_share', 0):.2f} / I {s.get('I_share', 0):.2f}; carriers {s.get('carrier_share', 0):.2f} "
              f"(raising {s.get('carriers_raising')} {s.get('sum_raising', 0):+.2e} vs lowering {s.get('carriers_lowering')} {s.get('sum_lowering', 0):+.2e}, "
              f"cancellation {s.get('cancellation_fraction', float('nan')):.2f}); tau {s.get('post_tau_ms')} ms, baseline {s.get('post_baseline')}, {s.get('normalisation')}, "
              f"pair gains {s.get('pair_gains_on_inputs')}, tiers {s.get('receptor_tiers')}")
        cols = ["pre_type", "pathway", "share_abs", "share_lif", "sign", "pair_gain", "pre_tau_ms", "pre_baseline", "receptor_tier", "pre_verdict", "pre_z", "pre_signed_figure", "term_optic"]
        common.print_table(g[cols], floatfmt="{:+.4f}")
    if len(lif_table):
        print("\nLIF-side inputs of the LC targets (the trace's lost_inputs; central-brain inputs scored on rate_hz):")
        common.print_table(lif_table[["post_type", "pre_type", "share_lif", "sign", "mv_per_volley", "pre_quantity", "pre_n_cells", "pre_stim_level", "pre_ctrl_level",
                                      "pre_stim_mean", "pre_null_mean", "pre_null_sd", "pre_z", "pre_verdict"]], floatfmt="{:+.4f}", max_rows=40)
    if len(pool):
        print("\nLC pooling (columns per cell, best-3 share):")
        common.print_table(pool, floatfmt="{:.3f}")
    print(f"written {path}")
    return 0


# ----------------------------------------------------------------------------------------------- analyse (CPU)
def cmd_analyse(args) -> int:
    """Every lesion arm's recordings -> its own trace (the same statistic and null as the baseline) -> the type x lesion
    matrix with scatter, the Neurome `sensitivity` table and the 'restores' calls; Result of tool 'lesion'."""
    c = load_c(args.cache_dir)
    d = Path(args.dir)
    lids = [x for x in args.lesions.split(",") if x] if args.lesions else [l for l in LESIONS if glob.glob(f"{d}/{l}_stim_r*.json")]
    traces, provs = {}, {}
    ew = None
    for lid in lids:
        if not glob.glob(f"{d}/{lid}_stim_r*.json"):
            print(f"[{lid}] no recordings; skipped"); continue
        dec = [t for t in args.decompose_at.split(",") if t] if (lid == args.baseline or args.decompose_all) else None
        res, stim = run_trace(c, d, lid, decompose_at=dec, ew=ew, per_body="lost" if lid == args.baseline else "none")
        if ew is None:
            ew = common.effective_weights(c, tr.params_from_provenance(stim[0].meta.get("provenance") or {}))
        res.files["generator"] = " ".join(sys.argv)
        res.summary["lesion"] = to_jsonable({k: v for k, v in (stim[0].meta.get("lesion") or {}).items() if k not in ("pre_bodies", "post_bodies")})
        p = OUT_JSON / f"trace_{lid}.json"
        res.save(p)
        traces[lid] = res; provs[lid] = stim[0].meta.get("provenance") or {}
        pt = res.table("per_type")
        heads = pt[pt.type.isin(["Mi4", "Mi1", "Tm3", "T3", "T2", "Tm5Y", "TmY21", "LC11", "LC10a", "LPLC2"])]
        print("  " + "; ".join(f"{r.type} {r.stim_mean:.4f} vs null {r.null_mean:.4f}+-{r.null_sd:.4f} z {r.z:+.1f} {r.verdict}" for r in heads.itertuples()))
        print(f"  -> {p} ({res.summary.get('seconds_cpu')} s)", flush=True)
    if args.baseline not in traces:
        raise SystemExit(f"baseline arm {args.baseline!r} has no recordings in {d}")
    base = traces[args.baseline].table("per_type"); brow = {t: r for t, r in zip(base.type, base.to_dict("records"))}
    rows, sens = [], []
    for lid, res in traces.items():
        pt = res.table("per_type"); prow = {t: r for t, r in zip(pt.type, pt.to_dict("records"))}
        les = res.summary.get("lesion") or {}
        for t in CHECK_TYPES:
            r = prow.get(t); b = brow.get(t)
            if r is None:
                continue
            delta = float(r["stim_mean"] - b["stim_mean"]) if b else float("nan")
            pooled = float(np.sqrt((r["stim_sd"] ** 2 + b["stim_sd"] ** 2) / 2)) if b else float("nan")
            moved = ("moved" if abs(delta) > 2 * pooled else "not moved" if abs(delta) <= pooled else "unclear") if (b and np.isfinite(pooled) and pooled > 0) else "n/a"
            rows.append({"type": t, "lesion": lid, "lesion_kind": les.get("kind"), "check": f"{t}.{STAT}", "quantity": r["quantity"], "unit_kind": r["unit_kind"],
                         "depth": r["depth"], "n_cells": r["n_cells"], "n_runs": int(len(r["stim_values"])), "stim_values": r["stim_values"], "stim_mean": r["stim_mean"],
                         "stim_sd": r["stim_sd"], "null_values": r["null_values"], "null_mean": r["null_mean"], "null_sd": r["null_sd"], "z": r["z"], "welch": r["welch"],
                         "U": r["U"], "p": r["p"], "p_floor": r["p_floor"], "verdict": r["verdict"], "note": r.get("note"),
                         "base_mean": b["stim_mean"] if b else float("nan"), "base_sd": b["stim_sd"] if b else float("nan"), "base_verdict": b["verdict"] if b else None,
                         "base_null_mean": b["null_mean"] if b else float("nan"), "base_null_sd": b["null_sd"] if b else float("nan"),
                         "delta_vs_base": delta, "pooled_sd": pooled, "moved_vs_base": moved,
                         "z_vs_base_null": float((r["stim_mean"] - b["null_mean"]) / b["null_sd"]) if (b and b["null_sd"] > 0) else float("nan"),
                         "restores": bool(r["verdict"] == "result" and b is not None and b["verdict"] != "result"),
                         "loses": bool(r["verdict"] != "result" and b is not None and b["verdict"] == "result")})
            if lid != args.baseline and b is not None:
                sens.append({"lesion_id": lid, "lesion_kind": les.get("kind"), "lesion_spec": json.dumps(to_jsonable(les.get("spec") or {k: les.get(k) for k in ("pre", "post", "factor") if k in les})),
                             "bodies": f"{les.get('n_pre_cells', '')} pre / {les.get('n_post_cells', '')} post cells" if les.get("kind") == "edges" else "",
                             "check": f"{t}.{STAT}", "baseline": b["stim_mean"], "value": r["stim_mean"], "delta": delta, "replicate_sd": r["stim_sd"],
                             "n_replicates": int(len(r["stim_values"])), "replicate_values": r["stim_values"], "null_mean": r["null_mean"], "null_sd": r["null_sd"],
                             "z": r["z"], "verdict": r["verdict"], "restores": rows[-1]["restores"], "loses": rows[-1]["loses"]})
    mat = pd.DataFrame(rows); sd = pd.DataFrame(sens)
    prov = dict(provs[args.baseline]); prov["analysis"] = {"flyverse_commit": common.git_state(), "lesions": {lid: provs[lid].get("stimulus", {}).get("lesion") for lid in traces},
                                                           "devices": sorted({str((provs[l].get("execution") or {}).get("device")) for l in traces}),
                                                           "traces": {lid: str(OUT_JSON / f"trace_{lid}.json") for lid in traces}}
    out = Result.new("lesion", prov)
    out.add_table("matrix", mat); out.add_table("sensitivity", sd)
    restores = mat[mat.restores]; loses = mat[mat.loses]
    out.summary = {"lesions": list(traces), "baseline": args.baseline, "stat": STAT, "check_types": CHECK_TYPES,
                   "restores": [{"lesion": r.lesion, "type": r.type, "z": r.z, "stim_mean": r.stim_mean, "null_mean": r.null_mean, "null_sd": r.null_sd, "p": r.p, "note": r.note} for r in restores.itertuples()],
                   "loses": [{"lesion": r.lesion, "type": r.type, "z": r.z} for r in loses.itertuples()],
                   "n_runs": {lid: int(traces[lid].summary.get("n_stim_runs", 0)) for lid in traces}, "p_floor": {lid: traces[lid].summary.get("p_floor") for lid in traces},
                   "carriers_by_depth": {lid: traces[lid].summary.get("counts_by_depth") for lid in traces}}
    out.replicates = {"n": int(traces[args.baseline].summary.get("n_stim_runs", 0)), "unit": "runs", "runs": traces[args.baseline].replicates.get("runs"),
                      "null": traces[args.baseline].replicates.get("null")}
    out.validation = dict(out.validation, measured={"note": "an application of the lesion pattern to the object protocol; the lesion tool's own validation is interp_lesion.md"}, status="not run")
    out.files = {"generator": " ".join(sys.argv), "recordings_dir": str(d), "traces": {lid: str(OUT_JSON / f"trace_{lid}.json") for lid in traces}}
    path = Path(args.json) if args.json else OUT_JSON / "lesions.json"
    out.save(path)
    print_matrix(mat, list(traces))
    print(f"\nrestores (verdict result where the baseline is not): {len(restores)} rows; loses: {len(loses)} rows")
    for r in restores.itertuples():
        print(f"  {r.lesion:12s} {r.type:8s} z {r.z:+.1f} stim {r.stim_mean:.4f} null {r.null_mean:.4f} +- {r.null_sd:.4f} p {r.p:.4f} {r.note or ''}")
    print(f"written {path}  (check: {out.check() or 'ok'})")
    return 0


def print_matrix(mat: pd.DataFrame, lids: list) -> None:
    print(f"\ntype x lesion: stim mean [z vs the arm's own null | verdict mark: * result, . null]")
    hdr = f"{'type':9s}" + "".join(f"{l:>20s}" for l in lids)
    print(hdr)
    for t in CHECK_TYPES:
        g = mat[mat.type == t]
        if not len(g):
            continue
        line = f"{t:9s}"
        for l in lids:
            r = g[g.lesion == l]
            if not len(r):
                line += f"{'--':>20s}"; continue
            r = r.iloc[0]
            z = f"{r.z:+.1f}" if np.isfinite(r.z) else "nan"
            line += f"{r.stim_mean:9.4f} [{z:>7s}]{'*' if r.verdict == 'result' else '.'}".rjust(20)
        print(line)


# ----------------------------------------------------------------------------------------------- the size ladder
def ladder_job(deg: float, arm: str, seed: int, out_dir: str, retina: bool, receptor_model: str) -> str:
    sid = size_id(deg); r = ball_radius_for(deg)
    stem = f"{out_dir}/{sid}_{arm}_s{seed}"
    flags = f"--receptor-model {receptor_model} --seed {seed} --ball-radius {r} --ahead {AHEAD} --half-sweep {HALF_SWEEP}" + (" --retina" if retina else "") + (" --null" if arm == "null" else "")
    return (f"mkdir -p {out_dir} && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"python scripts/interp_export.py record {flags} --out {stem} > {stem}.txt 2>&1; tail -3 {stem}.txt")


def cmd_ladder_plan(args) -> int:
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rel = args.out.replace("\\", "/").rstrip("/")
    sizes = [float(x) for x in args.sizes.split(",")] if args.sizes else SIZES_DEG
    jobs, geo = [], []
    for deg in sizes:
        g = angular_size_from_eye(ball_radius_for(deg)); g["nominal_deg"] = deg; g["id"] = size_id(deg); geo.append(g)
        for arm in ("stim", "null"):
            for s in range(args.runs):
                jobs.append(ladder_job(deg, arm, s, rel, retina=(s == 0), receptor_model=args.receptor_model))
    header = (f"# generated by scripts/interp_apply_object.py ladder-plan on {time.strftime('%Y-%m-%d %H:%M')}: {len(jobs)} jobs = "
              f"{len(sizes)} sizes x (stim + null) x {args.runs} runs of scripts/interp_export.py record (retina captured at seed 0 of each arm)\n")
    write_batch(out_dir / "batch.sh", args.name, args.minutes, jobs, rel, header)
    with open(out_dir / "ladder.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"sizes": geo, "runs": int(args.runs), "jobs": jobs, "receptor_model_flag": args.receptor_model, "out_dir": rel,
                   "cluster_log": f"out/{args.name}_cluster.log"}, f, indent=1)
    print(f"{len(jobs)} jobs -> {out_dir / 'batch.sh'}")
    common.print_table(pd.DataFrame(geo), floatfmt="{:.4f}")
    return 0


def retina_footprint(stim_npz, null_npz) -> dict:
    """The presented retinal sampling of one size: columns the ball dims by > 5 % at any frame, their azimuth /
    elevation extent, the per-frame mean count and the darkest ratio -- from the replayed radiance of the stim run
    against the blank run's (deterministic ray tracer: one blank frame is the blank)."""
    a = np.load(stim_npz); b = np.load(null_npz)
    ra = a["radiance"].sum(-1); rb = b["radiance"].sum(-1)
    rb0 = rb[0][None]
    rel = ra / np.maximum(rb0, 1e-9) - 1.0
    hit = rel < -0.05
    any_col = np.flatnonzero(hit.any(0))
    az_el = a["col_az_el"]
    return {"n_columns": int(ra.shape[1]), "n_frames": int(ra.shape[0]), "columns_dimmed_5pct_any_frame": int(len(any_col)),
            "columns_dimmed_5pct_per_frame_mean": float(hit.sum(1).mean()), "columns_dimmed_50pct_per_frame_mean": float((rel < -0.5).sum(1).mean()),
            "min_relative_radiance": float(1 + rel.min()), "azimuth_deg_min": float(az_el[any_col, 0].min()) if len(any_col) else float("nan"),
            "azimuth_deg_max": float(az_el[any_col, 0].max()) if len(any_col) else float("nan"),
            "elevation_deg_min": float(az_el[any_col, 1].min()) if len(any_col) else float("nan"),
            "elevation_deg_max": float(az_el[any_col, 1].max()) if len(any_col) else float("nan"),
            "blank_radiance_identical_over_frames": bool(np.allclose(rb, rb0)), "sampling": str(a["sampling"]) if "sampling" in a else ""}


def cmd_ladder(args) -> int:
    """The ladder's recorded runs -> one Neurome export per size (scripts/interp_export.py's own Result builder and
    serializer) + one summary Result (size x type, LC11 / LC10a / T2 / T3 / ..., the retinal footprint per size) ->
    its own export directory. Prints the run directories."""
    import interp_export as ie
    from flyverse.interp import export as ex
    d = Path(args.dir)
    lad = json.load(open(d / "ladder.json", encoding="utf-8")) if (d / "ladder.json").exists() else {"sizes": [dict(angular_size_from_eye(ball_radius_for(s)), nominal_deg=s, id=size_id(s)) for s in SIZES_DEG]}
    run_dirs, rows, foot, runs_tab = {}, [], [], []
    for g in lad["sizes"]:
        sid = g["id"]
        stim = ie._glob([f"{d}/{sid}_stim_s*.json"]); nul = ie._glob([f"{d}/{sid}_null_s*.json"])
        if not stim:
            print(f"[{sid}] no runs; skipped"); continue
        cells, ncells = ie._sibling(stim, "_cells.npz"), ie._sibling(nul, "_cells.npz")
        prov = (ie._sibling(stim, "_prov.json") or [None])[0]
        ret = (ie._sibling(stim, "_retina.npz") or [None])[0]
        res = ex.result_from_object_sweep(stim, nul, cells=cells, null_cells=ncells, provenance=prov, retina=ret,
                                          control_ids=[Path(p).stem for p in nul] or None,
                                          generator="scripts/interp_export.py record (ladder) + scripts/interp_apply_object.py ladder",
                                          run_id=f"objsize-{sid}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.urandom(4).hex()}")
        res.provenance["stimulus"]["size_ladder"] = to_jsonable(g)
        res.summary["size"] = to_jsonable(g)
        ie._stamp_commit(res)
        problems = res.check()
        if problems:
            print(f"[{sid}] Result.check(): {problems}")
        jp = OUT_JSON / f"ladder_{sid}.json"; res.save(jp)
        run_dir = ex.export(res, out_root=args.out, retina=ret, parquet_rows=1_000_000)
        ie._report(run_dir, res, SimpleNamespace(neurons=None, expect_counts='{"LC11": 143, "LC10a": 275}'))
        run_dirs[sid] = str(run_dir)
        per = res.table("per_type")
        for r in per.itertuples():
            rows.append({"size_id": sid, "nominal_deg": g["nominal_deg"], "angular_diameter_deg_probe": g["angular_diameter_deg_probe"],
                         "angular_diameter_deg_from_eye": g["angular_diameter_deg_from_eye"], "ball_radius_m": g["ball_radius_m"], "type": r.type,
                         "n_cells": r.n_cells, "statistic": r.statistic, "unit": getattr(r, "unit", ""), "stim_n": r.stim_n, "stim_values": r.stim_values, "stim_mean": r.stim_mean,
                         "stim_sd": r.stim_sd, "null_n": r.null_n, "null_values": r.null_values, "null_mean": r.null_mean, "null_sd": r.null_sd, "z": r.z,
                         "welch": r.welch, "U": r.U, "p": r.p, "verdict": r.verdict, "export_run_dir": str(run_dir)})
        nret = (ie._sibling(nul, "_retina.npz") or [None])[0]
        if ret and nret:
            f = retina_footprint(ret, nret); f.update({"size_id": sid, "nominal_deg": g["nominal_deg"], "stim_retina": ret, "null_retina": nret}); foot.append(f)
        runs_tab += [{"size_id": sid, "arm": "stim", "file": p} for p in stim] + [{"size_id": sid, "arm": "null", "file": p} for p in nul]
        print(f"[{sid}] {len(stim)} stim + {len(nul)} null runs -> {run_dir}", flush=True)
    if not run_dirs:
        raise SystemExit("nothing to summarise")
    tab = pd.DataFrame(rows)
    first = Result.load(OUT_JSON / f"ladder_{list(run_dirs)[0]}.json")
    prov = dict(first.provenance)
    prov["stimulus"] = {"protocol": "object_sweep size ladder", "params": {"sizes": lad["sizes"], "ahead_m": AHEAD, "half_sweep_m": HALF_SWEEP, "runs_per_arm": lad.get("runs")},
                        "control": "per size: the none-vs-none null of the same protocol (probe_object_sweep --null)", "per_size_exports": run_dirs}
    out = Result.new("export", prov)
    out.add_table("size_tuning", tab); out.add_table("retina_footprint", foot); out.add_table("runs", runs_tab)
    key = tab[tab.statistic.isin(["diff_max_over_cells_mean_mv", "diff_abs_best_cell_mean", "diff_rate_hz_max_cell"])]
    out.summary = {"per_size_exports": run_dirs, "sizes": lad["sizes"],
                   "lc_tuning": {t: {sid: {"z": float(r.z), "stim_mean": float(r.stim_mean), "null_mean": float(r.null_mean), "null_sd": float(r.null_sd), "verdict": r.verdict, "statistic": r.statistic}
                                     for sid, r in ((r.size_id, r) for r in key[key.type == t].itertuples()) if r.statistic in ("diff_max_over_cells_mean_mv", "diff_abs_best_cell_mean")}
                                 for t in ["LC11", "LC10a", "T2", "T3", "Tm5Y", "TmY21", "Mi1", "Mi4", "Tm3", "LPLC2", "LC10b", "LC16", "LC4"] if t in set(key.type)}}
    out.replicates = {"n": int(lad.get("runs", 0)), "unit": "runs", "runs": runs_tab, "null": {"per_size": True}}
    out.files = {"generator": " ".join(sys.argv), "per_size_results": {sid: str(OUT_JSON / f"ladder_{sid}.json") for sid in run_dirs}}
    out.validation = dict(out.validation, measured={"note": "the per-size exports each carry the round-trip / verify checks in checks.json"}, status="not run")
    ie._stamp_commit(out)
    jp = Path(args.json) if args.json else OUT_JSON / "ladder.json"
    out.save(jp)
    sum_dir = ex.export(out, out_root=args.out, parquet_rows=1_000_000)
    ie._report(sum_dir, out, SimpleNamespace(neurons=None, expect_counts=None))
    print("\nsize tuning (diff_max_over_cells_mean_mv for spiking, diff_abs_best_cell_mean for rate units; z vs the size's own null):")
    show = key[key.statistic.isin(["diff_max_over_cells_mean_mv", "diff_abs_best_cell_mean"])].pivot_table(index="type", columns="nominal_deg", values="z")
    print(show.round(1).to_string())
    print("\nmeans:")
    print(key[key.statistic.isin(["diff_max_over_cells_mean_mv", "diff_abs_best_cell_mean"])].pivot_table(index="type", columns="nominal_deg", values="stim_mean").round(4).to_string())
    if foot:
        common.print_table(pd.DataFrame(foot)[["size_id", "nominal_deg", "columns_dimmed_5pct_any_frame", "columns_dimmed_5pct_per_frame_mean", "columns_dimmed_50pct_per_frame_mean",
                                               "min_relative_radiance", "azimuth_deg_min", "azimuth_deg_max", "elevation_deg_min", "elevation_deg_max"]], floatfmt="{:.2f}")
    print(f"\nsummary Result {jp}\nsummary export {sum_dir}")
    for sid, rd in run_dirs.items():
        print(f"  {sid}: {rd}")
    return 0


# ----------------------------------------------------------------------------------------------- selftest (CPU)
def cmd_selftest(args) -> int:
    """The CPU pieces on tests/test_interp.py's 8-neuron graph: the edge lesion (factor 0 / -1 / 2) on c.W with the
    pattern kept and the receptor fast sign following the factor; the size geometry; the batch lines."""
    sys.path.insert(0, str(ROOT / "tests"))
    import test_interp as ti
    from flyverse import brain
    c = ti.graph()
    W0 = c.W.tocsr().copy()
    rec = apply_edge_lesion(c, "R1-R6", "Mi4", 0.0)
    assert rec["n_entries"] == 1 and rec["synapses"] == 40.0 and c.W.nnz == W0.nnz, rec
    assert c.W[1, 0] == 0.0 and c.W[2, 1] == W0[2, 1]
    c.W = W0.copy()
    rec = apply_edge_lesion(c, "LC4", "DNp01", -1.0)
    assert c.W[3, 2] == -W0[3, 2] and rec["n_entries"] == 1
    c.W = W0.copy()
    rec = apply_edge_lesion(c, "LC4", "DNp01|PEN_a", 2.0)
    assert rec["n_entries"] == 2 and c.W[3, 2] == 2 * W0[3, 2] and c.W[7, 2] == 2 * W0[7, 2]
    c.W = W0.copy()
    # the receptor wrapper: the fast sign of the block follows the factor (the synthetic graph is unmatched: tier fallback)
    from flyverse import connectome as cn
    orig_load, orig_rec = cn.load, brain._receptor
    try:
        c2 = ti.graph()
        # the wrapper's arithmetic on the synthetic graph by hand (install_edge_lesion loads the real cache)
        sel, _, _ = edge_selection(c2, "LC4", "DNp01")
        r = brain._receptor(c2, brain.LIFParams())
        assert r is not None and r.fast_sign[sel][0] == 1.0
        fs = np.array(r.fast_sign, copy=True); fs[sel] *= -1
        assert fs[sel][0] == -1.0 and (fs[~sel] == r.fast_sign[~sel]).all()
    finally:
        cn.load, brain._receptor = orig_load, orig_rec
    g = angular_size_from_eye(ball_radius_for(11.4))
    assert abs(g["angular_diameter_deg_probe"] - 11.4) < 0.05 and abs(g["ball_radius_m"] - 0.00499) < 2e-4, g
    for deg in SIZES_DEG:
        gg = angular_size_from_eye(ball_radius_for(deg)); assert abs(gg["angular_diameter_deg_probe"] - deg) < 0.05
    line = job_line("t3_off_held", "stim", 2, "out/x")
    assert line.startswith("mkdir -p out/x && python -c 'import torch; assert torch.cuda.is_available()' && ") and "--lesion t3_off_held --arm stim --seed 2" in line
    lj = ladder_job(30.0, "null", 0, "out/y", True, "default")
    assert "--null" in lj and "--retina" in lj and f"--ball-radius {ball_radius_for(30.0)}" in lj
    assert all(k in LESIONS for k in ("base", "fb0", "inl1", "outl2", "rect", "t3_off_held", "t3_on_held", "t3_off_flip"))
    # the per-run pieces on the synthetic arms of tests/test_interp.py (Mi4 graded / LC4 spiking carry, DNp01 does not)
    from flyverse.brain import LIFParams
    c3 = ti.graph(); p = LIFParams(receptor_model=None, event_driven=False)
    stim, ctrl, nul = ti.TraceTests()._arms(c3)
    ew = common.effective_weights(c3, p)
    res = tr.trace(c3, "R1-R6", stimulus=stim, control=ctrl, null=nul, params=p, stat=STAT, min_cells=1, decompose_at="first_lost", ew=ew)
    setup = figure_setup(c3, stim, ctrl, ew=ew)
    cp = cancellation_per_run(c3, stim, ctrl, res.table("lost_inputs"), setup=setup)
    g = cp[cp.target_type == "DNp01"]
    assert len(g) == 4 and (g.n_carriers == 1).all() and (g.cancellation_fraction == 0).all(), g          # one carrier (LC4, +): nothing to cancel
    assert (abs(g.linear_estimate - 0.5) < 0.1).all(), g.linear_estimate.tolist()                          # LC4's per-run drive figure ~ +0.5 mV x share 1
    pooled = {x["target_type"]: x for x in res.summary["cancellation"]}["DNp01"]["linear_estimate"]
    assert abs(g.linear_estimate.mean() - pooled) < 1e-9, (g.linear_estimate.mean(), pooled)             # the runs' mean is the pooled trace's number
    pj = deterministic_projection(c3, {"ref": (stim[:1], ctrl[:1]), "same": (stim[:1], ctrl[:1]), "neg": (ctrl[:1], stim[:1])}, "ref", types=["DNa02"], setup=setup)
    same = pj[pj.lesion == "same"]; neg = pj[pj.lesion == "neg"]
    assert len(same) == 1 and np.allclose(same.slope_on_ref, 1.0) and np.allclose(same.pearson_r, 1.0), same
    assert np.allclose(neg.slope_on_ref, -1.0) and np.allclose(neg.pearson_r, -1.0), neg                 # the reversed figure: slope -1
    assert np.allclose(same.value_at_ref_best_cell, same.ref_best_cell_value)
    print("selftest ok: edge lesion (0 / -1 / x2) keeps the pattern and edits the block; receptor sign follows the factor; per-run cancellation and projection; "
          f"sizes {[round(angular_size_from_eye(ball_radius_for(s))['angular_diameter_deg_from_eye'], 2) for s in SIZES_DEG]} deg from the eye; batch lines")
    return 0


# ----------------------------------------------------------------------------------------------- per-run scatter (CPU)
PROJECTION_TYPES = ["Mi1", "Mi4", "Tm3", "Tm1", "Tm4", "Tm2", "T4c", "T3", "T2", "T2a", "Tm6", "Tm12", "TmY18", "Tm5Y", "TmY21", "TmY13", "TmY5a",
                    "LC11", "LC10a", "LPLC2"]


def figure_setup(c, stim, ctrl, ew=None):
    """The pieces trace() builds before it scores (the type graph, the recorded cells' type ids, unit kinds, the scored
    quantity per type, the retinotopic column masks) so that a signed figure can be taken PER RUN with trace's own
    _signed_figure; the graph is the recordings' own model (params from the provenance)."""
    meta0 = stim[0].meta
    if ew is None:
        ew = common.effective_weights(c, tr.params_from_provenance(meta0.get("provenance") or {}))
    counts, _ = tr.full_raw_counts(c)
    tg = tr.TypeGraph(c, ew, counts)
    idx0 = np.asarray(stim[0].idx)
    for r in stim[1:] + ctrl:
        if len(r.idx) != len(idx0) or not np.array_equal(np.asarray(r.idx), idx0):
            raise ValueError("every recording must hold the same cells")
    inv = tg.inv[idx0]; nT = len(tg.keys)
    n_rec = np.bincount(inv, minlength=nT)
    kinds = common.unit_kinds(c, None)
    kind_of_type = np.array([kinds[idx0[inv == i]][0] if n_rec[i] else "spiking" for i in range(nT)], dtype=object)
    q_of_type = np.array([tr._quantity_for(kind_of_type[i], STAT, None, meta0) for i in range(nT)], dtype=object)
    obj = bg = None
    col = np.asarray(meta0.get("column", []), dtype=np.int64)
    if len(col) == len(idx0) and "columns_obj" in meta0:
        in_obj = np.zeros(int(max(col.max(), 0)) + 1, bool); in_obj[np.asarray(meta0["columns_obj"], dtype=np.int64)] = True
        in_bg = np.zeros_like(in_obj); in_bg[np.asarray(meta0["columns_bg"], dtype=np.int64)] = True
        has = col >= 0
        obj = has & in_obj[np.clip(col, 0, len(in_obj) - 1)]; bg = has & in_bg[np.clip(col, 0, len(in_bg) - 1)]
    return SimpleNamespace(tg=tg, ew=ew, idx0=idx0, inv=inv, nT=nT, kind_of_type=kind_of_type, q_of_type=q_of_type, obj=obj, bg=bg, meta0=meta0,
                           types_rec=tg.keys[inv], q_of_cell=q_of_type[inv])


def cancellation_per_run(c, stim, ctrl, lost_inputs: pd.DataFrame, setup=None) -> pd.DataFrame:
    """trace's lost-stage cancellation (the carriers' sign x share x signed figure, raising vs lowering) taken RUN BY
    RUN: run r's signed figure is (stimulus[r] - control[r]) alone; the carriers, signs and shares are the pooled
    trace's `lost_inputs` rows with pre_verdict 'result'. Rows: target_type, run, n_carriers, sum_positive_terms,
    sum_negative_terms, linear_estimate, cancellation_fraction, own_signed_figure. The pooled number the report quoted
    (T3 0.92, T2 0.99, Tm5Y 0.96, TmY21 0.45) is a single-draw-set number; this is its scatter (build skeptic, trace 1)."""
    s = setup or figure_setup(c, stim, ctrl)
    rows = []
    for r in range(min(len(stim), len(ctrl))):
        fig = tr._signed_figure(stim[r:r + 1], ctrl[r:r + 1], 1, s.inv, s.nT, s.q_of_type, s.kind_of_type, s.obj, s.bg, None)
        for t, g in lost_inputs.groupby("target_type", sort=False):
            gc = g[g.pre_verdict == "result"]
            terms = np.array([float(x.sign * x.share * fig[s.tg.pos[x.pre_type]]) for x in gc.itertuples() if x.pre_type in s.tg.pos], dtype=float)
            terms = terms[np.isfinite(terms)]
            pos_t = float(terms[terms > 0].sum()) if len(terms) else 0.0; neg_t = float(terms[terms < 0].sum()) if len(terms) else 0.0
            rows.append({"target_type": t, "run": r, "n_carriers": int(len(terms)), "sum_positive_terms": pos_t, "sum_negative_terms": neg_t,
                         "linear_estimate": pos_t + neg_t,
                         "cancellation_fraction": float(1 - abs(pos_t + neg_t) / max(abs(pos_t) + abs(neg_t), 1e-12)) if (pos_t or neg_t) else float("nan"),
                         "own_signed_figure": float(fig[s.tg.pos[t]]) if t in s.tg.pos else float("nan")})
    return pd.DataFrame(rows)


def per_cell_figure(rec_a, rec_b, q_of_cell: np.ndarray) -> np.ndarray:
    """Per recorded cell: the time-mean of its scored quantity in rec_a minus rec_b -- the best_cell statistic before
    its max over the cells of a type."""
    out = np.full(len(rec_a.idx), np.nan)
    for q in np.unique(q_of_cell):
        m = q_of_cell == q
        out[m] = (tr.cell_means(rec_a, q) - tr.cell_means(rec_b, q))[m]
    return out


def deterministic_projection(c, arms: dict, ref: str, types=None, setup=None) -> pd.DataFrame:
    """Every arm's per-cell figure regressed, type by type and RUN BY RUN, on the deterministic reference arm's (`ref`
    = the gain_fb 0 arm: one draw, its none-vs-none null is exactly 0): slope = <f_ref, f_run> / <f_ref, f_ref>, the
    Pearson r, the run's value at the reference's best cell, the run's own best cell. Slope ~ 1 with r ~ 1: the
    deterministic figure is present in full under the spiking feedback; slope < 1 with r > 0: attenuated; r ~ 0:
    absent or replaced. `arms` = {lesion_id: (stim_runs, ctrl_runs)}."""
    types = types or PROJECTION_TYPES
    ref_stim, ref_ctrl = arms[ref]
    s = setup or figure_setup(c, ref_stim, ref_ctrl)
    fref = per_cell_figure(ref_stim[0], ref_ctrl[0], s.q_of_cell)
    body = c.neurons.bodyId.to_numpy()[s.idx0]
    rows = []
    for lid, (stim, ctrl) in arms.items():
        for r in range(min(len(stim), len(ctrl))):
            f = per_cell_figure(stim[r], ctrl[r], s.q_of_cell)
            for t in types:
                m = (s.types_rec == t) & np.isfinite(f) & np.isfinite(fref)
                if m.sum() < 2:
                    continue
                a, b = fref[m], f[m]
                den = float(a @ a)
                k = int(np.argmax(a))
                rows.append({"lesion": lid, "type": t, "run": r, "n_cells": int(m.sum()), "quantity": str(s.q_of_cell[m][0]),
                             "slope_on_ref": float(a @ b / den) if den > 0 else float("nan"),
                             "pearson_r": float(np.corrcoef(a, b)[0, 1]) if (a.std() > 0 and b.std() > 0) else float("nan"),
                             "ref_best_cell": str(int(body[m][k])), "ref_best_cell_value": float(a[k]), "value_at_ref_best_cell": float(b[k]),
                             "own_best_cell": float(b.max()), "ref_rms": float(np.sqrt(np.mean(a ** 2))), "own_rms": float(np.sqrt(np.mean(b ** 2)))})
    return pd.DataFrame(rows)


def lesioned_carriers(lost_inputs: pd.DataFrame, lesion: dict) -> pd.DataFrame:
    """The baseline trace's `lost_inputs` under an 'edges' lesion (a pooled carrier table exists only for the baseline
    arm): held edges (factor 0) are dropped and the target's remaining shares renormalised, flipped edges (factor < 0)
    keep their share with the sign negated, a gain (|factor| != 1) scales the share and renormalises. Other kinds
    (optic overrides) leave the graph and the table unchanged."""
    if not lesion or lesion.get("kind") != "edges":
        return lost_inputs
    pre_re = re.compile("^(" + str(lesion["pre"]) + ")$"); post_re = re.compile("^(" + str(lesion["post"]) + ")$"); f = float(lesion["factor"])
    out = lost_inputs.copy()
    hit = out.pre_type.astype(str).str.match(pre_re) & out.target_type.astype(str).str.match(post_re)
    out.loc[hit, "share"] = out.loc[hit, "share"] * abs(f)
    if f < 0:
        out.loc[hit, "sign"] = -out.loc[hit, "sign"]
    for t in out.target_type[hit].unique():                       # the listed inputs' shares renormalised to their own total
        m = out.target_type == t
        before = float(lost_inputs.share[m].sum()); after = float(out.share[m].sum())
        if after > 0 and before > 0:
            out.loc[m, "share"] = out.loc[m, "share"] * (before / after)
    return out[out.share > 0].reset_index(drop=True)


def feedback_share(setup, types: list, top: int = 5) -> pd.DataFrame:
    """Per type: the share of its shaped-weight |input| (the LIF's A, i.e. the graph the rate lobe reads through W_rr /
    W_sr) that comes from SPIKING presynaptic units -- the entry points of the spiking -> rate feedback (gain_fb) --
    and the top spiking pre types with their shares. Structural, no recording involved."""
    tg = setup.tg
    spiking = np.array([k == "spiking" for k in setup.kind_of_type])
    rows = []
    for t in types:
        if t not in tg.pos:
            continue
        i = tg.pos[t]
        row = tg.share[i]
        cols, sh = row.indices, row.data
        keep = (cols != tg.untyped) if tg.untyped is not None else np.ones(len(cols), bool)
        cols, sh = cols[keep], sh[keep]
        sp_m = spiking[cols]
        order = np.argsort(-sh[sp_m])[:top]
        tops = [f"{tg.keys[cols[sp_m][k]]} {sh[sp_m][k]:.3f}" for k in order]
        rows.append({"type": t, "kind": str(setup.kind_of_type[i]), "n_cells": int(tg.n_cells[i]), "spiking_input_share": float(sh[sp_m].sum()),
                     "graded_input_share": float(sh[~sp_m].sum()), "n_spiking_pre_types": int(sp_m.sum()), "top_spiking_inputs": "; ".join(tops)})
    return pd.DataFrame(rows)


def _summ(df: pd.DataFrame, keys: list, cols: list) -> pd.DataFrame:
    g = df.groupby(keys, sort=False)
    out = g[cols].agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def cmd_perrun(args) -> int:
    """Replicate scatter for the report's two single-draw-set numbers (CPU, recordings only): (a) the lost-stage
    cancellation fractions per run on the trace recordings (out/trv, 5 runs) and on every lesion arm that has a trace
    JSON (4 runs each); (b) every stochastic arm's per-cell figure regressed on the deterministic fb0 arm's, per run
    -- is the deterministic small-field figure present, attenuated or absent under the spiking feedback? Result of
    tool 'trace' (tables cancellation_per_run / cancellation_summary / projection / projection_summary)."""
    c = load_c(args.cache_dir)
    d = Path(args.dir)
    lids = [x for x in args.lesions.split(",") if x]
    arms = {}
    for lid in lids:
        stim = tr.load_runs(f"{d}/{lid}_stim_r*"); ctrl = tr.load_runs(f"{d}/{lid}_ctrl_r*")
        if stim and ctrl:
            arms[lid] = (stim, ctrl)
        else:
            print(f"[{lid}] no recordings in {d}; skipped")
    sets = []
    if args.trace_dir and args.trace_json and Path(args.trace_json).exists():
        ts = tr.load_runs(f"{args.trace_dir}/{args.trace_prefix}_stim_r*"); tc = tr.load_runs(f"{args.trace_dir}/{args.trace_prefix}_ctrl_r*")
        if ts and tc:
            sets.append((args.trace_prefix, ts, tc, Path(args.trace_json)))
    base_li = None
    p_base = OUT_JSON / f"trace_{args.baseline}.json"
    if p_base.exists():
        base_li = Result.load(p_base).table("lost_inputs")
    for lid, (stim, ctrl) in arms.items():
        p = OUT_JSON / f"trace_{lid}.json"
        sets.append((lid, stim, ctrl, p if p.exists() else None))
    if not sets and not arms:
        raise SystemExit("nothing to do: no recordings")
    first = sets[0][1:3] if sets else next(iter(arms.values()))
    t0 = time.time()
    setup = figure_setup(c, first[0], first[1])
    print(f"setup {time.time() - t0:.0f} s: {setup.nT} types, {len(setup.idx0)} cells, columns {'yes' if setup.obj is not None else 'no'}", flush=True)
    canc = []
    for lid, stim, ctrl, p in sets:
        li = Result.load(p).table("lost_inputs") if p is not None else pd.DataFrame()
        source = str(p)
        if not len(li) and base_li is not None and len(base_li):        # the arm's trace was not decomposed: the baseline's carriers under the arm's lesion
            li = lesioned_carriers(base_li, stim[0].meta.get("lesion") or {}); source = f"{p_base} under lesion {lid}"
        if not len(li):
            continue
        cp = cancellation_per_run(c, stim, ctrl, li, setup=setup); cp.insert(0, "set", lid); cp["carriers_from"] = source
        canc.append(cp)
    canc = pd.concat(canc, ignore_index=True) if canc else pd.DataFrame()
    csum = _summ(canc, ["set", "target_type"], ["cancellation_fraction", "linear_estimate", "own_signed_figure"]) if len(canc) else pd.DataFrame()
    proj = deterministic_projection(c, arms, args.ref, setup=setup) if args.ref in arms else pd.DataFrame()
    psum = _summ(proj, ["lesion", "type"], ["slope_on_ref", "pearson_r", "value_at_ref_best_cell", "own_best_cell"]) if len(proj) else pd.DataFrame()
    fbs = feedback_share(setup, PROJECTION_TYPES + ["L1", "L2", "Mi9", "C3", "L5", "T4c", "T4d", "T5a", "Tm6", "Tm12", "TmY18", "T2a", "Li15"])
    prov = dict(first[0][0].meta.get("provenance") or {})
    prov["analysis"] = {"flyverse_commit": common.git_state(), "recordings_dir": str(d), "sets": [s[0] for s in sets], "arms": list(arms), "ref": args.ref,
                        "devices": sorted({str(((r.meta.get("provenance") or {}).get("execution") or {}).get("device")) for a in arms.values() for r in a[0] + a[1]})}
    res = Result.new("trace", prov)
    res.add_table("cancellation_per_run", canc); res.add_table("cancellation_summary", csum)
    res.add_table("projection", proj); res.add_table("projection_summary", psum); res.add_table("feedback_share", fbs)
    res.replicates = {"n": {lid: int(min(len(a[0]), len(a[1]))) for lid, a in arms.items()}, "unit": "runs", "note": "each arm's own independent draws; fb0 is deterministic (one draw repeated)"}
    res.summary = {"cancellation": {f"{r.set}/{r.target_type}": {"mean": r.cancellation_fraction_mean, "sd": r.cancellation_fraction_std, "n": int(r.cancellation_fraction_count)} for r in csum.itertuples()} if len(csum) else {},
                   "projection": {f"{r.lesion}/{r.type}": {"slope_mean": r.slope_on_ref_mean, "slope_sd": r.slope_on_ref_std, "r_mean": r.pearson_r_mean, "n": int(r.slope_on_ref_count)} for r in psum.itertuples()} if len(psum) else {},
                   "ref": args.ref, "stat": STAT}
    res.files = {"generator": " ".join(sys.argv), "recordings_dir": str(d), "traces": {s[0]: str(s[3]) for s in sets}}
    path = Path(args.json) if args.json else OUT_JSON / "perrun.json"
    res.save(path)
    if len(csum):
        print("\ncancellation fraction per run (mean +- sd over runs; the pooled trace's carriers):")
        print(common.print_table(csum, floatfmt="{:+.4f}", max_rows=80))
        print(common.print_table(canc[["set", "target_type", "run", "n_carriers", "sum_positive_terms", "sum_negative_terms", "cancellation_fraction", "own_signed_figure"]], floatfmt="{:+.2e}", max_rows=200))
    if len(psum):
        print(f"\nper-cell figure of each arm regressed on the deterministic {args.ref} arm (per type; mean +- sd over the arm's runs):")
        print(common.print_table(psum, floatfmt="{:+.3f}", max_rows=400))
    if len(fbs):
        print("\nshare of each type's shaped-weight |input| from spiking units (the spiking -> rate feedback's entry points; structural):")
        print(common.print_table(fbs, floatfmt="{:+.3f}", max_rows=80))
    print(f"written {path}  (check: {res.check() or 'ok'})")
    return 0


# ----------------------------------------------------------------------------------------------- argv
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trace", help="CPU: the object trace with the null, decomposed at the small-object stage and at LC11 / LC10a")
    t.add_argument("--dir", default="out/trv"); t.add_argument("--prefix", default="obj")
    t.add_argument("--decompose-at", default=",".join(STAGE_TYPES))
    t.add_argument("--no-rate", action="store_true", help="skip the second trace that scores spiking types on rate_hz")
    common.add_common_args(t); t.set_defaults(func=cmd_trace)

    s = sub.add_parser("stage", help="CPU: what is in force at the lost stage (channels, signs, tiers, gains, taus, baselines, normalisation)")
    s.add_argument("--trace", default=str(OUT_JSON / "trace_obj.json")); s.add_argument("--targets", default=",".join(STAGE_TYPES))
    s.add_argument("--top", type=int, default=16)
    common.add_common_args(s); s.set_defaults(func=cmd_stage)

    p = sub.add_parser("plan", help="CPU: resolve the lesions and write ONE cluster batch")
    p.add_argument("--out", default="out/apply_object/les"); p.add_argument("--runs", type=int, default=4); p.add_argument("--lesions", default=None)
    p.add_argument("--name", default="apobj-les"); p.add_argument("--minutes", type=int, default=45); p.add_argument("--series-every", type=int, default=5)
    common.add_common_args(p); p.set_defaults(func=cmd_plan)

    r = sub.add_parser("record", help="GPU: one arm of the object protocol under one lesion (interp_trace record's path)")
    r.add_argument("--lesion", required=True, choices=list(LESIONS)); r.add_argument("--arm", required=True, choices=list(ARMS))
    r.add_argument("--out", required=True); r.add_argument("--seconds", type=float, default=12.0); r.add_argument("--settle", type=float, default=3.0)
    r.add_argument("--quick", action="store_true"); r.add_argument("--allow-cpu", action="store_true"); r.add_argument("--series-every", type=int, default=5)
    common.add_common_args(r); r.set_defaults(func=cmd_record)

    a = sub.add_parser("analyse", help="CPU: per-lesion traces -> type x lesion matrix, sensitivity, restores")
    a.add_argument("--dir", default="out/apply_object/les"); a.add_argument("--lesions", default=None); a.add_argument("--baseline", default="base")
    a.add_argument("--decompose-at", default=",".join(STAGE_TYPES)); a.add_argument("--decompose-all", action="store_true")
    common.add_common_args(a); a.set_defaults(func=cmd_analyse)

    lp = sub.add_parser("ladder-plan", help="CPU: the size ladder batch (scripts/interp_export.py record per size / arm / run)")
    lp.add_argument("--out", default="out/apply_object/ladder"); lp.add_argument("--runs", type=int, default=5); lp.add_argument("--sizes", default=None)
    lp.add_argument("--name", default="apobj-lad"); lp.add_argument("--minutes", type=int, default=45)
    common.add_common_args(lp); lp.set_defaults(func=cmd_ladder_plan)

    la = sub.add_parser("ladder", help="CPU: the ladder runs -> one export per size + the summary export")
    la.add_argument("--dir", default="out/apply_object/ladder"); la.add_argument("--out", default="out/export")
    common.add_common_args(la); la.set_defaults(func=cmd_ladder)

    pr = sub.add_parser("perrun", help="CPU: replicate scatter -- cancellation fractions per run; every arm's per-cell figure regressed on the deterministic fb0 arm")
    pr.add_argument("--dir", default="out/apply_object/les"); pr.add_argument("--lesions", default=",".join(LESIONS)); pr.add_argument("--ref", default="fb0")
    pr.add_argument("--baseline", default="base", help="the arm whose decomposed trace supplies the carriers for arms whose trace was not decomposed")
    pr.add_argument("--trace-dir", default="out/trv"); pr.add_argument("--trace-prefix", default="obj"); pr.add_argument("--trace-json", default=str(OUT_JSON / "trace_obj.json"))
    common.add_common_args(pr); pr.set_defaults(func=cmd_perrun)

    st = sub.add_parser("selftest", help="CPU: the pieces on the synthetic graph"); st.set_defaults(func=cmd_selftest)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
