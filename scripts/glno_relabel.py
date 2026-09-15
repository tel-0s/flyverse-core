"""GLNO -> glutamate: the evidence case, the adoption suite and the rotation input (thread 5B; docs/audits/glno_relabel.md).

GLNO (4 LAL-NO1 cells) is 'unknown' in MaleCNS, so its 17,698 output synapses -- 84 GLNO -> PEN edges, 19.4 % of PEN's raw
input, every edge above the connection cap -- are explicit zeros in W (docs/audits/cx_glno.md section 1). This script
PREPARES the adoption case for the entry {"GLNO": "glutamate"} in connectome.TYPE_NT_OVERRIDE under the round-2 rule
(adopt only if the full benchmark suite changes no check's status in >= 3 draws). Nothing is adopted here: cache/,
TYPE_NT_OVERRIDE and every default are untouched; the candidate lives in a scratch cache.

CPU (this desktop, CUDA_VISIBLE_DEVICES=-1):
  python scripts/glno_relabel.py compile [--cache-dir out/cache_glno_glu]
      compile the candidate cache from the raw MaleCNS files with TYPE_NT_OVERRIDE + {GLNO: glutamate} (cx_wedge's
      scratch pattern: build in a temp dir, rename into place; a lock file serialises concurrent jobs) and build its
      sign0_counts.npz so that the cache carries the same files as the shipped one.
  python scripts/glno_relabel.py compare [--cache-dir out/cache_glno_glu] [--out out/cx5b/entry_compare.json]
      the shipped cache (cache/) against the candidate, entry by entry: the neuron table (nt / sign per cell), the sparsity
      pattern of W, the entries whose value differs (all must have a GLNO presynaptic cell), those entries by postsynaptic
      type with raw synapse counts, the PEN edges (count, min / max, per-PEN fan-in), the effective mV per spike after the
      connection cap and fan-in scale, and the loop sign.
  python scripts/glno_relabel.py plan [--dir out/cx5b] [--name cx5b] [--minutes 45] [--seeds 0,1,2,3] [--draws 3]
      write the ONE submission: predeclared.json (stamped, absolute UTC timestamp), tree_state.json (commit, status,
      sha256 of the simulation sources, the shipped probe_vnc_drive.py against HEAD's), jobs.json and batch.sh.
  python scripts/glno_relabel.py analyse [--dir out/cx5b] [--out out/cx5b/analysis]
      the suite status table (shipped vs glutamate, per check, per draw) and the adoption verdict; the compass flip rows
      (GLNO / PEN_a / PEN_b / PS196_b / AN04B003 / EPG), the bump drift, the chain rates, per-run scatter, the predeclared
      families with Holm; -> analysis/*.csv, analysis/analysis.md, analysis/summary.json.
  python scripts/glno_relabel.py smoke
      CPU smoke of the two job wrappers (a 2-frame compass run through probe_vnc_drive.py compass --quick --device cpu on
      the shipped cache, and the wrapper's post-processing); the numbers mean nothing.

GPU (the cluster jobs; every job line starts `mkdir -p out/cx5b && source .venv/bin/activate && ` and preserves the exit
code):
  python scripts/glno_relabel.py suite --arm shipped|glu --draw N --out out/cx5b/fam_suite/suite_<arm>_<N>.json
      scripts/benchmark.py --sections all --seeds 0,1,2 [--cache-dir out/cache_glno_glu] through benchmark.main() with the
      loaded connectome captured; appends `provenance` (flyverse.interp.common.provenance) and a `glno_relabel` block
      (cache, override table, GLNO nt / sign per cell read back from the cache the run used, device, tally); exit 3 when
      the cache the run used is not the one the arm asks for or the device is not a GPU.
  python scripts/glno_relabel.py compass --cache shipped|glu --gains exp|shipped --seed S --block fam_cS --out <stem>
      scripts/probe_vnc_drive.py compass --family level --arm C (all+leg_cycle) [--gains 2:15 | none] [--cache-dir ...],
      called as it is (runpy, its own argv; nothing in it is edited), then the same `glno_relabel` block appended to
      <stem>_run.json with the same exit-3 checks (plus family / arm / spec / gains read back from the JSON).

Design (predeclared before submission; docs/audits/glno_relabel.md section 3):
  suite      3 draws x {shipped, glu}                                                    6 jobs, block fam_suite
  compass    4 seeds x {shipped, glu} x {exp gains gE 2 / gD 15, shipped gains = none}   16 jobs, blocks fam_c<seed>
  22 jobs, one cluster_run submission (--arm-block fam), house target. Verdicts: flyverse.interp.common.compare over runs
  (4 v 4 on the compass; the suite is 3 v 3 = underpowered by construction, its verdict is the status rule), Holm within
  the declared families.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from flyverse import connectome  # noqa: E402
from flyverse.interp import common  # noqa: E402

OUT = ROOT / "out"
SCRATCH = OUT / "cache_glno_glu"
EXTRA = {"GLNO": "glutamate"}
GAINS = {"exp": "2:15", "shipped": "none"}          # the labelled instrument (gE 2 / gD 15) and the shipped path (no gains, no pulse)
FAMILY, ARM = "level", "C"                          # probe_vnc_drive.py family `level`, arm C = proprioception 'all+leg_cycle'
KEY_TYPES = ["GLNO", "PEN_a(PEN1)", "PEN_b(PEN2)", "PS196_b", "AN04B003", "EPG", "DNa02", "Delta7", "PEG", "ExR8", "FB4Y", "FB1C"]
PHASES = ["rest", "ccw", "rest2", "cw"]
N_CHECKS = 29


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    print(f"[glno_relabel {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def override_table() -> dict:
    """connectome.TYPE_NT_OVERRIDE (as shipped; TYPE_NT_OVERRIDE_DEFAULT must be True) + {GLNO: glutamate}."""
    if not connectome.TYPE_NT_OVERRIDE_DEFAULT:
        raise SystemExit("TYPE_NT_OVERRIDE_DEFAULT is False: the shipped cache would not carry the adopted table")
    table = dict(connectome.TYPE_NT_OVERRIDE)
    table.update(EXTRA)
    return table


# ---------------------------------------------------------------------------------------------- the scratch cache
def cache_complete(cache_dir: Path) -> bool:
    return all((cache_dir / f).exists() for f in ("W_post_pre.npz", "neurons.parquet", "TYPE_NT_OVERRIDE.json", connectome.SIGN0_COUNTS_FILE))


def ensure_scratch(cache_dir: Path = SCRATCH, wait_s: float = 3600.0) -> Path:
    """The candidate cache, compiled once. Concurrent jobs: the first takes `<cache_dir>.lock` (O_EXCL) and compiles into a
    temp dir beside it (renamed into place when complete); the others poll until the cache is complete. A stale lock older
    than `wait_s` is taken over."""
    cache_dir = Path(cache_dir)
    if cache_complete(cache_dir):
        return cache_dir
    lock = cache_dir.parent / (cache_dir.name + ".lock")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {utc()}\n".encode()); os.close(fd)
            break                                                   # we compile
        except FileExistsError:
            if cache_complete(cache_dir):
                return cache_dir
            age = time.time() - lock.stat().st_mtime if lock.exists() else 0.0
            if age > wait_s:
                log(f"lock {lock} is {age:.0f} s old: taking it over"); lock.unlink(missing_ok=True); continue
            if time.time() - t0 > wait_s:
                raise SystemExit(f"waited {wait_s:.0f} s for {cache_dir} (lock {lock})")
            log(f"another job holds {lock} ({age:.0f} s): waiting for {cache_dir}")
            time.sleep(20.0)
    try:
        if cache_complete(cache_dir):
            return cache_dir
        table = override_table()
        tmp = Path(tempfile.mkdtemp(prefix=cache_dir.name + ".tmp-", dir=cache_dir.parent))
        log(f"compiling {table} from the raw MaleCNS files into {tmp}")
        t1 = time.time()
        c = connectome.load(cache_dir=tmp, rebuild=True, verbose=True, type_nt_override=table)
        (tmp / "TYPE_NT_OVERRIDE.json").write_text(json.dumps(table, indent=1))
        connectome.build_sign0_counts(c, tmp, verbose=True)
        log(f"compiled: {c.n} cells, nnz {c.W.nnz}, sum|W| {float(abs(c.W).sum()):.0f} in {time.time() - t1:.0f} s")
        del c
        if cache_dir.exists() and cache_complete(cache_dir):
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            os.rename(tmp, cache_dir)
        return cache_dir
    finally:
        lock.unlink(missing_ok=True)


def glno_state(cache_dir: Path | None) -> dict:
    """GLNO's label and sign as the cache on disk stores them (neurons.parquet), plus the override table in force."""
    d = Path(cache_dir) if cache_dir else connectome.CACHE_DIR
    n = pd.read_parquet(d / "neurons.parquet", columns=["bodyId", "type", "instance", "nt", "sign"])
    g = n[n.type == "GLNO"].sort_values("bodyId")
    ov = d / "TYPE_NT_OVERRIDE.json"
    table = json.loads(ov.read_text()) if ov.exists() else ("connectome.TYPE_NT_OVERRIDE (the shipped default, TYPE_NT_OVERRIDE_DEFAULT True)"
                                                             if connectome.TYPE_NT_OVERRIDE_DEFAULT else {})
    return {"cache_dir": str(d), "override_table": table, "n_glno": int(len(g)),
            "glno": [{"bodyId": int(b), "instance": str(i), "nt": str(t), "sign": float(s)} for b, i, t, s in zip(g.bodyId, g.instance, g.nt, g.sign)],
            "glno_nt": sorted(set(g.nt.astype(str))), "glno_sign": sorted(set(float(s) for s in g.sign)),
            "nt_counts": {k: int(v) for k, v in n.nt.value_counts().items()}}


def expect_state(arm: str) -> dict:
    return {"shipped": {"glno_nt": ["unknown"], "glno_sign": [0.0]}, "glu": {"glno_nt": ["glutamate"], "glno_sign": [-1.0]}}[arm]


# ---------------------------------------------------------------------------------------------- compare (CPU)
def effective_mv(c, p):
    """A[post, pre] in mV per presynaptic spike as Brain installs it (cx_wedge.effective_weights' formula: the shaped
    weights, the fan-in scale, w_syn); the scale and the raw fan-in total per postsynaptic cell."""
    import scipy.sparse as sp
    from flyverse import brain
    W = brain._shaped_weights(c, p)
    tot = np.asarray(abs(W).sum(axis=1)).ravel()
    scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0).astype(np.float32)
    A = (sp.diags(scale) @ W).tocsr()
    A.data *= np.float32(p.w_syn)
    return A, scale, tot


def cmd_compare(args) -> int:
    from flyverse import brain
    scratch = ensure_scratch(Path(args.cache_dir))
    t0 = time.time()
    c0 = connectome.load(verbose=False)
    c1 = connectome.load(cache_dir=scratch, verbose=False)
    log(f"loaded shipped {connectome.CACHE_DIR} and candidate {scratch} in {time.time() - t0:.0f} s")
    n0, n1 = c0.neurons, c1.neurons
    res = {"shipped": {"cache_dir": str(connectome.CACHE_DIR)}, "candidate": {"cache_dir": str(scratch)}, "when": utc()}
    for name, c in (("shipped", c0), ("candidate", c1)):
        fp = common.connectome_fingerprint(c, c.cache_dir if hasattr(c, "cache_dir") else None)
        res[name].update({k: fp.get(k) for k in ("md5", "sum_abs_W", "nnz", "n_neurons")})
        res[name]["glno"] = glno_state(connectome.CACHE_DIR if name == "shipped" else scratch)
    # the neuron table
    same_ids = bool(len(n0) == len(n1) and (n0.bodyId.to_numpy() == n1.bodyId.to_numpy()).all())
    same_type = bool(same_ids and (n0.type.fillna("").to_numpy() == n1.type.fillna("").to_numpy()).all())
    nt_diff = np.flatnonzero(n0.nt.astype(str).to_numpy() != n1.nt.astype(str).to_numpy()) if same_ids else np.array([])
    sign_diff = np.flatnonzero(n0.sign.to_numpy() != n1.sign.to_numpy()) if same_ids else np.array([])
    ty = n0.type.fillna("").to_numpy()
    glno = np.flatnonzero(ty == "GLNO")
    res["neurons"] = {"same_bodyIds_and_order": same_ids, "same_types": same_type,
                      "nt_differs_cells": int(len(nt_diff)), "nt_differs_all_GLNO": bool(set(nt_diff.tolist()) == set(glno.tolist())),
                      "sign_differs_cells": int(len(sign_diff)), "sign_differs_all_GLNO": bool(set(sign_diff.tolist()) == set(glno.tolist())),
                      "nt_diff_types": sorted(set(ty[nt_diff].tolist()))}
    # W entry by entry
    W0 = c0.W.tocsr().copy(); W0.sort_indices(); W1 = c1.W.tocsr().copy(); W1.sort_indices()
    same_pattern = bool(W0.shape == W1.shape and W0.nnz == W1.nnz and np.array_equal(W0.indptr, W1.indptr) and np.array_equal(W0.indices, W1.indices))
    res["W"] = {"same_shape": W0.shape == W1.shape, "same_nnz": int(W0.nnz) == int(W1.nnz), "same_sparsity_pattern": same_pattern,
                "sum_abs_shipped": float(abs(W0).sum()), "sum_abs_candidate": float(abs(W1).sum())}
    if not same_pattern:
        log("the sparsity patterns differ: the entry comparison below is by (row, col) key")
        k0 = W0.tocoo(); k1 = W1.tocoo()
        key0 = k0.row.astype(np.int64) * c0.n + k0.col; key1 = k1.row.astype(np.int64) * c1.n + k1.col
        res["W"]["only_in_shipped"] = int(len(np.setdiff1d(key0, key1))); res["W"]["only_in_candidate"] = int(len(np.setdiff1d(key1, key0)))
        common_keys, i0, i1 = np.intersect1d(key0, key1, return_indices=True)
        d0, d1 = k0.data[i0], k1.data[i1]; rows, cols = k0.row[i0], k0.col[i0]
    else:
        coo = W1.tocoo(); d0, d1 = W0.data, W1.data; rows, cols = coo.row, coo.col
    diff = np.flatnonzero(d0 != d1)
    pre_is_glno = np.isin(cols[diff], glno)
    sign0 = connectome.sign0_counts(c0, W=W0, build=False)
    res["entries"] = {"differing": int(len(diff)), "differing_with_GLNO_pre": int(pre_is_glno.sum()), "all_differing_have_GLNO_pre": bool(pre_is_glno.all()),
                      "shipped_values_all_zero": bool((d0[diff] == 0).all()), "candidate_values_all_negative": bool((d1[diff] < 0).all()),
                      "candidate_abs_sum": float(np.abs(d1[diff]).sum()),
                      "candidate_abs_equals_shipped_sign0_counts": bool(sign0 is not None and same_pattern and np.allclose(np.abs(d1[diff]), sign0[diff])),
                      "glno_pre_entries_total": int(np.isin(cols, glno).sum()),
                      "glno_pre_entries_unchanged": int((np.isin(cols, glno) & (d0 == d1)).sum())}
    # by postsynaptic type
    post_ty = ty[rows[diff]]
    by = pd.DataFrame({"post_type": post_ty, "syn": np.abs(d1[diff])}).groupby("post_type").agg(entries=("syn", "size"), raw_syn=("syn", "sum")).sort_values("raw_syn", ascending=False)
    by["share_of_GLNO_output"] = by.raw_syn / by.raw_syn.sum()
    res["by_post_type"] = [{"post_type": t, "entries": int(r.entries), "raw_syn": float(r.raw_syn), "share": float(r.share_of_GLNO_output)} for t, r in by.iterrows()]
    # PEN edges and the loop
    pen = np.flatnonzero(np.char.startswith(ty.astype(str), "PEN_"))
    g2p = diff[np.isin(rows[diff], pen)]
    per_pen = pd.Series(rows[g2p]).value_counts()
    p2g_mask = np.isin(rows, glno) & np.isin(cols, pen)
    p = brain.LIFParams()
    A1, scale1, tot1 = effective_mv(c1, p)
    A0, scale0, tot0 = effective_mv(c0, p)
    a_g2p = np.asarray(A1[rows[g2p], cols[g2p]]).ravel()
    a_p2g = np.asarray(A1[rows[p2g_mask], cols[p2g_mask]]).ravel()
    res["pen"] = {"glno_to_pen_entries": int(len(g2p)), "glno_to_pen_raw_syn": float(np.abs(d1[g2p]).sum()),
                  "edge_min": float(np.abs(d1[g2p]).min()), "edge_max": float(np.abs(d1[g2p]).max()), "edge_mean": float(np.abs(d1[g2p]).mean()),
                  "edge_median": float(np.median(np.abs(d1[g2p]))), "edges_above_cap": int((np.abs(d1[g2p]) > p.conn_cap).sum()), "conn_cap": float(p.conn_cap), "w_syn": float(p.w_syn),
                  "glno_inputs_per_pen": {"min": int(per_pen.min()), "max": int(per_pen.max()), "n_pen_with_input": int(len(per_pen)), "n_pen": int(len(pen))},
                  "pen_raw_input_shipped": float(np.abs(d0[np.isin(rows, pen)]).sum() + (sign0[np.isin(rows, pen)].sum() if (sign0 is not None and same_pattern) else 0.0)),
                  "effective_mv_per_glno_spike_candidate": {"min": float(a_g2p.min()), "max": float(a_g2p.max()), "mean": float(a_g2p.mean())},
                  "effective_mv_per_glno_spike_shipped": 0.0,
                  "pen_fan_in_scale_candidate": {"min": float(scale1[pen].min()), "max": float(scale1[pen].max())}, "pen_fan_in_scale_shipped": {"min": float(scale0[pen].min()), "max": float(scale0[pen].max())},
                  "pen_fan_in_total_candidate": {"min": float(tot1[pen].min()), "max": float(tot1[pen].max())}, "pen_fan_in_total_shipped": {"min": float(tot0[pen].min()), "max": float(tot0[pen].max())},
                  "pen_to_glno_entries": int(p2g_mask.sum()), "pen_to_glno_W_sum_candidate": float(d1[p2g_mask].sum()), "pen_to_glno_W_sum_shipped": float(d0[p2g_mask].sum()),
                  "pen_to_glno_effective_mv_per_pen_spike": {"min": float(a_p2g.min()), "max": float(a_p2g.max())},
                  "glno_fan_in_scale_candidate": [float(x) for x in scale1[glno]], "glno_fan_in_scale_shipped": [float(x) for x in scale0[glno]],
                  "loop_sign": "PEN -> GLNO acetylcholine (+) -> GLNO -> PEN glutamate (-): a negative feedback loop closed contralaterally (cx_glno.md 3); shipped: the return leg is 0"}
    # the effective-weight matrices differ only on GLNO's output entries (plus every entry of a postsynaptic cell whose fan-in scale moved)
    dA = (A1 - A0).tocoo()
    dA_nonzero = dA.data != 0
    scale_moved = np.flatnonzero(scale1 != scale0)
    ratio = scale1[scale_moved] / scale0[scale_moved]
    res["effective"] = {"entries_differing": int(dA_nonzero.sum()), "with_GLNO_pre": int(np.isin(dA.col[dA_nonzero], glno).sum()),
                        "cells_whose_fan_in_scale_moved": int(len(scale_moved)), "scale_moved_types": {k: int(v) for k, v in pd.Series(ty[scale_moved]).value_counts().head(20).items()},
                        "scale_ratio_candidate_over_shipped": {"min": float(ratio.min()), "max": float(ratio.max())} if len(ratio) else None,
                        "scale_moved_cells": [{"type": str(ty[i]), "scale_shipped": float(scale0[i]), "scale_candidate": float(scale1[i]), "fan_in_shipped": float(tot0[i]),
                                               "fan_in_candidate": float(tot1[i]), "glno_syn_in": float(np.abs(d1[diff][rows[diff] == i]).sum())} for i in scale_moved],
                        "max_abs_change_mv_outside_GLNO_entries": float(np.abs(dA.data[dA_nonzero & ~np.isin(dA.col, glno)]).max()) if (dA_nonzero & ~np.isin(dA.col, glno)).any() else 0.0,
                        "note": "a post cell whose raw fan-in total crosses input_norm_ref once GLNO's synapses count changes the scale of ALL its inputs"}
    # the shipped receptor model (LIFParams.receptor_model 'sign', net rule 'abs') on the candidate's GLNO entries: the sign the
    # suite / compass runs really install on each of the 213 entries, by postsynaptic type
    rec = {}
    for rule in ("abs", "class"):
        rs = connectome.receptor_signs(c1, W=W1, net_rule=rule, with_counts=False)
        fs = rs.fast_sign[diff] if same_pattern else np.array([])
        tiers = np.array(connectome.RECEPTOR_TIERS)[rs.tier[diff]] if same_pattern else np.array([])
        by_t = pd.DataFrame({"post_type": post_ty, "fast_sign": fs, "syn": np.abs(d1[diff]), "tier": tiers}).groupby("post_type").agg(
            entries=("syn", "size"), syn=("syn", "sum"), neg=("fast_sign", lambda s: int((s < 0).sum())), pos=("fast_sign", lambda s: int((s > 0).sum())),
            zero=("fast_sign", lambda s: int((s == 0).sum())), tiers=("tier", lambda s: ",".join(sorted(set(s)))))
        rec[rule] = {"entries_neg": int((fs < 0).sum()), "entries_pos": int((fs > 0).sum()), "entries_zero": int((fs == 0).sum()),
                     "syn_pos": float(np.abs(d1[diff])[fs > 0].sum()), "syn_neg": float(np.abs(d1[diff])[fs < 0].sum()),
                     "by_post_type": [{"post_type": t, **{k: (int(v) if k != "syn" and k != "tiers" else (float(v) if k == "syn" else v)) for k, v in r.items()}} for t, r in by_t.iterrows() if r.pos or r.zero or t in ("PEN_a(PEN1)", "PEN_b(PEN2)", "GLNO", "ExR8", "FB4Y", "FB1C")]}
    res["receptor_model_on_GLNO_entries"] = {"note": "LIFParams default receptor_model 'sign' with net rule 'abs' (the suite / compass path); 'class' for reference; "
                                                     "cx_glno.md section 3-5 ran receptor_model None (= the NT_SIGN -1 on every entry)", **rec}
    # the sign-0 counts file of the candidate: the shipped file's entries minus the GLNO ones
    s1 = connectome.sign0_counts(c1, W=W1, build=False)
    res["sign0"] = {"shipped_total": float(sign0.sum()) if sign0 is not None else None, "candidate_total": float(s1.sum()) if s1 is not None else None,
                    "difference": float(sign0.sum() - s1.sum()) if (sign0 is not None and s1 is not None) else None}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(common.to_jsonable(res), indent=1), encoding="utf-8")
    # markdown
    L = [f"# Entry comparison: shipped cache vs the GLNO=glutamate candidate ({utc()})", "",
         f"shipped `{res['shipped']['cache_dir']}` md5 {res['shipped']['md5']} sum|W| {res['shipped']['sum_abs_W']:,.0f} nnz {res['shipped']['nnz']:,}; "
         f"candidate `{res['candidate']['cache_dir']}` md5 {res['candidate']['md5']} sum|W| {res['candidate']['sum_abs_W']:,.0f} nnz {res['candidate']['nnz']:,}", "",
         f"neurons: same bodyIds/order {same_ids}, same types {same_type}; nt differs on {len(nt_diff)} cells (all GLNO: {res['neurons']['nt_differs_all_GLNO']}); "
         f"sign differs on {len(sign_diff)} cells (all GLNO: {res['neurons']['sign_differs_all_GLNO']})", "",
         f"W: same sparsity pattern {same_pattern}; differing entries {len(diff):,} (with a GLNO presynaptic cell {int(pre_is_glno.sum()):,}; shipped value 0 in all: "
         f"{res['entries']['shipped_values_all_zero']}; candidate value negative in all: {res['entries']['candidate_values_all_negative']}; |candidate| = the shipped sign-0 counts: "
         f"{res['entries']['candidate_abs_equals_shipped_sign0_counts']}); GLNO presynaptic entries {res['entries']['glno_pre_entries_total']} of which unchanged {res['entries']['glno_pre_entries_unchanged']}", "",
         "| post type | entries | raw synapses | share of GLNO output |", "|---|---|---|---|"]
    L += [f"| {r['post_type']} | {r['entries']} | {r['raw_syn']:.0f} | {100 * r['share']:.1f} % |" for r in res["by_post_type"]]
    pe = res["pen"]
    L += ["", f"GLNO -> PEN: {pe['glno_to_pen_entries']} entries, {pe['glno_to_pen_raw_syn']:.0f} raw synapses; edges {pe['edge_min']:.0f}-{pe['edge_max']:.0f} (mean {pe['edge_mean']:.0f}, median "
          f"{pe['edge_median']:.0f}), {pe['edges_above_cap']} above the cap {pe['conn_cap']:.0f}; GLNO inputs per PEN {pe['glno_inputs_per_pen']['min']}-{pe['glno_inputs_per_pen']['max']} "
          f"({pe['glno_inputs_per_pen']['n_pen_with_input']}/{pe['glno_inputs_per_pen']['n_pen']} PEN); effective {pe['effective_mv_per_glno_spike_candidate']['min']:.2f}..{pe['effective_mv_per_glno_spike_candidate']['max']:.2f} mV per GLNO spike "
          f"(w_syn {pe['w_syn']} x cap {pe['conn_cap']:.0f} x PEN fan-in scale {pe['pen_fan_in_scale_candidate']['min']:.2f}-{pe['pen_fan_in_scale_candidate']['max']:.2f}; shipped 0); "
          f"PEN -> GLNO {pe['pen_to_glno_entries']} entries, W sum {pe['pen_to_glno_W_sum_candidate']:+.0f} (shipped {pe['pen_to_glno_W_sum_shipped']:+.0f}), "
          f"{pe['pen_to_glno_effective_mv_per_pen_spike']['min']:.2f}..{pe['pen_to_glno_effective_mv_per_pen_spike']['max']:.2f} mV per PEN spike; GLNO fan-in scale {pe['glno_fan_in_scale_candidate']} (shipped {pe['glno_fan_in_scale_shipped']})",
          "", f"effective-weight matrix (Brain's A, mV): {res['effective']['entries_differing']:,} entries differ, {res['effective']['with_GLNO_pre']:,} with a GLNO presynaptic cell; "
          f"{res['effective']['cells_whose_fan_in_scale_moved']} postsynaptic cells' fan-in scale moved ({res['effective']['scale_moved_types']})",
          "", f"sign-0 counts: shipped {res['sign0']['shipped_total']}, candidate {res['sign0']['candidate_total']}, difference {res['sign0']['difference']}", ""]
    for rule in ("abs", "class"):
        r = res["receptor_model_on_GLNO_entries"][rule]
        L += [f"receptor model sign/{rule} on the 213 GLNO entries: {r['entries_neg']} entries -1 ({r['syn_neg']:.0f} syn), {r['entries_pos']} entries +1 ({r['syn_pos']:.0f} syn), {r['entries_zero']} entries 0; "
              "by post type (those with a +1 / 0 entry, and the main targets): " + "; ".join(f"{t['post_type']} {t['entries']} entries {t['syn']:.0f} syn -> -1 x{t['neg']} +1 x{t['pos']} 0 x{t['zero']} (tier {t['tiers']})" for t in r["by_post_type"])]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    log(f"-> {out}, {out.with_suffix('.md')}")
    return 0


# ---------------------------------------------------------------------------------------------- the jobs (GPU)
def _assert_cuda(allow_cpu: bool):
    import torch
    if not allow_cpu:
        assert torch.cuda.is_available(), "CUDA is not available on this node (resubmit the job)"
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "cpu"


def _finish_block(path: Path, block: dict, problems: list):
    d = json.loads(path.read_text(encoding="utf-8"))
    block["problems"] = problems
    d["glno_relabel"] = block
    path.write_text(json.dumps(common.to_jsonable(d), indent=1), encoding="utf-8")
    log(f"glno_relabel block appended to {path}: " + (("INVALID -- " + "; ".join(problems)) if problems else "valid"))
    if problems:
        sys.exit(3)


def cmd_suite(args) -> int:
    t0 = time.time(); started = utc()
    dev, dev_name = _assert_cuda(args.allow_cpu)
    cache_dir = ensure_scratch(Path(args.cache_dir)) if args.arm == "glu" else None
    captured = []
    orig_load = connectome.load

    def capturing_load(*a, **k):
        c = orig_load(*a, **k); captured.append((c, k.get("cache_dir"))); return c
    connectome.load = capturing_load
    argv = ["benchmark.py", "--sections", args.sections, "--seeds", args.seeds, "--json", args.out] + (["--cache-dir", str(cache_dir)] if cache_dir else []) + (["--eager"] if args.allow_cpu else [])
    if args.fast:
        argv.append("--fast")
    log(f"[suite {args.arm} draw {args.draw}] {' '.join(argv)}")
    import benchmark
    sys.argv = argv
    try:
        benchmark.main()
    finally:
        connectome.load = orig_load
    out = Path(args.out)
    if not out.exists():
        raise SystemExit(f"benchmark.py wrote no JSON at {out}")
    if not captured:
        raise SystemExit("no connectome was loaded through connectome.load (exit 3)")
    c, used = captured[0]
    state = glno_state(cache_dir)
    d = json.loads(out.read_text(encoding="utf-8"))
    stim = {"protocol": f"scripts/benchmark.py --sections {args.sections} --seeds {args.seeds}" + (" (the 29-check suite)" if args.sections == "all" else ""),
            "params": {"sections": args.sections, "seeds": [int(s) for s in args.seeds.split(",")], "cache_dir": str(cache_dir) if cache_dir else str(connectome.CACHE_DIR),
                       "arm": args.arm, "draw": int(args.draw), "type_nt_override": state["override_table"]},
            "control": "suite_shipped_<draw> of the same block (the shipped cache through the same harness)"}
    prov = common.provenance(c, device=dev, seeds=[int(s) for s in args.seeds.split(",")], batch=1, stimulus=stim, cache_dir=str(cache_dir) if cache_dir else None)
    d["provenance"] = prov
    out.write_text(json.dumps(common.to_jsonable(d), indent=1), encoding="utf-8")
    checks = d.get("checks", [])
    tally = {"pass": sum(ch["status"].startswith("PASS") for ch in checks), "fail": sum(ch["status"] == "FAIL" for ch in checks),
             "gap": sum(ch["status"] == "KNOWN GAP" for ch in checks), "missing": sum(ch["status"] == "MISSING" for ch in checks), "n": len(checks)}
    problems = []
    exp = expect_state(args.arm)
    if state["glno_nt"] != exp["glno_nt"] or state["glno_sign"] != exp["glno_sign"]:
        problems.append(f"GLNO in the cache the run used is nt {state['glno_nt']} sign {state['glno_sign']}, arm {args.arm} needs {exp}")
    if str(used or "") != (str(cache_dir) if cache_dir else "") and not (used is None and cache_dir is None):
        problems.append(f"connectome.load was called with cache_dir {used!r}, the arm asks for {cache_dir!r}")
    if dev != "cuda" and not args.allow_cpu:
        problems.append(f"device {dev}")
    if d["config"].get("device") in (None, "cpu") and not args.allow_cpu:
        problems.append(f"benchmark config.device {d['config'].get('device')!r}")
    if len(checks) != N_CHECKS and args.sections == "all" and not args.fast:
        problems.append(f"{len(checks)} checks, expected {N_CHECKS}")
    block = {"kind": "suite", "arm": args.arm, "draw": int(args.draw), "cache": state, "requested_cache_dir": str(cache_dir) if cache_dir else None,
             "connectome_load_cache_dir": str(used) if used else None, "compiled_connectome_md5": prov["compiled_connectome"].get("md5"),
             "device": dev, "device_name": dev_name, "benchmark_device": d["config"].get("device"), "benchmark_cache_dir": d["config"].get("cache_dir"),
             "receptor": d["config"].get("receptor", {}).get("model"), "tally": tally, "started_utc": started, "finished_utc": utc(), "wall_s": round(time.time() - t0, 1),
             "argv": argv, "generator": "scripts/glno_relabel.py suite " + " ".join(sys.argv[1:] if sys.argv[0].endswith("glno_relabel.py") else [])}
    _finish_block(out, block, problems)
    print(f"suite {args.arm} draw {args.draw}: {tally} device {dev_name} md5 {block['compiled_connectome_md5']} GLNO {state['glno_nt']} {state['glno_sign']} {block['wall_s']} s", flush=True)
    return 0


def cmd_compass(args) -> int:
    import runpy
    t0 = time.time(); started = utc()
    dev, dev_name = _assert_cuda(args.allow_cpu)
    cache_dir = ensure_scratch(Path(args.cache_dir)) if args.cache == "glu" else None
    gains = GAINS[args.gains]
    probe = ROOT / "scripts" / "probe_vnc_drive.py"
    argv = ["probe_vnc_drive.py", "compass", "--family", FAMILY, "--arm", ARM, "--seed", str(args.seed), "--gains", gains, "--block", args.block or f"fam_c{args.seed}", "--out", args.out]
    argv += ["--cache-dir", str(cache_dir)] if cache_dir else []
    argv += ["--quick", "--sparse", "torch", "--device", "cpu"] if args.allow_cpu else []
    log(f"[compass {args.cache} {args.gains} seed {args.seed}] {' '.join(argv)}  (probe sha256 {sha256_lf(probe)[:12]})")
    sys.argv = argv
    rc = 0
    try:
        runpy.run_path(str(probe), run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0) if not isinstance(e.code, str) else 1
    if rc != 0:
        raise SystemExit(rc)
    run = Path(f"{args.out}_run.json")
    if not run.exists():
        raise SystemExit(f"probe_vnc_drive.py wrote no {run}")
    d = json.loads(run.read_text(encoding="utf-8"))
    state = glno_state(cache_dir)
    prov = d.get("provenance", {}); sp = prov.get("stimulus", {}).get("params", {})
    problems = []
    exp = expect_state(args.cache)
    if state["glno_nt"] != exp["glno_nt"] or state["glno_sign"] != exp["glno_sign"]:
        problems.append(f"GLNO in the cache the run used is nt {state['glno_nt']} sign {state['glno_sign']}, cache arm {args.cache} needs {exp}")
    want_gains = None if gains == "none" else [float(x) for x in gains.split(":")]
    if (d.get("gains") or None) != want_gains:
        problems.append(f"gains recorded {d.get('gains')} != requested {want_gains}")
    if d.get("family") != FAMILY or d.get("arm") != ARM or d.get("proprioception") != "all+leg_cycle":
        problems.append(f"family / arm / spec {d.get('family')} / {d.get('arm')} / {d.get('proprioception')} != {FAMILY} / {ARM} / all+leg_cycle")
    if not d.get("leg_cycle"):
        problems.append("leg_cycle is not on")
    rdev = str(prov.get("execution", {}).get("device") or d.get("device"))
    if "cuda" not in rdev and not args.allow_cpu:
        problems.append(f"realised device {rdev}")
    cd_recorded = prov.get("compiled_connectome", {}).get("cache_dir")
    block = {"kind": "compass", "cache": args.cache, "gains_arm": args.gains, "gains_requested": gains, "seed": int(args.seed), "block": args.block,
             "cache_state": state, "requested_cache_dir": str(cache_dir) if cache_dir else None, "provenance_cache_dir": cd_recorded,
             "compiled_connectome_md5": prov.get("compiled_connectome", {}).get("md5"), "sum_abs_W": prov.get("compiled_connectome", {}).get("sum_abs_W"),
             "family": d.get("family"), "arm": d.get("arm"), "proprioception": d.get("proprioception"), "leg_cycle": d.get("leg_cycle"), "haltere_sided": d.get("haltere_sided"),
             "mn_ref_hz": sp.get("mn_ref_hz", d.get("mn_ref_hz")), "channels": sp.get("channels"), "compass_adaptation": sp.get("compass_adaptation"),
             "background_hz": sp.get("background_hz"), "pulse_hz": sp.get("pulse_hz"), "dna02_hz": sp.get("dna02_hz"), "rate_dps": sp.get("rate_dps"),
             "device": rdev, "device_name": d.get("device_name"), "probe_sha256_lf": sha256_lf(probe), "started_utc": started, "finished_utc": utc(),
             "wall_s": round(time.time() - t0, 1), "argv": argv}
    _finish_block(run, block, problems)
    print(f"compass {args.cache} {args.gains} seed {args.seed}: device {rdev} md5 {block['compiled_connectome_md5']} GLNO {state['glno_nt']} {state['glno_sign']} "
          f"drift " + " ".join(f"{ph} {d['phases'][ph]['bump']['drift_wedges_per_s']:+.4f}" for ph in d.get("phases", {})) + f" {block['wall_s']} s", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- plan (CPU)
PREDECLARED = {
    "question": "Is the transmitter relabel {GLNO: glutamate} adoptable under the round-2 rule, and does a signed GLNO make the pinned fly's "
                "efferent self-turn report pass GLNO and move PEN / the compass bump?",
    "candidate": "connectome.TYPE_NT_OVERRIDE + {'GLNO': 'glutamate'}: only cells with NO usable fast label are relabelled (all 4 GLNO are "
                 "'unknown'); GLNO's 17,698 raw output synapses go from explicit 0 to sign -1 (NT_SIGN['glutamate']); nothing else changes "
                 "(verified entry by entry on CPU, out/cx5b/entry_compare.json)",
    "replicate_unit": "runs: one benchmark.py --sections all --seeds 0,1,2 process = one draw; one probe_vnc_drive.py compass process = one run (one seed)",
    "design": {"suite": "3 draws x {shipped, glu}; the same harness (scripts/benchmark.py through scripts/glno_relabel.py suite), the glu arm on the scratch cache",
               "compass": "4 seeds (0-3) x {shipped, glu} x {exp gains gE 2 / gD 15, shipped gains (none: no background, no pulse, adaptation as shipped)}; "
                          "probe_vnc_drive.py compass --family level --arm C (proprioception all+leg_cycle), DNa02_L / _R at 20 Hz for 10 s, 3 s skipped, phases rest / ccw / rest2 / cw",
               "jobs": "22 in ONE cluster_run submission on the house target (--arm-block fam: fam_suite = the 6 suite jobs; fam_c<seed> = the 4 compass arms of one seed)"},
    "adoption_rule": {"statement": "GLNO=glutamate is ADOPTABLE under the round-2 rule iff, in this submission, no check of the 29-check suite has a status in any "
                                   "glutamate draw that is not the status of every shipped draw (3 draws each). Statuses: PASS / PASS (gap closed) / KNOWN GAP / FAIL / "
                                   "MISSING; 'PASS (gap closed)' counts as PASS for a gap row. A check whose three shipped draws disagree among themselves is 'unstable "
                                   "under the shipped cache' and is reported, not counted against the candidate (it cannot be attributed to the relabel).",
                      "draw_matched_reading": "also reported: glutamate draw k vs shipped draw k (the same seeds), and the measured values with common.compare (3 v 3 = "
                                              "underpowered by construction; the scatter is what those rows carry).",
                      "not_here": "the room take-off protocol (the rate-half of the round-7 rule) is NOT run in this thread; the verdict is on the suite half only and says so."},
    "primaries": {"families": {
        "F1_glu_vs_shipped_at_exp_gains": {"members": ["GLNO flip (ccw - cw L-R)", "PEN_a(PEN1) flip", "PEN_b(PEN2) flip", "bump drift ccw - cw (wedges/s)"],
                                           "test": "common.compare(glu 4 runs, shipped 4 runs) per member; Holm m = 4", "call": "result iff |z| >= 3 and p_holm <= 0.05"},
        "F2_glu_vs_shipped_at_shipped_gains": {"members": ["GLNO flip", "PEN_a(PEN1) flip", "PEN_b(PEN2) flip"], "test": "as F1; Holm m = 3",
                                               "note": "no bump forms at the shipped gains (round 3), so the drift is reported, not a member"},
        "F3_signed_report_glu_exp": {"members": ["GLNO flip vs its null (rest2 - rest)", "PEN_a flip vs null", "PEN_b flip vs null"],
                                     "test": "interp_apply_rotation.flip_table on the glu / exp-gains arm: flips vs nulls 4 v 4 (common.compare); Holm m = 3",
                                     "call": "the signed self-turn report PASSES GLNO iff the GLNO member is result (|z| >= 3, p_holm <= 0.05)"},
        "F4_signed_report_glu_shipped": {"members": ["GLNO flip vs null", "PEN_a flip vs null", "PEN_b flip vs null"], "test": "as F3 on the glu / shipped-gains arm; Holm m = 3"},
        "F5_drift_glu_exp": {"members": ["ccw drift vs the rests (8 rest phases of the 4 runs)", "cw drift vs the rests"],
                             "test": "common.compare per member; Holm m = 2", "ideal": "+4.0 wedges/s at 90 deg/s (16 wedges per turn), opposite signs for ccw / cw"}}},
    "secondary_reported_not_called": ["the same rows for the shipped arms (F3 / F4 on shipped caches = round 4's reading re-run)", "PS196_b, AN04B003, EPG, DNa02 flips",
                                      "GLNO L-R per phase (round 4: +26.9..+28.1 Hz fixed)", "chain rates per phase", "bump vs / peak", "suite measured values (3 v 3)"],
    "expectations_written_before_the_runs": {"suite": "unknown; cx_glno.md 3-5 changed only ring rates (PEN -14..-16 %, bump -9 % at 2/15) with no bump loss, so the "
                                                      "compass.wedge_cells_persisting gap row is the one to watch",
                                             "compass": "rounds 2-4: nothing moves the bump under any arm (drift |mean| <= 0.005 w/s), GLNO L-R fixed at +27 Hz, "
                                                        "PEN flips <= 0.1 Hz; a signed GLNO closes the PEN <-> GLNO loop but PS196_b's input to GLNO is symmetric "
                                                        "(vnc_drive.md 6), so the prior is: GLNO flip still null, PEN flips still null, drift still ~0"},
    "provenance_required_per_run": ["provenance.compiled_connectome.md5 (must differ between the two caches)", "glno_relabel.cache_state.glno (nt / sign of the 4 cells read from the cache the run used)",
                                    "glno_relabel.mn_ref_hz / channels / gains (compass)", "execution.device cuda + device_name", "started / finished UTC"],
}


def tree_state(extra_files=()) -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"unavailable: {e!r}"
    files = sorted(glob.glob(str(ROOT / "flyverse" / "*.py")) + glob.glob(str(ROOT / "flyverse" / "interp" / "*.py")) + glob.glob(str(ROOT / "flyverse" / "backends" / "*.py")) +
                   [str(ROOT / "scripts" / f) for f in ("glno_relabel.py", "probe_vnc_drive.py", "interp_apply_rotation.py", "cx_wedge.py", "benchmark.py", "room_demo.py", "cluster_run.py")] +
                   [str(f) for f in extra_files])
    sha = {str(Path(f).relative_to(ROOT)).replace("\\", "/"): sha256_lf(Path(f)) for f in files if Path(f).exists()}
    head_probe = run(["git", "show", "HEAD:scripts/probe_vnc_drive.py"])
    head_sha = hashlib.sha256(head_probe.replace("\r\n", "\n").encode("utf-8") + b"\n").hexdigest() if head_probe and not head_probe.startswith("unavailable") else None
    return {"commit": run(["git", "rev-parse", "HEAD"]), "status_porcelain": run(["git", "status", "--porcelain"]).splitlines(),
            "diff_stat": run(["git", "diff", "--stat"]).splitlines()[-1:], "sha256_lf": sha,
            "probe_vnc_drive": {"shipped_sha256_lf": sha.get("scripts/probe_vnc_drive.py"), "head_sha256_lf": head_sha,
                                "shipped_is_head": sha.get("scripts/probe_vnc_drive.py") == head_sha,
                                "note": "cluster_run ships the working tree (git ls-files -m -o --exclude-standard), so the compass jobs run the working-tree file; "
                                        "it is called as it is (no edit by this thread); git diff HEAD -- scripts/probe_vnc_drive.py at submission is in probe_diff_vs_HEAD.txt"},
            "when": utc()}


def job_lines(d: str, seeds, draws):
    G = "python -c 'import torch; assert torch.cuda.is_available()'"
    pre = f"mkdir -p {d} && source .venv/bin/activate && {G} && "
    jobs = []
    for arm in ("shipped", "glu"):
        for n in range(1, draws + 1):
            D = f"{d}/fam_suite"; tag = f"suite_{arm}_{n}"
            cmd = (pre + f"mkdir -p {D} && PYTHONIOENCODING=utf-8 python scripts/glno_relabel.py suite --arm {arm} --draw {n} --out {D}/{tag}.json > {D}/{tag}.txt 2>&1; "
                   f"st=$?; test -s {D}/{tag}.json || st=1; tail -4 {D}/{tag}.txt; exit $st")
            jobs.append({"job": tag, "block": "fam_suite", "kind": "suite", "arm": arm, "draw": n, "json": f"{D}/{tag}.json", "line": cmd})
    for s in seeds:
        for cache in ("shipped", "glu"):
            for g in ("exp", "shipped"):
                D = f"{d}/fam_c{s}"; tag = f"compass_{cache}_{g}_r{s}"
                cmd = (pre + f"mkdir -p {D} && PYTHONIOENCODING=utf-8 python scripts/glno_relabel.py compass --cache {cache} --gains {g} --seed {s} --block fam_c{s} "
                       f"--out {D}/{tag} > {D}/{tag}.txt 2>&1; st=$?; test -s {D}/{tag}_run.json || st=1; tail -4 {D}/{tag}.txt; exit $st")
                jobs.append({"job": tag, "block": f"fam_c{s}", "kind": "compass", "cache": cache, "gains": g, "seed": s, "json": f"{D}/{tag}_run.json", "line": cmd})
    return jobs


def cmd_plan(args) -> int:
    import shlex
    d = args.dir.rstrip("/")
    Path(d).mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = job_lines(d, seeds, args.draws)
    pre = {"predeclared": PREDECLARED, "stamped_utc": utc(), "before_submission": True, "generator": "scripts/glno_relabel.py plan " + " ".join(sys.argv[2:]),
           "seeds": seeds, "draws": args.draws, "n_jobs": len(jobs), "gains": GAINS, "family": FAMILY, "arm": ARM, "cluster_name": args.name, "minutes": args.minutes}
    (Path(d) / "predeclared.json").write_text(json.dumps(pre, indent=1), encoding="utf-8")
    (Path(d) / "tree_state.json").write_text(json.dumps(tree_state(), indent=1), encoding="utf-8")
    try:
        diff = subprocess.run(["git", "diff", "HEAD", "--", "scripts/probe_vnc_drive.py"], cwd=str(ROOT), capture_output=True, text=True, timeout=60).stdout
        (Path(d) / "probe_diff_vs_HEAD.txt").write_text(diff, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        (Path(d) / "probe_diff_vs_HEAD.txt").write_text(f"unavailable: {e!r}\n", encoding="utf-8")
    (Path(d) / "jobs.json").write_text(json.dumps({"jobs": jobs, "written_utc": utc()}, indent=1), encoding="utf-8")
    log_path = f"out/{args.name}_cluster.log"
    sh = ["#!/bin/sh", f"# generated by scripts/glno_relabel.py plan on {utc()}: {len(jobs)} jobs, one submission", "set -e", f"mkdir -p {d} out",
          f'if [ -f {log_path} ]; then mv {log_path} "out/{args.name}_cluster.$(date +%Y%m%dT%H%M%S).log"; fi',
          f"PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name {args.name} --minutes {args.minutes} --arm-block fam \\"]
    sh += [f"  {shlex.quote(j['line'])} \\" for j in jobs]
    sh += [f"  --fetch {d}/ 2>&1 | tee {log_path}"]
    (Path(d) / "batch.sh").write_text("\n".join(sh) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(jobs)} jobs -> {d}/batch.sh (log {log_path}); predeclared.json stamped {pre['stamped_utc']}; tree_state.json, jobs.json, probe_diff_vs_HEAD.txt written")
    for j in jobs:
        print(f"  {j['block']:10s} {j['job']}")
    return 0


# ---------------------------------------------------------------------------------------------- analyse (CPU)
def holm(pvals):
    """Holm step-down adjusted p-values (NaN stays NaN and does not count in m)."""
    p = np.asarray(pvals, float); out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p)); m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for i, idx in enumerate(order):
        adj = min(1.0, (m - i) * p[idx]); running = max(running, adj); out[idx] = running
    return out


def status_rank(s: str) -> str:
    return "PASS" if str(s).startswith("PASS") else str(s)


def analyse_suite(d: str, out: Path, lines: list, summary: dict):
    runs = {arm: sorted(glob.glob(f"{d}/fam_suite/suite_{arm}_*.json")) for arm in ("shipped", "glu")}
    data = {arm: [json.loads(Path(f).read_text(encoding="utf-8")) for f in fs] for arm, fs in runs.items()}
    summary["suite"] = {"files": runs, "runs": {}}
    if not data["shipped"] or not data["glu"]:
        lines += ["## 1. The suite", "", f"missing runs: shipped {len(data['shipped'])}, glu {len(data['glu'])}", ""]
        summary["suite"]["verdict"] = "INCOMPLETE"; return
    keys = []
    for arm in data:
        for j in data[arm]:
            for ch in j["checks"]:
                if ch["key"] not in keys:
                    keys.append(ch["key"])
    lines += ["## 1. The 29-check suite (scripts/benchmark.py --sections all --seeds 0,1,2 through glno_relabel.py suite; shipped cache vs the GLNO=glutamate cache; 3 draws each)", ""]
    lines += ["| run | file | device | cache md5 | GLNO nt / sign | receptor | pass / fail / gap / missing | wall s | problems |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in data:
        for f, j in zip(runs[arm], data[arm]):
            b = j.get("glno_relabel", {}); cs = b.get("cache_state", b.get("cache", {})); t = b.get("tally", {})
            summary["suite"]["runs"][f] = {"arm": arm, "device": b.get("device_name"), "md5": b.get("compiled_connectome_md5"), "glno_nt": cs.get("glno_nt"), "glno_sign": cs.get("glno_sign"),
                                           "tally": t, "problems": b.get("problems"), "checks": {ch["key"]: [ch["measured"], ch["status"]] for ch in j["checks"]}}
            lines.append(f"| {arm} d{b.get('draw')} | {f} | {b.get('device_name')} | {str(b.get('compiled_connectome_md5'))[:8]} | {cs.get('glno_nt')} / {cs.get('glno_sign')} | "
                         f"{b.get('receptor')} | {t.get('pass')} / {t.get('fail')} / {t.get('gap')} / {t.get('missing')} | {b.get('wall_s')} | {b.get('problems')} |")
    lines += ["", "| check (criterion) | shipped d1 | d2 | d3 | glu d1 | d2 | d3 | shipped statuses | glu statuses | status changed (pooled) | changed (draw-matched) | measured: compare glu vs shipped (3 v 3) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    changed_pooled, unstable, changed_matched, rows = [], [], [], []
    for k in keys:
        cells, crit, st = {}, "", {}
        for arm in data:
            for i, j in enumerate(data[arm]):
                ch = next((c for c in j["checks"] if c["key"] == k), None)
                v = ch["measured"] if ch else None; s = ch["status"] if ch else "MISSING"; crit = ch["criterion"] if ch else crit
                cells[(arm, i)] = (v, s); st.setdefault(arm, []).append(status_rank(s))
        ship_set, glu_set = set(st["shipped"]), set(st["glu"])
        pooled = not glu_set.issubset(ship_set); matched = [i + 1 for i in range(min(len(st["shipped"]), len(st["glu"]))) if st["glu"][i] != st["shipped"][i]]
        if len(ship_set) > 1: unstable.append(k)
        if pooled and len(ship_set) == 1: changed_pooled.append(k)
        if matched: changed_matched.append((k, matched))
        a = [cells[("glu", i)][0] for i in range(len(data["glu"]))]; b = [cells[("shipped", i)][0] for i in range(len(data["shipped"]))]
        cmp_txt = ""
        if all(isinstance(x, (int, float)) and x is not None for x in a + b):
            r = common.compare(np.array(a, float), np.array(b, float))
            cmp_txt = f"diff {r['diff']:+.4g}, z {r['z']:.2g}, p {r['p']:.3g} ({r['verdict']})"
            rows.append({"check": k, "criterion": crit, "glu_runs": a, "shipped_runs": b, "diff": r["diff"], "z": r["z"], "p": r["p"], "verdict": r["verdict"],
                         "shipped_statuses": st["shipped"], "glu_statuses": st["glu"], "changed_pooled": pooled, "changed_matched": matched, "unstable_shipped": len(ship_set) > 1})
        else:
            rows.append({"check": k, "criterion": crit, "glu_runs": a, "shipped_runs": b, "shipped_statuses": st["shipped"], "glu_statuses": st["glu"], "changed_pooled": pooled, "changed_matched": matched, "unstable_shipped": len(ship_set) > 1})

        def fmt(v, s):
            tag = {"PASS": "P", "PASS (gap closed)": "P*", "KNOWN GAP": "G", "FAIL": "F", "MISSING": "M"}.get(s, s)
            return (f"{v:.4g}" if isinstance(v, (int, float)) and v is not None else str(v)) + f" {tag}"
        lines.append(f"| {k} ({crit}) | " + " | ".join(fmt(*cells.get(('shipped', i), (None, 'MISSING'))) for i in range(3)) + " | " +
                     " | ".join(fmt(*cells.get(('glu', i), (None, 'MISSING'))) for i in range(3)) + f" | {'/'.join(st['shipped'])} | {'/'.join(st['glu'])} | "
                     f"{'YES' if pooled else 'no'}{' (shipped unstable)' if len(ship_set) > 1 else ''} | {('d' + ','.join(map(str, matched))) if matched else 'no'} | {cmp_txt} |")
    pd.DataFrame(rows).to_csv(out / "suite_table.csv", index=False)
    n_ship, n_glu = len(data["shipped"]), len(data["glu"])
    complete = n_ship >= 3 and n_glu >= 3 and all(len(j["checks"]) == N_CHECKS for j in data["shipped"] + data["glu"])
    problems = [f for f, r in summary["suite"]["runs"].items() if r.get("problems")]
    verdict = ("ADOPTABLE by the suite half of the round-2 rule" if (complete and not changed_pooled and not problems) else
               "NOT adoptable by the rule" if (complete and not problems) else "INCOMPLETE (missing runs or invalid provenance)")
    lines += ["", "P PASS, P* PASS (gap closed), G KNOWN GAP, F FAIL, M MISSING. 'status changed (pooled)' = a glu draw's status is not the status of every shipped draw "
              "(gap rows: PASS (gap closed) and PASS count as the same status).", "",
              f"**Checks whose status changed under the rule: {changed_pooled or 'none'}**; draw-matched differences: {changed_matched or 'none'}; "
              f"checks unstable among the shipped draws (reported, not counted): {unstable or 'none'}; runs with provenance problems: {problems or 'none'}.", "",
              f"**Suite verdict ({n_ship} shipped + {n_glu} glu draws, {sum(len(j['checks']) for j in data['shipped'] + data['glu'])} check rows): {verdict}.**", ""]
    summary["suite"].update({"verdict": verdict, "changed_pooled": changed_pooled, "changed_matched": changed_matched, "unstable_shipped": unstable, "problems": problems,
                             "n_shipped": n_ship, "n_glu": n_glu, "complete": complete})


def load_compass_runs(d: str):
    runs = {}
    for p in sorted(glob.glob(f"{d}/fam_c*/compass_*_r*_run.json")):
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        b = j.get("glno_relabel", {})
        arm = f"{b.get('cache', '?')}_{b.get('gains_arm', '?')}"
        runs.setdefault(arm, []).append((p, j))
    for arm in runs:
        runs[arm].sort(key=lambda t: int(t[1]["seed"]))
    return runs


def analyse_compass(d: str, out: Path, lines: list, summary: dict, cache_dir=None):
    from flyverse.interp import trace as tr
    import interp_apply_rotation as rot
    runs = load_compass_runs(d)
    summary["compass"] = {"arms": {}, "families": {}}
    lines += ["## 2. The efferent compass (probe_vnc_drive.py compass --family level --arm C = all+leg_cycle; DNa02_L / _R 20 Hz for 10 s, 3 s skipped; 4 seeds per arm)", ""]
    if not runs:
        lines += ["no compass runs found", ""]; return
    c = connectome.load(cache_dir=Path(cache_dir), verbose=False) if cache_dir else connectome.load(verbose=False)
    lines += ["| arm | seed | file | device | cache md5 | GLNO nt / sign | gains | spec | mn_ref_hz | drift rest / ccw / rest2 / cw (w/s) | bump vs (rest) | peak Hz (rest) | heading ccw / cw (deg/s) | wall s | problems |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    per_arm = {}
    for arm, items in runs.items():
        summary["compass"]["arms"][arm] = {"runs": []}
        for p, j in items:
            b = j.get("glno_relabel", {}); cs = b.get("cache_state", {}); ph = j["phases"]
            dr = {k: ph[k]["bump"]["drift_wedges_per_s"] for k in PHASES if k in ph}
            summary["compass"]["arms"][arm]["runs"].append({"file": p, "seed": j["seed"], "device": j.get("device_name"), "md5": b.get("compiled_connectome_md5"), "glno_nt": cs.get("glno_nt"),
                                                            "glno_sign": cs.get("glno_sign"), "gains": j.get("gains"), "drift": dr, "problems": b.get("problems"),
                                                            "vs": {k: ph[k]["bump"]["vs"] for k in ph}, "peak": {k: ph[k]["bump"]["peak"] for k in ph}, "heading": {k: ph[k]["heading"]["rate_dps"] for k in ph}})
            lines.append(f"| {arm} | {j['seed']} | {p} | {j.get('device_name')} | {str(b.get('compiled_connectome_md5'))[:8]} | {cs.get('glno_nt')} / {cs.get('glno_sign')} | {j.get('gains')} | "
                         f"{j.get('proprioception')} | {b.get('mn_ref_hz')} | " + " / ".join(f"{dr.get(k, float('nan')):+.4f}" for k in PHASES) +
                         f" | {ph['rest']['bump']['vs']:.2f} | {ph['rest']['bump']['peak']:.0f} | {ph['ccw']['heading']['rate_dps']:+.1f} / {ph['cw']['heading']['rate_dps']:+.1f} | {j.get('wall_s')} | {b.get('problems')} |")
        # flip tables from the per-cell recordings
        cells = {ph: [] for ph in PHASES}
        for p, j in items:
            stem = p[: -len("_run.json")]
            for ph in PHASES:
                cells[ph].append(tr.load_run(Path(f"{stem}_{ph}.npz")))
        ft = rot.flip_table(c, cells); ft.to_csv(out / f"compass_flip_{arm}.csv", index=False)
        cr = rot.chain_rates(c, cells, types=KEY_TYPES); cr.to_csv(out / f"compass_chain_{arm}.csv", index=False)
        per_arm[arm] = {"ft": ft, "cr": cr, "items": items}
    lines.append("")
    # flip rows, key types, per arm, with the per-run values
    lines += ["### 2.1 Flip rows (L-R at ccw minus cw, against the run's own null rest2 - rest; 4 v 4, exact-U floor 0.029; per-run values in brackets)", "",
              "| arm | type | n_L / n_R | LR rest / ccw / rest2 / cw (Hz) | flip mean +- sd [per run] | null mean +- sd [per run] | z | p | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    flip_runs = {}
    for arm, pa in per_arm.items():
        ft = pa["ft"]
        for t in KEY_TYPES:
            r = ft[ft.type == t]
            if r.empty: continue
            r = r.iloc[0]
            fr = [float(x) for x in r["flip_runs"]] if "flip_runs" in r and isinstance(r["flip_runs"], (list, np.ndarray)) else None
            nr = [float(x) for x in r["null_runs"]] if "null_runs" in r and isinstance(r["null_runs"], (list, np.ndarray)) else None
            flip_runs[(arm, t)] = {"flips": fr, "nulls": nr, "z": float(r.z), "p": float(r.p), "verdict": str(r.verdict), "flip_mean": float(r.flip_mean), "null_mean": float(r.null_mean),
                                   "LR": {ph: float(r[f"LR_{ph}"]) for ph in PHASES if f"LR_{ph}" in r}}
            lines.append(f"| {arm} | {t} | {int(r.n_L)} / {int(r.n_R)} | " + " / ".join(f"{r[f'LR_{ph}']:+.2f}" for ph in PHASES) +
                         f" | {r.flip_mean:+.3f} +- {r.get('flip_sd', float('nan')):.3f} [{', '.join(f'{x:+.2f}' for x in fr) if fr else ''}] | "
                         f"{r.null_mean:+.3f} +- {r.get('null_sd', float('nan')):.3f} [{', '.join(f'{x:+.2f}' for x in nr) if nr else ''}] | {r.z:.2f} | {r.p:.3g} | {r.verdict} |")
    summary["compass"]["flip_rows"] = {f"{a}|{t}": v for (a, t), v in flip_runs.items()}
    lines.append("")
    # chain rates per phase
    lines += ["### 2.2 Rates per phase (L / R Hz, mean over runs; max single cell) -- never 'silent' for a cell with nonzero firing", "", "| arm | phase | type | L +- sd | R +- sd | L-R | max cell |", "|---|---|---|---|---|---|---|"]
    for arm, pa in per_arm.items():
        cr = pa["cr"]
        for _, r in cr.iterrows():
            lines.append(f"| {arm} | {r.phase} | {r.type} | {r.L:.2f} +- {r.L_sd:.2f} | {r.R:.2f} +- {r.R_sd:.2f} | {r.L - r.R:+.2f} | {r.max_cell:.1f} |")
    lines.append("")
    # drift per arm
    lines += ["### 2.3 Bump drift per phase (wedges/s; ideal +4.0 ccw / -4.0 cw at 90 deg/s; per-run values)", "", "| arm | phase | mean +- sd | per run | vs mean | peak Hz mean |", "|---|---|---|---|---|---|"]
    drift = {}
    for arm, pa in per_arm.items():
        for ph in PHASES:
            v = np.array([j["phases"][ph]["bump"]["drift_wedges_per_s"] for _, j in pa["items"]], float)
            vs = np.array([j["phases"][ph]["bump"]["vs"] for _, j in pa["items"]], float); pk = np.array([j["phases"][ph]["bump"]["peak"] for _, j in pa["items"]], float)
            drift[(arm, ph)] = v
            lines.append(f"| {arm} | {ph} | {np.nanmean(v):+.4f} +- {np.nanstd(v, ddof=1) if len(v) > 1 else float('nan'):.4f} | {', '.join(f'{x:+.4f}' for x in v)} | {np.nanmean(vs):.2f} | {np.nanmean(pk):.0f} |")
    lines.append("")
    # predeclared families
    fam_rows = []

    def member(fam, name, a, b, note=""):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if len(a) == 0 or len(b) == 0 or np.isnan(a).all() or np.isnan(b).all():
            fam_rows.append({"family": fam, "member": name, "a": a.tolist(), "b": b.tolist(), "diff": np.nan, "z": np.nan, "p": np.nan, "verdict": "missing", "note": note}); return
        r = common.compare(a, b)
        fam_rows.append({"family": fam, "member": name, "a": a.tolist(), "b": b.tolist(), "diff": r["diff"], "z": r["z"], "p": r["p"], "p_floor": r["p_floor"], "verdict": r["verdict"], "note": note})

    def flips_of(arm, t):
        v = flip_runs.get((arm, t)); return v["flips"] if v and v["flips"] is not None else []

    def nulls_of(arm, t):
        v = flip_runs.get((arm, t)); return v["nulls"] if v and v["nulls"] is not None else []

    def dcc(arm):
        return np.array(drift.get((arm, "ccw"), [])) - np.array(drift.get((arm, "cw"), []))
    for g, fam in (("exp", "F1_glu_vs_shipped_at_exp_gains"), ("shipped", "F2_glu_vs_shipped_at_shipped_gains")):
        for t in ("GLNO", "PEN_a(PEN1)", "PEN_b(PEN2)"):
            member(fam, f"{t} flip", flips_of(f"glu_{g}", t), flips_of(f"shipped_{g}", t), "glu runs vs shipped runs")
        if g == "exp":
            member(fam, "bump drift ccw - cw", dcc("glu_exp"), dcc("shipped_exp"), "glu runs vs shipped runs")
    for g, fam in (("exp", "F3_signed_report_glu_exp"), ("shipped", "F4_signed_report_glu_shipped")):
        for t in ("GLNO", "PEN_a(PEN1)", "PEN_b(PEN2)"):
            member(fam, f"{t} flip vs null", flips_of(f"glu_{g}", t), nulls_of(f"glu_{g}", t), "flip_table on the glu arm")
    rests = np.concatenate([drift.get(("glu_exp", "rest"), np.array([])), drift.get(("glu_exp", "rest2"), np.array([]))])
    member("F5_drift_glu_exp", "ccw drift vs rests", drift.get(("glu_exp", "ccw"), []), rests)
    member("F5_drift_glu_exp", "cw drift vs rests", drift.get(("glu_exp", "cw"), []), rests)
    # the same rows on the shipped arms, reported not called
    for g in ("exp", "shipped"):
        for t in ("GLNO", "PEN_a(PEN1)", "PEN_b(PEN2)", "PS196_b", "AN04B003", "EPG"):
            member(f"S_signed_report_shipped_{g}", f"{t} flip vs null", flips_of(f"shipped_{g}", t), nulls_of(f"shipped_{g}", t), "secondary: the shipped arm (round 4 re-run)")
        for t in ("PS196_b", "AN04B003", "EPG"):
            member(f"S_signed_report_glu_{g}", f"{t} flip vs null", flips_of(f"glu_{g}", t), nulls_of(f"glu_{g}", t), "secondary")
    fdf = pd.DataFrame(fam_rows)
    fdf["p_holm"] = np.nan; fdf["called"] = False
    for fam in fdf.family.unique():
        if fam.startswith("S_"): continue
        m = fdf.family == fam
        fdf.loc[m, "p_holm"] = holm(fdf.loc[m, "p"].to_numpy())
        fdf.loc[m, "called"] = (fdf.loc[m, "verdict"] == "result") & (fdf.loc[m, "p_holm"] <= 0.05) & (np.abs(fdf.loc[m, "z"]) >= 3)
    fdf.to_csv(out / "decision_table.csv", index=False)
    lines += ["### 2.4 Predeclared families (common.compare over runs, 4 v 4; Holm within each family; a member is CALLED iff verdict result, |z| >= 3 and p_holm <= 0.05)", "",
              "| family | member | treatment runs | reference runs | diff | z | p | p_holm | verdict | called |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in fdf.iterrows():
        lines.append(f"| {r.family} | {r.member} | {', '.join(f'{x:+.3f}' for x in r.a)} | {', '.join(f'{x:+.3f}' for x in r.b)} | {r['diff']:+.4f} | {r.z:.2f} | {r.p:.3g} | "
                     f"{'' if pd.isna(r.p_holm) else f'{r.p_holm:.3g}'} | {r.verdict} | {'' if r.family.startswith('S_') else ('YES' if r.called else 'no')} |")
    lines.append("")
    summary["compass"]["families"] = fdf.to_dict("records")
    summary["compass"]["drift"] = {f"{a}|{ph}": v.tolist() for (a, ph), v in drift.items()}


def cmd_analyse(args) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lines = [f"# GLNO=glutamate: suite and compass analysis (generated by scripts/glno_relabel.py analyse, {utc()})", ""]
    summary = {"when": utc(), "dir": args.dir}
    pre = Path(args.dir) / "predeclared.json"
    if pre.exists():
        summary["predeclared"] = json.loads(pre.read_text(encoding="utf-8"))
        lines += [f"Predeclaration: `{pre}` stamped {summary['predeclared'].get('stamped_utc')}", ""]
    analyse_suite(args.dir, out, lines, summary)
    if not args.skip_compass:
        analyse_compass(args.dir, out, lines, summary, cache_dir=args.cache_dir)
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(common.to_jsonable(summary), indent=1, default=str), encoding="utf-8")
    print("\n".join(lines)); print(f"\n-> {out}/analysis.md, summary.json, suite_table.csv, decision_table.csv, compass_flip_*.csv, compass_chain_*.csv")
    return 0


# ---------------------------------------------------------------------------------------------- smoke (CPU)
def cmd_smoke(args) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    d = Path(args.dir) / "smoke"; d.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", SDL_VIDEODRIVER="dummy")
    cmds = [[py, "scripts/glno_relabel.py", "compass", "--cache", "shipped", "--gains", "exp", "--seed", "0", "--block", "fam_smoke", "--allow-cpu", "--out", str(d / "compass_shipped_exp_r0")],
            [py, "scripts/glno_relabel.py", "compass", "--cache", "glu", "--gains", "shipped", "--seed", "0", "--block", "fam_smoke", "--allow-cpu", "--cache-dir", args.cache_dir, "--out", str(d / "compass_glu_shipped_r0")]]
    if args.suite:
        cmds.append([py, "scripts/glno_relabel.py", "suite", "--arm", "glu", "--draw", "1", "--allow-cpu", "--fast", "--sections", "rest", "--cache-dir", args.cache_dir, "--out", str(d / "suite_glu_1.json")])
    rc = 0
    for cmd in cmds:
        log("smoke: " + " ".join(cmd[1:]))
        r = subprocess.run(cmd, cwd=str(ROOT), env=env)
        log(f"exit {r.returncode}"); rc |= r.returncode
    for f in sorted(d.glob("*_run.json")) + sorted(d.glob("suite_*.json")):
        j = json.loads(f.read_text(encoding="utf-8")); b = j.get("glno_relabel", {})
        print(f"{f}: problems {b.get('problems')} cache {b.get('requested_cache_dir')} GLNO {b.get('cache_state', b.get('cache', {})).get('glno_nt')} md5 {b.get('compiled_connectome_md5')} "
              f"gains {j.get('gains')} spec {j.get('proprioception')} device {b.get('device')}")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compile"); p.add_argument("--cache-dir", default=str(SCRATCH))
    p = sub.add_parser("compare"); p.add_argument("--cache-dir", default=str(SCRATCH)); p.add_argument("--out", default=str(OUT / "cx5b" / "entry_compare.json"))
    p = sub.add_parser("plan"); p.add_argument("--dir", default="out/cx5b"); p.add_argument("--name", default="cx5b"); p.add_argument("--minutes", type=int, default=45)
    p.add_argument("--seeds", default="0,1,2,3"); p.add_argument("--draws", type=int, default=3)
    p = sub.add_parser("suite"); p.add_argument("--arm", required=True, choices=["shipped", "glu"]); p.add_argument("--draw", type=int, required=True); p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=str(SCRATCH)); p.add_argument("--sections", default="all"); p.add_argument("--seeds", default="0,1,2"); p.add_argument("--fast", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="smoke only: no CUDA assertion, --eager")
    p = sub.add_parser("compass"); p.add_argument("--cache", required=True, choices=["shipped", "glu"]); p.add_argument("--gains", required=True, choices=list(GAINS))
    p.add_argument("--seed", type=int, required=True); p.add_argument("--block", default=None); p.add_argument("--out", required=True); p.add_argument("--cache-dir", default=str(SCRATCH))
    p.add_argument("--allow-cpu", action="store_true", help="smoke only: --quick --sparse torch --device cpu")
    p = sub.add_parser("analyse"); p.add_argument("--dir", default="out/cx5b"); p.add_argument("--out", default="out/cx5b/analysis"); p.add_argument("--cache-dir", default=None)
    p.add_argument("--skip-compass", action="store_true")
    p = sub.add_parser("smoke"); p.add_argument("--dir", default="out/cx5b"); p.add_argument("--cache-dir", default=str(SCRATCH)); p.add_argument("--suite", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "compile":
        d = ensure_scratch(Path(a.cache_dir)); print(json.dumps(glno_state(d), indent=1)); return 0
    return {"compare": cmd_compare, "plan": cmd_plan, "suite": cmd_suite, "compass": cmd_compass, "analyse": cmd_analyse, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
