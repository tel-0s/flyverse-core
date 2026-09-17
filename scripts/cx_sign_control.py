"""Round 8, item 2 -- the V- arm: the sign-flipped counterpart of round 7's V arm (docs/audits/compass_sign_control.md,
batch cx9). The batch plan, the freeze and the analysis; the simulation is scripts/cx_wedge.py exactly as in cx8r.

    PYTHONIOENCODING=utf-8 python scripts/cx_sign_control.py --plan-batch out/cx9 --seeds 0-5 --minutes 30
        -> out/cx9/batch.sh (a DRAFT: nothing here submits; `set -o pipefail`, ONE cluster_run.py call of 18 jobs,
           `--target house --node "${CLUSTER_NODE_2:?}" --gpu-ids 4,5,6,7`, the docs/INTERP.md 10.3 job line) and
           out/cx9/arms.json (the arm table the analysis checks every run against).
    PYTHONIOENCODING=utf-8 python scripts/cx_sign_control.py --predeclare out/cx9
        -> out/cx9/predeclared.json (stamped before submission; never overwritten)
    PYTHONIOENCODING=utf-8 python scripts/cx_sign_control.py --analyse --runs out/cx9 --out out/cx9/analysis [--reference out/cx8r]
        -> runs.csv (one row per run with its checks), per_seed.csv (arm, key, seeds, values: rule 28), compare.csv (the
           three predeclared contrasts, common.compare 6 v 6 exact U, Holm m = 3), descriptive.csv (per-arm means of
           every key, per-side pedestals included), replication.csv (cx8r's S and V beside cx9's, per key: exact-equal
           count and max |diff| over the six seeds -- a cross-submission, cross-node check, never an inference) and
           analysis.md.

Arms (6 seeds each; cx8r's prescribed-turn protocol: full connectome, no world, compass adaptation 0, 10 Hz Poisson
background on the 46 EPG, 1 s settle, wedges 0-3 at +40 Hz for 2 s, 5 s free, a prescribed +90 deg/s turn over
0.5-3.5 s after the pulse end, fed to the afferent instrument when one is attached):

| arm | preset | instrument | role |
| S   | raw          | --                                   | reference (cx8r's S, re-run)              |
| V   | instrumented | sided_turn_afferent:k=0.5            | the afferent alone (cx8r's V, re-run)     |
| V-  | instrumented | sided_turn_afferent:k=0.5:sign=-1    | THE ARM: the sign-flipped afferent alone  |

Predeclared family (Holm, m = 3; 6 v 6 exact-U floor 0.0021645 x 3 = 0.0065, satisfiable):
  (1) GLNO_LR_hz   V- vs S   predicted NEGATIVE (the mirror of round 7's V minus S = +2.2183 Hz)
  (2) GLNO_LR_hz   V  vs V-  predicted POSITIVE (the sign flip)
  (3) PS196b_LR_hz V  vs V-  predicted NEGATIVE (V's PS196_b L-R was -18.16 Hz; the flipped afferent should reverse it)
Descriptive, no verdict: per-side turn-window rates (GLNO / PEN / PS196b / AFF / DNa02 _L / _R) -- the pedestal every
L-R sits on is quoted beside it. Verdicts come from flyverse.interp.common.compare (result / null / underpowered /
undetermined); nothing is adopted and no default moves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyverse.interp import common  # noqa: E402
import cx_velocity_route as r7  # noqa: E402  (the round-7 job line, protocol constants and per-run checks)

ARMS = [("S", None, "raw reference (cx8r's S, re-run)"),
        ("V", "sided_turn_afferent:k=0.5", "the afferent alone (cx8r's V, re-run)"),
        ("V-", "sided_turn_afferent:k=0.5:sign=-1", "the sign-flipped afferent alone")]
FAMILY = [("1_GLNO_LR_V-_vs_S", "GLNO_LR_hz", "V-", "S", "negative"),
          ("2_GLNO_LR_V_vs_V-", "GLNO_LR_hz", "V", "V-", "positive"),
          ("3_PS196b_LR_V_vs_V-", "PS196b_LR_hz", "V", "V-", "negative")]
KEYS = list(r7.KEYS)
SIDE_KEYS = [f"{g}_{s}_hz_turn" for g in ("GLNO", "PEN", "PS196b", "AFF", "DNa02") for s in ("L", "R")]
GPU_POOL = "4,5,6,7"
MAX_JOBS_PER_CALL = 24


def stale_sources(fp: dict, frozen_lf: dict) -> tuple[list, dict]:
    """(files the run loaded whose content is NOT the predeclared tree's, every hash the run recorded).

    The frozen hashes are LF-normalised. The fingerprint's `files_lf` is too, but `files_loaded` (the probe scripts the
    process really imported) is a raw hash, and a Windows checkout ships those files as CRLF bytes -- so a loaded file
    also passes when the local copy still has the frozen LF hash and the loaded hash is that copy's raw bytes."""
    loaded = dict(fp.get("files", {})); loaded.update(fp.get("files_lf", {}))
    for f, h in fp.get("files_loaded", {}).items():
        loaded.setdefault(f, h)
    stale = []
    for f, h in loaded.items():
        if f not in frozen_lf or frozen_lf[f] == h:
            continue
        local = ROOT / f
        if local.is_file():
            raw = local.read_bytes()
            if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == frozen_lf[f] and hashlib.sha256(raw).hexdigest() == h:
                continue                                       # the frozen content, shipped with CRLF line endings
        stale.append(f)
    return sorted(stale), loaded


def expected(label):
    for a in ARMS:
        if a[0] == label:
            return dict(preset="instrumented" if a[1] else "raw", instruments=[a[1].split(":")[0]] if a[1] else [],
                        glutamate=False, hold=None, spec=a[1])
    return None


def plan_batch(out_dir: Path, seeds, minutes=30, name="cx9"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rel = f"out/{out_dir.name}"
    if (out_dir / "predeclared.json").exists() or (out_dir / "submitted_at.txt").exists():
        raise ValueError("batch is frozen; use a new directory for a new predeclaration")
    jobs = []
    for seed in seeds:
        for label, spec, _ in ARMS:
            cmd, stem = r7.arm_command(label, False, None, spec, seed, rel)
            jobs.append(dict(seed=seed, arm=label, block=f"fam_s{seed}", line=r7.job_line(cmd, stem, rel)))
    if len(jobs) > MAX_JOBS_PER_CALL:
        raise ValueError(f"{len(jobs)} jobs exceed one call of {MAX_JOBS_PER_CALL}")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = ["#!/bin/bash", "set -o pipefail",
             f"# DRAFT -- NOT SUBMITTED. Round 8 / batch {name}: docs/audits/compass_sign_control.md.",
             f"# {len(jobs)} jobs = {len(ARMS)} arms x {len(seeds)} seeds, one arm per job, one cluster_run.py call; every job pinned to one GPU of the",
             f"# pool {GPU_POOL} (gpus 1 + gpu_ids [one id], round-robin). The second cluster node is named by the CLUSTER_NODE_2 environment",
             "# variable (docs/CLUSTER.md, git-ignored); committed text never carries the name. Generated by scripts/cx_sign_control.py on " + stamp + ".",
             f"# Read: '{len(jobs)} job(s), 0 failed' is NOT sufficient (the scheduler lists a crashed job as completed, exit_code None):",
             f"# then `ls {rel}/*_s?.json | wc -l` = {len(jobs)} and `grep -c 'device cuda' {rel}/*_s?.txt`, then --analyse (zero problems).",
             "# Arms: " + "; ".join(f"{a[0]} = {a[2]}" for a in ARMS),
             "python scripts/cluster_run.py --target house --node \"${CLUSTER_NODE_2:?set CLUSTER_NODE_2 to the name of the second cluster node}\" "
             f"--gpu-ids {GPU_POOL} --ship flyverse,scripts --name {name} --minutes {minutes} "
             + " ".join('"' + j["line"] + '"' for j in jobs) + f" --fetch {rel}/ 2>&1 | tee {rel}/client_stdout_0.txt || exit $?"]
    (out_dir / "batch.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    arms = {a[0]: dict(instrument=a[1], role=a[2], **{k: v for k, v in expected(a[0]).items() if k in ("preset", "instruments")}) for a in ARMS}
    (out_dir / "arms.json").write_text(json.dumps({"batch": name, "seeds": list(seeds), "turn_deg_s": float(r7.TURN[0]), "turn_window_s": r7.TURN[1],
                                                   "arms": arms, "family": FAMILY, "generated_utc": stamp, "gpu_pool": GPU_POOL,
                                                   "node": "<cluster-node-2>", "status": "DRAFT, not submitted", "protocol": r7.PROTOCOL}, indent=1),
                                       encoding="utf-8")
    print(f"{len(jobs)} jobs in 1 call -> {out_dir / 'batch.sh'} (DRAFT, not submitted); {out_dir / 'arms.json'}")


def resolved_lif_by_arm():
    from flyverse import brain
    from cx_wedge import COMPASS_RE, RING_RE
    gains = list(brain.DEFAULT_TYPE_PATH_GAIN) + [
        (r"^EPG$", r"^PEN_", 1.0), (r"^PEN_", r"^EPG$", 1.0), (r"^EPG$", r"^PEG$", 1.0), (r"^PEG$", r"^EPG$", 1.0),
        (r"^Delta7$", r"^(EPG$|PEN_)", 1.0), (RING_RE, r"^(EPG$|PEN_|PEG$)", 1.0)]
    params = brain.LIFParams(adapt_by_type={COMPASS_RE: 0.0}, type_path_gain=gains,
                             receptor_model=r7.PROTOCOL["receptor_model"], receptor_net_rule=r7.PROTOCOL["receptor_net_rule"])
    lif = common.to_jsonable(common.model_record(params)["lif"])
    return {a[0]: lif for a in ARMS}


def source_hashes():
    paths = sorted(set(ROOT.glob("flyverse/**/*.py")) | set(ROOT.glob("flyverse/data/*.csv")) |
                   {Path(__file__).resolve(), ROOT / "scripts/cx_wedge.py", ROOT / "scripts/probe_compass_room.py",
                    ROOT / "scripts/cx_velocity_route.py", ROOT / "scripts/cluster_run.py"})
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}


def predeclare(out_dir):
    out_dir = Path(out_dir)
    if (out_dir / "submitted_at.txt").exists() or list(out_dir.glob("*_s*.json")):
        raise ValueError("cannot predeclare after submission or results exist")
    plan = json.loads((out_dir / "arms.json").read_text(encoding="utf-8"))
    if plan["seeds"] != list(range(6)) or plan["family"] != [list(x) for x in FAMILY]:
        raise ValueError("cx9 requires the six seeds and the three primary contrasts")
    record = dict(plan, status="PREDECLARED, not submitted", written_before_submission=True,
                  stamped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  batch_sha256=hashlib.sha256((out_dir / "batch.sh").read_bytes()).hexdigest(),
                  arms_sha256=hashlib.sha256((out_dir / "arms.json").read_bytes()).hexdigest(),
                  source_sha256_lf=source_hashes(), resolved_lif_by_arm=resolved_lif_by_arm(),
                  replicate_unit="runs (one cx_wedge process per arm x seed)",
                  multiplicity="one Holm family, m = 3; a missing p counts toward m",
                  verdict_rule="flyverse.interp.common.compare: result needs |z| >= 3, p <= 0.05 and n_min >= 4; underpowered / null / "
                               "undetermined (deterministic reference) are its own words; a `result` whose sign is not the predicted one is "
                               "reported as `result, opposite sign` and is not a confirmation",
                  predictions={f[0]: f[4] for f in FAMILY},
                  descriptive="per-side turn-window rates (the pedestal) beside every L-R; no verdict on them",
                  replication_check="cx8r's S and V (out/cx8r, node 1, 2026-09-15) beside cx9's S and V, per key: count of seeds "
                                    "whose values are exactly equal and max |diff|. det1 found the cx_wedge protocol repeats exactly on "
                                    "one GPU; across two nodes this is an observation, not an inference, and no verdict rests on it",
                  outcome_rule="primaries 1 and 3 negative results and primary 2 a positive result = the GLNO side report follows the "
                               "afferent's sign; anything else = the sign specificity of round 7's primary 3 is not established",
                  nothing_adopted="no default changes; no gain selected; no compass room unlocked")
    with (out_dir / "predeclared.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(f"froze {out_dir / 'predeclared.json'} before submission")


def load_runs(runs_dir: Path, frozen: dict | None, arms=ARMS, seeds=range(6), strict=True):
    rows, problems, identities = [], [], set()
    for path in sorted(Path(runs_dir).glob("*_s*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, r in enumerate(data if isinstance(data, list) else [data]):
            if not isinstance(r, dict) or "metrics" not in r:
                problems.append(f"{path.name}: missing metrics"); continue
            arm, seed, m, prov = r.get("arm"), r.get("seed"), r["metrics"], r.get("provenance", {})
            run_id = f"{path.name}#{i}"
            row = dict(arm=arm, seed=seed, file=path.name, run_id=run_id, device=r.get("device"),
                       device_name=prov.get("execution", {}).get("device_name"), cuda_visible=None,
                       preset=r.get("preset"), instruments=",".join(r.get("instruments", [])),
                       md5=prov.get("compiled_connectome", {}).get("md5"), wall_s=r.get("wall_s"))
            for k in KEYS:
                v = m.get(k)
                row[k] = float(v) if isinstance(v, (int, float)) else float("nan")
            bad = []
            exp = expected(arm) if arm in {a[0] for a in arms} else None
            if strict:
                if exp is None or seed not in seeds:
                    bad.append("unexpected arm/seed")
                else:
                    bad.extend(r7.check_run(r, exp, path))
                    if frozen and prov.get("model", {}).get("lif") != frozen.get("resolved_lif_by_arm", {}).get(arm):
                        bad.append("resolved LIF differs from frozen protocol")
                    if not str(row["device_name"] or "").startswith("NVIDIA"):
                        bad.append("no CUDA device name in provenance")
                    stale, loaded = stale_sources(prov.get("source_fingerprint", {}), (frozen or {}).get("source_sha256_lf", {}))
                    if stale:
                        bad.append(f"loaded source differs from the predeclared tree: {stale[:4]}")
                    if frozen and not loaded:
                        bad.append("missing source fingerprint")
                if (arm, seed) in identities:
                    bad.append("duplicate arm/seed")
                identities.add((arm, seed))
            row["checks"] = "; ".join(bad)
            problems.extend(f"{run_id}: {b}" for b in bad)
            rows.append(row)
    if strict:
        wanted = {(a[0], s) for a in arms for s in seeds}
        if identities != wanted or len(rows) != len(wanted):
            problems.append(f"expected {len(wanted)} distinct arm/seed runs; got {len(rows)}; missing {sorted(wanted - identities)}")
    return rows, problems


def analyse(runs_dir: Path, out_dir: Path, reference: Path | None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    declaration = Path(runs_dir) / "predeclared.json"
    frozen = json.loads(declaration.read_text(encoding="utf-8")) if declaration.exists() else None
    rows, problems = load_runs(Path(runs_dir), frozen)
    if not rows:
        raise SystemExit(f"no *_s*.json runs under {runs_dir}")
    if frozen and frozen.get("family") != [list(x) for x in FAMILY]:
        problems.append("analysis disagrees with the frozen family")
    df = pd.DataFrame(rows); df.to_csv(out_dir / "runs.csv", index=False)
    per = []
    for arm, g in df.groupby("arm", sort=False):
        g = g.sort_values("seed")
        for k in KEYS:
            vals = g[k].to_numpy(float)
            per.append(dict(arm=arm, key=k, seeds=",".join(str(s) for s in g.seed), files=",".join(g.file), run_ids=",".join(g.run_id),
                            values=",".join(f"{v:.4f}" for v in vals),
                            mean=float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
                            sd=float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan"), n=int(np.isfinite(vals).sum())))
    per_df = pd.DataFrame(per); per_df.to_csv(out_dir / "per_seed.csv", index=False)
    order = [a[0] for a in ARMS if a[0] in set(df.arm)]
    desc = per_df.pivot_table(index="arm", columns="key", values="mean", aggfunc="first").reindex(order)
    desc.to_csv(out_dir / "descriptive.csv", index=True)
    comp = []
    for name, key, a, b, predicted in FAMILY:
        va = df[df.arm == a][key].to_numpy(float); vb = df[df.arm == b][key].to_numpy(float)
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        rec = dict(test=name, key=key, stim=a, null=b, predicted_sign=predicted, n_stim=int(len(va)), n_null=int(len(vb)),
                   stim_values=json.dumps([round(float(x), 4) for x in va]), null_values=json.dumps([round(float(x), 4) for x in vb]))
        if len(va) and len(vb):
            c = common.compare(va, vb)
            rec.update(verdict=c["verdict"], diff=c["diff"], z=c["z"], U=c["U"], p=c["p"], p_floor=c["p_floor"], null_sd_zero=c["null_sd_zero"], n_min=c["n_min"])
            # the symmetric statistic beside compare's reference-SD z (the round-7 skeptic: compare is orientation-dependent)
            sa, sb = np.std(va, ddof=1), np.std(vb, ddof=1)
            rec["welch_t"] = float((va.mean() - vb.mean()) / math.sqrt(sa ** 2 / len(va) + sb ** 2 / len(vb))) if (sa > 0 or sb > 0) else float("nan")
            rec["all_separated"] = bool(va.min() > vb.max() or va.max() < vb.min())
        else:
            rec.update(verdict="underpowered", diff=float("nan"), z=float("nan"), U=float("nan"), p=float("nan"), p_floor=float("nan"), null_sd_zero=None, n_min=0)
        comp.append(rec)
    adj = r7.holm({r["test"]: r["p"] for r in comp}, m=len(FAMILY))
    for r in comp:
        r["p_holm"] = adj[r["test"]]; r["m"] = len(FAMILY)
        holm_ok = r["verdict"] == "result" and np.isfinite(r["p_holm"]) and r["p_holm"] <= 0.05
        sign_ok = np.isfinite(r["diff"]) and ((r["diff"] < 0) if r["predicted_sign"] == "negative" else (r["diff"] > 0))
        r["verdict_holm"] = ("result" if (holm_ok and sign_ok) else "result, opposite sign" if holm_ok else
                             "null" if r["verdict"] == "result" else r["verdict"])
        if problems:
            r["unchecked_verdict"] = r["verdict_holm"]; r["verdict"] = r["verdict_holm"] = "undetermined (invalid batch)"
    comp_df = pd.DataFrame(comp); comp_df.to_csv(out_dir / "compare.csv", index=False)
    # replication against cx8r's S and V: per key, seeds equal exactly and max |diff|
    rep = []
    if reference and Path(reference).is_dir():
        ref_rows, _ = load_runs(Path(reference), None, strict=False)
        ref = pd.DataFrame(ref_rows)
        for arm in ("S", "V"):
            new = df[df.arm == arm].set_index("seed").sort_index(); old = ref[ref.arm == arm].set_index("seed").sort_index()
            seeds = sorted(set(new.index) & set(old.index))
            if not seeds:
                continue
            for k in KEYS:
                x, y = new.loc[seeds, k].to_numpy(float), old.loc[seeds, k].to_numpy(float)
                both_nan = np.isnan(x) & np.isnan(y)
                d = np.where(both_nan, 0.0, np.abs(x - y))
                rep.append(dict(arm=arm, key=k, n_seeds=len(seeds), n_exactly_equal=int(np.sum((x == y) | both_nan)),
                                max_abs_diff=float(np.nanmax(d)) if np.isfinite(d).any() else float("nan"),
                                cx9_values=",".join(f"{v:.4f}" for v in x), reference_values=",".join(f"{v:.4f}" for v in y),
                                cx9_device=",".join(str(v) for v in new.loc[seeds, "device_name"]), reference_device=",".join(str(v) for v in old.loc[seeds, "device_name"])))
        pd.DataFrame(rep).to_csv(out_dir / "replication.csv", index=False)
    summary = {"runs_dir": Path(runs_dir).as_posix(), "n_runs": int(len(df)), "arms": {a: int(n) for a, n in df.arm.value_counts().items()},
               "problems": problems, "family": comp, "replication_reference": str(reference) if reference else None,
               "replication_exact_keys": int(sum(r["n_exactly_equal"] == r["n_seeds"] for r in rep)), "replication_keys": len(rep),
               "valid_batch": not problems, "stamped_utc": (frozen or {}).get("stamped_utc"),
               "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "analysis.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    md = [f"# cx9 analysis of `{Path(runs_dir).as_posix()}`", "", f"n_runs {len(df)}; arms {dict(df.arm.value_counts())}; problems {len(problems)}"
          + (": " + "; ".join(problems[:20]) if problems else ""), "",
          "## The predeclared family (Holm, m = 3; `compare.csv`)", "",
          "| test | key | stim | null | predicted | n | verdict | verdict (Holm, sign) | diff | z (ref SD) | Welch t | p | p_holm | separated |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in comp:
        md.append(f"| {r['test']} | {r['key']} | {r['stim']} | {r['null']} | {r['predicted_sign']} | {r['n_stim']} v {r['n_null']} | {r['verdict']} | "
                  f"{r['verdict_holm']} | {r['diff']:+.4f} | {r['z']:+.2f} | {r.get('welch_t', float('nan')):+.2f} | {r['p']:.4f} | {r['p_holm']:.4f} | {r.get('all_separated')} |")
    md += ["", "## Per arm (`descriptive.csv`: mean over seeds; per-side pedestals)", "",
           "| arm | GLNO L-R | GLNO L / R | PEN L-R | PEN L / R | PS196b L-R | PS196b L / R | AFF L-R | AFF L / R | DNa02 L-R | frac confined post |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    sd = per_df.pivot_table(index="arm", columns="key", values="sd", aggfunc="first").reindex(order)
    for arm in order:
        def f(k, pm=True):
            v = desc.loc[arm, k]; s = sd.loc[arm, k]
            return (f"{v:+.4f} +/- {s:.4f}" if pm else f"{v:.2f}") if np.isfinite(v) else "NA"
        md.append(f"| {arm} | {f('GLNO_LR_hz')} | {f('GLNO_L_hz_turn', False)} / {f('GLNO_R_hz_turn', False)} | {f('PEN_LR_hz')} | "
                  f"{f('PEN_L_hz_turn', False)} / {f('PEN_R_hz_turn', False)} | {f('PS196b_LR_hz')} | {f('PS196b_L_hz_turn', False)} / {f('PS196b_R_hz_turn', False)} | "
                  f"{f('AFF_LR_hz')} | {f('AFF_L_hz_turn', False)} / {f('AFF_R_hz_turn', False)} | {f('DNa02_LR_hz')} | {f('frac_confined_post')} |")
    md += ["", "## Per-seed lists (`per_seed.csv`, columns arm,key,seeds,values -- pasted, never retyped: INTERP 10.4 rule 28)", "",
           "| arm | key | seeds | values |", "|---|---|---|---|"]
    for r in per:
        if r["key"] in ("GLNO_LR_hz", "PS196b_LR_hz", "PEN_LR_hz", "AFF_LR_hz", "DNa02_LR_hz", "frac_confined_post") or r["key"] in SIDE_KEYS:
            md.append(f"| {r['arm']} | {r['key']} | {r['seeds']} | {r['values']} |")
    if rep:
        md += ["", f"## Replication check against `{Path(reference).as_posix()}` (`replication.csv`): exact-equal seeds / 6 and max |diff| per key", "",
               "| arm | key | exactly equal / n | max abs diff | cx9 values | reference values |", "|---|---|---|---|---|---|"]
        for r in rep:
            if r["key"] in ("GLNO_LR_hz", "PS196b_LR_hz", "PEN_LR_hz", "AFF_LR_hz", "epg_in_mean_post", "GLNO_L_hz_turn", "GLNO_R_hz_turn"):
                md.append(f"| {r['arm']} | {r['key']} | {r['n_exactly_equal']} / {r['n_seeds']} | {r['max_abs_diff']:.4g} | {r['cx9_values']} | {r['reference_values']} |")
        md.append(f"\n{summary['replication_exact_keys']} of {summary['replication_keys']} (arm, key) rows are exactly equal in all six seeds.")
    (out_dir / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:12]))
    print(f"-> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-batch", default=None, metavar="DIR"); ap.add_argument("--seeds", default="0-5")
    ap.add_argument("--minutes", type=int, default=30); ap.add_argument("--name", default="cx9")
    ap.add_argument("--predeclare", default=None, metavar="DIR")
    ap.add_argument("--analyse", action="store_true"); ap.add_argument("--runs", default="out/cx9"); ap.add_argument("--out", default=None)
    ap.add_argument("--reference", default="out/cx8r", help="cx8r's runs for the replication check (skipped when absent)")
    a = ap.parse_args()
    if a.plan_batch:
        plan_batch(Path(a.plan_batch), r7.parse_seeds(a.seeds), minutes=a.minutes, name=a.name)
    if a.predeclare:
        predeclare(a.predeclare)
    if a.analyse:
        s = analyse(Path(a.runs), Path(a.out or (Path(a.runs) / "analysis")), Path(a.reference) if a.reference else None)
        if s["problems"]:
            raise SystemExit(2)
    if not (a.plan_batch or a.predeclare or a.analyse):
        ap.print_help()


if __name__ == "__main__":
    main()
