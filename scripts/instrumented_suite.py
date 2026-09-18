"""Round 8, item 3 -- the 29-check suite under `instrumented` with the three-instrument list of round 7
(docs/audits/instrumented_suite.md, batch suite-inst): `sided_turn_afferent:k=0.5`, `ring_dc_hold`, `glno_sign`,
three draws (seeds 0, 1, 2) beside `raw` three draws, the way docs/audits/compass_standin.md ran it
(`benchmark.py --sections all --draw-seed S --seeds S --preset ...`, one job per (preset, seed)).

    PYTHONIOENCODING=utf-8 python scripts/instrumented_suite.py plan --out out/suite-inst
        -> out/suite-inst/batch.sh (DRAFT: `set -o pipefail`, ONE cluster_run.py call of 6 jobs, the docs/INTERP.md 10.3
           job line, `--target house --node "${CLUSTER_NODE_2:?}" --gpu-ids 4,5,6,7 --ship flyverse,scripts`), jobs.json
    PYTHONIOENCODING=utf-8 python scripts/instrumented_suite.py predeclare out/suite-inst
        -> out/suite-inst/predeclared.json (stamped before submission; never overwritten)
    PYTHONIOENCODING=utf-8 python scripts/instrumented_suite.py analyse --runs out/suite-inst --out out/suite-inst/analysis
        -> suite_rows.csv (one row per check per seed: raw / instrumented value and status -- the per-draw lists every
           quoted value is pasted from, rule 28), suite_table.csv (one row per check: statuses, the status rule, the
           3 v 3 compare), runs.csv (one row per run with its checks), analysis.md, summary.json

The rule (PRESETS_SPEC section 2 item 5, the round-2 rule as compass_standin.md and glno_relabel.md applied it):
>= 3 draws; an instrumented draw whose status on a row is not the status of every raw draw is a STATUS CHANGE
(`PASS (gap closed)` and `PASS` count as one status); a status change on any row outside the rows the instruments are
declared to touch -- the compass rows, `compass.wedge_cells_persisting` -- is a REJECTION of the list, whatever its
direction (compass_standin's taste.MN9_hz FAIL -> PASS was rejected). Rows whose raw draws already disagree among
themselves are reported as unstable and not counted. The room rate-half (>= 6 runs per arm) is run only if this half
passes. Nothing is adopted either way; this records whether the list is admissible.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyverse.interp import common  # noqa: E402
from cx_sign_control import stale_sources  # noqa: E402

HOLD = "^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)"
INSTRUMENTS = "sided_turn_afferent:k=0.5"
NT_OVERRIDE = "GLNO=glutamate"
RECORD_NAMES = ["sided_turn_afferent", "ring_dc_hold", "glno_sign"]
DECLARED_GAP_ROWS = ["compass.wedge_cells_persisting"]     # the rows the three instruments are declared to touch
SEEDS = [0, 1, 2]
N_CHECKS = 29
GPU_POOL = "4,5,6,7"
CACHE_MD5 = {"raw": "ef23cc27bea13be7f6a96f3c04fd3737", "instrumented": "7a10d93ba2086f2c76bcdabdca79b4ec"}   # cx8r's two caches


def sh_token(s: str) -> str:
    return "'" + s + "'" if any(ch in s for ch in "()|$^") else s


def suite_command(preset: str, seed: int, rel: str) -> tuple[str, str]:
    stem = f"{rel}/suite_{preset}_s{seed}"
    parts = [f"python scripts/benchmark.py --sections all --draw-seed {seed} --seeds {seed} --preset {preset}"]
    if preset == "instrumented":
        parts += [f"--instruments {INSTRUMENTS}", f"--hold-edges {sh_token(HOLD)}", f"--nt-override {NT_OVERRIDE}"]
    parts.append(f"--json {stem}.json > {stem}.txt 2>&1")
    return " ".join(parts), stem


def job_line(cmd: str, stem: str, rel: str) -> str:
    return (f"mkdir -p {rel} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"{cmd}; st=\\$?; tail -3 {stem}.txt; exit \\$st")


def plan(out_dir: Path, minutes: int = 90, name: str = "suite-inst"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rel = f"out/{out_dir.name}"
    if (out_dir / "predeclared.json").exists():
        raise ValueError("batch is frozen; use a new directory for a new predeclaration")
    jobs = []
    for seed in SEEDS:
        for preset in ("raw", "instrumented"):
            cmd, stem = suite_command(preset, seed, rel)
            jobs.append(dict(preset=preset, seed=seed, stem=stem, command=cmd, line=job_line(cmd, stem, rel)))
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = ["#!/bin/bash", "set -o pipefail",
             f"# DRAFT -- NOT SUBMITTED. Round 8 / batch {name}: docs/audits/instrumented_suite.md.",
             f"# {len(jobs)} jobs = 2 presets x {len(SEEDS)} draws of the 29-check suite, one job each, one cluster_run.py call; every job pinned to one",
             f"# GPU of the pool {GPU_POOL} (gpus 1 + gpu_ids [one id], round-robin). The second cluster node is named by the CLUSTER_NODE_2",
             "# environment variable (docs/CLUSTER.md, git-ignored); committed text never carries the name. Generated by",
             "# scripts/instrumented_suite.py plan on " + stamp + ".",
             f"# Read: `ls {rel}/suite_*_s?.json | wc -l` = {len(jobs)}, `grep -c 'device' {rel}/suite_*_s?.txt`, then analyse (zero problems).",
             "python scripts/cluster_run.py --target house --node \"${CLUSTER_NODE_2:?set CLUSTER_NODE_2 to the name of the second cluster node}\" "
             f"--gpu-ids {GPU_POOL} --ship flyverse,scripts --name {name} --minutes {minutes} "
             + " ".join('"' + j["line"] + '"' for j in jobs) + f" --fetch {rel}/ 2>&1 | tee {rel}/client_stdout_0.txt || exit $?"]
    (out_dir / "batch.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "jobs.json").write_text(json.dumps({"batch": name, "seeds": SEEDS, "instruments": INSTRUMENTS, "hold_edges": HOLD,
                                                   "nt_override": NT_OVERRIDE, "record_names": RECORD_NAMES,
                                                   "declared_gap_rows": DECLARED_GAP_ROWS, "gpu_pool": GPU_POOL, "node": "<cluster-node-2>",
                                                   "jobs": jobs, "generated_utc": stamp, "status": "DRAFT, not submitted"}, indent=1),
                                       encoding="utf-8")
    print(f"{len(jobs)} jobs in 1 call -> {out_dir / 'batch.sh'} (DRAFT, not submitted); {out_dir / 'jobs.json'}")


def source_hashes() -> dict:
    paths = sorted(set(ROOT.glob("flyverse/**/*.py")) | set(ROOT.glob("flyverse/data/*.csv")) |
                   {Path(__file__).resolve(), ROOT / "scripts/benchmark.py", ROOT / "scripts/instrumented_benchmark.py",
                    ROOT / "scripts/cx_wedge.py", ROOT / "scripts/room_demo.py", ROOT / "scripts/cluster_run.py"})
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}


def predeclare(out_dir: Path):
    out_dir = Path(out_dir)
    if list(out_dir.glob("suite_*_s?.json")):
        raise ValueError("cannot predeclare after results exist")
    plan_rec = json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))
    record = dict(plan_rec, status="PREDECLARED, not submitted", written_before_submission=True,
                  stamped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  batch_sha256=hashlib.sha256((out_dir / "batch.sh").read_bytes()).hexdigest(),
                  jobs_sha256=hashlib.sha256((out_dir / "jobs.json").read_bytes()).hexdigest(),
                  source_sha256_lf=source_hashes(),
                  rule={"draws": "3 per preset (seeds 0, 1, 2; --draw-seed S --seeds S), the suite's own status per row",
                        "status_change": "an instrumented draw's status on a row is not the status of every raw draw; PASS (gap closed) "
                                         "and PASS are one status; rows whose raw draws disagree are `unstable`, reported, not counted",
                        "rejection": f"any status change on a row outside the declared gap rows {DECLARED_GAP_ROWS}, in either direction",
                        "values": "a 3 v 3 common.compare per row is descriptive only (underpowered by construction); no value is quoted "
                                  "beyond its across-draw spread (determinism_gate.md: the native path does not repeat)",
                        "validity": "29 checks per run; config.device a CUDA name; raw runs on the shipped cache "
                                    f"({CACHE_MD5['raw']}), instrumented runs on the GLNO=glutamate scratch cache ({CACHE_MD5['instrumented']}) "
                                    "with the hold in config.type_path_gain and the three records in every controller's provenance; "
                                    "every loaded source at the predeclared hash",
                        "room_rate_half": "run at >= 6 runs per arm ONLY if this half passes; otherwise not run",
                        "adoption": "none either way; this records whether the list is admissible"},
                  expectation="another rejection (compass_standin's own run was rejected on taste.MN9_hz); the point is the record")
    with (out_dir / "predeclared.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(f"froze {out_dir / 'predeclared.json'} before submission")


def status_rank(s: str) -> str:
    return "PASS" if str(s).startswith("PASS") else str(s)


def analyse(runs_dir: Path, out_dir: Path):
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((runs_dir / "predeclared.json").read_text(encoding="utf-8")) if (runs_dir / "predeclared.json").exists() else {}
    want = frozen.get("source_sha256_lf", {})
    data, run_rows, problems = {}, [], []
    for preset in ("raw", "instrumented"):
        for seed in SEEDS:
            p = runs_dir / f"suite_{preset}_s{seed}.json"
            if not p.is_file():
                problems.append(f"{p.name}: missing"); continue
            d = json.loads(p.read_text(encoding="utf-8"))
            cfg, checks = d.get("config", {}), d.get("checks", [])
            bad = []
            if len(checks) != N_CHECKS:
                bad.append(f"{len(checks)} checks, expected {N_CHECKS}")
            if not str(cfg.get("device", "")).startswith("NVIDIA"):
                bad.append(f"device {cfg.get('device')!r}")
            if cfg.get("preset") != preset:
                bad.append(f"preset {cfg.get('preset')!r}")
            ctrls = d.get("controllers", [])
            md5s = {c.get("compiled_connectome", {}).get("md5") for c in ctrls}
            if preset == "instrumented":
                if cfg.get("instruments") != [INSTRUMENTS] or cfg.get("nt_override") != {"GLNO": "glutamate"}:
                    bad.append(f"instruments / nt_override {cfg.get('instruments')} {cfg.get('nt_override')}")
                if [HOLD.split(":", 1)[0], HOLD.split(":", 1)[1], 0.0] not in [list(h) for h in cfg.get("type_path_gain", [])]:
                    bad.append("hold not in config.type_path_gain")
                if sorted(cfg.get("instrument_records", [])) != sorted(RECORD_NAMES):
                    bad.append(f"instrument_records {cfg.get('instrument_records')}")
                if md5s != {CACHE_MD5["instrumented"]}:
                    bad.append(f"compiled cache {md5s} != relabel cache")
                for i, c in enumerate(ctrls):
                    names = [r.get("name") for r in c.get("instruments", [])]
                    if c.get("preset") != "instrumented" or sorted(names) != sorted(RECORD_NAMES):
                        bad.append(f"controller {i} preset {c.get('preset')} records {names}"); break
                if not ctrls:
                    bad.append("no controller provenance")
            else:
                if cfg.get("instruments") or cfg.get("hold_edges") or cfg.get("nt_override"):
                    bad.append("raw run carries instruments / holds / relabel")
                if md5s and md5s != {CACHE_MD5["raw"]}:
                    bad.append(f"compiled cache {md5s} != shipped cache")
            for i, c in enumerate(ctrls):
                stale, loaded = stale_sources(c.get("source_fingerprint", {}), want)
                if stale:
                    bad.append(f"controller {i} loaded source differs from the predeclared tree: {stale[:4]}"); break
            console = p.with_suffix(".txt")
            if not (console.is_file() and "NVIDIA" in console.read_text(encoding="utf-8", errors="replace")):
                bad.append("console lacks the CUDA device line")
            data[(preset, seed)] = d
            tally = {"pass": sum(ch["status"].startswith("PASS") for ch in checks), "fail": sum(ch["status"] == "FAIL" for ch in checks),
                     "gap": sum(ch["status"] == "KNOWN GAP" for ch in checks), "missing": sum(ch["status"] == "MISSING" for ch in checks)}
            run_rows.append(dict(preset=preset, seed=seed, file=p.name, device=cfg.get("device"), cache_md5=",".join(sorted(str(m) for m in md5s)),
                                 backend=cfg.get("backend"), n_checks=len(checks), **tally, n_controllers=len(ctrls), checks_ok="; ".join(bad)))
            problems.extend(f"{p.name}: {b}" for b in bad)
    pd.DataFrame(run_rows).to_csv(out_dir / "runs.csv", index=False)
    keys = []
    for d in data.values():
        for ch in d.get("checks", []):
            if ch["key"] not in keys:
                keys.append(ch["key"])
    rows, table = [], []
    changed, unstable = [], []
    for k in keys:
        cells, st = {}, {"raw": [], "instrumented": []}
        crit = ""
        for preset in ("raw", "instrumented"):
            for seed in SEEDS:
                d = data.get((preset, seed))
                ch = next((c for c in d.get("checks", []) if c["key"] == k), None) if d else None
                v = ch["measured"] if ch else None; s = ch["status"] if ch else "MISSING"; crit = ch["criterion"] if ch else crit
                cells[(preset, seed)] = (v, s); st[preset].append(status_rank(s))
                rows.append(dict(check=k, seed=seed, preset=preset, value=v, status=s, criterion=crit))
        raw_set, inst_set = set(st["raw"]), set(st["instrumented"])
        pooled = not inst_set.issubset(raw_set)
        matched = [i for i, s in enumerate(SEEDS) if st["raw"][i] != st["instrumented"][i]]
        is_unstable = len(raw_set) > 1
        gap = k in DECLARED_GAP_ROWS
        if is_unstable:
            unstable.append(k)
        elif pooled:
            changed.append(k)
        a = [cells[("instrumented", s)][0] for s in SEEDS]; b = [cells[("raw", s)][0] for s in SEEDS]
        rec = dict(check=k, criterion=crit, raw_statuses="/".join(st["raw"]), instrumented_statuses="/".join(st["instrumented"]),
                   status_changed_pooled=pooled, changed_draws=",".join(str(SEEDS[i]) for i in matched), raw_unstable=is_unstable,
                   declared_gap_row=gap, rejecting=(pooled and not is_unstable and not gap),
                   raw_values=json.dumps(b), instrumented_values=json.dumps(a))
        if all(isinstance(x, (int, float)) and x is not None for x in a + b):
            r = common.compare(np.array(a, float), np.array(b, float))
            rec.update(diff=r["diff"], z=r["z"], p=r["p"], verdict=r["verdict"])
        table.append(rec)
    pd.DataFrame(rows).to_csv(out_dir / "suite_rows.csv", index=False)
    pd.DataFrame(table).to_csv(out_dir / "suite_table.csv", index=False)
    rejecting = [r["check"] for r in table if r["rejecting"]]
    gap_changes = [r["check"] for r in table if r["status_changed_pooled"] and r["declared_gap_row"] and not r["raw_unstable"]]
    complete = len(data) == 6 and all(r["n_checks"] == N_CHECKS for r in run_rows)
    verdict = ("INCOMPLETE or invalid (missing runs or provenance problems)" if (not complete or problems) else
               f"REJECTED: status change outside the declared gap rows on {rejecting}" if rejecting else
               "admissible by the suite half (no status change outside the declared gap rows); the room rate-half is owed")
    summary = dict(runs_dir=runs_dir.as_posix(), verdict=verdict, rejecting_rows=rejecting, gap_row_changes=gap_changes,
                   unstable_raw_rows=unstable, status_changes_pooled=changed, problems=problems, n_runs=len(data),
                   tallies={f"{p}_s{s}": {k: r[k] for k in ("pass", "fail", "gap", "missing")} for r in run_rows for p, s in [(r["preset"], r["seed"])]},
                   stamped_utc=frozen.get("stamped_utc"), analysis_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                   created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    md = [f"# suite-inst analysis of `{runs_dir.as_posix()}`", "", f"problems {len(problems)}" + (": " + "; ".join(problems[:12]) if problems else ""), "",
          f"**Verdict: {verdict}**", "",
          "| run | device | cache md5 | pass / fail / gap / missing | controllers | checks |", "|---|---|---|---|---|---|"]
    for r in run_rows:
        md.append(f"| {r['preset']} s{r['seed']} | {r['device']} | {str(r['cache_md5'])[:8]} | {r['pass']} / {r['fail']} / {r['gap']} / {r['missing']} | {r['n_controllers']} | {r['checks_ok'] or 'ok'} |")
    md += ["", "P PASS, P* PASS (gap closed), G KNOWN GAP, F FAIL, M MISSING; values by draw seed [0,1,2] (`suite_rows.csv`, columns value,status)", "",
           "| check (criterion) | raw [0,1,2] | instrumented [0,1,2] | raw / instrumented statuses | status change | rejecting |", "|---|---|---|---|---|---|"]
    abbr = {"PASS": "P", "PASS (gap closed)": "P*", "KNOWN GAP": "G", "FAIL": "F", "MISSING": "M"}
    for r in table:
        def fmt(vals, presets):
            out = []
            for s, v in zip(SEEDS, json.loads(vals)):
                st_ = next(x["status"] for x in rows if x["check"] == r["check"] and x["seed"] == s and x["preset"] == presets)
                out.append((f"{v:.6g}" if isinstance(v, (int, float)) and v is not None else str(v)) + " " + abbr.get(st_, st_))
            return ", ".join(out)
        md.append(f"| {r['check']} ({r['criterion']}) | {fmt(r['raw_values'], 'raw')} | {fmt(r['instrumented_values'], 'instrumented')} | "
                  f"{r['raw_statuses']} / {r['instrumented_statuses']} | {'YES' if r['status_changed_pooled'] else 'no'}"
                  f"{' (raw unstable)' if r['raw_unstable'] else ''}{' (declared gap row)' if r['declared_gap_row'] else ''} | {'REJECT' if r['rejecting'] else ''} |")
    (out_dir / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:6]))
    print(f"rejecting {rejecting}; gap-row changes {gap_changes}; unstable raw {unstable}; -> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out", default="out/suite-inst"); p.add_argument("--minutes", type=int, default=90); p.add_argument("--name", default="suite-inst")
    p = sub.add_parser("predeclare"); p.add_argument("dir")
    p = sub.add_parser("analyse"); p.add_argument("--runs", default="out/suite-inst"); p.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(Path(a.out), a.minutes, a.name)
    elif a.cmd == "predeclare":
        predeclare(Path(a.dir))
    else:
        s = analyse(Path(a.runs), Path(a.out or (Path(a.runs) / "analysis")))
        if s["problems"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
