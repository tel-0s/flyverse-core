"""The object sweep under the opt-in per-stream hooks of the rate optic lobe (docs/audits/optic_stream_hooks.md).

NOT a tuning run: one batch that shows each hook is LIVE on the full lobe and what its magnitude is, on the
scripts/probe_object_sweep.py protocol verbatim (pinned fly, the 1 cm black ball 5 cm ahead sweeping +-6 cm, 12 s scored
after 3 s settle, ball vs none), with the hook installed through OpticParams before the model is built.

Arms (`ARMS`; every arm is an OpticParams override, the graph is untouched):
  base      the shipped model (OpticParams defaults)
  rectify   stream_rectify [('^(Mi1|Tm3|Tm2)$', '^T3$', 'pos'), ('^(Tm1|Tm4)$', '^T3$', 'neg')]  -- the carriers of opposite
            figure sign onto T3 (docs/audits/deficit_object.md 0/4.3) half-wave rectified before T3's sum, signs preserved
  adapt     stream_adapt [('^(Mi1|Tm3|Tm2|Tm1|Tm4)$', '^T3$', 100.0, 1.0)]  -- the same streams with a 100 ms fast
            adaptation state subtracted (gain 1: the stream is high-passed)
  suppress  spatial_suppress [('^(Mi1|Tm1|Tm3|Tm4)$', 0.5, 10.0)]  -- centre-surround on those carriers' deviations, k 0.5
            within 10 deg, every target of the stream
  null      the shipped model, none vs none (probe_object_sweep --null): the run-to-run scatter of every diff_* field

    python scripts/probe_object_hooks.py plan --out out/hooks --runs 3 --name hooks          # -> out/hooks/batch.sh
    sh out/hooks/batch.sh                                                                     # one cluster_run call
    python scripts/probe_object_hooks.py record --arm rectify --seed 0 --out out/hooks/rectify_r0.json   (a job line)
    PYTHONIOENCODING=utf-8 python scripts/probe_object_hooks.py analyse --dir out/hooks --json out/interp/hooks/object_hooks.json

`record` writes probe_object_sweep's `summarize` statistics for the ball and none conditions, the population time
courses per type, the OpticLobe's `hook_info` (the split entry counts, the spatial operator's neighbourhood sizes,
whether the Torch substep ran) and the provenance block (resolved LIFParams / OpticParams, realised device, cache
fingerprint, commit). `analyse` reads every `<arm>_r*.json` and reports, per type (T3 / T2 / Tm5Y / TmY21 / LC11 /
LC10a) and statistic, `common.compare` of (i) each arm's (ball - none) statistic against the none-vs-none null,
(ii) each hook arm against the base arm on the same statistic, (iii) each hook arm against base on the ball arm's
absolute level. Three runs per arm read `underpowered` by construction (`common.p_floor(3, 3)` = 0.10): the
numbers are magnitudes with their scatter, never verdicts. Extra `--optic K=V` overrides (common.parse_kv grammar)
are applied on top of the arm's and recorded.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from flyverse.interp import common  # noqa: E402

RECT = [["^(Mi1|Tm3|Tm2)$", "^T3$", "pos"], ["^(Tm1|Tm4)$", "^T3$", "neg"]]
ADAPT = [["^(Mi1|Tm3|Tm2|Tm1|Tm4)$", "^T3$", 100.0, 1.0]]
SUPP = [["^(Mi1|Tm1|Tm3|Tm4)$", 0.5, 10.0]]
ARMS = {
    "base": {"optic": {}, "null": False, "note": "the shipped model (OpticParams defaults), ball vs none"},
    "rectify": {"optic": {"stream_rectify": RECT}, "null": False,
                "note": "per-stream half-wave rectification of the T3 carriers before T3's sum (Mi1/Tm3/Tm2 'pos', Tm1/Tm4 'neg'); weights and signs untouched"},
    "adapt": {"optic": {"stream_adapt": ADAPT}, "null": False, "note": "the same T3 streams minus a 100 ms fast-adaptation state (gain 1)"},
    "suppress": {"optic": {"spatial_suppress": SUPP}, "null": False, "note": "centre-surround (k 0.5, 10 deg) on Mi1/Tm1/Tm3/Tm4 deviations, every target"},
    "null": {"optic": {}, "null": True, "note": "the shipped model, none vs none: the null of every diff_* statistic"},
}
HOOK_ARMS = ("rectify", "adapt", "suppress")
RATE_TYPES = ["T3", "T2", "Tm5Y", "TmY21"]
SPK_TYPES = ["LC11", "LC10a"]
RATE_DIFF = ["diff_signed_best_cell", "diff_abs_best_cell_mean", "diff_signed_mean", "diff_abs_mean"]
RATE_LEVEL = ["dev_abs_best_cell_mean", "dev_abs_mean", "dev_mean"]
SPK_DIFF = ["diff_max_over_cells_mean_mv", "diff_mean_over_cells_mean_mv", "diff_rate_hz_max_cell", "diff_rate_hz_mean"]
SPK_LEVEL = ["drive_best_cell_mean_mv", "drive_mean_mv", "drive_peak_mv", "rate_hz_mean", "rate_hz_max_cell"]


def install_optic_overrides(overrides: dict):
    """Every optic.OpticParams built from here on (room_demo.Sim's included) carries `overrides` (the probes' pattern)."""
    if not overrides:
        return
    from flyverse import optic
    O = optic.OpticParams

    def make(**kw):
        p = O(**kw)
        for k, v in overrides.items():
            setattr(p, k, v)
        return p
    optic.OpticParams = make


def cmd_record(args) -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import torch
    assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    arm = ARMS[args.arm]
    overrides = dict(arm["optic"]); overrides.update(common.parse_kv(args.optic))
    install_optic_overrides(overrides)
    import probe_object_sweep as pos_
    from flyverse.interp import trace as tr
    if args.cache_dir:
        pos_.patch_cache(args.cache_dir)
    pos_.BALL_R, pos_.AHEAD, pos_.HALF_SWEEP = args.ball_radius, args.ahead, args.half_sweep
    with_ball = not arm["null"]
    print(f"object hooks: arm {args.arm} ({arm['note']}); overrides {overrides}; seed {args.seed}; {'none vs none' if arm['null'] else 'ball vs none'}; "
          f"ball r {pos_.BALL_R} m at {pos_.AHEAD} m ({2 * np.degrees(np.arctan(pos_.BALL_R / pos_.AHEAD)):.1f} deg), half sweep {pos_.HALF_SWEEP} m; "
          f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)
    t0 = time.time()
    sim_a, res_a = pos_.run(with_ball, args, args.seed)
    o = sim_a.fb.optic
    hook_info = common.to_jsonable(o.hook_info); hook_fb = common.to_jsonable(getattr(o, "hook_info_fb", None))
    hook_info.update({"torch_substep": not (o.cuda or o.metal), "optic_cuda_kernels": bool(o.cuda), "optic_metal_kernels": bool(o.metal),
                      "n_streams": len(o.streams), "adapt_state_shape": list(o.stream_adapt_state.shape)})
    print(f"  optic hooks: {json.dumps({k: v for k, v in hook_info.items() if k != 'streams'})}", flush=True)
    for s in hook_info.get("streams", []):
        print(f"    stream {s}", flush=True)
    stim = {"protocol": "object_sweep", "generator": "scripts/probe_object_sweep.py::run via scripts/probe_object_hooks.py record",
            "arm": args.arm, "arm_note": arm["note"], "optic_overrides": overrides, "condition_a": "none" if arm["null"] else "ball", "condition_b": "none",
            "ball_radius_m": pos_.BALL_R, "ahead_m": pos_.AHEAD, "half_sweep_m": pos_.HALF_SWEEP, "sweep_s": pos_.SWEEP_S,
            "angular_diameter_deg": float(2 * np.degrees(np.arctan(pos_.BALL_R / pos_.AHEAD))), "pos": list(pos_.POS), "heading_rad": float(pos_.HEADING),
            "seconds": args.seconds, "settle": args.settle, "control": "the same timeline with the ball parked out of the scene"}
    prov = common.provenance(sim_a.c, sim_a.fb.brain.p, o.p, fb=sim_a.fb, device="cuda", seeds=[args.seed], env_seeds=[args.seed], batch=1,
                             stimulus=stim, retina=tr.retina_record(sim_a.fb.retina, sim_a.c), cache_dir=args.cache_dir)
    del sim_a; torch.cuda.empty_cache()
    sim_b, res_b = pos_.run(False, args, args.seed)
    del sim_b; torch.cuda.empty_cache()
    S_a = pos_.summarize(res_a, res_b); S_b = pos_.summarize(res_b)

    def courses(res):
        out = {}
        for t in RATE_TYPES:
            tr_ = np.array(res["rate"][t].trace); out[t] = {"mean_signed": tr_[:, 0].tolist(), "mean_abs": tr_[:, 1].tolist()}
        for t in SPK_TYPES:
            tr_ = np.array(res["drive"][t].trace); mx = res["drive_frames"][t].max(1)
            out[t] = {"drive_mean_signed_mv": tr_[:, 0].tolist(), "drive_mean_abs_mv": tr_[:, 1].tolist(), "drive_max_cell_mv": mx.tolist()}
        return out
    out = {"schema": "flyverse.probe_object_hooks/1", "arm": args.arm, "arm_note": arm["note"], "null": arm["null"], "seed": args.seed,
           "optic_overrides": overrides, "generator": " ".join(sys.argv), "wall_s": time.time() - t0,
           "ball": S_a, "none": S_b, "hook_info": hook_info, "hook_info_fb": hook_fb, "radiance_check": res_a["radiance"],
           "time_course": {"a": courses(res_a), "b": courses(res_b), "frame_s": pos_.FRAME_S}, "provenance": prov}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable(out), f, indent=1)
    print(f"\narm {args.arm} seed {args.seed}: {'none vs none' if arm['null'] else 'ball vs none'}, {args.seconds:.0f} s window ({time.time() - t0:.0f} s wall)")
    for t in RATE_TYPES:
        a = S_a[t]
        print(f"  {t:6s} |dev| best cell {a['dev_abs_best_cell_mean']:.4f} (none {S_b[t]['dev_abs_best_cell_mean']:.4f})  diff signed best {a['diff_signed_best_cell']:+.5f}  "
              f"diff |dev| best {a['diff_abs_best_cell_mean']:+.5f}  diff signed mean {a['diff_signed_mean']:+.6f}")
    for t in SPK_TYPES:
        a = S_a[t]
        print(f"  {t:6s} drive best cell {a['drive_best_cell_mean_mv']:+.4f} mV (none {S_b[t]['drive_best_cell_mean_mv']:+.4f})  diff max-over-cells {a['diff_max_over_cells_mean_mv']:+.4f} mV  "
              f"rate mean {a['rate_hz_mean']:.4f} Hz  diff rate max cell {a['diff_rate_hz_max_cell']:+.4f} Hz")
    print(f"written {args.out}")
    return 0


def job_line(arm: str, seed: int, out_dir: str, extra: str = "") -> str:
    stem = f"{out_dir}/{arm}_r{seed}"
    return (f"mkdir -p {out_dir} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"python scripts/probe_object_hooks.py record --arm {arm} --seed {seed}{extra} --out {stem}.json > {stem}.txt 2>&1; tail -12 {stem}.txt")


def cmd_plan(args) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {list(ARMS)}")
    extra = "".join(f" --optic {shlex.quote(kv)}" for kv in args.optic)
    if args.ball_radius != 0.005:
        extra += f" --ball-radius {args.ball_radius}"
    jobs = [job_line(a, s, args.out, extra) for a in arms for s in common.replicate_seeds(args.runs, args.seed)]
    log = f"out/{args.name}_cluster.log"
    call = (f"python scripts/cluster_run.py --name {args.name} --minutes {int(args.minutes)} \\\n  " + " \\\n  ".join(shlex.quote(j) for j in jobs)
            + f" \\\n  --fetch {args.out}/")
    header = (f"# {len(jobs)} jobs = {len(arms)} arms x {args.runs} runs; ONE cluster_run call, fetched into the NAMED directory {args.out}/\n"
              f"# arms: " + "; ".join(f"{a} = {ARMS[a]['note']}" for a in arms) + "\n")
    (out / "batch.sh").write_text("#!/bin/sh\n" + header + f"set -e\nmkdir -p {args.out} out\n"
                                  f'if [ -f {log} ]; then mv {log} "{log[:-4]}.$(date +%Y%m%dT%H%M%S).log"; fi\n'
                                  f"{call} 2>&1 | tee {log}\n", encoding="utf-8", newline="\n")
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"arms": {a: ARMS[a] for a in arms}, "runs": args.runs, "seed0": args.seed, "extra_optic": args.optic, "ball_radius_m": args.ball_radius,
                   "jobs": jobs, "name": args.name, "git": common.git_state()}, f, indent=1)
    print(f"written {out / 'batch.sh'} ({len(jobs)} jobs) and {out / 'manifest.json'}")
    return 0


def load_arm(d: Path, arm: str) -> list[dict]:
    runs = []
    for p in sorted(glob.glob(str(d / f"{arm}_r*.json"))):
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
        if j.get("schema") != "flyverse.probe_object_hooks/1" or j.get("arm") != arm:
            raise ValueError(f"{p}: not a {arm} record of this tool")
        j["_file"] = p; runs.append(j)
    return runs


def verify_batch(d: Path, runs: dict) -> list[str]:
    """The console-vs-meta check before analysis: every JSON has a console, the console's realised device is cuda, the
    optic hooks of a hook arm ran on the Torch substep, and the recorded arm / seed match the file name."""
    problems = []
    for arm, rs in runs.items():
        for r in rs:
            p = Path(r["_file"]); txt = p.with_suffix(".txt")
            if not txt.exists():
                problems.append(f"{p.name}: no console"); continue
            console = txt.read_text(encoding="utf-8", errors="replace")
            if "device cuda" not in console:
                problems.append(f"{p.name}: console lacks 'device cuda'")
            if "written " not in console:
                problems.append(f"{p.name}: console has no 'written' line (died mid-run)")
            if r["provenance"]["execution"].get("device") != "cuda":
                problems.append(f"{p.name}: provenance device {r['provenance']['execution'].get('device')}")
            if p.name != f"{arm}_r{r['seed']}.json":
                problems.append(f"{p.name}: arm / seed mismatch ({arm}, {r['seed']})")
            hi = r["hook_info"]
            if arm in HOOK_ARMS and not (hi.get("active") and hi.get("torch_substep") and hi.get("n_streams", 0) > 0):
                problems.append(f"{p.name}: hook not live ({hi})")
            if arm not in HOOK_ARMS and hi.get("active"):
                problems.append(f"{p.name}: hooks active on a plain arm")
    return problems


def cmd_analyse(args) -> int:
    d = Path(args.dir)
    runs = {a: load_arm(d, a) for a in ARMS}
    runs = {a: r for a, r in runs.items() if r}
    if "base" not in runs:
        raise SystemExit(f"no base runs under {d}")
    problems = verify_batch(d, runs)
    print(f"runs per arm: {({a: len(r) for a, r in runs.items()})}; verify-batch: {'ok' if not problems else problems}")
    if problems and not args.force:
        raise SystemExit("verify-batch failed (--force to analyse anyway)")

    def vals(arm, t, key, cond="ball"):
        return [r[cond][t][key] for r in runs.get(arm, []) if key in r[cond][t]]
    rows = []
    for t in RATE_TYPES + SPK_TYPES:
        diffs = RATE_DIFF if t in RATE_TYPES else SPK_DIFF; levels = RATE_LEVEL if t in RATE_TYPES else SPK_LEVEL
        for key in diffs:
            null_v = vals("null", t, key)
            for arm in [a for a in ("base",) + HOOK_ARMS if a in runs]:
                v = vals(arm, t, key)
                if null_v:
                    c = common.compare(v, null_v)
                    rows.append({"type": t, "statistic": key, "arm": arm, "against": "null", **_flat(c)})
                if arm != "base":
                    c = common.compare(v, vals("base", t, key))
                    rows.append({"type": t, "statistic": key, "arm": arm, "against": "base", **_flat(c)})
        for key in levels:
            for arm in [a for a in HOOK_ARMS if a in runs]:
                c = common.compare(vals(arm, t, key), vals("base", t, key))
                rows.append({"type": t, "statistic": key + " (ball arm level)", "arm": arm, "against": "base", **_flat(c)})
    df = __import__("pandas").DataFrame(rows)
    # the hook_info of each hook arm (identical across runs by construction: same code, same graph)
    hooks = {a: {k: v for k, v in runs[a][0]["hook_info"].items()} for a in HOOK_ARMS if a in runs}
    prov = dict(runs["base"][0]["provenance"])
    prov["stimulus"] = dict(prov.get("stimulus", {}), arms={a: ARMS[a] for a in runs}, note="provenance of base_r<seed0>; every run file carries its own")
    res = common.Result.new("lesion", prov)
    res.replicates = {"n": {a: len(r) for a, r in runs.items()}, "unit": "runs", "runs": {a: [r["_file"] for r in rs] for a, rs in runs.items()},
                      "null": "null_r*.json (none vs none, shipped model)"}
    res.add_table("comparisons", df)
    res.add_table("per_run", [{"arm": a, "seed": r["seed"], "file": r["_file"], "wall_s": r.get("wall_s"),
                               **{f"{t}.{k}": r["ball"][t].get(k) for t in RATE_TYPES for k in RATE_DIFF + RATE_LEVEL},
                               **{f"{t}.{k}": r["ball"][t].get(k) for t in SPK_TYPES for k in SPK_DIFF + SPK_LEVEL}} for a, rs in runs.items() for r in rs])
    res.summary = {"arms": {a: ARMS[a]["note"] for a in runs}, "hook_info": hooks, "verify_batch": problems or "ok",
                   "reading": "3 runs per arm: every verdict is 'underpowered' by construction (p_floor 0.10); read diff, z and the per-run values",
                   "headline": {f"{t}.{key}": {a: common.ArmStats.of(vals(a, t, key)).record() for a in runs}
                                for t, key in [("T3", "diff_signed_best_cell"), ("T3", "dev_abs_best_cell_mean"), ("T2", "diff_signed_best_cell"),
                                               ("Tm5Y", "diff_signed_best_cell"), ("TmY21", "diff_signed_best_cell"),
                                               ("LC11", "diff_max_over_cells_mean_mv"), ("LC10a", "diff_max_over_cells_mean_mv")]}}
    res.validation = dict(res.validation, status="n/a", measured="a liveness / magnitude batch, not the lesion validation target")
    res.files = {"generator": "scripts/probe_object_hooks.py", "runs_dir": str(d), "batch": str(d / "batch.sh"), "cluster_log": f"out/{args.name}_cluster.log"}
    path = Path(args.json) if args.json else common.default_json_path("lesion", res.run_id)
    res.save(path)
    print("\ncomparisons (stim = the arm's runs, null = the 'against' arm's runs; verdicts at 3 v 3 are 'underpowered' by construction):")
    common.print_table(df[["type", "statistic", "arm", "against", "n_stim", "n_null", "stim_mean", "stim_sd", "null_mean", "null_sd", "diff", "z", "U", "p", "verdict"]],
                       floatfmt="{:+.4g}", max_rows=400)
    print(f"\nhook_info per arm:")
    for a, h in hooks.items():
        print(f"  {a}: " + json.dumps({k: v for k, v in h.items() if k != 'streams'}))
        for s in h.get("streams", []):
            print(f"     stream {json.dumps(s)}")
    print(f"written {path}")
    return 0


def _flat(c: dict) -> dict:
    return {"n_stim": c["stim"]["n"], "n_null": c["null"]["n"], "stim_mean": c["stim"]["mean"], "stim_sd": c["stim"]["sd"], "stim_values": c["stim"]["values"],
            "null_mean": c["null"]["mean"], "null_sd": c["null"]["sd"], "null_values": c["null"]["values"],
            "diff": c["diff"], "z": c["z"], "welch": c["welch"], "U": c["U"], "p": c["p"], "p_floor": c["p_floor"], "verdict": c["verdict"]}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="GPU: one arm, one seed (ball vs none, or none vs none for the null arm)")
    r.add_argument("--arm", required=True, choices=list(ARMS)); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--optic", action="append", default=[], metavar="KEY=VALUE", help="extra OpticParams override on top of the arm's (common.parse_kv)")
    r.add_argument("--seconds", type=float, default=12.0); r.add_argument("--settle", type=float, default=3.0)
    r.add_argument("--ball-radius", type=float, default=0.005); r.add_argument("--ahead", type=float, default=0.05); r.add_argument("--half-sweep", type=float, default=0.06)
    r.add_argument("--cache-dir", default=os.environ.get("FLYVERSE_CACHE") or None); r.add_argument("--out", required=True)
    p = sub.add_parser("plan", help="write the batch script (one cluster_run call) and the manifest")
    p.add_argument("--out", default="out/hooks"); p.add_argument("--runs", type=int, default=3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", default=",".join(ARMS)); p.add_argument("--name", default="hooks"); p.add_argument("--minutes", type=int, default=40)
    p.add_argument("--optic", action="append", default=[], metavar="KEY=VALUE"); p.add_argument("--ball-radius", type=float, default=0.005)
    a = sub.add_parser("analyse", help="CPU: compare the arms (common.compare) and write the Result JSON")
    a.add_argument("--dir", default="out/hooks"); a.add_argument("--json", default=None); a.add_argument("--name", default="hooks"); a.add_argument("--force", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"record": cmd_record, "plan": cmd_plan, "analyse": cmd_analyse}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
