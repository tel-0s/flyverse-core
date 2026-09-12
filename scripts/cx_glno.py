"""GLNO -> PEN sign and the receptor model in the compass ring (receptor integration round 2; docs/audits/cx_glno.md).

The ring attractor of docs/audits/cx_wedge.md (EPG <-> PEN and EPG <-> PEG x gE, Delta7 -> EPG x gD with
Delta7 -> PEN x1, compass adaptation 0, 10 Hz EPG background, 4 wedges at +40 Hz for 2 s, 5 s free) has one
large unsigned input: GLNO (4 LAL-NO1 cells, transmitter 'unknown' in every MaleCNS column and in no expression
source; 16,371 raw synapses = 19.4 % of PEN's raw input over 84 edges; its own input 37 % from PEN; T-bar
prediction glutamate 51 % / acetylcholine 37 %). Under NT_SIGN the loop PEN -> GLNO -> PEN is silent (sign 0).

Configs (x gains (2, 15) and (1.75, 15), x seeds 0, 1, 2):
  base        GLNO sign 0 (the adopted model: TYPE_NT_OVERRIDE = {TmY14, Mi19, aMe8}), receptor model off
  glu         TYPE_NT_OVERRIDE + {GLNO: glutamate}   (GLNO -> PEN fast -1 under NT_SIGN)
  ach         TYPE_NT_OVERRIDE + {GLNO: acetylcholine} (fast +1)
  sign-class  GLNO sign 0, LIFParams.receptor_model 'sign', net rule 'class'
  sign-abs    GLNO sign 0, receptor_model 'sign', net rule 'abs'
Every config compiles its connectome from the raw MaleCNS files into out/cache_<hash>/ (cx_wedge.load_connectome,
scratch=True) so that the cluster's shared cache -- which predates TYPE_NT_OVERRIDE -- is never used.

Cluster (one batch, one job per config; each job asserts CUDA):
  python scripts/cluster_run.py --name cx-glno --minutes 40 \
    "python scripts/cx_glno.py --run base > out/cx_glno_base.txt; cat out/cx_glno_base.txt" ... (glu, ach, sign-class, sign-abs)
    --fetch out/cx_glno_base.json out/cx_glno_base.txt ...
Report (CPU, local): python scripts/cx_glno.py --report   -> the config x seed table (out/cx_glno_table.md) and the
count of receptor-model entries changed on EPG / PEN / Delta7 (connectome.receptor_signs restricted to those cells).

Round 3 (docs/audits/cx_glno.md section 4): the GLNO=gaba gain scan, config `gaba`, gE {1.75, 2, 2.25, 2.5} x
gD {8, 15, 25, 40} (Delta7 -> EPG only), 3 seeds, one job per gE (12 runs each), plus `glu` at gE 2 / 2.25 on the
same gD grid and `base` at gE 2 / gD 15 from the shared cache (--no-scratch; it carries TYPE_NT_OVERRIDE since round 2):
  python scripts/cluster_run.py --name r3-glno-gaba --minutes 20 \
    "python -c 'import torch; assert torch.cuda.is_available()' && python scripts/cx_glno.py --run gaba --gains 1.75:8,1.75:15,1.75:25,1.75:40 --out out/cx_glno_gaba_gE1.75.json > out/cx_glno_gaba_gE1.75.txt; cat out/cx_glno_gaba_gE1.75.txt" ... --fetch out/
  python scripts/cx_glno.py --report --files "out/cx_glno_gaba_*.json" "out/cx_glno_glu_gE*.json" out/cx_glno_base_r3.json --table cx_glno_gaba_table
"""
from __future__ import annotations

import argparse
import json
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
CONFIGS = {
    "base": dict(nt_override={}, receptor_model=None, receptor_net_rule="class"),
    "glu": dict(nt_override={"GLNO": "glutamate"}, receptor_model=None, receptor_net_rule="class"),
    "ach": dict(nt_override={"GLNO": "acetylcholine"}, receptor_model=None, receptor_net_rule="class"),
    "sign-class": dict(nt_override={}, receptor_model="sign", receptor_net_rule="class"),
    "sign-abs": dict(nt_override={}, receptor_model="sign", receptor_net_rule="abs"),
    # round 3 (docs/audits/cx_glno.md section 4): the GLNO=gaba gain scan -- both EM predictions (MaleCNS T-bars, FlyWire
    # top_nt) favour an inhibitory GLNO; gE x gD grid, Delta7 -> EPG only, 3 seeds, 5 s
    "gaba": dict(nt_override={"GLNO": "gaba"}, receptor_model=None, receptor_net_rule="class"),
}
GAINS = [(2.0, 15.0), (1.75, 15.0)]
# round-3 grid: gE in {1.75, 2, 2.25, 2.5} x gD in {8, 15, 25, 40} for gaba; gE 2 / 2.25 for glu; base gE 2 / gD 15 anchor
GRID_GE = [1.75, 2.0, 2.25, 2.5]
GRID_GD = [8.0, 15.0, 25.0, 40.0]
SEEDS = [0, 1, 2]
THRESH_HZ = 22.0


# ---------------------------------------------------------------------------------------------- structure checks
def glno_structure(c, log=print) -> dict:
    """GLNO's cells, label and sign in this connectome, and the GLNO -> PEN / PEN -> GLNO entries of W."""
    n = c.neurons
    ty = n.type.fillna("").to_numpy()
    glno = np.flatnonzero(ty == "GLNO")
    pen = np.flatnonzero(np.char.startswith(ty.astype(str), "PEN_"))
    W = c.W.tocsr(); coo = W.tocoo()
    cnt = connectome.sign0_counts(c, W=W, build=False)   # raw counts of the explicit-zero (sign-0) entries, if the file is cached (never built here)
    raw = np.abs(coo.data).astype(np.float64)
    if cnt is not None:
        raw = np.where(coo.data == 0, cnt, raw)
    g2p = np.isin(coo.row, pen) & np.isin(coo.col, glno)
    p2g = np.isin(coo.row, glno) & np.isin(coo.col, pen)
    p_all = np.isin(coo.row, pen)
    d = dict(n_glno=int(len(glno)), glno_nt=n.nt.to_numpy()[glno].tolist(), glno_sign=n.sign.to_numpy()[glno].tolist(),
             glno_to_pen_edges=int(g2p.sum()), glno_to_pen_raw_syn=float(raw[g2p].sum()) if cnt is not None else None,
             glno_to_pen_W_sum=float(coo.data[g2p].sum()), pen_raw_input=float(raw[p_all].sum()) if cnt is not None else None,
             pen_to_glno_edges=int(p2g.sum()), pen_to_glno_W_sum=float(coo.data[p2g].sum()))
    log(f"GLNO: {d['n_glno']} cells, nt {sorted(set(d['glno_nt']))}, sign {sorted(set(d['glno_sign']))}; GLNO -> PEN {d['glno_to_pen_edges']} entries, "
        f"W sum {d['glno_to_pen_W_sum']:+.0f} (raw {d['glno_to_pen_raw_syn']}); PEN -> GLNO {d['pen_to_glno_edges']} entries, W sum {d['pen_to_glno_W_sum']:+.0f}")
    return d


def receptor_ring_changes(c, net_rule: str, log=print) -> dict:
    """connectome.receptor_signs(c, net_rule) restricted to the compass ring: entries whose fast sign differs from
    the presynaptic NT_SIGN, by postsynaptic type (EPG / PEN_a / PEN_b / Delta7, plus PEG, EPGt, ER/ExR, GLNO for
    context), and the table tier the ring's rows carry."""
    n = c.neurons
    ty = n.type.fillna("").to_numpy()
    rs = connectome.receptor_signs(c, net_rule=net_rule)
    coo = c.W.tocoo()
    pre_sign = n.sign.to_numpy(np.float32)[coo.col]
    changed = rs.fast_sign != pre_sign
    tiers = np.array(connectome.RECEPTOR_TIERS)[rs.tier]
    out = {"net_rule": net_rule, "post": {}}
    groups = {"EPG": ty == "EPG", "PEN_a": np.char.startswith(ty.astype(str), "PEN_a"), "PEN_b": np.char.startswith(ty.astype(str), "PEN_b"),
              "Delta7": ty == "Delta7", "PEG": ty == "PEG", "EPGt": ty == "EPGt",
              "ER/ExR": np.array([bool(re.match(cx_wedge.RING_RE, t)) for t in ty]), "GLNO": ty == "GLNO"}
    for name, m in groups.items():
        cells = np.flatnonzero(m)
        sel = np.isin(coo.row, cells)
        d = dict(cells=int(len(cells)), entries=int(sel.sum()), matched=int((rs.tier[sel] >= connectome.RECEPTOR_TIERS.index("nt_class")).sum()),
                 changed=int((changed & sel).sum()), changed_syn=float(np.abs(coo.data[changed & sel]).sum()),
                 tiers={t: int(v) for t, v in zip(*np.unique(tiers[sel], return_counts=True))})
        # changed entries by presynaptic type
        if d["changed"]:
            pre_ty = pd.Series(ty[coo.col[changed & sel]]).value_counts()
            d["changed_by_pre_type"] = {k: int(v) for k, v in pre_ty.head(12).items()}
        out["post"][name] = d
    core = ["EPG", "PEN_a", "PEN_b", "Delta7"]
    out["ring_core_changed"] = int(sum(out["post"][k]["changed"] for k in core))
    out["ring_core_entries"] = int(sum(out["post"][k]["entries"] for k in core))
    # also: entries whose PRE is a ring-core cell (their outputs elsewhere)
    core_cells = np.flatnonzero(np.isin(ty, ["EPG", "Delta7"]) | np.char.startswith(ty.astype(str), "PEN_"))
    pre_sel = np.isin(coo.col, core_cells)
    out["ring_core_as_pre_changed"] = int((changed & pre_sel).sum())
    out["ring_core_as_pre_entries"] = int(pre_sel.sum())
    log(f"receptor model ({net_rule}): entries with a changed fast sign, by postsynaptic type: " +
        ", ".join(f"{k} {v['changed']}/{v['entries']}" for k, v in out["post"].items()) +
        f"; ring core (EPG+PEN+Delta7 as post) {out['ring_core_changed']}/{out['ring_core_entries']}; "
        f"as pre {out['ring_core_as_pre_changed']}/{out['ring_core_as_pre_entries']}")
    return out


def ring_table_rows(log=print) -> pd.DataFrame:
    rt = connectome.read_receptor_table()
    sel = rt[rt.malecns_type.astype(str).str.match(r"^(EPG|PEN_a|PEN_b|Delta7|PEG|EPGt|GLNO)") &
             rt.transmitter.isin(["acetylcholine", "gaba", "glutamate"])]
    cols = [c for c in ["malecns_type", "transmitter", "tier", "source", "fast_sign", "fast_sign_abs", "fast_sign_nonmda",
                        "fast_gain_class", "fast_gain_class_abs", "slow_sign", "pool_mixed"] if c in sel.columns]
    log(sel[cols].to_string(index=False))
    return sel[cols]


# ---------------------------------------------------------------------------------------------- run (GPU)
def run(config: str, seeds, gains, out_path: Path, cuda_graphs=True, scratch=True):
    """scratch=False: a config without an override reads the default (shared) cache, which since round 2 carries
    TYPE_NT_OVERRIDE (sum|W| 121,460,584); configs with an override always compile into out/cache_<hash>/."""
    import torch
    assert torch.cuda.is_available(), "CUDA is not available on this node (resubmit the job)"
    print(f"device {torch.cuda.get_device_name(0)}; torch {torch.__version__}")
    cfg = CONFIGS[config]
    t0 = time.time()
    c, cache_dir, table = cx_wedge.load_connectome(cfg["nt_override"], scratch=scratch, verbose=False)
    print(f"sum|W| {float(abs(c.W).sum()):.0f}")
    print(f"connectome: {c.n} cells, nnz {c.W.nnz}; scratch cache {cache_dir}; TYPE_NT_OVERRIDE {table}; {time.time() - t0:.0f} s")
    st = glno_structure(c)
    extra = {}
    if cfg["receptor_model"]:
        extra["receptor_ring"] = receptor_ring_changes(c, cfg["receptor_net_rule"])
    cells = cx_wedge.compass_cells(c)
    rows = []
    for gE, gD in gains:
        for seed in seeds:
            r = cx_wedge.simulate(c, cells, [(gE, gD)], seed=seed, delta7_pen=False, cuda_graphs=cuda_graphs,
                                  receptor_model=cfg["receptor_model"], receptor_net_rule=cfg["receptor_net_rule"],
                                  nt_override=cfg["nt_override"], thresh_hz=THRESH_HZ)[0]
            r.update(config=config, cache_dir=str(cache_dir), glno_structure=st, **extra)
            rows.append(r)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=1)
    print(f"{len(rows)} rows -> {out_path}; {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------------------------- report (CPU)
def bump_at_driven(centre, start_wedge, width, n=16):
    """Circular distance of the bump centre (wedge units) from the driven block's centre, <= width / 2 + 0.5."""
    mid = start_wedge + (width - 1) / 2.0
    d = abs((centre - mid + n / 2) % n - n / 2)
    return d <= width / 2.0 + 0.5, d


def metrics(r: dict) -> dict:
    t5 = "t5.0"
    n_in, n_out = r["n_in"], r["n_out"]
    at_driven, dist = bump_at_driven(r[f"{t5}_centre_wedge"], r["start_wedge"], r["width"])
    pre_bump_elsewhere = r["pre_out_above"] >= 4 and r["pre_in_above"] <= 2
    pre_bump_at_driven = r["pre_in_above"] >= 6
    persist = r[f"{t5}_in_above"] >= 8 and r[f"{t5}_out_above"] <= 3
    if pre_bump_elsewhere:
        capture = "captured" if (at_driven and r[f"{t5}_in_above"] >= 6) else "not captured"
    elif pre_bump_at_driven:
        capture = "spontaneous at driven tile"
    else:
        capture = "no prior bump"
    return dict(config=r["config"], gE=r["gE"], gD=r["gD"], seed=r["seed"],
                pre=f"{r['pre_in_mean']:.0f}/{r['pre_out_mean']:.0f} ({r['pre_in_above']}/{n_in}, {r['pre_out_above']}/{n_out})",
                during=f"{r['during_in_mean']:.0f}/{r['during_out_mean']:.0f}",
                t05=f"{r['t0.5_in_mean']:.0f}/{r['t0.5_out_mean']:.0f} ({r['t0.5_in_above']}, {r['t0.5_out_above']})",
                t2=f"{r['t2.0_in_mean']:.0f}/{r['t2.0_out_mean']:.0f} ({r['t2.0_in_above']}, {r['t2.0_out_above']})",
                t5=f"{r[f'{t5}_in_mean']:.0f}/{r[f'{t5}_out_mean']:.0f}",
                in_above=f"{r[f'{t5}_in_above']}/{n_in}", out_above=f"{r[f'{t5}_out_above']}/{n_out}",
                persist="yes" if persist else "no", vs=round(r[f"{t5}_vector_strength"], 2),
                centre=round(r[f"{t5}_centre_wedge"], 1), capture=capture,
                bump_hz=round(r[f"{t5}_in_mean"], 1), out_hz=round(r[f"{t5}_out_mean"], 1),
                pen=round(r[f"{t5}_pen"], 1), delta7=round(r[f"{t5}_delta7"], 1), glno=round(r[f"{t5}_glno"], 1),
                glno_during=round(r["during_glno"], 1), peg=round(r[f"{t5}_peg"], 1), ring=round(r[f"{t5}_ring"], 1),
                rest=round(r[f"{t5}_rest"], 3), wall_s=r["wall_s"])


def report(out_dir: Path, configs=None, files=None, table="cx_glno_table"):
    """files: explicit JSON paths / globs (round 3: out/cx_glno_gaba_*.json ...); otherwise out/cx_glno_<config>.json.
    The summary groups by (gE, gD, config); the tables go to out/<table>.md / .csv."""
    rows = []
    paths = []
    if files:
        for pat in files:
            paths += sorted(Path().glob(pat)) if any(ch in pat for ch in "*?[") else [Path(pat)]
    else:
        paths = [out_dir / f"cx_glno_{cfg}.json" for cfg in (configs or CONFIGS)]
    for p in paths:
        if not p.exists():
            print(f"missing {p}"); continue
        for r in json.load(open(p)):
            rows.append(metrics(r))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows"); return None
    df = df.sort_values(["config", "gE", "gD", "seed"], key=lambda s: s.map(list(CONFIGS).index) if s.name == "config" else s,
                        ascending=[True, True, True, True]).reset_index(drop=True)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(df.to_string(index=False))
    # markdown table
    cols = ["config", "gE", "gD", "seed", "pre", "t05", "t2", "t5", "in_above", "out_above", "persist", "vs", "centre", "capture",
            "pen", "delta7", "glno", "glno_during", "ring", "rest", "wall_s"]
    hdr = ["config", "gE", "gD", "seed", "before pulse in/out (>22 Hz in, out)", "0.5 s in/out (>22)", "2 s", "5 s in/out Hz",
           "in >22 Hz", "out >22 Hz", "persists", "vs", "centre wedge", "capture", "PEN", "Delta7", "GLNO", "GLNO during pulse",
           "ER/ExR", "rest", "wall s"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md = "\n".join(lines)
    # per-config summary (mean over seeds, per gain)
    def rng(s):
        return f"{s.min():.0f}-{s.max():.0f}" if s.max() - s.min() >= 1 else f"{s.mean():.0f}"

    summ = df.groupby(["config", "gE", "gD"], sort=False).agg(
        n=("seed", "size"), persist=("persist", lambda s: int((s == "yes").sum())),
        bump_hz=("bump_hz", "mean"), bump_range=("bump_hz", rng), out_hz=("out_hz", "mean"), out_range=("out_hz", rng),
        in_above=("in_above", lambda s: "/".join(x.split("/")[0] for x in s)), out_above=("out_above", lambda s: "/".join(x.split("/")[0] for x in s)),
        vs=("vs", "mean"), vs_min=("vs", "min"), pen=("pen", "mean"), delta7=("delta7", "mean"), glno=("glno", "mean"),
        glno_during=("glno_during", "mean"),
        captured=("capture", lambda s: f"{int((s == 'captured').sum())}/{int(s.isin(['captured', 'not captured']).sum())}")).reset_index()
    print("\nsummary (mean over seeds):")
    print(summ.to_string(index=False))
    s_lines = ["| config | gE | gD | runs | persist (n) | bump Hz at 5 s (mean; range) | out Hz (mean; range) | in >22 Hz (/11, per seed) | out >22 Hz (/35, per seed) | vs (mean; min) | PEN | Delta7 | GLNO (5 s) | GLNO (pulse) | captured / with prior bump elsewhere |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        s_lines.append(f"| {r.config} | {r.gE} | {r.gD:.0f} | {r.n} | {r.persist} | {r.bump_hz:.0f}; {r.bump_range} | {r.out_hz:.1f}; {r.out_range} | {r.in_above} | {r.out_above} | "
                       f"{r.vs:.2f}; {r.vs_min:.2f} | {r.pen:.1f} | {r.delta7:.1f} | {r.glno:.1f} | {r.glno_during:.1f} | {r.captured} |")
    md_all = "## Per run\n\n" + md + "\n\n## Summary\n\n" + "\n".join(s_lines) + "\n"
    (out_dir / f"{table}.md").write_text(md_all, encoding="utf-8")
    df.to_csv(out_dir / f"{table}.csv", index=False)
    summ.to_csv(out_dir / f"{table}_summary.csv", index=False)
    print(f"\n-> {out_dir / (table + '.md')}, {out_dir / (table + '.csv')}, {out_dir / (table + '_summary.csv')}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(CONFIGS), default=None, help="run one config on the GPU (all gains x seeds)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--gains", default=",".join(f"{a}:{b}" for a, b in GAINS))
    ap.add_argument("--out", default=None, help="JSON (default out/cx_glno_<config>.json)")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--report", action="store_true", help="table from out/cx_glno_*.json (CPU)")
    ap.add_argument("--check-receptor", action="store_true",
                    help="count receptor-model entries changed on EPG / PEN / Delta7 in the local cache, all three net rules (CPU)")
    ap.add_argument("--configs", default=None, help="comma-separated subset for --report")
    ap.add_argument("--files", nargs="*", default=None, help="--report: explicit JSON paths / globs instead of out/cx_glno_<config>.json")
    ap.add_argument("--table", default="cx_glno_table", help="--report: output basename under out/")
    ap.add_argument("--no-scratch", action="store_true",
                    help="--run: a config without an override reads the default (shared) cache instead of compiling into out/cache_<hash>/")
    a = ap.parse_args()
    if a.run:
        seeds = [int(s) for s in a.seeds.split(",")]
        gains = [tuple(float(x) for x in g.split(":")) for g in a.gains.split(",")]
        run(a.run, seeds, gains, Path(a.out) if a.out else OUT / f"cx_glno_{a.run}.json", cuda_graphs=not a.no_graphs,
            scratch=not a.no_scratch)
    if a.check_receptor:
        c = connectome.load(verbose=False)
        print(f"local cache: {c.n} cells, nnz {c.W.nnz}")
        glno_structure(c)
        ring_table_rows()
        res = {rule: receptor_ring_changes(c, rule) for rule in connectome.RECEPTOR_NET_RULES}
        with open(OUT / "cx_glno_receptor_check.json", "w") as f:
            json.dump(res, f, indent=1)
        print(f"-> {OUT / 'cx_glno_receptor_check.json'}")
    if a.report:
        report(OUT, a.configs.split(",") if a.configs else None, files=a.files, table=a.table)


if __name__ == "__main__":
    main()
