"""Behaviour round 3, integrate (docs/audits/round3_integration.md): the three mechanisms of the round combined on
the plain-fly rollout and the two compass protocols, in ONE submission, nothing adopted.

The arms are the eight subsets of {A, B, C} on top of the shipped model:

    A  the sided proprioceptive transducer   senses.Proprioception('all+leg_cycle+haltere_sided') = arm D of
                                             docs/audits/body_sided_state.md (probe_vnc_drive.ARMS_BODY['D']); a body-model
                                             mechanism, opt-in
    B  the monoamine slow class at add_low   LIFParams(receptor_model 'full', net rule 'abs', gain classes 1/1/1, slow_mode
                                             'additive', slow_gain 0.02) = probe_monoamines.ARMS['add_low']['lif'] -- the one
                                             bracket of docs/audits/monoamine_slow_term.md that passed the suite 12/0/0 x 5 with
                                             every behaviour key null (0.2 / 1.0 additive run away; gain 0.2 hops)
    C  per-transmitter w_syn at 'high'       LIFParams(w_syn_by_nt = {acetylcholine 1.0, gaba 0.75, glutamate 0.75}) =
                                             probe_unitary.BRACKETS['high']. NO bracket of docs/audits/unitary_strength.md keeps
                                             the suite (walk.power_sustained_hz fails in 3/3 draws of every bracket); 'high' is
                                             the least-cost one (11/1) and the one the unitary thread itself put in the room, so
                                             it is run LABELLED as not suite-safe rather than left out.

Every flag is imported from the thread's own script (no arm is re-typed here), every protocol is the thread's own
generator called with the arm's LIFParams / sense in force (probe_vnc_drive.cmd_room / cmd_compass for the room and the
efferent compass, probe_unitary.cmd_compass for the shipped-gain cx_wedge compass, scripts/benchmark.py for the suite
sections), and every arm of every protocol runs on ONE backend -- the eager Torch path (cuda_kernels off, cuda_graphs
off, sparse matmul, cuSPARSE) that receptor_model 'full' forces in brain.Brain -- so no arm-vs-arm row is backend-crossed.
Each JSON is stamped with an `integrate` block: the arm, its mechanisms, the sense spec, the LIFParams overrides asked for
and the LIFParams fields the Brain actually carried (read back from the provenance; the job FAILS if they differ).

    room     (GPU)  16 flies x 60 s plain-fly rollout (probe_walk_straightness's protocol through probe_vnc_drive room)
    compass  (GPU)  the efferent rotation arm at gE 2 / gD 15 (probe_vnc_drive compass --family body): rest / ccw / rest2 / cw
    wedge    (GPU)  cx_wedge at the SHIPPED gains (probe_unitary compass): bump formation / survival / width after a pulse;
                    no body in the protocol, so the A-arms equal their brain-only counterpart by construction and only
                    shipped / B / C / BC are run
    bench    (GPU)  scripts/benchmark.py --eager, sections rest,taste,smell,walk,loom_escape (no body: same four configs)
    plan     (CPU)  writes out/r3int/batch.sh (ONE cluster_run.py call, --arm-block fam) and predeclared.json
    verify   (CPU)  counts the artefacts against the plan, checks every run's device / backend / arm flags
    analyse  (CPU)  the tables: room (every key x arm, scatter, common.compare vs shipped and the pairwise family),
                    per-frame sidedness, the efferent compass drift / flip tables, the wedge ledger rows, the suite checks,
                    and DNa02's input decomposed under the chosen arms

    PYTHONIOENCODING=utf-8 python scripts/probe_round3_integrate.py arms
    PYTHONIOENCODING=utf-8 python scripts/probe_round3_integrate.py plan --dir out/r3int --runs 5 --compass-seeds 3 --wedge-seeds 4 --draws 2
    bash out/r3int/batch.sh
    PYTHONIOENCODING=utf-8 python scripts/probe_round3_integrate.py verify --dir out/r3int
    PYTHONIOENCODING=utf-8 python scripts/probe_round3_integrate.py analyse --dir out/r3int --out out/r3int/analysis
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

if any(sys.argv[i] == "--device" and sys.argv[i + 1].startswith("cpu") for i in range(len(sys.argv) - 1)):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # a CPU smoke never touches this desktop's GPU; before torch is imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flyverse.interp import common  # noqa: E402
import probe_vnc_drive as pvd  # noqa: E402
import probe_monoamines as pmo  # noqa: E402
import probe_unitary as pun  # noqa: E402

# ---------------------------------------------------------------------------------------------- the arms (imported, not re-typed)
SENSE_A = pvd.ARMS_BODY["D"]                                   # 'all+leg_cycle+haltere_sided'
LIF_B = dict(pmo.ARMS["add_low"]["lif"])                       # receptor_model full / abs / gain 1,1,1 / additive / 0.02
BENCH_B = list(pmo.ARMS["add_low"]["bench"])                   # the same arm as benchmark.py flags
LIF_C = {"w_syn_by_nt": dict(pun.BRACKETS["high"])}            # ACh x1.0, GABA x0.75, glutamate x0.75
BACKEND = dict(cuda_graphs=False, cuda_kernels=False, event_driven=False, cuda_sparse="torch")   # the eager Torch path, every arm
ARMS = {"shipped": (), "A": ("A",), "B": ("B",), "C": ("C",), "AB": ("A", "B"), "AC": ("A", "C"), "BC": ("B", "C"), "ABC": ("A", "B", "C")}
BRAIN_ARMS = ("shipped", "B", "C", "BC")                       # the configurations a body-less protocol can distinguish
MECH_LABEL = {"A": f"sided transducer {SENSE_A!r} (body_sided_state.md arm D)",
              "B": "monoamine slow class add_low: receptor_model full/abs, gain classes 1/1/1, additive, slow_gain 0.02 (monoamine_slow_term.md)",
              "C": "per-transmitter w_syn 'high': ACh x1.0, GABA x0.75, glutamate x0.75 (unitary_strength.md; NOT suite-safe -- no bracket is)"}
PAIRS = (("A", "shipped"), ("B", "shipped"), ("C", "shipped"), ("AB", "A"), ("AB", "B"), ("AC", "A"), ("AC", "C"), ("BC", "B"), ("BC", "C"),
         ("ABC", "AB"), ("ABC", "AC"), ("ABC", "BC"), ("ABC", "A"), ("AB", "shipped"), ("AC", "shipped"), ("BC", "shipped"), ("ABC", "shipped"))
WEDGE_KEYS = ("survival_s", "frac_confined_post", "frac_confined_pre", "bump_hz_post", "width_half_post", "vs_post_all", "epg_in_mean_post", "epg_out_mean_post",
              "epg_in_mean_during", "epg_out_mean_during", "PEN_mean_post", "Delta7_mean_post", "Ring_mean_post", "rest_mean_post", "GLNO_mean_post", "epg_max_post")
BENCH_SECTIONS = pmo.BENCH_SECTIONS                            # rest,taste,smell,walk,loom_escape


def _log(msg):
    print(msg, flush=True)


def mech(arm):
    if arm not in ARMS:
        raise SystemExit(f"arm must be one of {list(ARMS)}")
    return ARMS[arm]


def sense_of(arm):
    return SENSE_A if "A" in mech(arm) else None


def lif_of(arm) -> dict:
    d = {}
    if "B" in mech(arm):
        d.update(copy.deepcopy(LIF_B))
    if "C" in mech(arm):
        d.update(copy.deepcopy(LIF_C))
    return d


def brain_arm(arm) -> str:
    """The brain-only configuration a body-less protocol sees (the A mechanism needs a body)."""
    return "".join(m for m in mech(arm) if m != "A") or "shipped"


def arm_label(arm) -> str:
    return "shipped (sense off, LIFParams())" if not mech(arm) else " + ".join(MECH_LABEL[m] for m in mech(arm))


def bench_flags(arm):
    return list(BENCH_B) if "B" in mech(arm) else ["--receptor-model", "default"]


@contextmanager
def patched(arm, what=("lif",)):
    """For the duration: brain.LIFParams builds instances carrying the arm's overrides (set after construction, so a caller's
    keywords cannot undo them: probe_unitary's pattern); BatchSim / room_demo.Sim / FlyBrain are subclasses that force
    BACKEND (the eager Torch path) whatever a generator asks for. Everything is restored on exit."""
    from flyverse import brain
    over = lif_of(arm)
    saved = {"lif": brain.LIFParams}
    L = brain.LIFParams

    def make(**kw):
        p = L(**kw)
        for k, v in over.items():
            setattr(p, k, copy.deepcopy(v))
        return p
    brain.LIFParams = make
    mods = {}
    try:
        if "batch" in what:
            from flyverse import batch_sim
            Bs = batch_sim.BatchSim; saved["batch"] = Bs

            class EagerBatchSim(Bs):
                def __init__(self, *a, **kw):
                    kw.update(BACKEND); super().__init__(*a, **kw)
            batch_sim.BatchSim = EagerBatchSim; mods["batch"] = batch_sim
        if "sim" in what:
            import room_demo
            Sm = room_demo.Sim; saved["sim"] = Sm

            class EagerSim(Sm):
                def __init__(self, *a, **kw):
                    kw.update(BACKEND); super().__init__(*a, **kw)
            room_demo.Sim = EagerSim; mods["sim"] = room_demo
        if "fly" in what:
            from flyverse import fly
            Fb = fly.FlyBrain; saved["fly"] = Fb

            class EagerFlyBrain(Fb):
                def __init__(self, c=None, **kw):
                    kw.update(cuda_graphs=False, cuda_kernels=False, cuda_sparse="torch"); super().__init__(c, **kw)
            fly.FlyBrain = EagerFlyBrain; mods["fly"] = fly
        yield
    finally:
        brain.LIFParams = saved["lif"]
        if "batch" in mods:
            mods["batch"].BatchSim = saved["batch"]
        if "sim" in mods:
            mods["sim"].Sim = saved["sim"]
        if "fly" in mods:
            mods["fly"].FlyBrain = saved["fly"]


LIF_KEYS_IN_FORCE = ("receptor_model", "receptor_net_rule", "receptor_gain", "slow_mode", "slow_gain", "slow_tau_ms", "w_syn_by_nt", "w_syn", "event_driven")


def _lif_in_force(prov: dict) -> dict:
    lif = (prov.get("model") or {}).get("lif") or {}
    return {k: lif.get(k) for k in LIF_KEYS_IN_FORCE}


def _check_in_force(arm, prov: dict, backend_realised: dict | None) -> list:
    """The LIFParams the Brain carried vs the arm's overrides; the realised backend vs BACKEND. Returns problems."""
    problems = []
    lif = (prov.get("model") or {}).get("lif") or {}
    want = lif_of(arm)
    for k, v in want.items():
        got = lif.get(k)
        if k == "receptor_gain":
            got = {kk: float(vv) for kk, vv in (got or {}).items()}; v = {kk: float(vv) for kk, vv in v.items()}
        if got != v:
            problems.append(f"{k}: in force {got!r} != asked {v!r}")
    if not want:
        if lif.get("receptor_model") != "sign" or lif.get("w_syn_by_nt") not in (None, {}):
            problems.append(f"shipped-brain arm carries receptor_model {lif.get('receptor_model')!r} w_syn_by_nt {lif.get('w_syn_by_nt')!r}")
    if "B" not in mech(arm) and lif.get("receptor_model") == "full":
        problems.append("receptor_model 'full' in force on an arm without B")
    if "C" not in mech(arm) and lif.get("w_syn_by_nt"):
        problems.append("w_syn_by_nt in force on an arm without C")
    ex = prov.get("execution") or {}
    dev = ex.get("device")
    if dev is None or "cpu" in str(dev):
        problems.append(f"device {dev!r} (a CPU run is a smoke, never a replicate)")
    if backend_realised:
        for k in ("cuda_kernels", "cuda_graphs"):
            if k in backend_realised and bool(backend_realised[k]):
                problems.append(f"backend {k} {backend_realised[k]!r} (the family runs the eager Torch path)")
    return problems


def _stamp(path: Path, arm, kind, argv, extra=None):
    """Post-edit a generator's JSON: the arm name, its mechanisms, the flags asked for, the LIFParams / backend in force,
    the generator line of THIS script; fail loudly when the flags did not reach the Brain."""
    j = json.loads(path.read_text(encoding="utf-8"))
    prov = j.get("provenance") or {}
    backend = (prov.get("execution") or {}).get("backend") or {}
    problems = _check_in_force(arm, prov, backend) if prov else ["no provenance block"]
    block = {"arm": arm, "mechanisms": list(mech(arm)), "arm_label": arm_label(arm), "brain_arm": brain_arm(arm), "protocol": kind,
             "sense_spec": sense_of(arm), "lif_overrides_asked": common.to_jsonable(lif_of(arm)), "lif_in_force": common.to_jsonable(_lif_in_force(prov)),
             "backend_policy": BACKEND, "backend_realised": backend, "device": (prov.get("execution") or {}).get("device"),
             "device_name": (prov.get("execution") or {}).get("device_name"), "problems": problems,
             "generator": "scripts/probe_round3_integrate.py " + " ".join(argv), "sources": {
                 "A": "probe_vnc_drive.ARMS_BODY['D'] (docs/audits/body_sided_state.md)", "B": "probe_monoamines.ARMS['add_low'] (docs/audits/monoamine_slow_term.md)",
                 "C": "probe_unitary.BRACKETS['high'] (docs/audits/unitary_strength.md)"}}
    if extra:
        block.update(extra)
    j["integrate"] = block
    j["vnc_arm"] = j.get("arm") if kind in ("room", "compass") else None
    j["arm"] = arm; j["arm_label"] = arm_label(arm)
    if isinstance(j.get("files"), dict):
        j["files"]["generator"] = block["generator"]; j["files"]["generator_inner"] = j["files"].get("generator_inner") or "see integrate.protocol"
    path.write_text(json.dumps(common.to_jsonable(j), indent=1), encoding="utf-8")
    if problems:
        _log(f"INTEGRATE PROBLEMS ({arm}, {kind}): " + "; ".join(problems))
    return problems


# ---------------------------------------------------------------------------------------------- arms (CPU)
def cmd_arms(a) -> int:
    from flyverse import brain
    _log("arm      mechanisms   sense spec                          brain arm  LIFParams overrides")
    for arm in ARMS:
        _log(f"{arm:8s} {','.join(mech(arm)) or '-':12s} {str(sense_of(arm)):36s} {brain_arm(arm):9s} {json.dumps(lif_of(arm), sort_keys=True)}")
    p0 = brain.LIFParams()
    assert lif_of("shipped") == {} and sense_of("shipped") is None
    assert lif_of("A") == {} and sense_of("A") == "all+leg_cycle+haltere_sided"
    assert lif_of("B") == pmo.ARMS["add_low"]["lif"] and brain._slow_spec(brain.LIFParams(**lif_of("B"))) is not None
    assert lif_of("C") == {"w_syn_by_nt": pun.BRACKETS["high"]} and pun.BRACKETS["high"] == {"acetylcholine": 1.0, "gaba": 0.75, "glutamate": 0.75}
    with patched("ABC"):
        p = brain.LIFParams()
    assert p.receptor_model == "full" and p.slow_gain == 0.02 and p.slow_mode == "additive" and p.w_syn_by_nt == pun.BRACKETS["high"], "the patch did not reach LIFParams"
    assert brain.LIFParams() == p0, "the patch was not restored"
    sp = brain._slow_spec(p)
    _log(f"\nshipped LIFParams(): receptor_model {p0.receptor_model}/{p0.receptor_net_rule}, slow_gain {p0.slow_gain} (slow term {'on' if brain._slow_spec(p0) else 'OFF'}), w_syn_by_nt {p0.w_syn_by_nt}")
    _log(f"ABC LIFParams in force: receptor_model {p.receptor_model}/{p.receptor_net_rule}, gain classes {brain._receptor_gain(p)}, slow {sp.mode} {sp.gain} tau {sp.tau} ms, w_syn_by_nt {p.w_syn_by_nt}")
    _log(f"backend policy (every arm, every protocol): {BACKEND}; benchmark flags B: {' '.join(BENCH_B)}")
    return 0


# ---------------------------------------------------------------------------------------------- room / compass / wedge / bench (GPU)
def cmd_room(a) -> int:
    letter = "D" if "A" in mech(a.arm) else "A"
    ns = argparse.Namespace(arm=letter, seed=a.seed, family="body", batch=a.batch, seconds=a.seconds, skip=a.skip, every=a.every, mean_every=a.mean_every,
                            device=a.device, cuda_sparse="torch", out=a.out, block=a.block)
    _log(f"[integrate room {a.arm}] mechanisms {mech(a.arm)} -> probe_vnc_drive room --family body --arm {letter} (spec {sense_of(a.arm)!r}), LIF overrides {lif_of(a.arm)}, backend {BACKEND}")
    with patched(a.arm, ("lif", "batch")):
        rc = pvd.cmd_room(ns)
    problems = _stamp(Path(a.out + ".json"), a.arm, "room", sys.argv[1:], {"block": a.block, "vnc_family_arm": letter})
    return rc or (3 if problems else 0)


def cmd_compass(a) -> int:
    letter = "D" if "A" in mech(a.arm) else "A"
    ns = argparse.Namespace(arm=letter, seed=a.seed, family="body", gains=a.gains, seconds=a.seconds, skip=a.skip, rate=90.0, dna02_hz=a.dna02_hz, sparse="torch",
                            quick=a.quick, device=a.device, cache_dir=None, out=a.out, block=a.block)
    _log(f"[integrate compass {a.arm}] mechanisms {mech(a.arm)} -> probe_vnc_drive compass --family body --arm {letter} (spec {sense_of(a.arm)!r}) gains {a.gains}, LIF overrides {lif_of(a.arm)}, backend {BACKEND}")
    with patched(a.arm, ("lif", "sim")):
        rc = pvd.cmd_compass(ns)
    problems = _stamp(Path(a.out + "_run.json"), a.arm, "compass", sys.argv[1:], {"block": a.block, "vnc_family_arm": letter, "gains": a.gains})
    return rc or (3 if problems else 0)


def cmd_wedge(a) -> int:
    if a.arm not in BRAIN_ARMS:
        raise SystemExit(f"wedge has no body: run it for {BRAIN_ARMS} (arm {a.arm} equals {brain_arm(a.arm)} by construction)")
    pun.BRACKETS[a.arm] = dict(LIF_C["w_syn_by_nt"]) if "C" in mech(a.arm) else {}
    if a.arm not in pun.ARMS:
        pun.ARMS.append(a.arm)
    orig = pun.lif_for

    def lif_for(arm_, **kw):
        from flyverse import brain
        return brain.LIFParams(**kw)                                # the patched maker carries the arm; kw = adapt_by_type off
    pun.lif_for = lif_for
    ns = argparse.Namespace(arm=a.arm, seed=a.seed, out=a.out, seconds=a.seconds, pulse_s=a.pulse_s, pulse_hz=a.pulse_hz, background=a.background,
                            width=4, start_wedge=0, adapt="off", device=a.device, no_graphs=True)
    _log(f"[integrate wedge {a.arm}] mechanisms {mech(a.arm)} -> probe_unitary compass (cx_wedge at the shipped gains), LIF overrides {lif_of(a.arm)}, backend eager")
    try:
        with patched(a.arm, ("lif", "fly")):
            pun.cmd_compass(ns)
    finally:
        pun.lif_for = orig
    problems = _stamp(Path(a.out), a.arm, "wedge", sys.argv[1:], {"block": a.block, "seed": a.seed})
    return 3 if problems else 0


def cmd_bench(a) -> int:
    if a.arm not in BRAIN_ARMS:
        raise SystemExit(f"bench has no body: run it for {BRAIN_ARMS} (arm {a.arm} equals {brain_arm(a.arm)} by construction)")
    import torch
    if not (a.device and str(a.device).startswith("cpu")):
        assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    import benchmark
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    argv = ["benchmark.py", "--eager", "--sections", a.sections, "--seeds", a.seeds, "--json", str(out)] + bench_flags(a.arm) + (["--fast"] if a.fast else [])
    _log(f"[integrate bench {a.arm}] mechanisms {mech(a.arm)} -> benchmark argv {argv[1:]}; LIF overrides {lif_of(a.arm)} (forced on every LIFParams the sections build)")
    saved_argv = sys.argv
    seen = []                                                            # every LIFParams the sections finalised (benchmark.Context._apply_lif)
    orig_apply = benchmark.Context._apply_lif

    def _apply(self, p):
        p = orig_apply(self, p)
        seen.append({k: common.to_jsonable(getattr(p, k, None)) for k in LIF_KEYS_IN_FORCE})
        return p
    try:
        sys.argv = argv
        benchmark.Context._apply_lif = _apply
        with patched(a.arm, ("lif",)):
            benchmark.main()
    finally:
        sys.argv = saved_argv; benchmark.Context._apply_lif = orig_apply
    d = json.loads(out.read_text(encoding="utf-8"))
    d["arm"] = a.arm; d["draw"] = a.draw; d["block"] = a.block
    uniq = [json.loads(s) for s in sorted({json.dumps(x, sort_keys=True) for x in seen})]
    d["integrate"] = {"arm": a.arm, "mechanisms": list(mech(a.arm)), "arm_label": arm_label(a.arm), "protocol": "bench", "sections": a.sections, "bench_flags": bench_flags(a.arm),
                      "lif_overrides_asked": common.to_jsonable(lif_of(a.arm)), "lif_in_force": uniq, "lif_params_built": len(seen),
                      "backend": "eager torch (--eager)", "device": d.get("config", {}).get("device"),
                      "generator": "scripts/probe_round3_integrate.py " + " ".join(sys.argv[1:])}
    cfg = d.get("config", {})
    problems = []
    if "B" in mech(a.arm) and "full" not in json.dumps(cfg):
        problems.append(f"bench config does not show receptor_model full: {cfg.get('receptor_model')!r}")
    if not seen:
        problems.append("no LIFParams passed through benchmark.Context._apply_lif")
    for u in uniq:
        for k, v in lif_of(a.arm).items():
            got = u.get(k); want = common.to_jsonable(v)
            if k == "receptor_gain":
                got = {kk: float(vv) for kk, vv in (got or {}).items()}; want = {kk: float(vv) for kk, vv in want.items()}
            if got != want:
                problems.append(f"{k}: in force {got!r} != asked {want!r}")
        if not lif_of(a.arm) and (u.get("receptor_model") != "sign" or u.get("w_syn_by_nt")):
            problems.append(f"shipped-brain arm carries receptor_model {u.get('receptor_model')!r} w_syn_by_nt {u.get('w_syn_by_nt')!r}")
    d["integrate"]["problems"] = problems
    out.write_text(json.dumps(d, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    _log(f"[integrate bench {a.arm} draw {a.draw}] device {cfg.get('device')}; checks " + "; ".join(f"{c_['key']} {c_['measured']} {c_['status']}" for c_ in d["checks"]))
    return 3 if problems else 0


# ---------------------------------------------------------------------------------------------- plan (CPU)
def cmd_plan(a) -> int:
    d = a.dir.rstrip("/")
    pre = f"mkdir -p {d} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "

    def job(cmd, log):
        return f"{cmd} > {log} 2>&1; st=$?; tail -4 {log}; exit $st"
    cmds = []; plan = {"room": [], "compass": [], "wedge": [], "bench": []}
    for s in range(a.runs):
        for arm in ARMS:
            stem = f"{d}/room_{arm}_r{s}"
            cmds.append(pre + job(f"python scripts/probe_round3_integrate.py room --arm {arm} --seed {s} --batch 16 --seconds 60 --block fam_r{s} --out {stem}", f"{stem}.txt"))
            plan["room"].append({"arm": arm, "seed": s, "stem": stem, "block": f"fam_r{s}"})
    for s in range(a.compass_seeds):                                    # one job per seed: the 8 arms sequential, each its own python process
        parts = []; sts = []
        for j, arm in enumerate(ARMS):
            stem = f"{d}/compass_{arm}_r{s}"
            parts.append(f"python scripts/probe_round3_integrate.py compass --arm {arm} --seed {s} --block fam_c{s} --out {stem} > {stem}.txt 2>&1; s{j}=$?; tail -3 {stem}.txt")
            sts.append(f"s{j}"); plan["compass"].append({"arm": arm, "seed": s, "stem": stem, "block": f"fam_c{s}"})
        cmds.append(pre + "; ".join(parts) + f"; exit $(({' | '.join(sts)}))")
    for barm in BRAIN_ARMS:                                             # one job per brain arm: the seeds sequential
        parts = []; sts = []
        for k in range(a.wedge_seeds):
            stem = f"{d}/wedge_{barm}_s{k}"
            parts.append(f"python scripts/probe_round3_integrate.py wedge --arm {barm} --seed {k} --block fam_wedge --out {stem}.json > {stem}.txt 2>&1; s{k}=$?; tail -3 {stem}.txt")
            sts.append(f"s{k}"); plan["wedge"].append({"arm": barm, "seed": k, "stem": stem, "block": "fam_wedge"})
        cmds.append(pre + "; ".join(parts) + f"; exit $(({' | '.join(sts)}))")
    for barm in BRAIN_ARMS:
        parts = []; sts = []
        for k in range(a.draws):
            stem = f"{d}/bench_{barm}_d{k}"
            parts.append(f"python scripts/probe_round3_integrate.py bench --arm {barm} --draw {k} --block fam_bench --out {stem}.json > {stem}.txt 2>&1; s{k}=$?; tail -6 {stem}.txt")
            sts.append(f"s{k}"); plan["bench"].append({"arm": barm, "draw": k, "stem": stem, "block": "fam_bench"})
        cmds.append(pre + "; ".join(parts) + f"; exit $(({' | '.join(sts)}))")

    def quoted(c_):
        return '"' + c_.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'
    targets = f" --targets {a.targets}" if a.targets else ""
    line = (f"python scripts/cluster_run.py --name {a.name} --minutes {a.minutes} --arm-block fam{targets} " + " ".join(quoted(c_) for c_ in cmds)
            + f" --fetch {d}/ 2>&1 | tee out/{a.name}_cluster.log")
    Path(d).mkdir(parents=True, exist_ok=True)
    Path(d, a.script).write_text("#!/bin/bash\n# ONE submission: " + f"{len(cmds)} jobs = {len(ARMS) * a.runs} room (8 arms x {a.runs} seeds; blocks fam_r<seed>) + {a.compass_seeds} compass jobs "
                                 f"(8 arms sequential per seed; blocks fam_c<seed>) + {len(BRAIN_ARMS)} wedge jobs ({a.wedge_seeds} seeds sequential; fam_wedge) + "
                                 f"{len(BRAIN_ARMS)} bench jobs ({a.draws} draws sequential; fam_bench); targets {a.targets or 'default (house)'}\n" + line + "\n", encoding="utf-8", newline="\n")
    pre_decl = {"schema": "flyverse.round3_integration.predeclared/1", "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "name": a.name, "dir": d,
                "arms": {arm: {"mechanisms": list(mech(arm)), "sense_spec": sense_of(arm), "lif_overrides": common.to_jsonable(lif_of(arm)), "brain_arm": brain_arm(arm), "label": arm_label(arm)} for arm in ARMS},
                "sources": {"A": "docs/audits/body_sided_state.md (probe_vnc_drive.ARMS_BODY['D'])", "B": "docs/audits/monoamine_slow_term.md (probe_monoamines.ARMS['add_low'])",
                            "C": "docs/audits/unitary_strength.md (probe_unitary.BRACKETS['high']) -- labelled NOT suite-safe: no bracket keeps walk.power_sustained_hz"},
                "backend": BACKEND, "protocols": {"room": {"batch": 16, "seconds": 60, "window_s": [5, 60], "runs_per_arm": a.runs, "replicate_unit": "job"},
                                                  "compass": {"gains": "2:15", "phases": "rest/ccw/rest2/cw 10 s, 3 s skipped", "dna02_hz": 20, "seeds": a.compass_seeds},
                                                  "wedge": {"protocol": "cx_wedge at gE=gD=gR=1, adapt off, 10 Hz background, +40 Hz x 2 s on 4 wedges, 5 s free", "seeds": a.wedge_seeds, "arms": list(BRAIN_ARMS)},
                                                  "bench": {"sections": BENCH_SECTIONS, "seeds": "0,1,2", "draws": a.draws, "arms": list(BRAIN_ARMS), "eager": True}},
                "primary_keys": {"room": ["yaw_sd_clean_deg_s", "straightness", "DNa02_L_hz", "DNa02_R_hz", "DNa02_LR_hz", "leg_LR_hz", "leg_LR_neg_flies", "hops", "n_left_table", "speed_cmd_mean_m_s", "airborne_frac"],
                                 "compass": ["drift ccw/cw vs rests", "flip AN04B003 / PS196_b / GLNO / PEN / EPG"], "wedge": ["compass.EPG.bump_survival_s", "bump_rate_hz", "bump_width_wedges", "PEN_mean_post"],
                                 "bench": ["every check's status vs shipped; walk.power_sustained_hz, taste.MN9_hz, loom_escape.GF_peak_hz"]},
                "comparisons": [list(p) for p in PAIRS], "reading_rules": "common.compare over runs (result needs >= 4 runs per arm, |z| >= 3, p <= 0.05; 5 v 5 floor p 0.0079); zero-SD null -> undetermined; "
                                                                           "compass 3 v 6 per phase -> underpowered by rule, per-seed lists carried; wedge 4 v 4; bench 2 draws -> magnitudes and statuses only. "
                                                                           "Every room / compass / wedge JSON is stamped with the LIFParams in force and fails when they differ from the arm's. GPU rows are B200 draws; nothing is a bit check.",
                "jobs": len(cmds), "plan": plan, "generator": "scripts/probe_round3_integrate.py plan " + " ".join(sys.argv[2:])}
    Path(d, "predeclared.json").write_text(json.dumps(pre_decl, indent=1), encoding="utf-8")
    _log(f"wrote {d}/{a.script} ({len(cmds)} jobs, name {a.name}, targets {a.targets or 'default'}) and {d}/predeclared.json (stamped {pre_decl['stamped_utc']})")
    return 0


# ---------------------------------------------------------------------------------------------- verify (CPU)
def _load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cmd_verify(a) -> int:
    d = a.dir.rstrip("/")
    plan = _load_json(Path(d, "predeclared.json"))["plan"]
    problems = []; rows = []
    for kind, items in plan.items():
        for it in items:
            stem = it["stem"]
            jp = Path(stem + (".json" if kind in ("room",) else "_run.json" if kind == "compass" else ".json"))
            if kind in ("wedge", "bench"):
                jp = Path(stem + ".json")
            txt = Path(stem + ".txt")
            row = {"kind": kind, "arm": it["arm"], "seed": it.get("seed", it.get("draw")), "json": jp.exists(), "console": txt.exists()}
            if not jp.exists():
                problems.append(f"missing {jp}"); rows.append(row); continue
            j = _load_json(jp)
            ib = j.get("integrate") or {}
            prov = j.get("provenance") or {}
            ex = prov.get("execution") or {}
            row.update(arm_json=j.get("arm"), device=ex.get("device") or j.get("device") or (j.get("config") or {}).get("device"), device_name=ex.get("device_name") or (j.get("config") or {}).get("device"),
                       block=j.get("block") or ib.get("block"), mechanisms="".join(ib.get("mechanisms", [])) or "-", in_force=json.dumps(ib.get("lif_in_force", {}), sort_keys=True)[:160],
                       backend=json.dumps(ib.get("backend_realised", {}), sort_keys=True), stamp_problems="; ".join(ib.get("problems", [])))
            if j.get("arm") != it["arm"]:
                problems.append(f"{jp.name}: arm {j.get('arm')!r} != planned {it['arm']!r}")
            if ib.get("problems"):
                problems.append(f"{jp.name}: " + "; ".join(ib["problems"]))
            if kind == "room":
                for suf in ("_body.npz", "_rec.npz", "_flies.npz", "_max.npz"):
                    if not Path(stem + suf).exists():
                        problems.append(f"missing {stem + suf}")
                spec = j.get("proprioception")
                if spec != sense_of(it["arm"]):
                    problems.append(f"{jp.name}: sense spec {spec!r} != {sense_of(it['arm'])!r}")
                if str(row["device"]) != "cuda" and "cuda" not in str(row["device"]):
                    problems.append(f"{jp.name}: device {row['device']}")
            if kind == "compass":
                for ph in ("rest", "ccw", "rest2", "cw"):
                    if not Path(f"{stem}_{ph}.npz").exists():
                        problems.append(f"missing {stem}_{ph}.npz")
                if j.get("proprioception") != sense_of(it["arm"]):
                    problems.append(f"{jp.name}: sense spec {j.get('proprioception')!r} != {sense_of(it['arm'])!r}")
            if kind == "bench":
                if "eager" not in str((j.get("config") or {}).get("backend", "")):
                    problems.append(f"{jp.name}: backend {(j.get('config') or {}).get('backend')!r}")
                if "B" in mech(it["arm"]) and "full" not in json.dumps(j.get("config", {})):
                    problems.append(f"{jp.name}: no 'full' in the bench config")
            rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(a.out or Path(d, "verify"))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out, "verify_runs.csv"), index=False)
    counts = df.groupby(["kind", "arm"]).agg(n=("json", "sum"), devices=("device_name", lambda s: ",".join(sorted(set(map(str, s)))))).reset_index()
    common.print_table(counts, max_rows=60)
    _log(f"\n{len(df)} planned runs, {int(df.json.sum())} JSONs present, devices {sorted(set(map(str, df.get('device_name', pd.Series()).dropna())))}")
    for p in problems:
        _log("  PROBLEM " + p)
    Path(out, "verify.json").write_text(json.dumps({"problems": problems, "n_planned": int(len(df)), "n_present": int(df.json.sum()),
                                                   "counts": common.to_jsonable(counts.to_dict("records")), "generator": "scripts/probe_round3_integrate.py verify " + " ".join(sys.argv[2:])}, indent=1), encoding="utf-8")
    _log(f"problems: {len(problems)} -> {out}/verify.json")
    return 1 if problems else 0


# ---------------------------------------------------------------------------------------------- analyse (CPU)
def load_rooms(d):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "room_*_r*.json"))):
        m = re.fullmatch(r"room_([A-Za-z]+)_r(\d+)\.json", Path(p).name)
        if not m:
            continue
        j = _load_json(p)
        body = Path(p[:-5] + "_body.npz")
        if body.exists():
            j["run"].update(pvd.robust_room(body, skip_f=int(round(j.get("skip_s", 5.0) * 100))))
        runs.setdefault(j["arm"], []).append((p, j))
    return runs


def _cmp_row(vals, t, r, row, tag=None):
    tag = tag or f"{t}v{r}"
    if len(vals.get(t, [])) and len(vals.get(r, [])):
        cmp_ = common.compare(vals[t], vals[r])
        row.update({f"{tag}_diff": cmp_["diff"], f"{tag}_z": cmp_["z"], f"{tag}_p": cmp_["p"], f"{tag}_verdict": cmp_["verdict"]})


def room_analysis(runs, out, c):
    arms = [a for a in ARMS if a in runs]
    # per-frame sidedness (probe_vnc_drive.sided_frames; needs the _rec / _body npz)
    for arm in arms:
        for p, j in runs[arm]:
            try:
                j["run"].update(pvd.sided_frames(p, c))
            except Exception as e:  # noqa: BLE001
                _log(f"  sided_frames failed on {Path(p).name}: {e}")
    keys = list(pvd.ROOM_KEYS) + [("speed_cmd_mean_m_s", "commanded walking speed (m/s, window mean)"), ("yaw_sd_all_deg_s", "yaw-rate SD over ALL frames (deg/s; carries the edge / hop artefact)")] \
        + list(pvd.ROBUST_KEYS) + list(pvd.CYCLE_KEYS) + list(pvd.SIDED_KEYS)
    j0 = runs[arms[0]][0][1]
    for ch in j0["channels"]:
        keys += [(f"commanded_{ch}_hz", f"channel {ch} commanded Hz"), (f"measured_{ch}_hz", f"channel {ch} measured Hz")]
    for ch in ("chordotonal", "hair_plate", "campaniform", "haltere"):
        keys += [(f"commanded_{ch}_LR_hz", f"channel {ch} commanded L-R (Hz)"), (f"commanded_{ch}_absLR_hz", f"channel {ch} commanded |L-R| per frame (Hz)")]
    for w in sorted({w for items in runs.values() for _, j in items for w in j["watch"]}):
        keys.append((f"{w}_hz", f"{w} (Hz)"))
    keys = [(k, l) for k, l in keys if any(j["run"].get(k) is not None for items in runs.values() for _, j in items)]
    rows = []
    for key, label in keys:
        row = {"key": key, "label": label}
        vals = {}
        for arm in arms:
            v = np.array([j["run"][key] for _, j in runs[arm] if j["run"].get(key) is not None], float)
            vals[arm] = v
            row[f"{arm}_mean"] = float(v.mean()) if len(v) else np.nan; row[f"{arm}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            row[f"{arm}_n"] = int(len(v)); row[f"{arm}_runs"] = [round(float(x), 4) for x in v]
        for arm in arms:
            if arm != "shipped":
                _cmp_row(vals, arm, "shipped", row, tag=f"{arm}_vs_shipped")
        for t, r in PAIRS:
            _cmp_row(vals, t, r, row)
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(Path(out, "room_table.csv"), index=False)
    lines = [f"{'key':34s} " + " | ".join(f"{a:^19s}" for a in arms)]
    for _, r in df.iterrows():
        cells = [f"{r[f'{a}_mean']:8.3f}+-{r[f'{a}_sd']:7.3f}" if np.isfinite(r[f"{a}_mean"]) else " " * 17 for a in arms]
        lines.append(f"{r['key']:34s} " + " | ".join(cells))
        vs = "   ".join(f"{a}:{r.get(f'{a}_vs_shipped_diff', np.nan):+.3f} z{r.get(f'{a}_vs_shipped_z', np.nan):+.1f} p{r.get(f'{a}_vs_shipped_p', np.nan):.3f} {r.get(f'{a}_vs_shipped_verdict', '')}" for a in arms if a != "shipped")
        lines.append(" " * 36 + "vs shipped: " + vs)
    text = "\n".join(lines); Path(out, "room_table.txt").write_text(text, encoding="utf-8"); _log(text)
    # the pairwise family on the headline keys
    show = ["yaw_sd_clean_deg_s", "yaw_median_abs_clean_deg_s", "yaw_signed_mean_deg_s", "straightness", "net_turns", "n_left_table", "hops", "airborne_frac", "speed_cmd_mean_m_s",
            "DNa02_L_hz", "DNa02_R_hz", "DNa02_LR_hz", "DNa02_abs_LR_hz", "DNa02_active_frac", "DNa02_LR_pos_flies", "leg_L_hz", "leg_R_hz", "leg_LR_hz", "leg_LR_neg_flies",
            "haltere_LR_hz", "power_sustained_hz", "gf_max_hz", "commanded_chordotonal_hz", "AN04B003_hz", "PS059_hz", "PS196_b_hz", "GLNO_hz", "corr_dna02LR_chordLR", "an04LR_given_chordLR_pos_minus_neg"]
    _log("\n== pairwise verdicts (common.compare; runs per arm as listed):")
    plines = []
    for _, r in df[df.key.isin(show)].iterrows():
        cells = [f"{t}v{ref} {r[f'{t}v{ref}_diff']:+.3f} z{r[f'{t}v{ref}_z']:+.1f} p{r[f'{t}v{ref}_p']:.3f} {r[f'{t}v{ref}_verdict']}" for t, ref in PAIRS if isinstance(r.get(f"{t}v{ref}_verdict"), str)]
        plines.append(f"  {r.key:34s} " + " | ".join(cells))
    _log("\n".join(plines)); Path(out, "pairwise.txt").write_text("\n".join(plines), encoding="utf-8")
    return df


def compass_analysis(runs, out, c):
    df, flips, chain = pvd.analyse_compass(c, runs, out)
    fl = []
    for arm in ARMS:
        f = Path(out, f"compass_flip_{arm}.csv")
        if f.exists():
            ft = pd.read_csv(f); ft = ft[ft.type.isin(pvd.CHAIN)].copy(); ft.insert(0, "arm", arm); fl.append(ft)
    if fl:
        fdf = pd.concat(fl); fdf.to_csv(Path(out, "compass_flip_chain.csv"), index=False)
        _log("\n== compass flip table, chain types, every arm (L-R at ccw minus cw against the rest2 - rest null; per-seed values in compass_flip_chain.csv)")
        common.print_table(fdf[fdf.type.isin(["AN04B003", "AN06A026", "PS047_b", "PS196_b", "GLNO", "PEN_a(PEN1)", "PEN_b(PEN2)", "EPG", "PS059", "DNa02"])]
                           [["arm", "type", "n_L", "n_R", "LR_rest", "LR_ccw", "LR_rest2", "LR_cw", "flip_mean", "flip_sd", "null_mean", "null_sd", "p", "verdict"]], max_rows=120)
    return df


def wedge_analysis(d, out):
    files = sorted(glob.glob(os.path.join(d, "wedge_*_s*.json")))
    if not files:
        return None
    by = {}
    for f in files:
        j = _load_json(f); by.setdefault(j["arm"], []).append((f, j))
    rows = []
    for arm in BRAIN_ARMS:
        if arm not in by:
            continue
        items = by[arm]
        row = {"arm": arm, "n": len(items), "devices": ";".join(sorted({str(j.get("provenance", {}).get("execution", {}).get("device_name")) for _, j in items})),
               "ledger_survival": ",".join(j["ledger"]["compass.EPG.bump_survival_s"]["status"] for _, j in items),
               "ledger_rate": ",".join(j["ledger"]["compass.EPG.bump_rate_hz"]["status"] for _, j in items),
               "ledger_width": ",".join(j["ledger"]["compass.EPG.bump_width_wedges"]["status"] for _, j in items)}
        for k in WEDGE_KEYS:
            v = np.array([j["metrics"].get(k, np.nan) for _, j in items], float)
            row[f"{k}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan; row[f"{k}_runs"] = [float(x) for x in v]
            if arm != "shipped" and "shipped" in by:
                nv = np.array([j["metrics"].get(k, np.nan) for _, j in by["shipped"]], float)
                cmp_ = common.compare(v[np.isfinite(v)], nv[np.isfinite(nv)]) if np.isfinite(v).any() and np.isfinite(nv).any() else None
                if cmp_:
                    row[f"{k}_vs_shipped"] = f"{cmp_['diff']:+.4f} z{cmp_['z']:+.2f} p{cmp_['p']:.3f} {cmp_['verdict']}"
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(Path(out, "wedge_table.csv"), index=False)
    _log("\n== cx_wedge compass at the shipped gains (probe_unitary compass protocol; runs = jobs; unrounded values compared)")
    cols = ["arm", "n", "devices", "ledger_survival", "ledger_rate", "ledger_width"] + [f"{k}_mean" for k in ("survival_s", "frac_confined_post", "epg_in_mean_post", "epg_out_mean_post", "PEN_mean_post", "Delta7_mean_post", "Ring_mean_post", "rest_mean_post", "GLNO_mean_post")]
    common.print_table(df[cols], max_rows=8)
    for _, r in df.iterrows():
        vs = [f"{k}: {r[f'{k}_vs_shipped']}" for k in ("survival_s", "frac_confined_post", "epg_in_mean_post", "PEN_mean_post", "Delta7_mean_post", "Ring_mean_post", "rest_mean_post", "GLNO_mean_post") if isinstance(r.get(f"{k}_vs_shipped"), str)]
        if vs:
            _log(f"  {r.arm} vs shipped: " + " | ".join(vs))
        _log(f"  {r.arm} per-seed survival {r['survival_s_runs']} PEN post {[round(x, 3) for x in r['PEN_mean_post_runs']]} Delta7 post {[round(x, 2) for x in r['Delta7_mean_post_runs']]} EPG in post {[round(x, 2) for x in r['epg_in_mean_post_runs']]}")
    return df


def bench_analysis(d, out):
    files = sorted(glob.glob(os.path.join(d, "bench_*_d*.json")))
    if not files:
        return None
    rows = []
    for f in files:
        j = _load_json(f)
        for ch in j["checks"]:
            rows.append({"arm": j["arm"], "draw": j.get("draw"), "key": ch["key"], "measured": ch.get("measured"), "status": ch.get("status"), "criterion": ch.get("criterion") or ch.get("op"),
                         "bound": ch.get("bound") or ch.get("threshold"), "device": (j.get("config") or {}).get("device"), "backend": (j.get("config") or {}).get("backend"), "file": Path(f).name})
    df = pd.DataFrame(rows); df.to_csv(Path(out, "bench_checks.csv"), index=False)
    piv = df.pivot_table(index="key", columns=["arm", "draw"], values="measured", aggfunc="first")
    st = df.pivot_table(index="key", columns=["arm", "draw"], values="status", aggfunc="first")
    tally = df.groupby(["arm", "draw"]).status.value_counts().unstack(fill_value=0).reset_index()
    _log("\n== benchmark sections " + BENCH_SECTIONS + " (--eager; 2 draws per brain arm = magnitudes and statuses, not verdicts)")
    common.print_table(tally, max_rows=12)
    lines = []
    for key in piv.index:
        cells = []
        for arm in BRAIN_ARMS:
            if arm in piv.columns.get_level_values(0):
                vals = piv.loc[key, arm]; sts = st.loc[key, arm]
                cells.append(f"{arm}: " + " / ".join(f"{v:.3f}{('' if s == 'PASS' else ' ' + str(s))}" if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else f"{v} {s}" for v, s in zip(vals.values, sts.values)))
        lines.append(f"  {key:28s} " + " | ".join(cells))
    _log("\n".join(lines)); Path(out, "bench_table.txt").write_text("\n".join(lines), encoding="utf-8")
    worse = []
    if "shipped" in st.columns.get_level_values(0):
        rank = {"PASS": 0, "KNOWN GAP": 1, "KNOWN_GAP": 1, "FAIL": 2, "MISSING": 3}
        for key in st.index:
            base = max(rank.get(str(s), 0) for s in st.loc[key, "shipped"].values)
            for arm in BRAIN_ARMS:
                if arm != "shipped" and arm in st.columns.get_level_values(0):
                    for dr, s in st.loc[key, arm].items():
                        if rank.get(str(s), 0) > base:
                            worse.append({"arm": arm, "draw": dr, "key": key, "status": s, "shipped_worst": base})
    _log(f"  checks worse in status than the shipped arm's worst draw: {worse if worse else 'none'}")
    pd.DataFrame(worse).to_csv(Path(out, "bench_worse.csv"), index=False)
    return df


def dna02_decompose(runs, out, c, arms):
    from flyverse.interp import decompose as dc, trace as tr
    if "shipped" not in runs:
        return None
    null = {"shipped": pvd.load_flies(runs, "shipped")}
    rec = {a: pvd.load_flies(runs, a) for a in arms if a in runs}
    lp_of = {a: tr.params_from_provenance(runs[a][0][1]["provenance"]) for a in list(rec) + ["shipped"]}
    lp0 = lp_of["shipped"]
    ew0 = common.effective_weights(c, lp0)
    results = {}
    for tgt, spec in (("DNa02_L", "type=DNa02&somaSide=L"), ("DNa02_R", "type=DNa02&somaSide=R")):
        t0 = time.time()
        res = dc.decompose(c, spec, recording=rec, null_recording=null, params=lp0, by=("type", "side"), tiers=False, top=30, ew=ew0, keep_links=False,
                           arm_weights_override=lp_of)                # each arm decomposed with ITS OWN shaped weights (C: w_syn_by_nt), the null with the shipped ones
        res.files["generator"] = "scripts/probe_round3_integrate.py analyse (decompose)"
        path = Path(out, f"decompose_{tgt}.json"); res.save(path)
        pt = res.table("per_type"); pt.to_csv(Path(out, f"decompose_{tgt}_per_type.csv"), index=False)
        results[tgt] = pt
        _log(f"\n== decompose {tgt} by (type, side), arms {list(rec)} vs null shipped, each arm on its own weights ({time.time() - t0:.0f} s) -> {path}; check {res.check()}")
        for arm_, v in (res.summary.get("dynamic", {}) or {}).items():
            for t_, d_ in v.items():
                cp = d_.get("cancelling_pair", {})
                _log(f"  arm {arm_} {t_}: E {d_['E_total']:+.1f} I {d_['I_total']:+.1f} net {d_['net']:+.1f} mV/s per cell; cancelling pair {cp.get('E')} {cp.get('E_value', 0):+.1f} vs {cp.get('I')} {cp.get('I_value', 0):+.1f}")
    dd = pvd.decompose_summary(out, "DNa02", ["shipped"] + list(rec))
    if len(dd):
        dd.to_csv(Path(out, "dna02_decompose_summary.csv"), index=False)
        cols = ["post", "arm", "rate_hz", "E_total", "I_total", "net", "PS059/L", "PS059/R", "AN04B003/L", "AN04B003/R", "IN12B014/L", "IN12B014/R", "IN19A003/L", "IN19A003/R", "GNG562/L", "GNG562/R", "LT51/L", "LT51/R"]
        _log("\n== DNa02 rate-weighted input per arm (mV/s per post cell) -> dna02_decompose_summary.csv")
        common.print_table(dd[[c_ for c_ in cols if c_ in dd]], max_rows=20)
        for _, r in dd.iterrows():
            _log(f"  {r.post} {r.arm}: top E {r.top_E}\n      top I {r.top_I}")
    return dd


def cmd_analyse(a) -> int:
    from flyverse import connectome
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = a.dir.rstrip("/")
    runs = load_rooms(d)
    _log(f"room runs per arm: {[(k, len(v)) for k, v in runs.items()]}  ({d}/room_<arm>_r<seed>.json)")
    for arm, items in runs.items():
        devs = sorted({str(j.get("device_name")) for _, j in items}); specs = sorted({str(j.get("proprioception")) for _, j in items})
        inf = sorted({json.dumps((j.get("integrate") or {}).get("lif_in_force"), sort_keys=True) for _, j in items})
        be = sorted({json.dumps((j.get("integrate") or {}).get("backend_realised"), sort_keys=True) for _, j in items})
        _log(f"  arm {arm:8s} n {len(items)} device {devs} spec {specs} blocks {sorted({str(j.get('block')) for _, j in items})}\n    lif in force {inf}\n    backend {be}")
        if any("cpu" in str(j.get("device")) for _, j in items):
            _log(f"  WARNING: arm {arm} has a cpu run")
    summary = {"generator": "scripts/probe_round3_integrate.py analyse " + " ".join(sys.argv[2:]), "room_runs": {k: [p for p, _ in v] for k, v in runs.items()}}
    c = connectome.load(verbose=False)
    if runs:
        df = room_analysis(runs, out, c); summary["room_table"] = common.to_jsonable(df.to_dict("records"))
    cr = pvd.load_compass(d)
    if cr:
        _log(f"\ncompass runs per arm: {[(k, len(v)) for k, v in cr.items()]}")
        for arm, items in cr.items():
            _log(f"  arm {arm:8s} device {sorted({str(j.get('device_name')) for _, j in items})} spec {sorted({str(j.get('proprioception')) for _, j in items})} "
                 f"lif in force {sorted({json.dumps((j.get('integrate') or {}).get('lif_in_force'), sort_keys=True) for _, j in items})}")
        cdf = compass_analysis(cr, out, c); summary["compass_table"] = common.to_jsonable(cdf.to_dict("records"))
    wd = wedge_analysis(d, out)
    if wd is not None:
        summary["wedge_table"] = common.to_jsonable(wd.to_dict("records"))
    bd = bench_analysis(d, out)
    if bd is not None:
        summary["bench_checks"] = common.to_jsonable(bd.to_dict("records"))
    if runs and not a.skip_decompose:
        arms = [x.strip() for x in a.decompose_arms.split(",") if x.strip()]
        dd = dna02_decompose(runs, out, c, arms)
        if dd is not None and len(dd):
            summary["dna02_decompose"] = common.to_jsonable(dd.to_dict("records"))
    summary["wall_s"] = round(time.time() - t0, 1)
    Path(out, "summary.json").write_text(json.dumps(common.to_jsonable(summary), indent=1), encoding="utf-8")
    _log(f"\nwrote {out}/summary.json ({time.time() - t0:.0f} s)")
    return 0


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("arms", help="print and check the resolved arms (CPU)")
    r = sub.add_parser("room", help="one plain-fly room run under one arm (GPU)")
    r.add_argument("--arm", required=True, choices=list(ARMS)); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--batch", type=int, default=16); r.add_argument("--seconds", type=float, default=60.0); r.add_argument("--skip", type=float, default=5.0)
    r.add_argument("--every", type=int, default=2); r.add_argument("--mean-every", type=int, default=5)
    r.add_argument("--device", default=None); r.add_argument("--block", default=None); r.add_argument("--out", required=True)
    k = sub.add_parser("compass", help="the efferent rotation arm (gE 2 / gD 15) under one arm (GPU)")
    k.add_argument("--arm", required=True, choices=list(ARMS)); k.add_argument("--seed", type=int, default=0); k.add_argument("--gains", default="2:15")
    k.add_argument("--seconds", type=float, default=10.0); k.add_argument("--skip", type=float, default=3.0); k.add_argument("--dna02-hz", type=float, default=20.0)
    k.add_argument("--quick", action="store_true"); k.add_argument("--device", default=None); k.add_argument("--block", default=None); k.add_argument("--out", required=True)
    w = sub.add_parser("wedge", help="cx_wedge at the shipped gains under one brain arm (GPU)")
    w.add_argument("--arm", required=True, choices=list(BRAIN_ARMS)); w.add_argument("--seed", type=int, default=0)
    w.add_argument("--seconds", type=float, default=5.0); w.add_argument("--pulse-s", type=float, default=2.0); w.add_argument("--pulse-hz", type=float, default=40.0)
    w.add_argument("--background", type=float, default=10.0); w.add_argument("--device", default=None); w.add_argument("--block", default=None); w.add_argument("--out", required=True)
    b = sub.add_parser("bench", help="scripts/benchmark.py sections under one brain arm (GPU)")
    b.add_argument("--arm", required=True, choices=list(BRAIN_ARMS)); b.add_argument("--draw", type=int, default=0); b.add_argument("--sections", default=BENCH_SECTIONS)
    b.add_argument("--seeds", default="0,1,2"); b.add_argument("--fast", action="store_true"); b.add_argument("--device", default=None); b.add_argument("--block", default=None); b.add_argument("--out", required=True)
    p = sub.add_parser("plan", help="write the ONE cluster submission and predeclared.json (CPU)")
    p.add_argument("--dir", default="out/r3int"); p.add_argument("--runs", type=int, default=5); p.add_argument("--compass-seeds", type=int, default=3)
    p.add_argument("--wedge-seeds", type=int, default=4); p.add_argument("--draws", type=int, default=2); p.add_argument("--minutes", type=int, default=150)
    p.add_argument("--name", default="r3int"); p.add_argument("--targets", default=None); p.add_argument("--script", default="batch.sh")
    v = sub.add_parser("verify", help="count the artefacts against the plan and check every run's device / flags (CPU)")
    v.add_argument("--dir", default="out/r3int"); v.add_argument("--out", default=None)
    an = sub.add_parser("analyse", help="the tables (CPU)")
    an.add_argument("--dir", default="out/r3int"); an.add_argument("--out", default="out/r3int/analysis")
    an.add_argument("--decompose-arms", default="A,B,C,AB,AC,BC,ABC"); an.add_argument("--skip-decompose", action="store_true")
    a = ap.parse_args(argv)
    return {"arms": cmd_arms, "room": cmd_room, "compass": cmd_compass, "wedge": cmd_wedge, "bench": cmd_bench, "plan": cmd_plan, "verify": cmd_verify, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
