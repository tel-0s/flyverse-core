"""Round 8, item 3, the room rate-half: batch_sustain.py's take-off protocol under `raw` and under the three-instrument
list (docs/audits/instrumented_suite.md section 3; batch suite-inst-room). A wrapper in the sense of
scripts/guard_suites.sh's embedded guard_wrap.py: batch_sustain.py is run as it is (its own argv, nothing in it edited)
with its BatchSim rebound to a subclass that, for the instrumented arm, supplies the GLNO=glutamate scratch cache
(cx_wedge.load_connectome), `preset="instrumented"` and fresh instrument objects (cx_wedge.build_instruments: the
afferent spec, the `ring_dc_hold` record, the `glno_sign` record), and with flyverse.brain.LIFParams rebound to a
subclass that appends the hold to `type_path_gain` (the record verifies it at attach). The JSON then gets a `room`
block (what was resolved, problems -> exit 3) and `provenance`.

    python scripts/instrumented_room.py --arm raw|instrumented --json OUT.json -- <batch_sustain.py arguments>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

HOLD = ("^(ExR6|ER6|ER4m)$", "^(PEN_|EPG$)", 0.0)
INSTRUMENT_SPECS = ["sided_turn_afferent:k=0.5"]
NT_OVERRIDE = {"GLNO": "glutamate"}
RECORD_NAMES = ["sided_turn_afferent", "ring_dc_hold", "glno_sign"]
CAPTURED: list = []


def parse(argv):
    if "--" not in argv:
        raise SystemExit("usage: instrumented_room.py --arm raw|instrumented --json OUT -- <batch_sustain.py args>")
    i = argv.index("--")
    head, rest = argv[:i], argv[i + 1:]
    arm = head[head.index("--arm") + 1]
    out = head[head.index("--json") + 1]
    if arm not in ("raw", "instrumented"):
        raise SystemExit(f"--arm must be raw or instrumented, got {arm!r}")
    return arm, out, rest


def rebind(arm, c):
    """batch_sustain's BatchSim and flyverse.brain's LIFParams, rebound for the instrumented arm; untouched for raw."""
    import batch_sustain
    from flyverse import brain
    from flyverse.batch_sim import BatchSim as Base
    import cx_wedge

    class RoomBatchSim(Base):
        def __init__(self, *a, **kw):
            if arm == "instrumented":
                kw["c"] = c
                kw["preset"] = "instrumented"
                kw["instruments"] = cx_wedge.build_instruments(c, INSTRUMENT_SPECS, "instrumented", hold_edges=[HOLD], nt_override=NT_OVERRIDE)
            super().__init__(*a, **kw)
            CAPTURED.append(self)

    RoomBatchSim.__name__ = RoomBatchSim.__qualname__ = "BatchSim"
    batch_sustain.BatchSim = RoomBatchSim
    if arm == "instrumented":
        Base_LIF = brain.LIFParams

        class HeldLIFParams(Base_LIF):
            def __init__(self, **kw):
                super().__init__(**kw)
                base = list(brain.DEFAULT_TYPE_PATH_GAIN if self.type_path_gain is None else self.type_path_gain)
                if HOLD not in base:
                    self.type_path_gain = base + [HOLD]

        HeldLIFParams.__name__ = HeldLIFParams.__qualname__ = "LIFParams"
        brain.LIFParams = HeldLIFParams


def finish(arm, out, rest, t0, cache_dir):
    import torch
    from flyverse import brain
    from flyverse.interp import common
    path = Path(out)
    d = json.loads(path.read_text(encoding="utf-8"))
    if not CAPTURED:
        raise SystemExit("instrumented_room: no BatchSim was built (exit 3)")
    sim = CAPTURED[-1]
    lif = sim.fb.brain.p
    tpg = [list(t) for t in (brain.DEFAULT_TYPE_PATH_GAIN if lif.type_path_gain is None else lif.type_path_gain)]
    names = [getattr(i, "name", None) for i in sim.fb.instruments.values()]
    nts = sorted(set(sim.c.neurons.nt.to_numpy()[sim.c.neurons.type.fillna("").to_numpy().astype(str) == "GLNO"].tolist()))
    dev = str(sim.fb.device)
    resolved = {"preset": sim.fb.preset, "instruments": names, "hold_in_type_path_gain": list(HOLD) in tpg, "glno_nt": nts,
                "type_path_gain": tpg, "receptor_model": lif.receptor_model, "receptor_net_rule": lif.receptor_net_rule,
                "cache_dir": str(cache_dir) if cache_dir else None, "proprioception": sim.proprioception,
                "proprioception_sense_attached": getattr(sim.fb, "proprioception_sense", None) is not None,
                "flight": {"gf_hz": float(sim.flights[0].gf_hz), "takeoff_power_hz": float(sim.flights[0].takeoff_power_hz),
                           "takeoff_hold_s": float(sim.flights[0].takeoff_hold_s)}}
    problems = []
    if arm == "instrumented":
        if sim.fb.preset != "instrumented" or sorted(names) != sorted(RECORD_NAMES):
            problems.append(f"preset {sim.fb.preset!r} instruments {names}")
        if list(HOLD) not in tpg:
            problems.append("hold not in the brain's type_path_gain")
        if nts != ["glutamate"]:
            problems.append(f"GLNO nt {nts} in the cache the run used")
    else:
        if sim.fb.preset != "raw" or names:
            problems.append(f"raw arm resolved preset {sim.fb.preset!r} instruments {names}")
        if list(HOLD) in tpg:
            problems.append("raw arm carries the hold")
        if nts != ["unknown"]:
            problems.append(f"raw arm GLNO nt {nts}")
    if "cuda" not in dev and os.environ.get("INSTRUMENTED_ROOM_ALLOW_CPU") != "1":
        problems.append(f"realised device {dev} is not a GPU")
    seed = int(rest[rest.index("--seed") + 1]) if "--seed" in rest else 0
    stim = {"protocol": "batch_sustain.py --program cx --fruit apple --fence --energy 0.9, live escape route (the room take-off rate-half)",
            "params": {"batch": sim.B, "env_seeds": list(sim.seeds), "program": sim.program_names[0], "fence": sim.fence, "fruit_set": sim.fruit_set,
                       "arm": arm, "instruments": INSTRUMENT_SPECS if arm == "instrumented" else [], "hold_edges": [list(HOLD)] if arm == "instrumented" else [],
                       "nt_override": NT_OVERRIDE if arm == "instrumented" else {}},
            "control": "the raw run of the same brain seed / env seeds"}
    prov = common.provenance(sim.c, fb=sim.fb, device=dev, seeds=[seed], env_seeds=list(sim.seeds), batch=sim.B, stimulus=stim,
                             cache_dir=str(cache_dir) if cache_dir else None)
    d["room"] = {"arm": arm, "resolved": resolved, "problems": problems, "device": dev,
                 "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "wall_s": round(time.time() - t0, 1),
                 "wrapper": "scripts/instrumented_room.py", "tool_argv": rest}
    d["provenance"] = prov
    path.write_text(json.dumps(common.to_jsonable(d), indent=1) + "\n", encoding="utf-8")
    print(f"room: arm {arm} preset {sim.fb.preset} instruments {names} hold {resolved['hold_in_type_path_gain']} GLNO {nts} device {dev} "
          f"{d['room']['device_name']} md5 {prov['compiled_connectome'].get('md5')} hops {d.get('hops_total')} = escape {d.get('hops_escape_total')} "
          f"+ voluntary {d.get('hops_voluntary_total')} over {d.get('fly_s')} fly-s", flush=True)
    if problems:
        print("room: INVALID -- " + "; ".join(problems), flush=True)
        sys.exit(3)


def main():
    t0 = time.time()
    arm, out, rest = parse(sys.argv[1:])
    import torch
    allow_cpu = os.environ.get("INSTRUMENTED_ROOM_ALLOW_CPU") == "1"       # a desktop smoke of the rebinding only; never a result
    assert allow_cpu or torch.cuda.is_available(), "GPU required"
    print("device", "cuda" if torch.cuda.is_available() else "cpu (SMOKE ONLY)", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
          "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), "arm", arm, flush=True)
    c, cache_dir = None, None
    if arm == "instrumented":
        import cx_wedge
        c, cache_dir, table = cx_wedge.load_connectome(NT_OVERRIDE)
        print(f"connectome from scratch cache {cache_dir} (TYPE_NT_OVERRIDE = {table})", flush=True)
    rebind(arm, c)
    import batch_sustain
    sys.argv = ["batch_sustain.py"] + rest + ["--json", out]
    batch_sustain.main()
    finish(arm, out, rest, t0, cache_dir)


if __name__ == "__main__":
    main()
