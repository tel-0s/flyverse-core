"""Attribute CUDA work to profile_room scopes by runtime correlation, including graph replay."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def summarize(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    scopes = [e for e in events if e.get("cat") == "user_annotation"]
    counts = Counter(e["name"] for e in scopes)
    runtime = {}
    result = defaultdict(Counter)
    for e in events:
        if e.get("cat") != "cuda_runtime":
            continue
        parents = [s for s in scopes if s["tid"] == e["tid"] and s["pid"] == e["pid"] and
                   s["ts"] <= e["ts"] < s["ts"]+s["dur"]]
        if not parents:
            continue
        name = min(parents,key=lambda s:s["dur"])["name"]
        runtime[e["args"]["correlation"]] = name
        if "Launch" in e["name"]:
            result[name]["host_launches"] += 1
        if "Synchronize" in e["name"]:
            result[name]["host_syncs"] += 1
    kernels = defaultdict(Counter)
    for e in events:
        cat = e.get("cat")
        if cat not in ("kernel","gpu_memcpy"):
            continue
        name = runtime.get(e.get("args",{}).get("correlation"))
        if name is None:
            continue
        if cat == "kernel":
            result[name]["kernels"] += 1
            result[name]["kernel_us"] += e["dur"]
            kernels[name][e["name"]] += 1
        elif "DtoH" in e["name"]:
            result[name]["d2h_copies"] += 1
            result[name]["d2h_bytes"] += e["args"].get("bytes",0)
    return {"trace":str(path),"scopes":{name:dict(samples=counts[name],
        per_call={k:v/counts[name] for k,v in row.items()},
        kernel_names=dict(kernels[name])) for name,row in result.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()
    result = [summarize(path) for path in args.traces]
    if args.json:
        Path(args.json).write_text(json.dumps(result,indent=2))
    for row in result:
        print(row["trace"])
        for name,scope in row["scopes"].items():
            print(f"  {name}: {json.dumps(scope['per_call'])}")
