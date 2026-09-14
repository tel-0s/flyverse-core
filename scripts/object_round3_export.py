"""Round-three Neurome delivery and independent checks (CPU only).

`pack-rect --raw DIR --out DIR` preserves every LC body's frame series from the
original synthetic recordings before a fetch. It never changes the recordings.
The revision-two exporter supplies the interchange schema and its verifier.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyverse.interp import common, export as ex
import object_round2_export as e2
import object_round2_compare as c2
import object_round3_rectangles as rect

LC = ("LC11", "LC10a")
EXPORT_CODE_BYTES = Path(__file__).read_bytes()
EXPORT_CODE_SHA256 = hashlib.sha256(EXPORT_CODE_BYTES).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(common.to_jsonable(value), indent=1), encoding="utf-8")


def pack_rect(args):
    """Lossless selection of LC columns; 10 ms sampling, both original arms."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary in sorted(Path(args.raw).glob("*_summary.json")):
        stem = str(summary)[:-len("_summary.json")]
        name = Path(stem).name; sm = load(summary)
        if rect.parse_stem(name) is None:
            continue
        dest = out / (name + "_lc.npz")
        side = out / (name + "_lc.json")
        if dest.exists() and side.exists() and load(side).get("sha256") == ex.sha256_file(dest):
            rows.append(load(side)); continue
        arrays, sources = {}, []
        for label, arm in zip(("a", "b"), sm["arms"]):
            source = Path(stem + "_" + arm + ".npz")
            record = common.Recording.load(source)
            idx = np.flatnonzero(np.isin(record.types.astype(str), LC))
            if label == "a":
                arrays.update(body_ids=record.body_ids[idx], types=record.types[idx], t_ms=record.t_ms,
                              idx=record.idx[idx])
            else:
                assert np.array_equal(arrays["body_ids"], record.body_ids[idx])
                assert np.array_equal(arrays["t_ms"], record.t_ms)
            arrays[label + "__drive_mv"] = record.quantities["drive_mv"][:, idx]
            arrays[label + "__spikes"] = rect.per_frame_spikes(record.quantities["spike_count"])[:, idx]
            sources.append({"file": str(source), "sha256": ex.sha256_file(source), "arm": arm})
        assert {t: int((arrays["types"] == t).sum()) for t in LC} == {"LC11": 143, "LC10a": 275}
        np.savez_compressed(dest, **arrays)
        row = {"schema": "flyverse.object_lc_frames/1", "file": str(dest), "sha256": ex.sha256_file(dest),
               "source_recordings": sources, "provenance": load(stem + "_prov.json"),
               "generator": " ".join(sys.argv), "analysis_sha256": ex.sha256_file(__file__),
               "n_frames": len(arrays["t_ms"]), "n_bodies": len(arrays["body_ids"]),
               "spikes": "per-frame counts differenced from cumulative counts; initial previous count is zero"}
        save(side, row); rows.append(row)
        print(name, "packed", dest.stat().st_size, flush=True)
    if not rows:
        raise ValueError("no synthetic rectangle recordings found")
    save(out / "index.json", {"generator": " ".join(sys.argv), "runs": rows, "n_runs": len(rows)})
    print(f"packed {len(rows)} LC frame records", flush=True)
    return 0


def verify_lc(args):
    rows=[]; problems=[]
    for p in sorted(Path(args.out).glob("*_lc.json")):
        side=load(p); data=p.with_suffix(".npz")
        ok=data.exists() and ex.sha256_file(data)==side["sha256"]
        if ok:
            with np.load(data,allow_pickle=False) as z:
                counts={t:int((z["types"]==t).sum()) for t in LC}
                ok=counts=={"LC11":143,"LC10a":275} and len(z["t_ms"])==side["n_frames"]
                ok &= all(z[k].shape==(side["n_frames"],418) for k in ("a__drive_mv","b__drive_mv","a__spikes","b__spikes"))
        if not ok: problems.append(str(p))
        rows.append({"file":str(data),"ok":bool(ok),"device_name":side["provenance"]["execution"].get("device_name")})
    if len(rows)!=args.expected: problems.append(f"{len(rows)} records, expected {args.expected}")
    save(Path(args.out)/"verify.json",{"schema":"flyverse.object_lc_frames_check/1","generator":" ".join(sys.argv),
         "runs":rows,"problems":problems,"index_sha256":ex.sha256_file(Path(args.out)/"index.json")})
    print(f"LC frame archive: {len(rows)} records, {len(problems)} problems")
    return int(bool(problems))


def receipts(args):
    """Recover scheduler completion evidence when an original polling client died."""
    import cluster_run as cr
    cfg, _ = cr.load_config(); target = cfg[args.target]
    console = Path(args.log).read_text(encoding="utf-8", errors="replace")
    ids = list(dict.fromkeys(re.findall(r"job ([0-9a-f]+) queued", console)))
    if not ids:
        raise ValueError("no submitted job IDs in the original console")
    code = ("import urllib.request,json\nrows=[]\n"
            f"for id in {ids!r}:\n"
            f" d=json.load(urllib.request.urlopen({target.remote_api!r}+'/jobs/'+id,timeout=30))\n"
            " d=d.get('job',d); r={k:d.get(k) for k in ['id','name','status','created_at','submitted_at','started_at','completed_at','exit_code','error','node']}\n"
            " r['requested_node']=d.get('spec',{}).get('node'); rows.append(r)\n"
            "print(json.dumps(rows))")
    proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-p", str(target.port), target.host,
                           "python3 -c " + shlex.quote(code)], check=True, capture_output=True, text=True)
    rows = json.loads(proc.stdout)
    failed = [r for r in rows if r["status"] != "completed"]
    save(Path(args.out) / "scheduler_receipt.json", {"schema": "flyverse.scheduler_receipt/1", "jobs": rows,
         "source_log": args.log, "source_log_sha256": ex.sha256_file(args.log), "target": args.target,
         "recovered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generator": " ".join(sys.argv)})
    log = Path(args.out) / "recovered_cluster.log"
    log.write_text("# Recovered from Heimdall by job ID; the original client did not finish polling.\n"
                   + f"# Evidence: {args.out}/scheduler_receipt.json\n"
                   + f"{len(rows)} job(s), {len(failed)} failed (recovered scheduler status; non-completed counted as failed)\n", encoding="utf-8")
    print(log.read_text(), flush=True)
    return int(bool(failed))


def fetch(args):
    """Resume a completed batch's named directory with one tar stream, checking hashes."""
    import cluster_run as cr
    cfg, _ = cr.load_config(); target = cfg[args.target]
    for value in (args.run, args.subdir):
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", value) or ".." in value.split("/") or value.startswith("/"):
            raise ValueError("run/subdir must be safe relative names")
    source = f"{target.runs}/{args.run}/out/{args.subdir}"
    ssh = ["ssh", "-o", "BatchMode=yes", "-p", str(target.port), target.host]
    code = ("from pathlib import Path; import json,hashlib; print(json.dumps({str(p):"
            "{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} "
            "for p in Path('.').rglob('*') if p.is_file()}))")
    proc = subprocess.run(ssh + [f"cd {shlex.quote(source)} && python3 -c {shlex.quote(code)}"], check=True, capture_output=True, text=True)
    sizes = json.loads(proc.stdout); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name in sizes:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe archive member")
    def matches(name, info):
        p = out / name
        return p.exists() and p.stat().st_size == info["size"] and ex.sha256_file(p) == info["sha256"]
    wanted = [name for name, info in sizes.items() if not matches(name, info)]
    print(f"{len(sizes)} remote files; {len(wanted)} missing or hash-mismatched", flush=True)
    if wanted:
        listing = out / ".round3_fetch_list.txt"; listing.write_bytes(("\n".join(wanted) + "\n").encode("utf-8"))
        with listing.open("rb") as handle:
            sender = subprocess.Popen(ssh + [f"cd {shlex.quote(source)} && tar czf - -T -"], stdin=handle, stdout=subprocess.PIPE)
            receiver = subprocess.Popen(["tar", "xzf", "-", "-C", str(out)], stdin=sender.stdout)
            sender.stdout.close(); receiver.wait(); sender.wait()
        listing.unlink()
        if receiver.returncode or sender.returncode:
            raise RuntimeError(f"tar exit codes {sender.returncode}/{receiver.returncode}")
    missing = [name for name, info in sizes.items() if not matches(name, info)]
    save(out / "round3_fetch_receipt.json", {"source": source, "target": args.target, "files": sizes,
         "n_files": len(sizes), "missing_or_hash_mismatched": missing, "generator": " ".join(sys.argv)})
    print(f"hash verification: {len(sizes)} files, {len(missing)} problems", flush=True)
    return int(bool(missing))


def time_course(paths, kind="sphere"):
    """Run-mean and run-SD of paired frame series, without pooling bodies."""
    values = {}; bodies = types = times = None
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            b = z["lc_body_ids"] if kind == "sphere" else z["body_ids"]
            ty = z["lc_types"].astype(str) if kind == "sphere" else z["types"].astype(str)
            mask = np.isin(ty, LC)
            if kind == "sphere":
                pairs = {"drive_mv": (z["a__lc_drive_mv"][:, mask], z["b__lc_drive_mv"][:, mask]),
                         "spikes_per_frame": (z["a__lc_spikes"][:, mask], z["b__lc_spikes"][:, mask])}
                t = np.arange(len(pairs["drive_mv"][0]), dtype=float) * common.FRAME_MS
            else:
                pairs = {"drive_mv": (z["a__drive_mv"][:, mask], z["b__drive_mv"][:, mask]),
                         "spikes_per_frame": (z["a__spikes"][:, mask], z["b__spikes"][:, mask])}
                t = z["t_ms"]
            if bodies is None:
                bodies, types, times = b[mask], ty[mask], t
            assert np.array_equal(bodies, b[mask]) and np.array_equal(times, t), "frame/body alignment differs"
            for q, (a, blank) in pairs.items():
                for arm, v in (("object", a), ("blank", blank), ("difference", a.astype(float) - blank)):
                    values.setdefault(q + "_" + arm, []).append(v.astype(float))
    if bodies is None:
        raise ValueError("no frame records")
    df = pd.DataFrame({"frame": np.repeat(np.arange(len(times)), len(bodies)),
                       "t_ms": np.repeat(times, len(bodies)), "bodyId": np.tile(bodies.astype(str), len(times)),
                       "type": np.tile(types, len(times)), "n_runs": len(paths)})
    for key, v in values.items():
        a = np.stack(v)
        df[key + "_mean"] = a.mean(0).ravel()
        df[key + "_sd_runs"] = a.std(0, ddof=1).ravel() if len(v) > 1 else np.nan
    return df


def write_delivery(res, prefix, root, extras=(), **kwargs):
    """Write through revision 2; supplementary tables share its hashes/verifier."""
    e2.stamp_commit(res)
    rid = e2.run_id(prefix)
    run_dir = ex.export(res, out_root=root, run_id=rid, parquet_rows=100_000, **kwargs)
    man = load(run_dir / "manifest.json")
    device = (res.provenance.get("execution") or {}).get("device_name")
    for name, df in extras:
        if not len(df):
            continue
        df = df.copy()
        if "device_name" not in df and device:
            df["device_name"] = device
        df = ex._dataset_columns(df, res.provenance)
        man["tables"].append(ex.write_table(df, run_dir, name, 100_000, role="interchange"))
    man["files"]["round3_export_code_sha256"] = EXPORT_CODE_SHA256
    save(run_dir / "manifest.json", man)
    check = ex.verify(run_dir, neurons=True)
    trip = ex.round_trip_check(res, run_dir)
    for table, row in trip.items():
        if row["max_abs_diff"] > 1e-6 or not row.get("ids_match", True) or not row["text_columns_match"]:
            check["problems"].append(f"round trip {table}: {row}")
    save(run_dir / "checks.json", {"verify": check, "round_trip": trip})
    print(rid, "tables", len(man["tables"]), "problems", check["problems"], flush=True)
    if check["problems"]:
        raise ValueError(check["problems"])
    return {"run_id": rid, "run_dir": str(run_dir).replace("\\", "/"), "tables": man["tables"],
            "problems": check["problems"], "round_trip": trip}


def recorded_device_maps(source, spec_root):
    maps = {s: {} for s in ("sphere", "spec", "bench")}
    arms = sorted(source.get("summary", {}).get("arms", c2.ARMS), key=len, reverse=True)
    def arm_of(run):
        return next(a for a in arms if str(run).startswith(a + "_"))
    for row in source.get("summary", {}).get("verify", {}).get("sphere", {}).get("runs", []):
        arm = arm_of(row["run"])
        maps["sphere"].setdefault(arm, set()).add(row["device_name"])
    for row in source.get("tables", {}).get("spec_per_run", []):
        p = Path(spec_root) / (row["run"] + "_prov.json")
        maps["spec"].setdefault(row["arm"], set()).add(load(p)["execution"]["device_name"])
    for row in source.get("tables", {}).get("bench_per_run", []):
        p = load(row["file"])
        maps["bench"].setdefault(row["arm"], set()).add(p["provenance"]["execution"]["device_name"])
    return {section: {a: sorted(v) for a, v in entries.items()} for section, entries in maps.items()}


def finalize(args):
    """Join row provenance to its actual source, then hash and verify the delivery.

    This also completes revision-two-generated retinal tables with device labels.
    No simulation measurements, test statistics or inference families are changed.
    """
    import ast
    directories = []; inventory = []; problems = []; device_cache = {}
    for index_path in args.indices:
        index = load(index_path)
        for entry in index["directories"]:
            path = Path(entry["run_dir"]); man = load(path / "manifest.json")
            raw_result = load(path / man["source_result"]["file"])
            result_path = man["files"].get("analysis_result", {}).get("file")
            source = load(result_path) if result_path else {}
            if result_path and result_path not in device_cache:
                device_cache[result_path] = recorded_device_maps(source, args.spec_root)
            maps = device_cache.get(result_path, {})
            sphere_map = maps.get("sphere", {})
            group = man["summary"].get("group")
            family = bool(sphere_map) and group is None
            device = ("|".join(sorted({d for ds in sphere_map.values() for d in ds})) if family
                      else man["execution"].get("device_name"))
            same = sphere_map.get(group) == sphere_map.get("base") if sphere_map and group else not family
            if not device:
                raise ValueError(f"unknown device in {path}")
            updated = []
            for table in man["tables"]:
                name = table["name"]
                needs = (bool(sphere_map) or name.startswith("transition_") or name == "benchmark_sidecars"
                         or "device_name" not in table["columns"]
                         or "same_device_as_reference" not in table["columns"])
                if not needs:
                    updated.append(table); continue
                # Start with the Result where available: metadata completion
                # must never turn the literal verdict "null" into missing data.
                if name in raw_result["tables"]:
                    df = ex._dataset_columns(pd.DataFrame(raw_result["tables"][name]), raw_result["provenance"])
                elif name in source.get("tables", {}):
                    df = ex._dataset_columns(pd.DataFrame(source["tables"][name]), source["provenance"])
                    if group and "arm" in df: df = df[df.arm == group]
                else:
                    df = ex.read_table(path, name)
                if name.startswith("transition_"):
                    arm_names = sorted(maps["spec"], key=len, reverse=True)
                    def arm_of(run):
                        return next(a for a in arm_names if str(run).startswith(a + "_"))
                    df["arm"] = df.run.map(arm_of)
                    # Read each recorded run's provenance rather than infer its
                    # GPU from the family directory's representative first run.
                    records = {}
                    for run in df.run.unique():
                        candidates = list(Path(args.spec_root).glob(str(run) + "_prov.json"))
                        if len(candidates) != 1:
                            raise ValueError(f"missing unique transition provenance for {run}")
                        records[run] = load(candidates[0])["execution"]["device_name"]
                    df["device_name"] = df.run.map(records)
                    reference = set(maps["spec"]["base"])
                    df["same_device_as_reference"] = df.device_name.map(lambda d: {d} == reference)
                elif name == "benchmark_sidecars":
                    def parse_sidecar(value):
                        if not isinstance(value, str): return value
                        try: return json.loads(value)
                        except json.JSONDecodeError: return ast.literal_eval(value)
                    sidecars = [parse_sidecar(v) for v in df.sidecar]
                    df["sidecar"] = [json.dumps(v, separators=(",", ":")) for v in sidecars]
                    df["same_device_as_reference"] = df.device_name.map(lambda d: {d} == set(maps["bench"]["base"]))
                else:
                    section = "spec" if name.startswith("spec_") else "bench" if name.startswith("bench_") else "sphere"
                    dm = maps.get(section, {})
                    arm_column = next((c for c in ("arm", "group") if c in df and df[c].isin(dm).all()), None)
                    if arm_column:
                        df["device_name"] = df[arm_column].map(lambda a: "|".join(dm[a]))
                        df["same_device_as_reference"] = df[arm_column].map(lambda a: dm[a] == dm["base"])
                    else:
                        df["device_name"] = device
                        df["same_device_as_reference"] = same
                    if "against" in df:
                        own_null = df.against.eq("null")
                        df.loc[own_null, "same_device_as_reference"] = True
                        df["device_reference"] = np.where(own_null, "own arm blank/blank runs", "base arm")
                if df.device_name.isna().any() or df.device_name.eq("").any():
                    raise ValueError(f"missing row device in {path}/{name}")
                if name in raw_result["tables"]:
                    raw_result["tables"][name] = common.to_jsonable(df.to_dict("records"))
                updated.append(ex.write_table(df, path, name, 100_000, role=table["role"]))
            man["tables"] = updated
            man["files"]["round3_finalizer_code_sha256"] = EXPORT_CODE_SHA256
            if sphere_map:
                recordings = man["files"].get("recordings", [])
                if group is None:
                    records = source["summary"]["verify"]["sphere"]["runs"]
                    man["replicates"] = {"unit": "runs", "n": len(records), "runs": records}
                else:
                    stim = [p for p in recordings if "_obj_s" in Path(p).stem]
                    nulls = [p for p in recordings if "_null_s" in Path(p).stem]
                    man["replicates"] = {"unit": "runs", "n": len(stim), "runs": [{"file": p} for p in stim],
                                         "n_null": len(nulls), "null": nulls}
            man["summary"]["device_label_reading"] = ("GPU model of the row's recorded source. A pipe-separated set denotes a multi-device aggregate; "
                "same_device_as_reference uses the explicitly named comparison reference where present, otherwise the base arm of that experiment "
                "(shipped lobe for rectangle/RF maps). It does not imply equality of seeds, host or numerical trajectories.")
            raw_result["replicates"] = man["replicates"]
            save(path / man["source_result"]["file"], raw_result)
            man["source_result"]["sha256"] = ex.sha256_file(path / man["source_result"]["file"])
            save(path / "manifest.json", man)
            verified = ex.verify(path, neurons=True)
            prior_checks = load(path / "checks.json")
            prior_checks["verify"] = verified
            prior_checks["round_trip"] = ex.round_trip_check(common.Result.load(path / man["source_result"]["file"]), path)
            for table_name, check in prior_checks["round_trip"].items():
                if check["max_abs_diff"] > 1e-6 or not check.get("ids_match", True) or not check["text_columns_match"]:
                    verified["problems"].append(f"round trip {table_name}: {check}")
            save(path / "checks.json", prior_checks)
            entry.update(tables=updated, problems=verified["problems"], round_trip=prior_checks["round_trip"])
            problems.extend(f"{path}: {p}" for p in verified["problems"])
            directories.append(entry)
            inventory.extend({"run_id": entry["run_id"], **t} for t in updated)
            print(path.name, len(updated), "tables; problems", verified["problems"], flush=True)
        save(index_path, index)
    save(args.index, {"schema": ex.EXPORT_SCHEMA, "export_revision": 2, "generator": " ".join(sys.argv),
         "source_indices": args.indices, "directories": directories, "n_directories": len(directories),
         "n_tables": len(inventory), "problems": problems, "analysis_sha256": EXPORT_CODE_SHA256})
    save(Path(args.index).with_name("objr3_tables.json"), inventory)
    Path(args.index).with_name("object_round3_export_snapshot.py").write_bytes(EXPORT_CODE_BYTES)
    return int(bool(problems))


def compare_export(args):
    """Reuse revision two's rung/readout builder; extend its writer in this process."""
    source = load(args.result)
    runs = e2.load_runs(args.out, "arm", c2.parse_sphere_name)
    if len(runs) == 0:
        raise ValueError("no complete sphere records")
    devices = {}
    for r in runs.itertuples():
        devices.setdefault(r.arm, set()).add(load(r.json)["provenance"]["execution"]["device_name"])
    base_device = devices["base"]
    original = e2.write_run_dir

    def writer(res, prefix, export_root, retina=None, retina_blank=None, track=None, track_report=None,
               paired_ids="", null_ids="", **unused):
        g = res.summary.get("group"); diam = res.summary.get("angular_diameter_deg")
        res.tool_version = "object_round3_export/1"
        res.files["analysis_result"] = {"file": args.result, "sha256": ex.sha256_file(args.result)}
        res.files["generator"] = " ".join(sys.argv)
        res.summary["export_reading"] = ("Original stamped analysis families and Holm values retained; device_name and "
            "same_device_as_reference accompany rows. Individual LC frame series are run means and run SDs, "
            "not maxima; all bodies, including unwindowed bodies, are retained.")
        dn = "|".join(sorted(devices.get(g, set().union(*devices.values()))))
        same = devices.get(g) == base_device if g else False
        for name in list(res.tables):
            frame = pd.DataFrame(res.tables[name])
            if not len(frame):
                continue
            if "analysis_family" in frame:
                for key in ("family", "p_holm", "survives_holm"):
                    if "analysis_" + key in frame:
                        frame[key] = frame["analysis_" + key]
                frame["p_holm_source"] = "Original stamped analysis family, copied without recomputing"
            if "device_name" not in frame:
                frame["device_name"] = (frame["arm"].map(lambda a: "|".join(sorted(devices.get(a, []))))
                                        if "arm" in frame else dn)
            if "same_device_as_reference" not in frame:
                frame["same_device_as_reference"] = (frame["arm"].map(lambda a: devices.get(a) == base_device)
                                                      if "arm" in frame else same)
            res.add_table(name, frame)
        extras = []
        if g is not None and diam is not None:
            obj = runs[(runs.arm == g) & ~runs.null & np.isclose(runs.diam.astype(float), diam)]
            tc = time_course(obj.npz.tolist())
            tc["arm"] = g; tc["diam_deg"] = diam; tc["same_device_as_reference"] = same
            extras.append(("lc_per_body_time_course", tc))
            rb = pd.DataFrame(res.tables["readout_per_body"])
            ww = rb[(rb["window_n_frames"] > 0) & rb.type.isin(LC)].drop_duplicates("bodyId")
            coverage = ww.groupby(["type", "window_source"]).agg(n_bodies_windowed=("bodyId", "nunique")).reset_index()
            coverage["arm"] = g; coverage["diam_deg"] = diam
            coverage["same_device_as_reference"] = same
            extras.append(("window_coverage", coverage))
            first = load(obj.iloc[0].json)
            if not res.provenance["model"].get("optic_hook_info"):
                spec = sorted(Path(args.out, "spec").glob(g + "*_summary.json"))
                if spec:
                    res.provenance["model"]["optic_hook_info"] = load(spec[0]).get("hook_info")
                    res.files["hook_info_source"] = {"file": str(spec[0]), "sha256": ex.sha256_file(spec[0])}
            res.summary["lc_time_course"] = {"n_bodies": int(tc.bodyId.nunique()), "n_frames": int(tc.frame.nunique()),
                "n_runs": len(obj), "frame_ms": common.FRAME_MS, "time_origin": "first frame after settling",
                "records": obj.stem.tolist(), "scatter": "sample SD over runs, ddof=1; paired difference computed within each run"}
        # Levels and held-out battery are useful alongside each rung, with raw
        # benchmark/provenance sidecars included in the family directory below.
        for name in ("levels", "levels_runs", "spec_per_run", "spec_comparisons", "bench_per_run", "bench_comparisons"):
            frame = pd.DataFrame(source["tables"].get(name, []))
            if len(frame):
                if g and "arm" in frame:
                    frame = frame[frame.arm == g]
                section = "spec" if name.startswith("spec") else "bench" if name.startswith("bench") else "sphere"
                devmap = source.get("summary", {}).get("devices_by_arm", {}).get(section, devices)
                if isinstance(devmap, dict):
                    frame["device_name"] = frame["arm"].map(lambda a: "|".join(sorted(devmap.get(a, [])))) if "arm" in frame else dn
                    if "same_device_as_reference" not in frame and "arm" in frame:
                        frame["same_device_as_reference"] = frame.arm.map(lambda a: bool(devmap.get(a)) and devmap.get(a) == devmap.get("base"))
                extras.append((name, frame))
        if g is None:
            if args.transitions and Path(args.transitions).exists():
                tr = load(args.transitions)
                for name, rows in tr["tables"].items():
                    frame = pd.DataFrame(rows)
                    if len(frame):
                        smap = source.get("summary", {}).get("devices_by_arm", {}).get("spec", {})
                        if "arm" in frame:
                            frame["device_name"] = frame.arm.map(lambda a: "|".join(smap.get(a, [])))
                            frame["same_device_as_reference"] = frame.arm.map(lambda a: bool(smap.get(a)) and smap.get(a) == smap.get("base"))
                        extras.append(("transition_" + name, frame))
                res.files["transition_result"] = {"file": args.transitions, "sha256": ex.sha256_file(args.transitions)}
            sidecars = []
            for p in sorted(Path(args.out, "bench").glob("*_prov.json")):
                d = load(p); pv = d.get("provenance", {})
                sidecars.append({"file": str(p), "sha256": ex.sha256_file(p), "arm": d.get("arm"), "seed": d.get("seed"),
                                 "device_name": pv.get("execution", {}).get("device_name"), "sidecar": d})
            if sidecars:
                extras.append(("benchmark_sidecars", pd.DataFrame(sidecars)))
        if track is not None and len(track):
            extras.append(("retina_object_track", track))
            res.summary["object_track"] = track_report
        # Names cannot duplicate an existing manifest table.
        extras = [(n, f) for n, f in extras if n not in res.tables]
        tag = args.prefix + (f"-{g}-d{round(diam * 10):03d}" if g is not None else "-family")
        return write_delivery(res, tag, export_root, extras,
                              retina=retina, retina_blank=retina_blank, retina_in_loop=retina is not None,
                              paired_control_ids=paired_ids or None, null_reference_ids=null_ids or None)

    e2.write_run_dir = writer
    try:
        argv = ["compare", "--out", args.out, "--baseline", args.result, "--export-root", args.export_root,
                "--index", args.index, "--family", "none", "--parquet-rows", "100000"]
        if args.group:
            argv += ["--group", args.group]
        if args.sizes:
            argv += ["--sizes", *map(str, args.sizes)]
        return e2.main(argv)
    finally:
        e2.write_run_dir = original


def check_stats(args):
    """Independent enumeration of label assignments, moments, and family Holm.

    Deliberately does not call a round-two/three comparison or Holm reducer.
    Tests the recorded run vectors; raw-to-vector checks are a separate step.
    """
    from itertools import combinations
    from math import comb
    from scipy.stats import rankdata, mannwhitneyu
    source = load(args.result); findings = []; checked = 0; families = {}; combos = {}

    def equal(x, y):
        if x is None: x = np.nan
        if y is None: y = np.nan
        return bool(np.isclose(float(x), float(y), atol=1e-10, rtol=1e-8, equal_nan=True))

    for table, rows in source["tables"].items():
        for i, row in enumerate(rows):
            if "stim_values" not in row or "null_values" not in row:
                continue
            a = np.asarray(row["stim_values"], float); b = np.asarray(row["null_values"], float)
            a = a[np.isfinite(a)]; b = b[np.isfinite(b)]; n, m = len(a), len(b)
            calc = {"stim_n": n, "null_n": m}
            for label, v in (("stim", a), ("null", b)):
                calc[label + "_mean"] = v.mean() if len(v) else np.nan
                calc[label + "_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            diff = calc["stim_mean"] - calc["null_mean"]; sd = calc["null_sd"]
            calc.update(diff=diff, z=diff / sd if sd > 0 else np.nan)
            if n and m:
                ranks = rankdata(np.r_[a, b]); rank_sum = ranks[:n].sum(); centre = n * (n + m + 1) / 2
                if (n, m) not in combos:
                    if comb(n+m, n) > 200_000:
                        raise ValueError("independent exact check exceeds enumeration bound")
                    combos[n, m] = np.array(list(combinations(range(n+m), n)))
                sums = ranks[combos[n, m]].sum(axis=1)
                calc["p"] = float((np.abs(sums-centre) >= abs(rank_sum-centre)-1e-9).mean())
                calc["U_tie"] = rank_sum-n*(n+1)/2
                calc["n_tied"] = n+m-len(np.unique(np.r_[a,b]))
                # The inherited common.compare verdict uses SciPy's exact p;
                # the stamped family uses the tie-aware p separately.
                p0 = float(mannwhitneyu(a,b,method="exact").pvalue)
                det = sd <= 1e-12 * max(1,abs(calc["null_mean"]))
                verdict = ("underpowered" if min(n,m) < 4 or 2/comb(n+m,n) > .05 else
                           "undetermined" if det and diff != 0 and p0 <= .05 else
                           "result" if np.isfinite(calc["z"]) and abs(calc["z"]) >= 3 and p0 <= .05 else "null")
            else:
                calc["p"] = np.nan; verdict = "underpowered"
            for key, value in calc.items():
                if key in row and not equal(value, row[key]):
                    findings.append({"table": table, "row": i, "field": key, "reported": row[key], "recomputed": value})
            if row.get("verdict") != verdict:
                findings.append({"table": table, "row": i, "field": "verdict", "reported": row.get("verdict"), "recomputed": verdict})
            if row.get("family") and "p_holm" in row:
                families.setdefault((table,row["family"]), []).append((i,row,calc["p"]))
            checked += 1
    n_families_checked = 0
    for (table, family), members in families.items():
        if all(r.get("p_holm") is None for _,r,_ in members):
            continue
        n_families_checked += 1
        order = sorted(range(len(members)), key=lambda k: members[k][2] if np.isfinite(members[k][2]) else np.inf)
        previous = 0.0
        for rank, k in enumerate(order):
            i,row,p = members[k]
            if not np.isfinite(p): continue
            previous = min(1., max(previous,(len(members)-rank)*p))
            if not equal(previous,row["p_holm"]):
                findings.append({"table":table,"row":i,"field":"p_holm","reported":row["p_holm"],"recomputed":previous})
    stamp = load(Path(args.out)/"predeclared.json") if args.out else {}
    receipt = load(Path(args.out)/"scheduler_receipt.json") if args.out and (Path(args.out)/"scheduler_receipt.json").exists() else {}
    submitted = [j["submitted_at"] for j in receipt.get("jobs",[]) if j.get("submitted_at")]
    stamp_before = bool(submitted and stamp.get("stamped_utc") and stamp["stamped_utc"] < min(submitted))
    report = {"schema":"flyverse.object_independent_statistics/1", "result":args.result,
              "result_sha256":ex.sha256_file(args.result), "provenance":source["provenance"],
              "analysis_sha256":ex.sha256_file(__file__), "generator":" ".join(sys.argv),
              "rows_checked":checked,"families_checked":n_families_checked,"family_labels_seen":len(families),"differences":findings,
              "predeclaration_before_submission":stamp_before,"stamped_utc":stamp.get("stamped_utc"),
              "first_submission_utc":min(submitted) if submitted else None,
              "verdict":"sound" if not findings and checked else "unsound",
              "scope":"Independent numeric reduction of recorded run vectors; not a substitute for checking raw arrays or fresh-device replication."}
    save(args.json,report)
    print(f"independent check: {checked} rows, {n_families_checked} Holm families, {len(findings)} differences; stamp precedes submission {stamp_before}")
    return int(bool(findings))


def check_preference(args):
    """Independent rank/label-permutation checks of rectangle preference tables."""
    from scipy.stats import rankdata
    result = load(args.result); errors = []; schedules = {}; checked = 0
    def permutations(n, draws):
        if (n, draws) not in schedules:
            rng = np.random.RandomState(0)
            schedules[n, draws] = np.array([rng.permutation(n) for _ in range(draws)])
        return schedules[n, draws]
    def rank_test(x, y, draws):
        keep = np.isfinite(x) & np.isfinite(y); x, y = x[keep], y[keep]
        if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0: return np.nan, np.nan
        a, b = rankdata(x), rankdata(y); a -= a.mean(); b -= b.mean()
        scale = np.linalg.norm(a) * np.linalg.norm(b); rho = float(a @ b / scale)
        perm = (b[permutations(len(x), draws)] * a).sum(1) / scale
        return rho, float((1 + np.count_nonzero(abs(perm) >= abs(rho)-1e-12)) / (draws+1))
    def contrast(a, b):
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if not len(a) or not len(b): return np.nan, np.nan
        z = np.r_[a, b]; diff = float(a.mean()-b.mean()); k = len(a)
        shuffled = z[permutations(len(z), 20000)]
        pdiff = shuffled[:, :k].mean(1)-shuffled[:, k:].mean(1)
        return diff, float((1 + np.count_nonzero(abs(pdiff) >= abs(diff)-1e-12)) / 20001)
    for i, row in enumerate(result["tables"]["preference"]):
        q = row["statistic"]; table = "per_run" if row["type"] in LC else "upstream_runs"
        df = pd.DataFrame(result["tables"][table]); col = "rung_" + row["ladder"]
        df = df[(df.lobe == row["lobe"]) & (df.type == row["type"]) & df[col].notna()]
        a = df[~df.null & (df.contrast_name == row["contrast_name"])]; b = df[df.null]
        x, y = a[col].to_numpy(float), a[q].to_numpy(float)
        rho, p = rank_test(x, y, 20000)
        nr, npv = rank_test(b[col].to_numpy(float), b[q].to_numpy(float), 2000)
        small = np.isin(x, (2.2,4.4,8.8)); effect, cp = contrast(y[small], y[~small])
        values = dict(spearman_rho=rho, spearman_p_perm=p, null_spearman_rho=nr,
                      null_spearman_p_perm=npv, small_minus_large=effect, small_minus_large_p_perm=cp)
        if row["peak_rungs"]:
            target = np.isin(x, row["peak_rungs"])
            values["peak_contrast"], values["peak_p_perm"] = contrast(y[target], y[~target])
        for key, v in values.items():
            old = row[key] if row[key] is not None else np.nan
            if not np.isclose(old, v, atol=1e-10, rtol=1e-8, equal_nan=True):
                errors.append({"row": i, "field": key, "reported": old, "recomputed": v})
            checked += 1
    save(args.json, {"result": args.result, "result_sha256": ex.sha256_file(args.result),
         "analysis_sha256": EXPORT_CODE_SHA256, "generator": " ".join(sys.argv),
         "statistics_checked": checked, "differences": errors, "verdict": "sound" if not errors else "unsound"})
    print(checked, "preference statistics;", len(errors), "differences", flush=True)
    return int(bool(errors))


def check_delivery(args):
    """Audit every table's row labels; independently check representative LC traces."""
    index = load(args.index); problems = []; groups = {}; tables = 0; frame_dirs = 0; trace_checks = []
    for entry in index["directories"]:
        path = Path(entry["run_dir"]); man = load(path / "manifest.json")
        for table in man["tables"]:
            columns = ("device_name", "same_device_as_reference")
            if not all(c in table["columns"] for c in columns):
                problems.append(f"{path}/{table['name']}: missing device columns"); continue
            file = path / table["file"]
            df = (pd.read_parquet(file, columns=list(columns)) if table["format"] == "parquet" else
                  pd.read_csv(file, usecols=list(columns), keep_default_na=False, na_values=[""]))
            if df.device_name.isna().any() or df.same_device_as_reference.isna().any():
                problems.append(f"{path}/{table['name']}: missing row provenance")
            tables += 1
        tc = next((t for t in man["tables"] if t["name"] == "lc_per_body_time_course"), None)
        if tc:
            frame_dirs += 1
            if tc["rows"] != 418 * 1200: problems.append(f"{path}: incomplete LC frame lattice")
            category = "rectangle" if "lc_sources" in man["files"] else "same_device" if "samedevice" in path.name else "round2"
            groups.setdefault(category, []).append((path, man, tc))
    for category, entries in groups.items():
        for j in sorted({0, len(entries)//2, len(entries)-1}):
            path, man, tc = entries[j]; df = ex.read_table(path, tc["name"])
            values = []
            if category == "rectangle":
                paths = [Path(r["file"]) for r in man["files"]["lc_sources"]]
            else:
                paths = [Path(p).with_suffix(".npz") for p in man["files"]["recordings"] if "_obj_s" in Path(p).stem]
            for p in paths:
                with np.load(p, allow_pickle=False) as z:
                    if category == "rectangle":
                        ids = z["body_ids"].astype(str); a = z["a__drive_mv"]; b = z["b__drive_mv"]
                    else:
                        mask = np.isin(z["lc_types"].astype(str), LC); ids = z["lc_body_ids"][mask].astype(str)
                        a = z["a__lc_drive_mv"][:, mask]; b = z["b__lc_drive_mv"][:, mask]
                    values.append(a.astype(float)-b.astype(float))
                    if not np.array_equal(df.bodyId.astype(str).to_numpy().reshape(1200,418)[0], ids):
                        problems.append(f"{path}: body order differs from {p}")
            mean = sum(values)/len(values)
            sd = np.sqrt(sum((a-mean)**2 for a in values)/(len(values)-1))
            dev_mean = float(np.max(abs(mean.ravel()-df.drive_mv_difference_mean)))
            dev_sd = float(np.max(abs(sd.ravel()-df.drive_mv_difference_sd_runs)))
            if max(dev_mean,dev_sd) > 1e-10: problems.append(f"{path}: LC trace differs from paired raw frames")
            trace_checks.append({"run_id": path.name, "n_runs": len(values), "n_values": int(mean.size),
                                 "max_abs_mean_difference": dev_mean, "max_abs_sd_difference": dev_sd})
    save(args.json, {"index": args.index, "index_sha256": ex.sha256_file(args.index), "generator": " ".join(sys.argv),
         "n_directories": len(index["directories"]), "n_tables": tables, "n_lc_frame_directories": frame_dirs,
         "trace_checks": trace_checks, "problems": problems, "analysis_sha256": EXPORT_CODE_SHA256})
    print(len(index["directories"]), "directories;", tables, "tables;", len(trace_checks), "independent frame checks;", len(problems), "problems")
    return int(bool(problems))


def check_contrast(args):
    """Recompute the reported rectangle contrast/coverage directly from radiance."""
    result = load(args.result); problems = []; checked = 0; calculated = []
    fields = ("max_coverage", "col_equiv_median", "extreme_rel_change_median")
    def check(tag, key, old, new):
        nonlocal checked
        checked += 1
        if not np.isclose(float(old), float(new), rtol=1e-8, atol=1e-10):
            problems.append({"row": tag, "field": key, "reported": old, "recomputed": new})
    for row in result["tables"]["effective_contrast_runs"]:
        with np.load(Path(args.out) / "syn" / (row["run"] + "_radiance.npz"), allow_pickle=False) as z:
            pixels = z["radiance"].astype(float); background = z["blank"].astype(float)
            relative = pixels[:,:,0]/background[:,0]-1
            # The declared luminance metric sums channels, adds a 1e-9
            # denominator floor and excludes frames without >5% change.
            luminance_change = (pixels-background).sum(2)/(background.sum(1)+1e-9)
            extrema = abs(luminance_change).max(1)
            changed_frames = extrema > .05
        magnitude = abs(relative)/abs(row["weber_contrast"])
        v = {"max_coverage": magnitude.max(), "col_equiv_median": np.median(magnitude.sum(1)),
             "extreme_rel_change_median": np.sign(row["weber_contrast"])*np.median(extrema[changed_frames])}
        for key in fields: check(row["run"], key, row[key], v[key])
        calculated.append({**{k:row[k] for k in ("width_deg","height_deg","contrast_name")}, **v})
    df = pd.DataFrame(calculated)
    for row in result["tables"]["effective_contrast"]:
        part = df[(df.width_deg == row["width_deg"]) & (df.height_deg == row["height_deg"]) & (df.contrast_name == row["contrast_name"])]
        for key in fields:
            check(str((row["width_deg"],row["height_deg"],row["contrast_name"])), key, row[key], part[key].mean())
            check(str((row["width_deg"],row["height_deg"],row["contrast_name"])), key+"_sd_runs", row[key+"_sd_runs"], part[key].std())
    save(args.json, {"result": args.result, "result_sha256": ex.sha256_file(args.result), "generator": " ".join(sys.argv),
         "statistics_checked": checked, "problems": problems, "analysis_sha256": EXPORT_CODE_SHA256})
    print(checked, "radiance statistics;", len(problems), "problems")
    return int(bool(problems))


def link_replications(args):
    """Attach separately labeled native Results; do not pool them with the H200 exports."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); rows = []
    names = [Path(p).parent.name + "-" + Path(p).name for p in args.results]
    if len(names) != len(set(names)): raise ValueError("replication filenames collide")
    for source_path, name in zip(args.results, names):
        result = common.Result.load(source_path)
        problems = result.check()
        if problems: raise ValueError(f"{source_path}: {problems}")
        dest = out / name; dest.write_bytes(Path(source_path).read_bytes())
        rows.append({"file": str(dest).replace("\\", "/"), "sha256": ex.sha256_file(dest),
                     "source_file": source_path, "device_name": result.provenance["execution"]["device_name"],
                     "tool": result.tool, "problems": problems})
    save(out / "index.json", {"schema": "flyverse.object_replication_delivery/1", "generator": " ".join(sys.argv),
         "reading": "Fresh-seed B200 replication Results, kept separate from the original H200/r2 revision-two directories. "
                    "Native Result JSONs preserve all tables and full provenance; external raw-recording paths remain source references.",
         "results": rows, "n_results": len(rows), "analysis_sha256": EXPORT_CODE_SHA256})
    index = load(args.index)
    index["replication_evidence"] = {"file": str(out / "index.json").replace("\\", "/"),
                                     "sha256": ex.sha256_file(out / "index.json"), "n_results": len(rows)}
    save(args.index, index)
    print(len(rows), "verified native replication Results linked from", args.index)
    return 0


def check_rf(args):
    """Recompute pooled centres/widths/false fits directly from node arrays.

    Uses a scalar reducer and spherical dot products, independently of fit_boxed.
    The anatomical prior is read from the delivered map and is not re-estimated.
    """
    result = load(args.result); table = pd.DataFrame(result["tables"]["rf_map"])
    az = el = None; response = []; blank = []
    for run in result["replicates"]["runs"]:
        path = Path(run["file"])
        for dest, p in ((response,path),(blank,Path(str(path).replace("_nodes.npz","_nodes_blank.npz")))):
            with np.load(p,allow_pickle=False) as z:
                assert np.array_equal(z["body_ids"].astype(str),table.bodyId.astype(str))
                az,el = z["node_az_deg"].astype(float),z["node_el_deg"].astype(float)
                dest.append(z["resp__"+result["summary"]["window"]].astype(float)-z["resp__base"])
    rp = np.mean(response,axis=0); rb = np.mean(blank,axis=0)
    nr = len(response); diffs=[]; records=[]
    ae,ee = np.deg2rad(az),np.deg2rad(el)
    xyz = np.c_[np.cos(ee)*np.cos(ae),np.cos(ee)*np.sin(ae),np.sin(ee)]
    spacing = result["summary"]["localizer_params"]["grid_spacing_deg"]
    radius = result["summary"]["box_radius_deg"]

    def one(v, mask, threshold):
        ix=np.flatnonzero(mask); vals=v[ix]
        if len(ix)<12 or not np.isfinite(vals).all(): return False,np.nan,np.nan,np.nan,0
        pk=vals[np.argmax(np.abs(vals))]; noise=1.4826*np.median(np.abs(vals-np.median(vals)))
        fitted=bool(abs(pk)>1e-9 and (noise==0 or abs(pk)>=threshold*noise))
        if not fitted: return False,np.nan,np.nan,np.nan,0
        aligned=vals*(1 if pk>=0 else -1); keep=aligned>=abs(pk)/2
        weights=aligned[keep]-abs(pk)/2+1e-12*abs(pk); cells=ix[keep]
        return True,float(np.average(az[cells],weights=weights)),float(np.average(el[cells],weights=weights)),max(2*np.sqrt(len(cells)*spacing**2/np.pi),spacing),len(cells)

    for j,row in table.iterrows():
        ca,ce=np.deg2rad([row.anat_input_az_deg,row.anat_input_el_deg])
        point=np.array([np.cos(ce)*np.cos(ca),np.cos(ce)*np.sin(ca),np.sin(ce)])
        mask=xyz@point>=np.cos(np.deg2rad(radius))
        got=one(rp[:,j],mask,row.z_min_used); null=one(rb[:,j],mask,row.z_min_used)
        want=(row.fitted,row.az_deg,row.el_deg,row.width_deg,row.n_nodes_above_threshold)
        for field,a,b in zip(("fitted","az_deg","el_deg","width_deg","n_nodes_above_threshold"),got,want):
            if not np.isclose(float(a),float(b) if b is not None else np.nan,atol=1e-8,rtol=1e-8,equal_nan=True):
                diffs.append({"bodyId":row.bodyId,"field":field,"recomputed":a,"reported":b})
        per=sum(one(v[:,j],mask,row.z_min_used)[0] for v in response)
        if per != row.n_runs_fitted:
            diffs.append({"bodyId":row.bodyId,"field":"n_runs_fitted","recomputed":per,"reported":row.n_runs_fitted})
        if bool(null[0]) != bool(row.blank_fitted_at_z):
            diffs.append({"bodyId":row.bodyId,"field":"blank_fitted_at_z"})
        records.append({"bodyId":row.bodyId,"type":row.type,"fitted":got[0],"blank_fitted":null[0],"n_runs_fitted":per})
    save(args.json,{"schema":"flyverse.rf_independent_check/1","result":args.result,"result_sha256":ex.sha256_file(args.result),
                    "provenance":result["provenance"],"analysis_sha256":ex.sha256_file(__file__),"generator":" ".join(sys.argv),
                    "n_runs":nr,"bodies_checked":len(table),"differences":diffs,"per_body":records,
                    "scope":"Independent reduction of every recorded body's pooled node response; supplied anatomical prior held fixed.",
                    "verdict":"sound" if not diffs else "unsound"})
    print(f"independent RF check: {len(table)} bodies x {nr} runs; {len(diffs)} differences",flush=True)
    return int(bool(diffs))


def rf_export(args):
    records=[]
    for path in args.results:
        res=common.Result.load(path); lobe=res.summary["lobe"]
        res.files["source_result"]={"file":path,"sha256":ex.sha256_file(path)}
        runs=res.replicates["runs"]; values={}; bodies=types=az=el=None
        for run in runs:
            for arm,p in (("stimulus",run["file"]),("blank",run["file"].replace("_nodes.npz","_nodes_blank.npz"))):
                with np.load(p,allow_pickle=False) as z:
                    mask=np.isin(z["types"].astype(str),LC)
                    if bodies is None:
                        bodies=z["body_ids"][mask].astype(str); types=z["types"][mask].astype(str)
                        az=z["node_az_deg"]; el=z["node_el_deg"]
                    assert np.array_equal(bodies,z["body_ids"][mask].astype(str))
                    for window in ("on","on1","off","base"):
                        values.setdefault(f"{arm}_{window}_mv",[]).append(z["resp__"+window][:,mask].astype(float))
        df=pd.DataFrame({"node":np.repeat(np.arange(len(az)),len(bodies)),"az_deg":np.repeat(az,len(bodies)),
                         "el_deg":np.repeat(el,len(bodies)),"bodyId":np.tile(bodies,len(az)),"type":np.tile(types,len(az)),
                         "n_runs":len(runs),"lobe":lobe})
        for key, arrays in values.items():
            a=np.stack(arrays); df[key+"_mean"]=a.mean(0).ravel(); df[key+"_sd_runs"]=a.std(0,ddof=1).ravel()
        dev=res.provenance["execution"].get("device_name")
        for name, rows in list(res.tables.items()):
            frame=pd.DataFrame(rows); frame["device_name"]=dev; frame["same_device_as_reference"]=True
            res.add_table(name,frame)
        res.summary["node_series_reading"]="Per-body received drive in each recorded role window, pooled by spatial node over shuffled run orders; this is not a 10-ms frame series."
        records.append(write_delivery(res,args.prefix+"-"+lobe,args.export_root,[("lc_per_body_node_responses",df)]))
    save(args.index,{"generator":" ".join(sys.argv),"directories":records,"schema":ex.EXPORT_SCHEMA})
    return 0


def rectangle_readout(obj, nulls, win, width, height):
    """Whole-window paired readouts and exact lattice-window means, run by run."""
    collected=[]
    for group,runs in (("object",obj),("null",nulls)):
        for row in runs.itertuples():
            with np.load(row.stem+"_reduced.npz",allow_pickle=False) as z:
                sel=z["lc_idx"]; types=z["types"].astype(str)[sel]; ids=z["body_ids"][sel].astype(str)
                pb=rect.per_body_from_reduced(z,win,width,height).set_index("bodyId").loc[ids]
                for q,a_key,b_key,unit,scale,window_a,window_b in (
                    ("upstream_drive_mV","cell__mean_drive_a","cell__mean_drive_b","mV",1.,"drive_a_win","drive_b_win"),
                    ("output_Hz","cell__spikes_a","cell__spikes_b","Hz",1./float(z["win_s"]),"spikes_a_win","spikes_b_win")):
                    a=z[a_key][sel]*scale; b=z[b_key][sel]*scale
                    wa=pb[window_a].to_numpy(); wb=pb[window_b].to_numpy()
                    if unit=="Hz":
                        divisor=pb.n_frames_win.to_numpy()*common.FRAME_MS/1000
                        wa=np.divide(wa,divisor,out=np.full_like(wa,np.nan),where=divisor>0)
                        wb=np.divide(wb,divisor,out=np.full_like(wb,np.nan),where=divisor>0)
                    collected.append(pd.DataFrame({"bodyId":ids,"type":types,"model_index":z["idx"][sel],"quantity":q,"unit":unit,
                         "group":group,"run":row.run,"a":a,"b":b,"difference":a-b,"window_a":wa,"window_b":wb,
                         "window_difference":wa-wb,"window_source":pb.window_source.to_numpy(),"window_n_frames":pb.n_frames_win.to_numpy()}))
    whole=pd.concat(collected,ignore_index=True); rows=[]
    for (bid,typ,q),g in whole.groupby(["bodyId","type","quantity"],sort=False):
        o=g[g.group=="object"]; n=g[g.group=="null"]; cmp=common.compare(o.difference.tolist(),n.difference.tolist())
        r={"bodyId":bid,"type":typ,"quantity":q,"model_index":int(g.model_index.iloc[0]),"unit":g.unit.iloc[0],"unit_kind":"spiking",
           "window_start_s":0.,"window_end_s":12.,"stimulus_value":o.a.mean(),"control_value":o.b.mean(),
           "stimulus_minus_control":o.difference.mean(),"trial_sd":o.difference.std(ddof=1),"n_trials":len(o),
           "null_mean":n.difference.mean(),"null_sd":n.difference.std(ddof=1),"n_null":len(n),"z_vs_null":cmp["z"],
           "verdict":cmp["verdict"],"p":cmp["p"],"reading":"Per-body exploratory readout, no multiplicity claim; primary inference is per population with runs as replicates.",
           "paired_control_ids":"|".join(o.run+ex.ARM_B_SUFFIX),"null_reference_ids":"|".join(n.run),"control_ids":"|".join(n.run),
           "window_source":o.window_source.iloc[0],"window_n_frames":o.window_n_frames.mean(),"window_unit":g.unit.iloc[0]}
        for field,source in (("stimulus_value","window_a"),("control_value","window_b"),("stimulus_minus_control","window_difference")):
            r["window_"+field]=o[source].mean()
        r["window_trial_sd"]=o.window_difference.std(ddof=1); r["window_null_mean"]=n.window_difference.mean(); r["window_null_sd"]=n.window_difference.std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows),whole


def check_rect_raw(args):
    result=load(args.result); win=pd.DataFrame(result["tables"]["windows"]).set_index("bodyId")
    runs=rect.load_reduced_runs(args.out); lc_rows=pd.DataFrame(result["tables"]["per_run"])
    up_rows=pd.DataFrame(result["tables"]["upstream_runs"]); differences=[]; checked=0
    for r in runs.itertuples():
        with np.load(r.stem+"_reduced.npz",allow_pickle=False) as z:
            for block in ("lc","up"):
                if block=="lc":
                    with np.load(Path(args.lc)/(r.run+"_lc.npz"),allow_pickle=False) as f:
                        ids=f["body_ids"].astype(str); types=f["types"].astype(str)
                        signal=f["a__drive_mv"].astype(float)-f["b__drive_mv"]
                    az=z["az_deg"]; el=z["el_deg"]; counts=np.ones(len(az))
                    expected=lc_rows[lc_rows.run==r.run].set_index("type")
                else:
                    sel=z["up_idx"]; types=z["types"].astype(str)[sel]; ids=z["body_ids"][sel].astype(str)
                    signal=z["lat_dr_diff"].astype(float); az=z["lattice_az"]; el=np.full(len(az),z["el_deg"][0]); counts=z["lattice_count"]
                    expected=up_rows[up_rows.run==r.run].set_index("type")
                prior=win.reindex(ids)
                mask=((abs(az[:,None]-prior.az_deg.to_numpy()[None])<=(prior.width_deg.to_numpy()[None]+r.width)/2)
                      &(abs(el[:,None]-prior.el_deg.to_numpy()[None])<=(prior.height_deg.to_numpy()[None]+r.height)/2))
                mask[:,prior.window_source.isna().to_numpy()|(prior.window_source=="whole").to_numpy()]=True
                n=(mask*counts[:,None]).sum(0)
                values=np.divide((signal*mask).sum(0),n,out=np.full(len(ids),np.nan),where=n>0)
                for typ in (LC if block=="lc" else rect.UPSTREAM):
                    valid=(types==typ)&(n>0); vals=values[valid]
                    if block=="up": vals=abs(vals)
                    q="drive_median" if block=="lc" else "abs_drive_median"
                    actual=float(np.median(vals)) if len(vals) else np.nan; want=expected.loc[typ,q]
                    if not np.isclose(actual,want,atol=1e-8,rtol=1e-7,equal_nan=True):
                        differences.append({"run":r.run,"type":typ,"statistic":q,"actual":actual,"reported":want})
                    if int(valid.sum())!=expected.loc[typ,"n_bodies_windowed"]:
                        differences.append({"run":r.run,"type":typ,"field":"n_bodies_windowed"})
                    checked+=2
    save(args.json,{"schema":"flyverse.rectangle_independent_raw_check/1","provenance":result["provenance"],
        "generator":" ".join(sys.argv),"analysis_code_sha256":ex.sha256_file(__file__),"values_checked":checked,"differences":differences,
        "scope":"All LC primaries recomputed from original per-frame records; all upstream companions from the verified sweep-lattice sums; windowed body counts checked separately.",
        "verdict":"sound" if not differences else "unsound"})
    print(f"independent rectangle input check: {checked} values, {len(differences)} differences",flush=True)
    return int(bool(differences))


def rectangles_export(args):
    source=load(args.result); runs=rect.load_reduced_runs(args.out)
    win=pd.DataFrame(source["tables"]["windows"]); win["bodyId"]=win.bodyId.astype(str)
    records=[]
    for lobe in rect.LOBES:
        for width,height in rect.RECTS:
            for contrast in rect.CONTRASTS:
                obj=runs[(runs.lobe==lobe)&~runs.null&(runs.width==width)&(runs.height==height)&(runs.contrast_name==contrast)]
                nulls=runs[(runs.lobe==lobe)&runs.null&(runs.width==width)&(runs.height==height)]
                if not len(obj): continue
                if len(obj)<6 or len(nulls)<6: raise ValueError("incomplete rectangle group")
                if args.group and args.group!=rect.stem_of(lobe,width,height,contrast,0)[:-3]: continue
                first=obj.iloc[0]; prov=load(first.stem+"_prov.json"); dev=prov["execution"]["device_name"]
                res=common.Result.new("export",prov); res.tool_version="object_round3_export/1"
                res.files={"analysis_result":{"file":args.result,"sha256":ex.sha256_file(args.result)},"generator":" ".join(sys.argv)}
                rb,per_body=rectangle_readout(obj,nulls,win,width,height)
                rb["device_name"]=dev; rb["same_device_as_reference"]=True
                res.add_table("readout_per_body",rb)
                res.summary={"lobe":lobe,"width_deg":width,"height_deg":height,"contrast":contrast,
                             "predeclared":source["summary"]["predeclared"],"n_runs":len(obj),"n_null":len(nulls),
                             "window_rule":source["summary"]["windows"],"same_device_as_reference":True}
                res.replicates={"unit":"runs","n":len(obj),"stimulus":obj.run.tolist(),"null":nulls.run.tolist()}
                for name in ("per_run","upstream_runs","comparisons","preference","effective_contrast","effective_contrast_runs"):
                    frame=pd.DataFrame(source["tables"].get(name,[]))
                    for col,value in (("lobe",lobe),("width_deg",width),("height_deg",height),("contrast_name",contrast)):
                        if col in frame: frame=frame[frame[col]==value]
                    if "ladder" in frame:
                        mask=np.zeros(len(frame),bool)
                        for ladder,rung,in_ladder in (("hlad",height,width==rect.FIXED_WIDTH),("wlad",width,height==rect.FIXED_HEIGHT)):
                            if in_ladder:
                                m=(frame.ladder==ladder).to_numpy(copy=True)
                                if "rung" in frame: m &= (frame.rung==rung).to_numpy()
                                mask |= m
                        frame=frame[mask]
                    frame["device_name"]=dev; frame["same_device_as_reference"]=True
                    if len(frame): res.add_table(name,frame)
                paths=[Path(args.lc)/ (r+"_lc.npz") for r in obj.run]
                if any(not p.exists() for p in paths): raise FileNotFoundError("missing original LC frame records")
                tc=time_course(paths,"rectangle"); tc["same_device_as_reference"]=True
                ww=rb[(rb.window_n_frames>0)].drop_duplicates("bodyId")
                coverage=ww.groupby(["type","window_source"]).agg(n_bodies_windowed=("bodyId","nunique")).reset_index()
                res.summary["lc_time_course"]={"n_bodies":tc.bodyId.nunique(),"n_frames":tc.frame.nunique(),"n_runs":len(obj),
                   "records":[str(p) for p in paths],"scatter":"sample SD across independent runs; paired difference within each run"}
                res.files["lc_sources"]=[load(p.with_suffix(".json")) for p in paths]
                with np.load(first.stem+"_radiance.npz",allow_pickle=False) as z:
                    maps=e2.retina_maps(prov["retina"],z["col_az_el"],z["col_side"])
                    maps.pop("n_columns",None)
                    times=np.arange(len(z["radiance"]))*float(z["dt_s"])
                    radiance={"radiance":z["radiance"],"t_s":times,**maps,"sampling":np.asarray("in-loop synthetic column radiance")}
                    blank={"radiance":np.broadcast_to(z["blank"],z["radiance"].shape),"t_s":times,**maps,"sampling":np.asarray("matched blank in-loop radiance")}
                    tag=args.prefix+"-"+rect.stem_of(lobe,width,height,contrast,0)[:-3]
                    records.append(write_delivery(res,tag,args.export_root,[("lc_per_body_time_course",tc),("window_coverage",coverage)],
                                                   retina=radiance,retina_blank=blank,retina_in_loop=True))
    save(args.index,{"generator":" ".join(sys.argv),"directories":records,"schema":ex.EXPORT_SCHEMA})
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch"); p.add_argument("--target", default="house"); p.add_argument("--run", required=True)
    p.add_argument("--subdir", required=True); p.add_argument("--out", required=True); p.set_defaults(func=fetch)
    p = sub.add_parser("finalize"); p.add_argument("--indices", nargs="+", required=True)
    p.add_argument("--index", default="out/export/objr3_index.json"); p.add_argument("--spec-root", default="out/objr2c/spec")
    p.set_defaults(func=finalize)
    p = sub.add_parser("pack-rect"); p.add_argument("--raw", required=True); p.add_argument("--out", required=True)
    p.set_defaults(func=pack_rect)
    p=sub.add_parser("verify-lc"); p.add_argument("--out",required=True); p.add_argument("--expected",type=int,default=324); p.set_defaults(func=verify_lc)
    p = sub.add_parser("receipts"); p.add_argument("--target", required=True); p.add_argument("--log", required=True); p.add_argument("--out", required=True)
    p.set_defaults(func=receipts)
    p = sub.add_parser("compare")
    p.add_argument("--out", default="out/objr2c"); p.add_argument("--result", default="out/interp/objr2c/compare.json")
    p.add_argument("--prefix", default="objr3-r2compare"); p.add_argument("--index", default="out/export/objr3_r2compare_index.json")
    p.add_argument("--export-root", default="out/export"); p.add_argument("--transitions", default="out/interp/objr2c/spec_transitions.json")
    p.add_argument("--no-transitions", action="store_const", const=None, dest="transitions", help="omit transition sidecars for a sphere-only batch")
    p.add_argument("--group"); p.add_argument("--sizes", type=float, nargs="+"); p.set_defaults(func=compare_export)
    p = sub.add_parser("link-replications"); p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--index", default="out/export/objr3_index.json"); p.add_argument("--out", default="out/export/objr3_house_results")
    p.set_defaults(func=link_replications)
    p = sub.add_parser("check-contrast"); p.add_argument("--out", default="out/objr3rect")
    p.add_argument("--result", default="out/interp/objr3rect/rectangles.json"); p.add_argument("--json", default="out/objr3rect/skeptic_contrast.json")
    p.set_defaults(func=check_contrast)
    p = sub.add_parser("check-delivery"); p.add_argument("--index", default="out/export/objr3_index.json")
    p.add_argument("--json", default="out/export/objr3_skeptic.json"); p.set_defaults(func=check_delivery)
    p = sub.add_parser("check-preference"); p.add_argument("--result", required=True); p.add_argument("--json", required=True)
    p.set_defaults(func=check_preference)
    p = sub.add_parser("check-stats"); p.add_argument("--result", required=True); p.add_argument("--out"); p.add_argument("--json", required=True)
    p.set_defaults(func=check_stats)
    p = sub.add_parser("check-rf"); p.add_argument("--result", required=True); p.add_argument("--json", required=True)
    p.set_defaults(func=check_rf)
    p=sub.add_parser("rfmap"); p.add_argument("--results",nargs="+",required=True); p.add_argument("--export-root",default="out/export")
    p.add_argument("--prefix",default="objr3-rfmap"); p.add_argument("--index",default="out/export/objr3_rfmap_index.json"); p.set_defaults(func=rf_export)
    p=sub.add_parser("rectangles"); p.add_argument("--out",default="out/objr3rect"); p.add_argument("--result",default="out/interp/objr3rect/rectangles.json")
    p.add_argument("--lc",default="out/objr3rect_lc"); p.add_argument("--export-root",default="out/export"); p.add_argument("--prefix",default="objr3-rect")
    p.add_argument("--index",default="out/export/objr3_rectangles_index.json"); p.add_argument("--group"); p.set_defaults(func=rectangles_export)
    p=sub.add_parser("check-rect-raw"); p.add_argument("--out",default="out/objr3rect"); p.add_argument("--lc",default="out/objr3rect_lc")
    p.add_argument("--result",default="out/interp/objr3rect/rectangles.json"); p.add_argument("--json",default="out/objr3rect/skeptic_raw.json")
    p.set_defaults(func=check_rect_raw)
    return ap


if __name__ == "__main__":
    a = parser().parse_args(); raise SystemExit(a.func(a))
