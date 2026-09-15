"""Reproduce backend acceptance: CPU anatomy/columns or a house-GPU room frame.

Every column check is named for the spec 2.5 validation it performs and every one is in `checks`, including the
strict per-cell rim membership of 2.5 (i), which does NOT hold: it is carried as a recorded expected failure
(EXPECTED_CHECKS) and asserted to be exactly that, rather than being reported one level up where the gate never
looked (connectome_backends_review B4). The DRA check therefore tests orientation by enrichment AND records that it
does not pretend every community DRA annotation is a correctly assigned rim cell.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse import connectome as cn, regions
from flyverse.backends.common import sha256
from flyverse.interp import common
from flyverse.motor import wing_groups
from flyverse.retina import build_retina
from flyverse.senses import Proprioception

# Spec 2.5 validations whose documented value is not True. 2.5 (i) strict: 100/126 non-putative community DRA labels
# sit in the flat height band and 118/126 within two rows of the curved dorsal envelope, so per-cell rim membership
# fails while the population orientation passes by a 24:1 margin (docs/audits/connectome_backends.md).
EXPECTED_CHECKS = {"i_dra_all_labels_on_rim": False}

# Spec 2.3: "make the mapping's cell counts part of the acceptance test". These are the MaleCNS *selections*
# Proprioception makes (LEG_NERVES-gated), not the labelled-cell counts the spec's prose quotes; the audit records
# both readings. BANC's are recorded beside them, not asserted: an independent release has its own populations.
SPEC_2_3_MALECNS_SELECTED = {"chordotonal": 615, "hair_plate": 113, "campaniform": 12, "haltere": 201}
# Spec 2.3 again: the wing motor groups must not be silently empty on a release that has those cells (B1).
WING_GROUPS_MALECNS = {"steer_L": 16, "steer_R": 16, "power": 24}


def wing_group_counts(c):
    wg = wing_groups(c)
    return {name: int(len(getattr(wg, name))) for name in ("steer_L", "steer_R", "power", "haltere", "ttm", "gf")}


def t4_offsets(c):
    n = c.neurons
    rows = {}
    for side in ("L", "R"):
        for t in ("T4a", "T4b", "T4c", "T4d"):
            post = c.select(type=t, somaSide=side)
            means = {}; ok = np.ones(len(post), bool)
            for pre_type in ("Mi1", "Mi4"):
                pre = c.select(type=pre_type, somaSide=side)
                pre = pre[n.hex1.notna().to_numpy()[pre]]
                w = abs(c.W[post][:, pre]); total = np.asarray(w.sum(axis=1)).ravel()
                ok &= total > 0
                means[pre_type] = w @ n.iloc[pre][["hex1", "hex2"]].to_numpy() / np.maximum(total[:, None], 1)
            offset = (means["Mi4"] - means["Mi1"])[ok].mean(axis=0)
            rows[f"{side}/{t}"] = dict(n=int(ok.sum()), offset=offset.tolist())
    return rows


def columns(c, male):
    r = build_retina(c); n = c.neurons
    labels_path = cn.data_directory("fafb") / "labels.csv.gz"
    labels = pd.read_csv(labels_path, usecols=["root_id", "label"])
    labels = labels[labels.root_id.isin(n.loc[n.type.isin(cn.PHOTORECEPTOR_TYPES), "bodyId"])
        & labels.label.str.contains("DRA|dorsal rim", case=False, na=False)
        & ~labels.label.str.contains("putative|or |difficult", case=False, na=False)]
    dra = {}
    for side in ("L", "R"):
        grid = n[n.hex_side == side][["hex1", "hex2"]].dropna().drop_duplicates()
        boundary = float(np.quantile(grid.sum(axis=1), .8))
        cells = n[n.bodyId.isin(labels.root_id) & n.hex_side.eq(side) & n.hex1.notna()]
        height = cells.hex1 + cells.hex2
        # A curved dorsal rim extends below a single whole-eye height cutoff.
        # Measure depth below the dorsal envelope at the same anterior coordinate
        # as well. One lattice row at fixed (hex1-hex2) changes height by two.
        envelope = pd.DataFrame(dict(anterior=grid.hex1 - grid.hex2, height=grid.hex1 + grid.hex2)).groupby("anterior").height.max()
        depth = ((cells.hex1 - cells.hex2).map(envelope) - height) / 2
        dra[side] = dict(n=len(cells), dorsal_boundary_q80=boundary, median_height=float(height.median()),
                        rim_fraction=float((height >= boundary).mean()),
                        off_rim_body_ids=cells.loc[height < boundary, "bodyId"].astype(str).tolist(),
                        dorsal_envelope=dict(median_depth_rows=float(depth.median()), max_depth_rows=float(depth.max()),
                            within_two_rows_fraction=float((depth <= 2).mean()),
                            deeper_cells=[dict(bodyId=str(row.bodyId), depth_rows=float(depth.loc[i]), hex_source=row.hex_source)
                                          for i, row in cells.loc[depth > 2].iterrows()]),
                        orientation_pass=bool(height.median() >= boundary and (height >= boundary).mean() > .5))
    a, b = t4_offsets(c), t4_offsets(male)
    for key, row in a.items():
        x, y = np.asarray(row["offset"]), np.asarray(b[key]["offset"])
        row.update(malecns_offset=y.tolist(), cosine=float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
    # Compare actual viewing directions at hex coordinates present in both eyes.
    eye = {}
    for side in ("L", "R"):
        mask = r.col_side == side
        eye[side] = {tuple(xy): v for xy, v in zip(r.col_hex[mask], r.col_dir[mask])}
    shared = sorted(set(eye["L"]) & set(eye["R"]))
    dots = [np.dot(eye["L"][key], eye["R"][key] * [1, -1, 1]) for key in shared]
    error = np.degrees(np.arccos(np.clip(dots, -1, 1)))
    all_on_rim = all(not v["dorsal_envelope"]["deeper_cells"] for v in dra.values())
    return dict(n_columns=r.n_columns, coverage=r.coverage(), per_side={s: int((r.col_side == s).sum()) for s in ("L", "R")},
        dra=dict(source=str(labels_path.name), sha256=sha256(labels_path), selection="non-putative DRA community labels on photoreceptors",
                 sides=dra, all_labels_on_rim=all_on_rim),
        mirror=dict(shared_columns=len(shared), max_error_deg=float(error.max()), mean_error_deg=float(error.mean())),
        # One entry per spec 2.5 validation, named for it. 2.5 (i) is two questions and both are in the gate: the
        # population-orientation one it passes, and the strict per-cell one it does not (EXPECTED_CHECKS).
        t4=a, checks=dict(i_dra_population_orientation=all(v["orientation_pass"] for v in dra.values()),
                         i_dra_all_labels_on_rim=bool(all_on_rim),
                         ii_lr_mirror=bool(error.max() < r.geometry.interommatidial_deg),
                         iii_t4_direction=all(v["cosine"] > .95 for v in a.values()),
                         iv_column_count=1500 <= r.n_columns <= 1650))


def cpu():
    from flyverse.brain import Brain
    from flyverse.fly import FlyBrain
    male = cn.load(verbose=False)
    proprio = Proprioception(male).counts()
    report = {"malecns_fingerprint": common.connectome_fingerprint(male), "datasets": {},
              # Spec 2.3 reference selections, recorded so the gate can assert them rather than only print them.
              "malecns_selections": dict(proprioception=proprio,
                                         proprioception_selected={k: v["n"] for k, v in proprio.items()},
                                         expected_proprioception_selected=SPEC_2_3_MALECNS_SELECTED,
                                         wing_groups=wing_group_counts(male), expected_wing_groups=WING_GROUPS_MALECNS)}
    for dataset in ("fafb", "banc"):
        c = cn.load(dataset=dataset, verbose=False)
        post = c.select(type="DNa02")[0]
        row = c.W[post]
        idx = np.unique(np.r_[post, row.indices[np.argsort(-abs(row.data))[:199]]])
        sub = c.subset(idx)
        b = Brain(sub, device="cpu"); b.step(100)
        rs = cn.receptor_signs(c)
        d = dict(n=c.n, nnz=c.W.nnz, nt_counts=c.neurons.nt.value_counts().to_dict(),
            cpu_smoke=dict(n=sub.n, ms=100, finite=bool(np.isfinite(b.v.cpu().numpy()).all())),
            receptor_tiers={t: int((rs.tier == i).sum()) for i, t in enumerate(cn.RECEPTOR_TIERS)},
            regions=pd.Series(regions.labels(c)).value_counts().to_dict(),
            provenance=common.provenance(c, device="cpu", stimulus={"protocol": "backend_acceptance", "params": {}, "control": None}))
        if dataset == "fafb":
            d["columns"] = columns(c, male)
        else:
            d["proprioception"] = Proprioception(c).counts()
            d["proprioception_selected"] = {k: v["n"] for k, v in d["proprioception"].items()}
            d["motor_subclasses"] = c.neurons.loc[c.neurons.superclass.eq("vnc_motor"), "subclass"].value_counts().to_dict()
            # B1: the selected wing groups, not only the subclass census that hid their emptiness.
            d["wing_groups"] = wing_group_counts(c)
            fb = FlyBrain(sub, device="cpu", optic=None); fb.step(10)
            d["optic_none"] = fb.optic is None
        report["datasets"][dataset] = d
    report["cache_md5"] = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in cn.CACHE_DIR.iterdir() if p.is_file()}
    return report


def gpu():
    import torch
    from flyverse.fly import FlyBrain
    from flyverse import world
    assert torch.cuda.is_available(), "run --gpu on the house cluster"
    c = cn.load(dataset="fafb", verbose=False)
    fb = FlyBrain(c, device="cuda", cuda_kernels=True, cuda_sparse="warp")
    scene, info = world.make_room(0, "all")
    # Use the same world ray tracer as the room. Body pose is fixed for this one frame.
    rays, weights = fb.retina.ray_directions()
    scene.device = "cuda"
    dirs = torch.as_tensor(rays.reshape(-1, 3), dtype=torch.float32, device="cuda")
    origin = torch.tensor([-.5, .05, info["table_top_z"] + .002], dtype=torch.float32, device="cuda").expand_as(dirs)
    light = scene.trace(origin, dirs).reshape(fb.retina.n_columns, -1, 4)
    radiance = (light * torch.as_tensor(weights, device="cuda", dtype=torch.float32)[None, :, None]).sum(1)
    fb.vision(radiance); fb.step(10); torch.cuda.synchronize()
    return dict(provenance=common.provenance(c, fb=fb, device="cuda", seeds=[0], env_seeds=[0], batch=1,
        stimulus={"protocol": "room_frame", "params": {"ms": 10}, "control": None}),
        n_columns=fb.retina.n_columns, rate_units=len(fb.optic.rate_idx),
        finite=bool(torch.isfinite(fb.brain.v).all() and torch.isfinite(fb.optic.v).all()), ms=10)


def compare_caches(baseline_root, rebuild=False):
    """Audit a cache update against a separate retained baseline; never overwrite it."""
    from flyverse.brain import LIFParams, _shaped_weights
    baseline_root = Path(baseline_root).resolve()
    cases = [("malecns", "threshold"), ("fafb", "threshold"), ("banc", "threshold"), ("fafb", "no_threshold")]
    directories = []
    for dataset, edges in cases:
        relative = Path() if dataset == "malecns" else Path(dataset)
        if edges != "threshold":
            relative /= edges
        old_dir = baseline_root / relative
        new_dir = cn.default_cache_directory(dataset, edges).resolve()
        if old_dir == new_dir:
            raise ValueError("comparison requires separate baseline and current caches")
        if not (old_dir / "W_post_pre.npz").exists():
            raise ValueError(f"missing baseline cache: {old_dir}")
        directories.append((dataset, edges, old_dir, new_dir))
    report = {}
    for dataset, edges, old_dir, new_dir in directories:
        old = cn.load(old_dir, verbose=False)
        new = cn.load(new_dir, dataset=dataset, edges=edges,
                      rebuild=rebuild and dataset != "malecns", verbose=False)
        same_matrix = old.W.shape == new.W.shape and all(
            getattr(old.W, k).dtype == getattr(new.W, k).dtype and np.array_equal(getattr(old.W, k), getattr(new.W, k))
            for k in ("data", "indices", "indptr"))
        changed_columns = {k: int((~(old.neurons[k].eq(new.neurons[k]) |
                                    (old.neurons[k].isna() & new.neurons[k].isna()))).sum())
                           for k in old.neurons if not old.neurons[k].equals(new.neurons[k])}
        row = dict(n=new.n, nnz=new.W.nnz, same_matrix=same_matrix, changed_columns=changed_columns,
            same_column_order=old.neurons.columns.tolist() == new.neurons.columns.tolist(),
            same_cache_files={name: sha256(old_dir / name) == sha256(new_dir / name)
                              for name in ("W_post_pre.npz", "sign0_counts.npz")},
            provenance={label: common.provenance(c, device="cpu", stimulus={"protocol": "cache_update_comparison",
                         "params": {"edges": edges}, "control": str(baseline_root)}) for label, c in (("before", old), ("after", new))})
        assert same_matrix and row["same_column_order"] and all(row["same_cache_files"].values())
        if dataset == "malecns":
            assert old.neurons.equals(new.neurons)
            before, after = (common.connectome_fingerprint(c) for c in (old, new))
            assert {k: v for k, v in before.items() if k != "cache_dir"} == {k: v for k, v in after.items() if k != "cache_dir"}
        elif edges == "threshold":
            rs_old, rs_new = (cn.receptor_signs(c) for c in (old, new))
            row["receptor_changed_entries"] = {k: int(np.count_nonzero(getattr(rs_old, k) != getattr(rs_new, k)))
                for k in ("fast_sign", "fast_gain", "slow_sign", "slow_gain", "slow_class", "tier")}
            w_old, w_new = _shaped_weights(old, LIFParams(), rs_old), _shaped_weights(new, LIFParams(), rs_new)
            assert np.array_equal(w_old.indices, w_new.indices) and np.array_equal(w_old.indptr, w_new.indptr)
            row["default_shaped_weight_changes"] = int(np.count_nonzero(w_old.data != w_new.data))
        report[f"{dataset}/{edges}"] = row
        print(dataset, edges, "changed columns:", changed_columns,
              "receptor:", row.get("receptor_changed_entries"), "shaped weights:", row.get("default_shaped_weight_changes"))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--compare-cache", type=Path, help="compare against a separate cache root, including FAFB no_threshold")
    ap.add_argument("--rebuild", action="store_true", help="recompile current female caches before --compare-cache")
    ap.add_argument("--out", default="out/connectome_backends/acceptance.json")
    args = ap.parse_args()
    if args.gpu and args.compare_cache or args.rebuild and not args.compare_cache:
        ap.error("--compare-cache is CPU-only; --rebuild requires --compare-cache")
    report = compare_caches(args.compare_cache, args.rebuild) if args.compare_cache else gpu() if args.gpu else cpu()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(common.to_jsonable(report), indent=2), encoding="utf-8")
    print(f"wrote {out}")
    if args.compare_cache:
        pass  # comparison assertions run before the report is returned
    elif args.gpu:
        assert report["finite"]
    else:
        assert all(d["cpu_smoke"]["finite"] for d in report["datasets"].values())
        checks = report["datasets"]["fafb"]["columns"]["checks"]
        for name, value in checks.items():
            want = EXPECTED_CHECKS.get(name, True)
            print(f"  {name}: {value}" + ("" if want else f"  (EXPECTED FAILURE, documented value {want})"))
            assert value == want, f"spec 2.5 check {name} is {value}, documented value {want}"
        selections = report["malecns_selections"]
        print("  MaleCNS spec 2.3 selections:", selections["proprioception_selected"], selections["wing_groups"])
        print("  BANC spec 2.3 selections:", report["datasets"]["banc"]["proprioception_selected"],
              report["datasets"]["banc"]["wing_groups"])
        assert selections["proprioception_selected"] == SPEC_2_3_MALECNS_SELECTED
        assert {k: selections["wing_groups"][k] for k in WING_GROUPS_MALECNS} == WING_GROUPS_MALECNS
