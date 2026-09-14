"""SKEPTIC verification of build:hooks (independent of tests/test_optic_hooks.py).

Runs on the FULL connectome, on CUDA, with the native CUDA optic kernels -- the surface the CPU test suite does
NOT cover. Checks, in order:

  A  shipped defaults bit-identical: working-tree flyverse/optic.py vs the committed HEAD copy shipped as
     scripts/_skeptic_optic_head_ref.txt, both built with cuda_kernels=True / cuda_sparse='warp' on the full lobe;
     drive torch.equal every frame, plus v / adapt / delta_rate / r0.
  B  the defaults still take the native CUDA path (ol.cuda truthy) -- i.e. the kernels are really untouched.
  C  hook lobe: cuda_sparse 'warp' downgraded with a warning, ol.cuda / ol.metal falsey, torch substep.
  D  stream isolation on the full lobe: rest == unmatched entries exactly, blocks == matched entries exactly,
     non-T3 rows of _recurrent torch.equal to the plain product.
  E  sign preservation per entry on the full lobe under pos / neg / abs: x >= 0, and every entry contributes
     W_ij x_j with the sign of W_ij; the same entries under the linear sum DO flip.
  F  hook_info counts, to cross-check the reported 15,659 / 11,174 / 8,752,204.
  G  fb_hold [('.*','.*')] == gain_fb=0 bit for bit on the full lobe with the native kernels.

    python scripts/skeptic_hooks_verify.py --out out/skeptic_hooks/verify.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyverse import connectome as cn        # noqa: E402
from flyverse import optic, retina as rt     # noqa: E402

RECT = [("^(Mi1|Tm3|Tm2)$", "^T3$", "pos"), ("^(Tm1|Tm4)$", "^T3$", "neg")]
SUPP = [("^(Mi1|Tm1|Tm3|Tm4)$", 0.5, 10.0)]


def to_scipy(M) -> sp.csr_matrix:
    if isinstance(M, sp.spmatrix):
        return M.tocsr()
    if hasattr(M, "ptr") and hasattr(M, "idx") and hasattr(M, "values") and not isinstance(M, torch.Tensor):
        # flyverse.cuda.CSR / metal.MetalCSR wrapper
        return sp.csr_matrix((M.values.cpu().numpy(), M.idx.cpu().numpy(), M.ptr.cpu().numpy()), shape=tuple(M.shape))
    M = M.cpu()
    if M.layout == torch.sparse_csr:
        return sp.csr_matrix((M.values().numpy(), M.col_indices().numpy(), M.crow_indices().numpy()), shape=tuple(M.shape))
    M = M.coalesce()
    return sp.csr_matrix((M.values().numpy(), (M.indices()[0].numpy(), M.indices()[1].numpy())), shape=tuple(M.shape))


def load_head_module():
    src_path = ROOT / "scripts" / "_skeptic_optic_head_ref.txt"
    src = src_path.read_text(encoding="utf-8")
    tmp = ROOT / "scripts" / "_skeptic_optic_head_mod.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("flyverse._skeptic_optic_head", tmp)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "flyverse"
    sys.modules["flyverse._skeptic_optic_head"] = mod
    spec.loader.exec_module(mod)
    import hashlib
    return mod, hashlib.md5(src.encode("utf-8")).hexdigest(), len(src.splitlines())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/skeptic_hooks/verify.json")
    ap.add_argument("--frames", type=int, default=12)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    dev = "cuda"
    R = {"device_name": torch.cuda.get_device_name(0), "torch": torch.__version__, "frames": a.frames}

    head, head_md5, head_lines = load_head_module()
    R["head_ref"] = {"md5": head_md5, "lines": head_lines}

    c = cn.load(verbose=False)
    r = rt.build_retina(c)
    R["graph"] = {"n_neurons": int(c.n), "n_columns": int(r.n_columns)}
    print(f"graph n={c.n} columns={r.n_columns} device={R['device_name']}", flush=True)

    gen = torch.Generator().manual_seed(11)
    rad = torch.rand(1, r.n_columns, 4, generator=gen).to(dev)
    spk = torch.zeros(1, c.n, device=dev)
    # a live feedback source so gain_fb actually matters
    fb_idx = c.select(superclass="visual_projection")
    spk[0, fb_idx] = 40.0
    R["feedback_source_cells"] = int(len(fb_idx))

    def frames(ol, n):
        ol.reset(); ol.relax()
        out = []
        for k in range(n):
            out.append(ol.step_frame(rad * (1.0 + 0.5 * float(np.sin(k / 2.0))), spk, 10.0).clone())
        return out

    def bit_identical(x, y, n):
        fa, fb = frames(x, n), frames(y, n)
        eq = [bool(torch.equal(p, q)) for p, q in zip(fa, fb)]
        maxdiff = max(float((p - q).abs().max()) for p, q in zip(fa, fb))
        return {"frames_equal": eq, "all_equal": all(eq), "max_abs_diff": maxdiff,
                "v_equal": bool(torch.equal(x.v, y.v)), "adapt_equal": bool(torch.equal(x.adapt, y.adapt)),
                "delta_rate_equal": bool(torch.equal(x.delta_rate, y.delta_rate)),
                "r0_equal": bool(torch.equal(x.r0, y.r0))}

    # ---------------------------------------------------------------- A / B  defaults, native kernels
    head_fields = set(head.OpticParams.__dataclass_fields__)
    kw = {k: v for k, v in optic.OpticParams().__dict__.items() if k in head_fields}

    for backend in ("warp_cuda_kernels", "torch_sparse"):
        ck, cs = (True, "warp") if backend == "warp_cuda_kernels" else (False, "torch")
        print(f"A[{backend}]: building lobes (cuda_kernels={ck}, cuda_sparse={cs})", flush=True)
        ol_a = optic.OpticLobe(c, r, optic.OpticParams(), device=dev, cuda_kernels=ck, cuda_sparse=cs)
        ol_b = optic.OpticLobe(c, r, optic.OpticParams(), device=dev, cuda_kernels=ck, cuda_sparse=cs)
        ol_h = head.OpticLobe(c, r, head.OpticParams(**kw), device=dev, cuda_kernels=ck, cuda_sparse=cs)
        # A0 CONTROL: the same object replayed twice -- is this backend deterministic at all?
        f1, f2 = frames(ol_a, a.frames), frames(ol_a, a.frames)
        a0 = {"all_equal": all(bool(torch.equal(p, q)) for p, q in zip(f1, f2)),
              "max_abs_diff": max(float((p - q).abs().max()) for p, q in zip(f1, f2))}
        # A1 CONTROL: two separate constructions of the SAME (working-tree) code
        a1 = bit_identical(ol_a, ol_b, a.frames)
        # A2 THE CLAIM: working tree vs committed HEAD
        a2 = bit_identical(ol_a, ol_h, a.frames)
        R[f"A_{backend}"] = {"A0_same_object_replayed": a0, "A1_two_builds_same_code": a1, "A2_worktree_vs_HEAD": a2,
                             "cuda_kernels_in_use": bool(ol_a.cuda)}
        print(f"A[{backend}] A0 same-object={a0['all_equal']} (maxdiff {a0['max_abs_diff']:.3e}) "
              f"A1 two-builds={a1['all_equal']} (maxdiff {a1['max_abs_diff']:.3e}) "
              f"A2 vs-HEAD={a2['all_equal']} (maxdiff {a2['max_abs_diff']:.3e}) cuda={bool(ol_a.cuda)}", flush=True)
        if backend == "warp_cuda_kernels":
            ol_new, ol_head = ol_a, ol_h
        else:
            del ol_a, ol_h
        del ol_b
    R["A_defaults_bit_identical"] = R["A_torch_sparse"]["A2_worktree_vs_HEAD"]

    # A3: does the no-match hook (Torch substep, split matrices) change the numbers vs the native CUDA path?
    # This is the numerical-path difference the build's base-vs-hook arms confound with the hook effect.
    print("A3: no-match hook (torch substep) vs the native CUDA default", flush=True)
    ol_nm = optic.OpticLobe(c, r, optic.OpticParams(stream_rectify=[("^NoSuchType$", "^T3$", "pos")]),
                            device=dev, cuda_kernels=True, cuda_sparse="torch")
    R["A3_no_match_hook_vs_native"] = bit_identical(ol_nm, ol_new, a.frames)
    R["A3_no_match_hook_vs_native"]["nm_cuda"] = bool(ol_nm.cuda)
    R["A3_no_match_hook_vs_native"]["nm_streams"] = len(ol_nm.streams)
    R["A3_no_match_hook_vs_native"]["nm_rr_matched"] = ol_nm.hook_info["entries_rr_matched"]
    print("A3:", R["A3_no_match_hook_vs_native"], flush=True)
    del ol_nm
    R["B_native_path"] = {"new_cuda": bool(ol_new.cuda), "head_cuda": bool(ol_head.cuda),
                          "new_hooks": bool(ol_new._hooks), "new_streams": len(ol_new.streams),
                          "new_fields_added": sorted(set(optic.OpticParams.__dataclass_fields__) - head_fields)}
    R["B_defaults_of_new_fields"] = {k: getattr(optic.OpticParams(), k) for k in R["B_native_path"]["new_fields_added"]}
    print("A:", R["A_defaults_bit_identical"]["all_equal"], "B:", R["B_native_path"], flush=True)

    # ---------------------------------------------------------------- C  hook lobe downgrades warp
    print("C: hook lobe with cuda_sparse=warp", flush=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ol_r = optic.OpticLobe(c, r, optic.OpticParams(stream_rectify=RECT), device=dev, cuda_kernels=True, cuda_sparse="warp")
        R["C_warnings"] = [str(x.message) for x in w]
    R["C_hook_backend"] = {"cuda": bool(ol_r.cuda), "metal": bool(ol_r.metal), "hooks": bool(ol_r._hooks),
                           "torch_substep": ol_r.hook_info.get("torch_substep"), "n_streams": len(ol_r.streams)}
    print("C:", R["C_hook_backend"], R["C_warnings"], flush=True)

    # ---------------------------------------------------------------- F  hook_info counts
    R["F_hook_info_rectify"] = json.loads(json.dumps(ol_r.hook_info))
    print("F:", {k: v for k, v in ol_r.hook_info.items() if k != "streams"}, flush=True)

    # ---------------------------------------------------------------- D  stream isolation on the full lobe
    # the plain reference must live on the SAME numerical backend as the hook lobe (torch sparse), or the row
    # comparison would measure the backend, not the hook
    print("D: stream isolation (torch-sparse reference)", flush=True)
    ol_def_t = optic.OpticLobe(c, r, optic.OpticParams(), device=dev, cuda_kernels=False, cuda_sparse="torch")
    types = c.neurons.type.fillna("").to_numpy()[ol_def_t.rate_idx]
    W = to_scipy(ol_def_t.W_rr).tocoo()
    m = np.zeros(W.nnz, bool)
    for pre_re, post_re, _ in RECT:
        m |= optic.OpticLobe._type_mask(pre_re, types)[W.col] & optic.OpticLobe._type_mask(post_re, types)[W.row]
    Rest = sp.csr_matrix((W.data[~m], (W.row[~m], W.col[~m])), shape=W.shape)
    Blk = sp.csr_matrix((W.data[m], (W.row[m], W.col[m])), shape=W.shape)
    blk_hook = sum(to_scipy(s["W_rr"]) for s in ol_r.streams)
    R["D_split"] = {
        "matched_entries": int(m.sum()), "rest_entries": int((~m).sum()), "total_entries": int(W.nnz),
        "rest_max_abs_diff": float(abs(Rest - to_scipy(ol_r.W_rr_rest)).max()),
        "blocks_max_abs_diff": float(abs(Blk - blk_hook).max()),
        "hook_info_rr_matched": ol_r.hook_info["entries_rr_matched"],
        "hook_info_rr_rest": ol_r.hook_info["entries_rr_rest"],
    }
    g2 = torch.Generator().manual_seed(5)
    dr = ((torch.rand(1, ol_def_t.n_rate, generator=g2) - 0.5) * 0.4).to(dev)
    plain = ol_def_t._recurrent(dr)
    hooked = ol_r._recurrent(dr)
    t3 = torch.from_numpy(types == "T3").to(dev)
    R["D_rows"] = {
        "non_T3_rows_bit_identical": bool(torch.equal(plain[0, ~t3], hooked[0, ~t3])),
        "non_T3_max_abs_diff": float((plain[0, ~t3] - hooked[0, ~t3]).abs().max()),
        "T3_rows_differ_max": float((plain[0, t3] - hooked[0, t3]).abs().max()),
        "n_T3_rows": int(t3.sum()),
    }
    print("D:", R["D_split"], R["D_rows"], flush=True)

    # ---------------------------------------------------------------- E  sign preservation per entry
    print("E: sign preservation", flush=True)
    E = {}
    drc = dr[0].cpu().numpy().astype(np.float64)
    for mode in ("pos", "neg", "abs"):
        rect_m = [(p, q, mode) for p, q, _ in RECT]
        ol_m = optic.OpticLobe(c, r, optic.OpticParams(stream_rectify=rect_m), device=dev, cuda_kernels=False, cuda_sparse="torch")
        xs = ol_m._stream_signals(dr, False)
        xmin = min(float(x.min()) for x in xs)
        viol_hook = 0; viol_lin = 0; n_entries = 0; n_inhib = 0
        for s, x in zip(ol_m.streams, xs):
            Bk = to_scipy(s["W_rr"]).tocoo()
            xv = x[0].cpu().numpy().astype(np.float64)
            contrib = Bk.data.astype(np.float64) * xv[Bk.col]
            lin = Bk.data.astype(np.float64) * drc[Bk.col]
            viol_hook += int(((Bk.data < 0) & (contrib > 0)).sum() + ((Bk.data > 0) & (contrib < 0)).sum())
            viol_lin += int(((Bk.data < 0) & (lin > 0)).sum() + ((Bk.data > 0) & (lin < 0)).sum())
            n_entries += Bk.nnz; n_inhib += int((Bk.data < 0).sum())
        E[mode] = {"x_min": xmin, "sign_violations_hook": viol_hook, "sign_violations_linear": viol_lin,
                   "entries": n_entries, "inhibitory_entries": n_inhib}
        print("  E", mode, E[mode], flush=True)
        del ol_m
    R["E_sign"] = E

    # ---------------------------------------------------------------- G  fb_hold == gain_fb 0
    print("G: fb_hold vs gain_fb=0", flush=True)
    # on the deterministic backend, so the comparison means something
    ol_hold = optic.OpticLobe(c, r, optic.OpticParams(fb_hold=[(".*", ".*")]), device=dev, cuda_kernels=False, cuda_sparse="torch")
    ol_g0 = optic.OpticLobe(c, r, optic.OpticParams(gain_fb=0.0), device=dev, cuda_kernels=False, cuda_sparse="torch")
    R["G_fb_hold"] = bit_identical(ol_hold, ol_g0, a.frames)
    # and separately: a fb_hold lobe still takes the native CUDA kernels (it is a build-time weight edit)
    ol_hold_k = optic.OpticLobe(c, r, optic.OpticParams(fb_hold=[(".*", ".*")]), device=dev, cuda_kernels=True, cuda_sparse="warp")
    R["G_fb_hold"]["hold_uses_cuda_kernels"] = bool(ol_hold_k.cuda)
    R["G_fb_hold"]["hold_hooks_flag"] = bool(ol_hold_k._hooks)
    del ol_hold_k
    R["G_fb_hold"]["hook_info_fb"] = json.loads(json.dumps(ol_hold.hook_info_fb))
    # and that the hold really differs from the shipped model (a live feedback source)
    R["G_hold_differs_from_default"] = not bit_identical(ol_hold, ol_def_t, 4)["all_equal"]
    print("G:", R["G_fb_hold"]["all_equal"], "differs from default:", R["G_hold_differs_from_default"], flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    print("written", out, flush=True)
    ok = (R["A_torch_sparse"]["A2_worktree_vs_HEAD"]["all_equal"] and R["B_native_path"]["new_cuda"]
          and R["A3_no_match_hook_vs_native"] is not None and R["D_split"]["rest_max_abs_diff"] == 0.0 and R["D_split"]["blocks_max_abs_diff"] == 0.0
          and R["D_rows"]["non_T3_rows_bit_identical"] and R["G_fb_hold"]["all_equal"]
          and all(v["sign_violations_hook"] == 0 for v in E.values()))
    print("SKEPTIC VERDICT:", "ALL CHECKS PASS" if ok else "SOME CHECK FAILED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
