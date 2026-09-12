"""Per-op determinism probe for the round-4 `slowdet` finding (docs/audits/slow_term.md section 8.3): under
`torch.use_deterministic_algorithms(True)` the walk / motion sections are still not bit-reproducible and torch
raises nothing.  Section 8.5 names the candidates -- the sparse CSR x dense products behind `brain.Brain._transmit`
and `optic.OpticLobe._mv`, and the ray tracer.  This probe runs those ops alone, on the real connectome matrix,
with and without the flag, and reports whether repeated evaluations are bit-identical.

GPU job (cluster only):

    python scripts/cluster_run.py --name r4-detops --minutes 15 \
      "python -c 'import torch; assert torch.cuda.is_available()' && CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/skeptic_det_ops.py --deterministic > out/r4_detops_det.txt; cat out/r4_detops_det.txt" \
      "python -c 'import torch; assert torch.cuda.is_available()' && python scripts/skeptic_det_ops.py > out/r4_detops_nd.txt; cat out/r4_detops_nd.txt" --fetch out/
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from flyverse import connectome as cn  # noqa: E402


def h(t: torch.Tensor) -> str:
    return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    if a.deterministic:
        if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    print(f"deterministic={a.deterministic} torch={torch.__version__} "
          f"cublas={os.environ.get('CUBLAS_WORKSPACE_CONFIG')} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = cn.load(verbose=False)
    W = c.W.tocsr()
    print(f"W: n={c.n} nnz={W.nnz}", flush=True)
    Wt = torch.sparse_csr_tensor(torch.from_numpy(W.indptr.astype(np.int32)).to(dev),
                                 torch.from_numpy(W.indices.astype(np.int32)).to(dev),
                                 torch.from_numpy(W.data.astype(np.float32)).to(dev),
                                 size=(c.n, c.n))
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.rand((1, c.n), generator=g, device=dev, dtype=torch.float32)

    def run(name, fn):
        hs = []
        raised = None
        for _ in range(a.reps):
            try:
                hs.append(h(fn()))
            except RuntimeError as e:            # what use_deterministic_algorithms(True) raises
                raised = str(e).splitlines()[0]
                break
        if raised:
            print(f"{name:28s} RAISED: {raised}", flush=True)
        else:
            u = sorted(set(hs))
            print(f"{name:28s} reps={len(hs)} distinct={len(u)} "
                  f"{'BIT-IDENTICAL' if len(u) == 1 else 'NON-DETERMINISTIC ' + str(u[:4])}", flush=True)

    run("csr @ dense (W @ x.T).T", lambda: (Wt @ x.T).T)
    run("csr @ dense batch=8", lambda: (Wt @ torch.rand((8, c.n), generator=g, device=dev).T).T * 0
        + (Wt @ x.repeat(8, 1).T).T)
    d = torch.rand((2048, 2048), generator=g, device=dev)
    run("dense @ dense 2048", lambda: d @ d)
    idx = torch.randint(0, c.n, (1 << 20,), generator=g, device=dev)
    src = torch.rand((1 << 20,), generator=g, device=dev)
    run("index_add_ 1M", lambda: torch.zeros(c.n, device=dev).index_add_(0, idx, src))


if __name__ == "__main__":
    main()
