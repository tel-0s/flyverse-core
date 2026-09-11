"""Torch backend selection: CUDA if present, else Apple MPS, else CPU.

Everything that creates a device or a sparse weight matrix goes through here so the choice is made once.
MPS caveats handled: no CSR sparse tensors (COO sparse @ dense works), no float64.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
import torch


def default_device() -> torch.device:
    """FLYVERSE_DEVICE if set (e.g. "cpu", "cuda:1", "mps"), else cuda > mps > cpu."""
    env = os.environ.get("FLYVERSE_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve(device=None) -> torch.device:
    return torch.device(device) if device is not None else default_device()


def sparse_matrix(D: sp.spmatrix, device, dtype=torch.float32) -> torch.Tensor:
    """scipy sparse -> torch sparse on `device`, for `M @ dense` products. CSR on cuda/cpu (fastest spmm);
    COO on mps, where torch has no compressed-sparse kernels."""
    device = torch.device(device)
    if device.type == "mps":
        C = D.tocoo()
        idx = torch.from_numpy(np.vstack([C.row, C.col]).astype(np.int64))
        vals = torch.from_numpy(np.asarray(C.data)).to(dtype)
        return torch.sparse_coo_tensor(idx, vals, size=C.shape).coalesce().to(device)
    C = D.tocsr()
    # cuSPARSE reads fewer index bytes with int32. Retain int64 for oversized
    # matrices and for CPU; the MPS path above is independent.
    itype = np.int32 if device.type == "cuda" and max(*C.shape, C.nnz) < 2**31 else np.int64
    return torch.sparse_csr_tensor(torch.from_numpy(C.indptr.astype(itype)), torch.from_numpy(C.indices.astype(itype)),
                                   torch.from_numpy(np.asarray(C.data)).to(dtype), size=C.shape).to(device)
