"""Optional CUDA kernels, compiled with nvcc and called on Torch's current stream.

No Python/Torch C++ ABI dependency: the small shared library accepts raw device pointers.
Tensor validation, storage offsets, stream selection and ownership stay on the Python side.
Enable explicitly or with FLYVERSE_CUDA_KERNELS=1; CPU and Metal paths are independent.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import numpy as np
import scipy.sparse as sp
import torch

_LIBS = {}
_LOCK = threading.Lock()


def use(device, explicit=None):
    enabled = os.environ.get("FLYVERSE_CUDA_KERNELS", "0") == "1" if explicit is None else explicit
    if enabled and torch.device(device).type != "cuda":
        if explicit:
            raise ValueError("CUDA kernels require a CUDA device")
        return False
    if enabled:
        lib(device)  # compilation and runtime initialization must precede graph capture
    return bool(enabled)


def lib(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA kernels require a CUDA device")
    capability = torch.cuda.get_device_capability(device)
    with _LOCK:
        if capability in _LIBS:
            return _LIBS[capability]
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("CUDA kernels need nvcc and a host C++ compiler; disable --cuda-kernels to use Torch")
        source = Path(__file__).with_name("kernels") / "neural.cu"
        version = subprocess.check_output([nvcc, "--version"])
        flags = ["-Xcompiler", "/MD"] if sys.platform == "win32" else ["-Xcompiler", "-fPIC"]
        compile_flags = ["--shared", "-O3", "--fmad=false", "-std=c++17",
                         f"-arch=sm_{capability[0]}{capability[1]}", *flags]
        digest = hashlib.sha256(source.read_bytes() + version + repr(compile_flags).encode()).hexdigest()[:20]
        root = Path(os.environ.get("FLYVERSE_CUDA_CACHE", Path.home() / ".cache" / "flyverse" / "cuda"))
        folder = root / digest
        folder.mkdir(parents=True, exist_ok=True)
        suffix = ".dll" if sys.platform == "win32" else ".so"
        binary = folder / ("neural" + suffix)
        if not binary.exists():
            temporary = folder / (f"neural_{os.getpid()}" + suffix)
            command = [nvcc, *compile_flags, str(source), "-o", str(temporary)]
            result = subprocess.run(command, cwd=folder, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("CUDA kernel compilation failed:\n" + result.stdout + result.stderr)
            try:
                temporary.replace(binary)
            except OSError:
                if not binary.exists():
                    raise
                temporary.unlink(missing_ok=True)
        library = ct.CDLL(str(binary))
        P, I, F, S = ct.POINTER(ct.c_void_p), ct.c_int, ct.c_float, ct.c_void_p
        signatures = {"launch_lif": [P,I,I,I,S], "launch_optic_dr": [P,I,I,S],
                      "launch_optic": [P,I,I,F,F,F,S], "launch_csr": [P,I,I,I,I,S],
                      "launch_events": [P,I,I,I,I,S], "launch_means": [P,I,I,I,I,S]}
        for name, signature in signatures.items():
            fn = getattr(library, name); fn.argtypes = signature; fn.restype = I
        _LIBS[capability] = library
        return library


def _call(name, tensors, *args):
    device = tensors[0].device
    for t in tensors:
        if t.device != device or device.type != "cuda" or not t.is_contiguous():
            raise ValueError("CUDA kernels require contiguous tensors on the same CUDA device")
    pointers = (ct.c_void_p * len(tensors))(*(t.data_ptr() for t in tensors))
    with torch.cuda.device(device):
        error = getattr(lib(device), name)(pointers, *args, torch.cuda.current_stream(device).cuda_stream)
    if error:
        raise RuntimeError(f"{name}: CUDA error {error}")


def _float_shapes(tensors, shapes):
    for t, shape in zip(tensors, shapes, strict=True):
        if t.dtype != torch.float32 or tuple(t.shape) != tuple(shape):
            raise ValueError(f"expected float32 tensor of shape {tuple(shape)}, got {t.dtype} {tuple(t.shape)}")
        if t.numel() >= 2**31:
            raise ValueError("CUDA kernels require fewer than 2**31 elements")


def lif_update(brain, rnd, phase=None):
    tensors = [brain.v, brain.g, brain.drive, brain.adapt, brain.refrac, brain.active,
               brain.poisson_p, rnd, brain.res, brain.std_u_vec, brain.spikes,
               brain.spike_buf[brain.buf_pos], brain.rate, brain.spike_counts, brain._cuda_P]
    flags = int(brain._poisson_on) | (2*brain._std_on) | (4*brain.record_activity) | (8*(brain.p.adapt_jump > 0))
    tensors.append(phase if phase is not None else brain._cuda_P)
    flags |= 16 if phase is not None else 0
    flags |= 32 if brain.cuda_compact else 0
    B, N = brain.B, brain.n
    _float_shapes(tensors, [(B,N)]*5 + [(N,), (B,N), (B,N), (B,N), (N,)] +
                  [(B,N)]*4 + [(11,), (8,N) if phase is not None else (11,)])
    _call("launch_lif", tensors, brain.n, brain.B, flags)


def optic_dr(v, b, dr):
    if v.ndim != 2:
        raise ValueError("optic voltage must have shape (batch, neurons)")
    _float_shapes([v,b,dr], [v.shape,(v.shape[1],),v.shape])
    _call("launch_optic_dr", [v,b,dr], v.shape[1], v.shape[0])


def optic_update(optic, y, pr, sk):
    p = optic.p
    tensors = [optic.v,optic.adapt,optic._cuda_dr,y.contiguous(),pr.contiguous(),sk.contiguous(),optic._a,optic.b_vec]
    _float_shapes(tensors, [(optic.B,optic.n_rate)]*6 + [(optic.n_rate,)]*2)
    _call("launch_optic", tensors,
          optic.n_rate, optic.B, p.gain_rr, p.adapt_gain, optic._a_ad)


class CSR:
    """Experimental warp-per-row product, benchmarked against the default cuSPARSE path."""
    def __init__(self, matrix, device, dtype=torch.float32):
        if dtype not in (torch.float32, torch.float16):
            raise ValueError("CUDA CSR weights must be float32 or float16")
        matrix = sp.csr_matrix(matrix, copy=True); matrix.sum_duplicates(); matrix.sort_indices()
        matrix.check_format(full_check=True)
        if max(matrix.shape, default=0) >= 2**31 or matrix.nnz >= 2**31:
            raise ValueError("CUDA CSR requires 32-bit dimensions and indices")
        self.shape = matrix.shape
        self.ptr = torch.as_tensor(matrix.indptr.astype(np.int32), device=device)
        self.idx = torch.as_tensor(matrix.indices.astype(np.int32), device=device)
        self.values = torch.as_tensor(matrix.data, dtype=dtype, device=device)

    def matvec(self, x):
        if x.ndim != 2 or x.shape[1] != self.shape[1] or x.dtype != torch.float32:
            raise ValueError("CSR input must be float32 (batch, columns)")
        if max(x.numel(), x.shape[0]*self.shape[0]) >= 2**31:
            raise ValueError("CUDA CSR batch is too large")
        y = torch.empty(x.shape[0], self.shape[0], dtype=torch.float32, device=x.device)
        _call("launch_csr", [self.ptr,self.idx,self.values,x.contiguous(),y],
              *self.shape, x.shape[0], int(self.values.dtype == torch.float16))
        return y


def event_scatter(ptr, idx, weights, x, g, pre_ids=None):
    """Scatter valid CSC buffers built by Brain.set_weights; no host read of index values."""
    if ptr.dtype != torch.int32 or idx.dtype != torch.int32:
        raise ValueError("CUDA events require int32 indices")
    if x.ndim != 2 or ptr.shape != (x.shape[1]+1,) or idx.ndim != 1 or weights.shape != idx.shape:
        raise ValueError("invalid CSC buffers or input shape")
    if weights.dtype not in (torch.float16, torch.float32):
        raise ValueError("CUDA event weights must be float32 or float16")
    _float_shapes([x,g], [x.shape,x.shape])
    if pre_ids is not None and (pre_ids.dtype != torch.int32 or pre_ids.ndim != 1):
        raise ValueError("compact presynaptic indices must be an int32 vector")
    _call("launch_events", [ptr,idx,weights,x,g,ptr if pre_ids is None else pre_ids], x.shape[1], x.shape[0],
          int(weights.dtype == torch.float16), -1 if pre_ids is None else pre_ids.numel())


def group_means(ptr, idx, rates, spikes=None):
    """Reduce validated selectors packed by motor.py; no host read of index values."""
    if ptr.dtype != torch.int32 or idx.dtype != torch.int32 or ptr.ndim != 1 or idx.ndim != 1 or rates.ndim != 2:
        raise ValueError("group indices must be int32 vectors; rates must be a matrix")
    _float_shapes([rates], [rates.shape])
    if ptr.numel() < 1 or rates.shape[0]*(ptr.numel()-1) >= 2**31:
        raise ValueError("invalid or oversized group table")
    groups = ptr.numel()-1
    result = torch.empty(rates.shape[0], groups+int(spikes is not None), dtype=torch.float64, device=rates.device)
    _call("launch_means", [ptr,idx,rates,result], rates.shape[1], groups, rates.shape[0], result.shape[1])
    if spikes is not None:
        _float_shapes([spikes], [rates.shape])
        torch.sum(spikes,dim=1,out=result[:,-1])
    return result
