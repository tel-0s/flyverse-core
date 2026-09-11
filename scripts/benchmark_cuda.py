"""CUDA sparse backend experiments on the current full hybrid brain's actual matrices.

GPU event timings are still subject to other processes. Report distributions, alternate
backend order, and keep these results separate from room/UI throughput.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
import torch
from flyverse import cuda
from flyverse.fly import FlyBrain


def timed(function, repeats, rounds):
    for _ in range(3):
        function()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            function()
    def run():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); graph.replay(); end.record(); end.synchronize()
        return start.elapsed_time(end)*1000/repeats
    return run, graph  # retain graph-owned outputs and allocations


def scipy_csr(t):
    return sp.csr_matrix((t.values().cpu().numpy(),t.col_indices().cpu().numpy(),t.crow_indices().cpu().numpy()),shape=t.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="out/cuda_sparse.json")
    ap.add_argument("--batches", type=int, nargs="+", default=[1,8])
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=12)
    args = ap.parse_args()
    cuda.lib("cuda")
    fb = FlyBrain(device="cuda",cuda_kernels=False)
    matrices = {"lif_pruned":fb.brain._W_cpu,"optic_recurrent":scipy_csr(fb.optic.W_rr)}
    result = {"device":torch.cuda.get_device_name(),"torch":torch.__version__,"shared_machine":True,"cases":[]}
    rng = np.random.default_rng(8)
    for name, matrix in matrices.items():
        for batch in args.batches:
            for dtype in (torch.float32,torch.float16) if name == "lif_pruned" else (torch.float32,):
                native = cuda.CSR(matrix,"cuda",dtype)
                for activity in ([.001,.01,.1,1.] if name == "lif_pruned" else [1.]):
                    raw = (rng.random((batch,matrix.shape[1])) < activity).astype(np.float32)
                    x = torch.tensor(raw,device="cuda")
                    xt = x.T.to(dtype).contiguous()
                    out = torch.empty(matrix.shape[0],batch,device="cuda")
                    methods = {}
                    for itype in (torch.int64,torch.int32):
                        W = torch.sparse_csr_tensor(native.ptr.to(itype),native.idx.to(itype),native.values,size=matrix.shape,device="cuda")
                        methods[str(itype).split(".")[1]] = lambda W=W: torch.mm(W,xt,out=out)
                    methods["warp_csr"] = lambda: native.matvec(x)
                    rows = np.flatnonzero(np.diff(matrix.indptr))
                    packed = matrix[rows]
                    Wr = torch.sparse_csr_tensor(torch.tensor(packed.indptr,device="cuda",dtype=torch.int32),
                         torch.tensor(packed.indices,device="cuda",dtype=torch.int32),
                         torch.tensor(packed.data,device="cuda",dtype=dtype),size=packed.shape,device="cuda")
                    yr = torch.empty(len(rows),batch,device="cuda")
                    methods["compact_rows_product"] = lambda: torch.mm(Wr,xt,out=yr)
                    if name == "lif_pruned":
                        csc = matrix.tocsc()
                        ptr = torch.tensor(csc.indptr,device="cuda",dtype=torch.int32)
                        idx = torch.tensor(csc.indices,device="cuda",dtype=torch.int32)
                        weights = torch.tensor(csc.data,device="cuda",dtype=dtype)
                        g = torch.empty_like(x)
                        def events():
                            g.zero_(); cuda.event_scatter(ptr,idx,weights,x,g)
                        methods["events_including_clear"] = events
                        pre_ids = torch.tensor(np.flatnonzero(np.diff(csc.indptr)),device="cuda",dtype=torch.int32)
                        def compact_events():
                            g.zero_(); cuda.event_scatter(ptr,idx,weights,x,g,pre_ids)
                        methods["compact_events_including_clear"] = compact_events
                    runners = {key:timed(fn,args.repeats,args.rounds) for key,fn in methods.items()}
                    samples = {key:[] for key in methods}
                    for _ in range(args.rounds):
                        for key in rng.permutation(list(methods)):
                            samples[key].append(runners[key][0]())
                    case = dict(matrix=name,shape=matrix.shape,nnz=matrix.nnz,nonempty_rows=len(rows),batch=batch,
                                dtype=str(dtype),activity=activity,us={key:dict(p10=float(np.percentile(v,10)),
                                median=float(np.median(v)),p90=float(np.percentile(v,90))) for key,v in samples.items()})
                    result["cases"].append(case)
                    print(json.dumps(case),flush=True)
                    del runners
    path = Path(args.json); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
