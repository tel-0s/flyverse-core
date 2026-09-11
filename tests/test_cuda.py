"""Native CUDA contracts. Opt in to compilation with FLYVERSE_CUDA_TESTS=1."""
import os
import unittest
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from flyverse import cuda
from flyverse.brain import Brain, LIFParams
from flyverse.fly import FlyBrain
from flyverse.optic import OpticParams
from flyverse.motor import read_motor
from test_control import graph


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("FLYVERSE_CUDA_TESTS") == "1",
                     "set FLYVERSE_CUDA_TESTS=1 to compile and test CUDA kernels")
class CudaTests(unittest.TestCase):
    def test_lif_state_and_rng(self):
        for batch, std, adapt, clocks in [(1,0,0,False), (3,.2,1.37,False), (2,.4,1.5,True)]:
            with self.subTest(batch=batch, std=std, clocks=clocks):
                p = LIFParams(std_u=std, std_u_by_type={}, adapt_jump=adapt, event_driven=False)
                a = Brain(graph(), p, device="cuda", batch=batch, seed=31, cuda_kernels=False)
                b = Brain(graph(), p, device="cuda", batch=batch, seed=31, cuda_kernels=True)
                for brain in (a,b):
                    brain.record_activity = True
                    brain.freeze([4])
                    brain.set_poisson([0,4], [750,300])  # forcing a frozen unit is supported
                    brain.set_drive([2,3], [33,17])
                    if clocks:
                        brain.set_clocks([1,2,3,1,2,3])
                for _ in range(40):
                    a.step(); b.step()
                    for key in FlyBrain.BRAIN_TENSORS:
                        torch.testing.assert_close(getattr(a,key), getattr(b,key), atol=1e-5, rtol=1e-6, msg=key)
                self.assertTrue(torch.equal(a.gen.get_state(), b.gen.get_state()))
                self.assertEqual((a.buf_pos,a.t,a.step_count), (b.buf_pos,b.t,b.step_count))

    def test_optic_held_inputs_and_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            gen = torch.Generator(device="cuda").manual_seed(6)
            rand = lambda *shape: torch.rand(*shape, device="cuda", generator=gen)
            for batch in (1,3):
                n = 259  # partial block, several batches
                v, adapt = rand(batch,n)-.5, rand(batch,n)
                o = SimpleNamespace(v=v.clone(), adapt=adapt.clone(), B=batch, n_rate=n, p=OpticParams(),
                                    _a=rand(n), b_vec=rand(n), _a_ad=.997, _cuda_dr=torch.empty_like(v))
                cuda.optic_dr(o.v,o.b_vec,o._cuda_dr)
                for _ in range(10):
                    y, pr, sk = rand(batch,n), rand(batch,n), rand(batch,n)
                    dr = (v+o.b_vec).clamp(0,1)-o.b_vec
                    inp = o.p.gain_rr*y+pr-o.p.adapt_gain*adapt
                    inp = inp+sk
                    v = inp+(v-inp)*o._a
                    adapt = dr+(adapt-dr)*o._a_ad
                    cuda.optic_update(o,y,pr,sk)
                torch.testing.assert_close(o.v,v,rtol=0,atol=0)
                torch.testing.assert_close(o.adapt,adapt,rtol=0,atol=0)
        stream.synchronize()

    def test_sparse_products_and_events(self):
        rng = np.random.default_rng(5)
        matrix = sp.random(277,277,density=.04,random_state=rng,format="csr",dtype=np.float32)
        matrix.data -= .5
        for dtype in (torch.float32, torch.float16):
            for batch in (1,8):
                csr = cuda.CSR(matrix,"cuda",dtype)
                x = torch.as_tensor(rng.random((batch,277),dtype=np.float32), device="cuda")
                x[:,::3] = 0
                dense = torch.as_tensor(matrix.toarray(),device="cuda").to(dtype).float()
                refx = x.half().float() if dtype == torch.float16 else x
                torch.testing.assert_close(csr.matvec(x),refx@dense.T,rtol=2e-5,atol=2e-6)
                csc = matrix.tocsc()
                ptr = torch.tensor(csc.indptr,dtype=torch.int32,device="cuda")
                idx = torch.tensor(csc.indices,dtype=torch.int32,device="cuda")
                w = torch.tensor(csc.data,dtype=dtype,device="cuda")
                g = torch.ones_like(x)
                cuda.event_scatter(ptr,idx,w,x,g)
                torch.testing.assert_close(g,1+x@dense.T,rtol=2e-5,atol=3e-6)

    def test_capture_restore_and_pulses(self):
        for events in (False,True):
            p = LIFParams(event_driven=events)
            a = FlyBrain(graph(),device="cuda",seed=7,lif_params=p,cuda_kernels=True)
            b = FlyBrain(graph(),device="cuda",seed=7,lif_params=p,cuda_kernels=True,cuda_graphs=True)
            for fb in (a,b):
                fb.stimulate([0,2],400,7)
            a.step(3); b.step(3)
            saved = b.state_dict()
            a.step(9); b.step(9)
            for key in FlyBrain.BRAIN_TENSORS:
                torch.testing.assert_close(getattr(a.brain,key),getattr(b.brain,key),atol=1e-5,rtol=1e-6)
            final = b.brain.v.clone()
            b.load_state_dict(saved); b.step(9)
            torch.testing.assert_close(b.brain.v,final,atol=1e-5,rtol=1e-6)

    def test_compact_frozen_state_and_sparse_backends(self):
        for events, sparse in ((True,"torch"),(False,"warp"),(False,"torch")):
            a = Brain(graph(),LIFParams(event_driven=events),device="cuda",batch=3,cuda_kernels=True,cuda_compact=False,
                      cuda_sparse=sparse)
            b = Brain(graph(),LIFParams(event_driven=events),device="cuda",batch=3,cuda_kernels=True,cuda_compact=True,
                      cuda_sparse=sparse)
            for brain in (a,b):
                brain.freeze([2,3,4]); brain.prune([2,3,4])
                brain.record_activity = True
            for frame in range(4):
                for brain in (a,b):
                    if frame == 1:
                        brain.set_poisson([2],1000)
                        brain.v[:,3] = -40  # direct probe edits must fall through the rest fast path
                        brain.drive[:,4] = 20
                    if frame == 2:
                        brain.reset([1])
                    brain.step(8)
                for key in FlyBrain.BRAIN_TENSORS:
                    torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=0,atol=0)

    def test_motor_groups_batch_empty_and_snapshot(self):
        for batch in (1,3):
            fb = FlyBrain(graph(),device="cuda",batch=batch,cuda_kernels=True)
            fb.brain.rate.copy_(torch.arange(batch*6,device="cuda").view(batch,6)*1.37)
            fb.brain.spikes[:,::2] = 1
            actual = fb.motor()
            np.testing.assert_array_equal(fb.brain.total_spikes(),3 if batch == 1 else np.full(batch,3))
            fb.brain.spikes[:,1] = 1  # direct edits invalidate the packed spike-count snapshot
            np.testing.assert_array_equal(fb.brain.total_spikes(),4 if batch == 1 else np.full(batch,4))
            fb.brain.cuda = False
            expected = fb.motor()
            fb.brain.cuda = True
            for f in fields(actual):
                a, b = getattr(actual,f.name), getattr(expected,f.name)
                if isinstance(a,dict):
                    self.assertEqual(a.keys(),b.keys())
                    for key in a:
                        np.testing.assert_allclose(a[key],b[key],rtol=2e-7,atol=1e-6)
                else:
                    np.testing.assert_allclose(a,b,rtol=2e-7,atol=1e-6)
            saved = actual.row(0)
            fb.brain.rate.zero_()
            self.assertEqual(fb.motor().row(0).gf,0)
            self.assertEqual(actual.row(0).gf,saved.gf)
            fb.wings.gf = np.array([],dtype=int)
            self.assertEqual(fb.motor().row(0).gf,0)

    def test_clock_checkpoint_and_graph_weight_replacement(self):
        fb = FlyBrain(graph(),device="cuda",cuda_kernels=True,cuda_graphs=True)
        fb.brain.set_clocks([1,2,3,1,2,3])
        fb.stimulate([0,2],500,100)
        fb.step(4)  # leaves slow-clock spike accumulators partially filled
        saved = fb.state_dict()
        self.assertGreater(sum(float(v.sum()) for v in saved["clock_accumulators"].values()),0)
        fb.step(6)
        expected = fb.state_dict()
        fb.load_state_dict(saved); fb.step(6)
        for key in FlyBrain.BRAIN_TENSORS:
            torch.testing.assert_close(getattr(fb.brain,key).cpu(),expected["brain"][key],rtol=0,atol=0)
        for k,v in fb.brain._acc.items():
            torch.testing.assert_close(v.cpu(),expected["clock_accumulators"][k],rtol=0,atol=0)
        fb = FlyBrain(graph(),device="cuda",cuda_kernels=True,cuda_graphs=True)
        fb.stimulate([0],1000,100); fb.step(10)
        old = tuple(fb._graphs.values())
        fb.brain.set_weights(sp.csr_matrix((6,6),dtype=np.float32)); fb.step(10)
        self.assertTrue(all(g not in old for g in fb._graphs.values()))

    def test_validation_and_empty_rows(self):
        cuda.lib("cuda")
        x = torch.zeros(2,4,device="cuda")
        b = torch.zeros(4,device="cuda")
        with self.assertRaises(ValueError):
            cuda.optic_dr(x.double(),b,x)
        with self.assertRaises(ValueError):
            cuda.optic_dr(x[:,::2],b[:2],x[:,::2])
        with self.assertRaises(ValueError):
            cuda.optic_dr(x,b[:2],x)
        csr = cuda.CSR(sp.csr_matrix((3,4)),"cuda")
        torch.testing.assert_close(csr.matvec(x),torch.zeros(2,3,device="cuda"))
        v = torch.empty(2,0,device="cuda")
        cuda.optic_dr(v,b[:0],v)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("FLYVERSE_CUDA_TESTS") == "1" and
                     os.environ.get("FLYVERSE_INTEGRATION") == "1", "opt-in full CUDA integration")
class FullCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flyverse import connectome
        cls.c = connectome.load(verbose=False)

    def test_full_hybrid_backends_and_checkpoint(self):
        for batch,events,sparse,dtype,dt,clocks in (
                (1,False,"torch","float32",.5,{"vnc":1.}),
                (2,False,"torch","float16",1.,None),
                (1,True,"warp","float32",.5,None)):
            with self.subTest(batch=batch,events=events,sparse=sparse,dtype=dtype):
                p = LIFParams(event_driven=events,weight_dtype=dtype,dt=dt,dt_by_module=clocks)
                a = FlyBrain(self.c,device="cuda",seed=5,batch=batch,lif_params=p,cuda_kernels=False)
                b = FlyBrain(self.c,device="cuda",seed=5,batch=batch,lif_params=p,cuda_kernels=True,
                             cuda_sparse=sparse,cuda_graphs=True)
                for fb in (a,b):
                    fb.smell({"DM1":1.},{"DM1":.2})
                    fb.stimulate([int(fb.optic.rate_idx[0])],200,20)  # original index and full checkpoint layout
                for frame in range(3):
                    rad = np.full((a.retina.n_columns,4),.2+frame*.04,dtype=np.float32)
                    a.vision(rad); b.vision(rad)
                    a.step(6); b.step(6)
                    for name in FlyBrain.BRAIN_TENSORS:
                        tol = dict(rtol=2e-5,atol=3e-4) if name in ("v","g","drive") else dict(rtol=0,atol=0)
                        torch.testing.assert_close(getattr(a.brain,name),getattr(b.brain,name),**tol,msg=name)
                    torch.testing.assert_close(a.optic.delta_rate,b.optic.delta_rate,rtol=1e-5,atol=3e-6)
                saved = b.state_dict()
                b.step(6); expected = b.state_dict()
                b.load_state_dict(saved); b.step(6)
                for name in FlyBrain.BRAIN_TENSORS:
                    tol = dict(rtol=2e-5,atol=3e-4) if name in ("v","g","drive") else dict(rtol=0,atol=0)
                    torch.testing.assert_close(getattr(b.brain,name).cpu(),expected["brain"][name],**tol,msg=name)
                self.assertEqual(b.brain.n,167106)
                del a,b

    def test_batched_env_native_events(self):
        from flyverse.env import EnvParams, FlyRoomEnv
        env = FlyRoomEnv(batch=2,device="cuda",params=EnvParams(cuda_kernels=True,cuda_graphs=True,event_driven=True))
        env.reset()
        for _ in range(2):
            obs,reward,done,info = env.step(np.zeros((2,2),np.float32))
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(reward).all())
        self.assertGreater(env.fb.t,0)


if __name__ == "__main__":
    unittest.main()
