"""Small deterministic contracts; no connectome dataset or GPU required."""
import tempfile
import unittest
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from unittest.mock import patch

from flyverse.connectome import Connectome, save, load
from flyverse.brain import Brain, LIFParams
from flyverse.fly import FlyBrain
from flyverse.motor import MotorRates
from flyverse.body import Locomotion, FlyState
from flyverse import regions
from flyverse import AsyncFlyBrain


def graph():
    n = pd.DataFrame({"bodyId": np.arange(10, 16), "type": ["ORN_DM1", "DM1_lPN", "DNa02", "DNa02", "DNp01", "DLMn"],
        "superclass": ["cb_sensory", "cb_intrinsic", "descending_neuron", "descending_neuron", "descending_neuron", "vnc_motor"],
        "class": ["olfactory", "ALPN", "", "", "", ""], "subclass": ["", "", "", "", "", "wm"],
        "somaSide": ["", "L", "L", "R", "R", "L"]})
    w = np.array([[0,0,0,0,0,0], [400,0,0,0,0,0], [0,100,0,100,0,0],
                  [0,0,100,0,0,0], [0,0,40,0,0,0], [0,0,500,0,40,0]], dtype=np.float32)
    return Connectome(n, sp.csr_matrix(w), pd.Series(np.arange(6), index=n.bodyId))


class SubsetTests(unittest.TestCase):
    def test_reindex_nested_and_persist(self):
        c = graph(); sub = c.subset([5, 2, 0]).subset([1, 0])
        np.testing.assert_array_equal(sub.neurons.bodyId, [12, 15])
        np.testing.assert_array_equal(sub.index_of([15,12]), [1,0])
        np.testing.assert_array_equal(sub.W.toarray(), c.W[[2,5]][:,[2,5]].toarray())
        self.assertIs(sub.reference, c)
        with tempfile.TemporaryDirectory() as folder:
            save(sub, Path(folder)); restored = load(Path(folder))
        np.testing.assert_array_equal(restored.reference.W.toarray(), c.W.toarray())
        np.testing.assert_array_equal(restored.neurons.in_syn, sub.neurons.in_syn)

    def test_weights_invariant_with_custom_parameters(self):
        c = graph(); idx = [5,2,4]
        for p in [LIFParams(input_norm_ref=20, event_driven=False),
                  LIFParams(conn_cap=0, same_type_gain=.3, path_gain=[], type_path_gain=[],
                            input_norm_ref=40, input_norm_alpha=.7, event_driven=False)]:
            full = Brain(c, p, device="cpu")
            reduced = Brain(c.subset(idx), p, device="cpu")
            torch.testing.assert_close(reduced.W.to_dense(), full.W.to_dense()[idx][:,idx], rtol=0, atol=0)

    def test_selection_validation(self):
        c=graph()
        for idx in ([1,1], [-1], [6], [1.5], [True,False]):
            with self.assertRaises(ValueError): c.subset(idx)
        self.assertEqual(c.subset([]).n, 0)

    def test_modules_partition_and_paths(self):
        c=graph()
        np.testing.assert_array_equal(regions.select(c, regions.MODULES), np.arange(c.n))
        np.testing.assert_array_equal(regions.pare(c,[0],[5],max_hops=3).neurons.bodyId, [10,11,12,15])
        self.assertEqual(regions.pare(c,[5],[0]).n,0)
        with self.assertRaises(ValueError): regions.select(c,["typo"])
        # An explicit zero in sparse storage is not a traversable edge.
        c.W[0,5]=0
        self.assertEqual(regions.pare(c,[5],[0]).n,0)


class ControlTests(unittest.TestCase):
    def make(self, batch=1):
        return FlyBrain(graph(), batch=batch, device="cpu", seed=4)

    def test_senses_and_missing_groups(self):
        fb=self.make(batch=2)
        self.assertEqual(fb.available_senses,("smell",))
        fb.smell({"DM1": np.array([1.,2.])},{"DM1":0.})
        self.assertGreater(float(fb.brain.poisson_p[1,0]),float(fb.brain.poisson_p[0,0]))
        for call in (lambda: fb.vision([]), lambda: fb.wind(0,0), lambda: fb.taste(1)):
            with self.assertRaises(ValueError): call()
        motor=fb.motor()
        self.assertEqual(motor.row(1).proboscis,0.)
        fb.brain.rate.fill_(20)
        self.assertEqual(motor.row(0).fwd_dn,0.)

    def test_pulse_expiration_restores_sensory_input(self):
        fb=self.make(); fb.smell({"DM1":1.},{"DM1":1.})
        baseline=fb.brain.poisson_p.clone()
        fb.stimulate([0],200,1.0)
        self.assertGreater(float(fb.brain.poisson_p[0,0]),float(baseline[0,0]))
        fb.step(.5); fb.step(.5)
        torch.testing.assert_close(fb.brain.poisson_p,baseline,rtol=0,atol=0)

    def test_fractional_time_and_resume(self):
        fb=self.make(); fb.stimulate([2],150,10)
        self.assertEqual(fb.step(.2),0)
        self.assertEqual(fb.step(.3),.5)
        fb.step(1); state=fb.state_dict()
        fb.step(10); expected=fb.state_dict()
        fb.load_state_dict(state); fb.step(10); actual=fb.state_dict()
        for k in expected['brain']:
            torch.testing.assert_close(actual['brain'][k],expected['brain'][k],rtol=0,atol=0)
        self.assertEqual(actual['brain_scalars'],expected['brain_scalars'])

    def test_reset_and_activity(self):
        fb=self.make(batch=2); fb.stimulate([2],2000,2); fb.step(2)
        self.assertTrue(fb.activity_mask(100)[0,2])
        fb.reset([0]); self.assertEqual(float(fb.brain.spike_counts[0].sum()),0)
        self.assertGreater(float(fb.brain.spike_counts[1].sum()),0)
        fb.reset(); self.assertEqual(fb.t,0)
        self.assertFalse(bool(fb.brain.poisson_p.any()))

    def test_body_accepts_rates_without_neural_objects(self):
        loco=Locomotion(); pose=FlyState()
        loco.step(pose,MotorRates(fwd_dn=20),.01,(-1,1,-1,1))
        self.assertGreater(pose.x,0)
        self.assertEqual(loco.readout(MotorRates())['mode'],'searching')

    def test_optic_substeps_carry_fractional_time(self):
        c=graph()
        c.neurons.loc[0,['type','superclass','class']]=['R1-R6','ol_sensory','']
        c.neurons.loc[1,['type','superclass','class']]=['L1','ol_intrinsic','']
        c.neurons['hex1']=np.nan; c.neurons['hex2']=np.nan; c.neurons['hex_side']=''
        c.neurons.loc[0,['hex1','hex2','hex_side']]=[1.,1.,'L']
        fb=FlyBrain(c,device='cpu')
        fb.vision(np.full((fb.retina.n_columns,4),.2))
        with patch.object(fb.optic,'_substep',wraps=fb.optic._substep) as substep:
            fb.step(.5); self.assertEqual(substep.call_count,0)
            state=fb.state_dict()
            fb.step(.5); self.assertEqual(substep.call_count,1)
            fb.load_state_dict(state)
            self.assertEqual(fb.optic._pending_ms,.5)
            fb.vision(np.full((fb.retina.n_columns,4),.3))
            torch.testing.assert_close(state['radiance'],torch.full_like(state['radiance'],.2),rtol=0,atol=0)
            fb.step(.5); self.assertEqual(substep.call_count,2)

    def test_rl_env_uses_available_senses_in_a_subset(self):
        from flyverse.env import FlyRoomEnv, EnvParams
        with patch('flyverse.connectome.load',return_value=graph()):
            env=FlyRoomEnv(batch=2,device='cpu',params=EnvParams(modules=('antennal_lobe','descending'),obs='pn'))
        self.assertIsNone(env.optic)
        obs=env.reset()
        self.assertEqual(obs.shape,(2,2))
        obs,reward,done,info=env.step(np.zeros((2,2)))
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(reward.shape,(2,))

    def test_budget_reports_completed_time(self):
        fb=self.make()
        zero=fb.step_budget(0)
        self.assertEqual(zero.steps,0)
        result=fb.step_budget(5,frame_ms=1)
        self.assertGreater(result.steps,0)
        self.assertEqual(result.simulated_ms,result.steps*fb.brain.p.dt)
        self.assertAlmostEqual(result.time_dilation,result.simulated_ms/result.wall_ms)

    def test_async_update_and_close(self):
        fb=self.make()
        with AsyncFlyBrain(fb,frame_ms=1) as worker:
            worker.submit(smell=({'DM1':1.},{'DM1':0.}))
            worker.stimulate([2],150,10)
            update=worker.wait_for_update(0)
            self.assertGreater(update.time_ms,0)
            self.assertGreaterEqual(worker.motor().time_ms,update.time_ms)
        self.assertFalse(worker._thread.is_alive())
        with self.assertRaises(RuntimeError): worker.submit(smell=({},{}))

    def test_async_read_does_not_wait_for_running_frame(self):
        fb=self.make(); entered=threading.Event(); release=threading.Event()
        def slow_step(ms):
            entered.set()
            if not release.wait(2): raise TimeoutError('test frame was not released')
            return ms
        with patch.object(fb,'step',side_effect=slow_step):
            with AsyncFlyBrain(fb,frame_ms=1) as worker:
                try:
                    worker.submit(smell=({},{}))
                    self.assertTrue(entered.wait(2))
                    self.assertEqual(worker.motor().time_ms,0)
                finally:
                    release.set()

    def test_async_propagates_errors(self):
        with AsyncFlyBrain(self.make(),frame_ms=1) as worker:
            worker.submit(smell=({'DM1':-1.},{}))
            with self.assertRaises(RuntimeError): worker.wait_for_update(0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class CudaTests(unittest.TestCase):
    def test_graph_rng_and_delay_phase(self):
        for dtype in ('float32','float16'):
            a=FlyBrain(graph(),device='cuda',seed=5,lif_params=LIFParams(weight_dtype=dtype))
            b=FlyBrain(graph(),device='cuda',seed=5,lif_params=LIFParams(weight_dtype=dtype),cuda_graphs=True)
            for fb in (a,b): fb.smell({'DM1':1.},{'DM1':.2})
            for _ in range(5):
                a.step(1.5); b.step(1.5)
                for name in a.BRAIN_TENSORS:
                    torch.testing.assert_close(getattr(a.brain,name),getattr(b.brain,name),rtol=0,atol=0)
            self.assertEqual(a.t,b.t)
            self.assertEqual(a.brain.buf_pos,b.brain.buf_pos)

    def test_half_accumulates_into_float32(self):
        c=graph()
        b=Brain(c,LIFParams(weight_dtype='float16',input_norm_alpha=0,conn_cap=0),device='cuda')
        # Deliberately exceed the half output range: fp32 accumulation must stay finite.
        b.set_weights(sp.csr_matrix(np.full((c.n,c.n),20000,dtype=np.float32)))
        b._add_synaptic_input(torch.ones_like(b.g))
        self.assertEqual(b.g.dtype,torch.float32)
        torch.testing.assert_close(b.g,torch.full_like(b.g,120000),rtol=0,atol=0)

    def test_event_backend_matches_sparse_input(self):
        c=graph()
        a=Brain(c,LIFParams(event_driven=False),device='cuda',batch=2)
        b=Brain(c,LIFParams(event_driven=True),device='cuda',batch=2)
        x=torch.tensor([[0,1,0,1,0,1],[1,0,1,0,1,0]],device='cuda',dtype=torch.float32)
        a._add_synaptic_input(x); b._add_synaptic_input(x)
        torch.testing.assert_close(a.g,b.g)


if __name__ == '__main__':
    unittest.main()
