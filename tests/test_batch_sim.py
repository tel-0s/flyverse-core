"""Batch isolation and scalar-reference equivalence. GPU checks are explicitly opt-in."""
import copy
from dataclasses import asdict, fields
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flyverse import BatchSim
from flyverse import air, body, programs, surfaces, world
from flyverse.batch_air import BatchAir
from flyverse.batch_body import BatchBody, frames
from flyverse.batch_sim import _RowStimulus
from flyverse.batch_world import BatchWorld
from flyverse.motor import MotorRates
from test_control import graph


def assert_nested(test, a, b, tol=2e-12):
    if isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for k in a: assert_nested(test,a[k],b[k],tol)
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b): assert_nested(test,x,y,tol)
    elif isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray) and a.dtype.kind in "fiu" or isinstance(a,(float,np.floating)):
        np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    else: test.assertEqual(a,b)


def scalar_body_step(model, commands, wings, tasting, dt):
    for i,(f,l,flight,m) in enumerate(zip(model.flies,model.locos,model.flights,model.metabolisms)):
        if f.airborne:
            model.feeding[i] = False; m.update(False,0.,dt)
            if model.fence is None: flight.step(f,wings[i],dt,model.surfaces.scalar,(-2,2,-2,2,2.6))
            else:
                x0,x1,y0,y1,z = model.fence
                flight.step(f,wings[i],dt,lambda x,y:z,(x0,x1,y0,y1,2.6))
        elif not flight.maybe_takeoff(f,wings[i],dt):
            model.feeding[i] = m.update(bool(tasting[i]),f.speed,dt)
            cmd = dict(commands[i],speed=0.,yaw=0.) if model.feeding[i] else commands[i]
            l.step(f,cmd,dt,model.surfaces.scalar if model.fence is None else model.fence[:4])


def motor_sample(batch, rng):
    values = {f.name:rng.uniform(0,20,batch).astype(np.float32) for f in fields(MotorRates) if f.name not in ("lh_odour","pn_glomeruli","pn_glom_cells","time_ms")}
    values.update(lh_odour={"apple":rng.uniform(0,30,batch),"berry":rng.uniform(0,30,batch)},
                  pn_glomeruli={"DM1":rng.uniform(0,40,batch)},pn_glom_cells={"DM1":5})
    return MotorRates(**values)


class BatchPhysicsTests(unittest.TestCase):
    def test_body_frames_include_flight_and_edge_blend(self):
        flies = [body.FlyState(heading=.2),body.FlyState(airborne=True,pitch=.3,roll=-.4,_heading=.6),body.FlyState()]
        flies[2].place(.6,0,.74,normal=(1,0,0))
        flies[2].begin_edge(np.array([0,0,1.]),np.array([1,0,0.]))
        flies[2]._edge_left *= .7
        actual = frames(flies)
        for k,name in enumerate(("eye_pos","forward","left","up")):
            np.testing.assert_allclose(actual[k],[getattr(f,name) for f in flies],rtol=1e-14,atol=1e-14)

    def test_mixed_body_matches_scalar_through_edges_takeoff_landing_and_feeding(self):
        _,info = world.make_room()
        geo = surfaces.Surfaces(info["room"],info["solids"],info["solid_labels"])
        for fence in (None,(*info["table_extent"],.75)):
            flies = [body.FlyState(x=0,y=0,z=.75,heading=.15) for _ in range(9)]
            flies[1].x = .59995; flies[1].heading = 0.
            flies[2].place(-.576,-.35,0.,heading=0.)
            for i,pos,velocity in ((3,(0,0,.754),(0,0,-.2)),(4,(.2,.2,1.),(.1,-.03,.1)),(8,(1.999,0,1.),(.4,0,0))):
                flies[i].set_pos(pos); flies[i].airborne=True; flies[i].air_time=.1
                flies[i].vx,flies[i].vy,flies[i].vz=velocity
            if fence:
                flies[2].place(-.3,-.1,.75)
                flies[8].set_pos((.599,0,.8))
            actual = BatchBody(flies,geo,fence)
            actual.metabolisms[5].energy=.9499
            actual.flights[7]._power_hold=.29
            for i,l in enumerate(actual.locos): l.k_opto=.01*i
            reference = BatchBody(copy.deepcopy(flies),geo,fence)
            reference.locos = copy.deepcopy(actual.locos)
            reference.metabolisms = copy.deepcopy(actual.metabolisms)
            reference.flights = copy.deepcopy(actual.flights)
            rng = np.random.default_rng(3)
            for step in range(100):
                motor = motor_sample(len(flies),rng)
                commands,wings = actual.readout(motor,.01)
                refcmd = [l.readout(motor.row(i)) for i,l in enumerate(reference.locos)]
                refw = [f.readout(motor.row(i)) for i,f in enumerate(reference.flights)]
                assert_nested(self,commands,refcmd)
                assert_nested(self,wings,refw)
                if step == 0:
                    wings[6]["gf"] = refw[6]["gf"] = 80.
                    wings[7]["power"] = refw[7]["power"] = 60.
                if step == 50:
                    actual.metabolisms[5].energy = reference.metabolisms[5].energy = .6999
                taste = np.zeros(len(flies)); taste[5] = 1.
                actual.step(commands,wings,taste,.01)
                scalar_body_step(reference,refcmd,refw,taste,.01)
                with self.subTest(fence=fence is not None,step=step):
                    for a,b in zip(actual.flies,reference.flies): assert_nested(self,asdict(a),asdict(b))
                    for a,b in zip(actual.metabolisms,reference.metabolisms): assert_nested(self,asdict(a),asdict(b))
                    np.testing.assert_array_equal(actual.feeding,reference.feeding)

    def test_air_has_independent_scenes_phases_and_clocks(self):
        airs = []
        for i in range(3):
            _,info = world.make_room(i)
            a = air.Air([(n,c,r/.02,r) for n,c,r in info['fruit']],air.WindParams(speed=.2*i,direction_deg=150+30*i),seed=i)
            a.t = .17*i
            airs.append(a)
        batched = BatchAir(airs)
        eye = np.array([[.26,.15,.7512],[-.04,.28,.7512],[-.5,-.3,1.]])
        forward = np.array([[1,0,0],[0,1,0],[0,0,1.]])
        left = np.array([[0,1,0],[-1,0,0],[-1,0,0.]])
        for _ in range(3):
            batched.step(.01)
            actual = batched.antennae(eye,left,forward)
            wind = batched.deflections(forward,left)
            for i,a in enumerate(airs):
                expected = a.antennae(eye[i],left[i],forward[i])
                for side in (0,1):
                    for k in expected[side]: np.testing.assert_allclose(actual[side][k][i],expected[side][k][0],rtol=1e-13,atol=1e-13)
                    np.testing.assert_allclose(wind[side][i],a.deflections(forward[i],left[i])[side][0],rtol=1e-14,atol=1e-14)
        empty = BatchAir([air.Air([]),air.Air([])])
        self.assertEqual(empty.concentration(np.zeros((2,3,3))).shape,(2,3,0))


class BatchSceneTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(1)

    def tearDown(self): torch.set_num_threads(self.threads)

    def compare(self, device, capture=False):
        scenes = [world.make_room(i)[0] for i in range(3)]
        for w in scenes: w.device=device
        scenes[1].spheres[0].material='black'
        scenes[2].light_pos=(1,0,2)
        batched = BatchWorld(scenes,device)
        gen = torch.Generator().manual_seed(12)
        dirs = torch.randn(3,74,3,generator=gen)[:,::2]
        dirs /= dirs.norm(dim=-1,keepdim=True)
        org = torch.tensor([-.5,.05,.7512]).expand_as(dirs)
        original = batched.trace(org,dirs,cuda_graphs=capture)
        for i,w in enumerate(scenes): torch.testing.assert_close(original[i],w.trace(org[i],dirs[i]),rtol=3e-5,atol=2e-5)
        saved = original.clone()
        batched.move_sphere(0,(-.45,.05,.77),(.05,)*3,rows=[1])
        changed = batched.trace(org,dirs,cuda_graphs=capture)
        torch.testing.assert_close(changed[0],original[0],rtol=0,atol=0)
        torch.testing.assert_close(changed[2],original[2],rtol=0,atol=0)
        self.assertFalse(torch.equal(changed[1],original[1]))
        torch.testing.assert_close(original,saved,rtol=0,atol=0)
        for i,w in enumerate(scenes): torch.testing.assert_close(changed[i],w.trace(org[i],dirs[i]),rtol=3e-5,atol=2e-5)
        if capture:
            torch.testing.assert_close(changed,batched.trace(org,dirs),rtol=0,atol=0)
            self.assertEqual(len(batched._trace_graphs),1)

    def test_independent_scenes_match_scalar_rays(self): self.compare('cpu')

    def test_empty_scenes_and_bad_shapes(self):
        b = BatchWorld([world.World(device='cpu'),world.World(device='cpu')])
        torch.testing.assert_close(b.trace(torch.zeros(2,3,3),torch.ones(2,3,3)),torch.zeros(2,3,4),rtol=0,atol=0)
        with self.assertRaises(ValueError): b.trace(torch.zeros(3,3),torch.zeros(3,3))
        b.worlds[1].spheres.append(world.Sphere((1,0,0),(.1,)*3,'apple'))
        with self.assertRaises(ValueError): b.trace(torch.zeros(2,3,3),torch.ones(2,3,3))

    @unittest.skipUnless(os.environ.get('FLYVERSE_BATCH_CUDA')=='1' and torch.cuda.is_available(),'opt-in batch CUDA check')
    def test_cuda_capture_and_scene_isolation(self): self.compare('cuda',True)


class BatchSimTests(unittest.TestCase):
    def test_one_brain_step_per_frame_and_row_reset_isolation(self):
        sim = BatchSim(3,c=graph(),device='cpu',program=['none','anemotaxis','klinotaxis'])
        with patch.object(sim.fb,'step',wraps=sim.fb.step) as step:
            sim.step(); sim.step()
            self.assertEqual(step.call_count,2)
        self.assertEqual(sim.brain.rate.shape,(3,6))
        before = [asdict(f) for f in sim.flies]
        rate = sim.brain.rate.clone(); time = sim.fb.t
        sim.reset([1])
        self.assertEqual(sim.fb.t,time)
        for i in (0,2):
            assert_nested(self,asdict(sim.flies[i]),before[i])
            torch.testing.assert_close(sim.brain.rate[i],rate[i],rtol=0,atol=0)
            self.assertEqual(sim.airs[i].t,.02)
        self.assertEqual(sim.airs[1].t,0.)
        self.assertEqual(sim.episode_frames.tolist(),[2,0,2])
        for bad in ([-1],[3],[1,1],[.5]):
            with self.assertRaises(ValueError): sim.reset(bad)
        sim.reset()
        self.assertEqual(sim.fb.t,0.)

    def test_program_stimulation_is_coalesced_and_row_specific(self):
        sim = BatchSim(3,c=graph(),device='cpu')
        sim._row_inputs[0].stimulate({'type':'DNp01'},50.,15.)
        sim._row_inputs[2].stimulate({'type':'DNp01'},80.,15.)
        sim._row_inputs[0].stimulate({'type':'DNp01'},20.,15.)
        self.assertEqual(len(sim._program_pulses),1)
        idx,hz,ms = next(iter(sim._program_pulses.values()))
        np.testing.assert_array_equal(hz[:,0],[50,0,80])
        sim.fb.stimulate(idx,hz,ms)
        np.testing.assert_allclose(sim.brain.poisson_p[:,4],[.025,0,.04])

    def test_checkpoint_restores_program_rng_body_and_brain(self):
        sim = BatchSim(2,c=graph(),device='cpu',program='anemotaxis+klinotaxis',escape_gating=True)
        initial = sim.state_dict()
        sim.step(); expected_initial_step = sim.state_dict()
        sim.load_state_dict(initial); sim.step()
        assert_nested(self,sim.state_dict()['programs'],expected_initial_step['programs'])
        for _ in range(4): sim.step()
        sim.start_loom(rows=[1])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'batch.pt'
            sim.save_state(path)
            for _ in range(8): sim.step()
            expected = sim.state_dict()
            sim.load_state(path)
            for _ in range(8): sim.step()
        actual = sim.state_dict()
        for k in ('controller','programs','gatings','commands','wcommands','feeding','episode_frames','loom_t'):
            assert_nested(self,actual[k],expected[k])
        for a,b in zip(sim.flies,expected['flies']): assert_nested(self,asdict(a),asdict(b))


@unittest.skipUnless(os.environ.get('FLYVERSE_BATCH_INTEGRATION')=='1' and torch.cuda.is_available(),'opt-in full-connectome batch integration')
class FullBatchTests(unittest.TestCase):
    def test_single_row_agrees_with_room_demo(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from room_demo import Sim
        options = dict(seed=9,cuda_graphs=True,cuda_kernels=True,event_driven=False,cuda_sparse='torch',
                       program='anemotaxis+klinotaxis',escape_gating=True)
        scalar = Sim(**options,trail_seconds=0)
        batched = BatchSim(1,device='cuda',**options)
        for k in range(12):
            if k==3: scalar.start_loom(); batched.start_loom()
            scalar.step(); batched.step()
            torch.testing.assert_close(batched.col_rad[0],scalar.col_rad,rtol=3e-5,atol=2e-5)
            torch.testing.assert_close(batched.brain.rate,scalar.brain.rate,rtol=1e-5,atol=2e-4)
            assert_nested(self,asdict(batched.flies[0]),asdict(scalar.fly),tol=1e-8)
            assert_nested(self,batched.commands[0],scalar.cmd,tol=1e-5)
            assert_nested(self,asdict(batched.metabolisms[0]),asdict(scalar.metabolism),tol=1e-12)

    def test_single_row_cx_pulses_agree_with_demo(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from room_demo import Sim
        options = dict(seed=5,cuda_graphs=True,cuda_kernels=True,event_driven=False,program='cx+klinotaxis',fruit_set='apple',fence=True)
        scalar = Sim(**options,trail_seconds=0)
        batched = BatchSim(1,device='cuda',**options)
        for _ in range(12):
            scalar.step(); batched.step()
            torch.testing.assert_close(batched.brain.poisson_p,scalar.brain.poisson_p,rtol=0,atol=0)
            torch.testing.assert_close(batched.brain.rate,scalar.brain.rate,rtol=1e-5,atol=2e-4)
            assert_nested(self,batched.commands[0],scalar.cmd,tol=1e-5)
            assert_nested(self,asdict(batched.flies[0]),asdict(scalar.fly),tol=1e-8)

    def test_native_batched_cx_and_checkpoint(self):
        sim = BatchSim(4,device='cuda',cuda_graphs=True,cuda_kernels=True,event_driven=True,program='cx')
        for i,f in enumerate(sim.flies): f.heading=i*.8
        for _ in range(4): sim.step()
        self.assertEqual(sim.col_rad.shape,(4,sim.r.n_columns,4))
        self.assertTrue(torch.isfinite(sim.brain.v).all())
        self.assertEqual(len(sim.commands),4)
        self.assertLessEqual(len(sim._program_pulses),3)
        state = sim.state_dict()
        sim.step(); sim.load_state_dict(state)
        for name in sim.fb.BRAIN_TENSORS:
            torch.testing.assert_close(getattr(sim.brain,name).cpu(),state['controller']['brain'][name],rtol=0,atol=0)
        sim.reset([2]); sim.step()
        self.assertEqual(sim.episode_frames.tolist(),[5,5,1,5])


if __name__ == '__main__': unittest.main()
