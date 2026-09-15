"""The compass stand-in: angular memory, neural boundary, isolation and batch lifecycle."""
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from flyverse.brain import LIFParams
from flyverse.compass import CompassDriver, epg_columns
from flyverse.connectome import Connectome
from flyverse.fly import FlyBrain


def graph():
    rows=[]
    for side in ('L','R'):
        for g in range(1,9):
            for _ in range(3):
                rows.append(dict(bodyId=len(rows)+1,type='EPG',instance=f'EPG(PB08)_{side}{g}',somaSide=side,
                                 superclass='cb_intrinsic',nt='acetylcholine',**{'class':'','subclass':''}))
    n=pd.DataFrame(rows)
    return Connectome(n,sp.csr_matrix((len(n),len(n)),dtype=np.float32),pd.Series(np.arange(len(n)),index=n.bodyId))


def make(c=None, **kwargs):
    return FlyBrain(c if c is not None else graph(),device='cpu',optic=None,
                    lif_params=LIFParams(receptor_model=None),**kwargs)


def test_raw_guard_and_empty_instruments_are_identical():
    c=graph()
    with pytest.raises(ValueError,match='raw'):
        make(c,instruments=['compass'])
    raw=make(c,seed=4)
    with pytest.raises(ValueError,match='instrumented'):
        raw.attach(CompassDriver(c))
    assert raw.attached_modules=={} and raw.instruments=={}
    empty=make(c,seed=4,preset='instrumented',instruments=[])
    for fb in (raw,empty):
        fb.stimulate(np.arange(10),40.,200.)
        fb.step(100.)
    for name in FlyBrain.BRAIN_TENSORS:
        torch.testing.assert_close(getattr(raw.brain,name),getattr(empty.brain,name),rtol=0,atol=0)


def test_mapping_targets_and_parent_order_guard():
    c=graph(); idx,w=epg_columns(c)
    assert w[:6].tolist()==[0,0,0,2,2,2]
    assert w[24:30].tolist()==[15,15,15,13,13,13]
    driver=CompassDriver(c)
    changed=graph();changed.neurons=changed.neurons.iloc[::-1].reset_index(drop=True)
    fb=make(changed,preset='instrumented')
    with pytest.raises(ValueError,match='row order'):
        fb.attach(driver)
    bad=graph();bad.neurons.loc[0,'instance']='EPG_unknown'
    with pytest.raises(ValueError,match='glomeruli'):
        CompassDriver(bad)


def test_signed_turn_memory_wrap_and_no_absolute_heading_input():
    fb=make(batch=3,preset='instrumented',instruments=['compass'])
    driver=fb.instruments['compass']
    assert 'proprioception' in fb.available_senses
    for _ in range(200):
        fb.proprioception(0,0,0,False,yaw_rate=np.deg2rad([90.,-45.,0.]))
        fb.step(10.)
    np.testing.assert_allclose(driver.phase.numpy().ravel(),[np.pi,3*np.pi/2,0],atol=4e-5)
    fb.proprioception(0,0,0,False,yaw_rate=0.)
    before=driver.phase.clone();fb.step(300.)
    torch.testing.assert_close(driver.phase,before,rtol=0,atol=0)
    # Actual spike-rate phase after settling is close to the represented angle, not just the module's output.
    z=fb.brain.rate.numpy()@np.exp(1j*driver.columns*2*np.pi/16)
    error=np.angle(z*np.exp(-1j*before.numpy().ravel()))
    assert np.max(np.abs(error))<np.deg2rad(20)
    assert np.all(np.abs(z)/fb.brain.rate.numpy().sum(axis=1)>.6)
    with pytest.raises(ValueError,match='finite'):
        driver.observe_turn([np.nan]*3)


def test_checkpoint_partial_reset_and_detach():
    fb=make(batch=2,preset='instrumented',instruments=['compass'])
    c=fb.instruments['compass']
    fb.proprioception(0,0,0,False,yaw_rate=[1.,-2.]);fb.step(80.)
    state=fb.state_dict()
    fb.step(50.);expected=fb.brain.rate.clone();phase=c.phase.clone()
    fb.load_state_dict(state);fb.step(50.)
    torch.testing.assert_close(fb.brain.rate,expected,rtol=0,atol=0)
    torch.testing.assert_close(c.phase,phase,rtol=0,atol=0)
    fb.reset(rows=[0]);assert c.phase[0]==0 and c.yaw_rate[0]==0
    torch.testing.assert_close(c.phase[1],phase[1],rtol=0,atol=0)
    fb.detach('compass')
    assert fb.instrument_records()==[] and fb.attached_modules=={} and 'proprioception' not in fb.available_senses
    assert fb._extensions is None
    fb.reset();baseline=make(batch=2,preset='instrumented');baseline.step(20.);fb.step(20.)
    torch.testing.assert_close(fb.brain.rate,baseline.brain.rate,rtol=0,atol=0)


def test_explicit_module_attach_has_instrument_provenance():
    c=graph();fb=make(c,preset='instrumented');driver=CompassDriver(c)
    fb.attach(driver)
    assert fb.instrument_records()[0]['law']=='unverified'
    assert fb.module_records()[0]['n_writes']=={'epg':48}
    assert fb.instrument_records()[0]['body_ids']==c.neurons.bodyId.tolist()
    with pytest.raises(ValueError,match='unique'):
        fb.attach(CompassDriver(c))
    other=make(c,preset='instrumented',instruments=[CompassDriver(c,velocity_gain=-1)])
    with pytest.raises(ValueError,match='preset/instruments'):
        other.load_state_dict(fb.state_dict())


def test_legacy_benchmark_adapter_runs_module_and_preserves_external_drive():
    from scripts.instrumented_benchmark import InstrumentedBenchmarkBrain
    b=InstrumentedBenchmarkBrain(graph(),LIFParams(receptor_model=None),device='cpu',
                                 preset='instrumented',instruments=['compass'])
    b.drive=torch.full_like(b.drive,1.25)
    b.fb.proprioception(0,0,0,False,yaw_rate=1.)
    b.run_ms(200.)
    assert b.fb.instruments['compass'].phase.item()==pytest.approx(.2,abs=1e-6)
    torch.testing.assert_close(b.drive,torch.full_like(b.drive,1.25),rtol=0,atol=0)
    assert b.rate.max()>10  # the attached module, not the subthreshold constant current, emits the bump
    b.reset();assert b.fb.instruments['compass'].phase.item()==0


def test_batch_body_feeds_realized_turn_without_enabling_afferents():
    from flyverse import BatchSim
    sim=BatchSim(2,c=graph(),device='cpu',preset='instrumented',instruments=['compass'])
    sim.flies[0].yaw_rate=1.;sim.flies[1].yaw_rate=-2.
    sim.step()
    np.testing.assert_allclose(sim.fb.instruments['compass'].phase.numpy().ravel(),[.01,2*np.pi-.02],atol=1e-6)
