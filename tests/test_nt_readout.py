"""Optional live NT contracts and physical-field projection; no NT model or dataset."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from flyverse import FlyBrain, NTChannel, NTSnapshot
from flyverse.brainmap import BrainMap
from test_control import graph


def frame():
    return NTSnapshot(10., np.array([15,10,999]),
                      (NTChannel("dopamine","nM",0.,100.), NTChannel("serotonin","a.u.",0.,1.)),
                      [[80.,.2],[20.,.4],[90.,.8]])


class NTReadoutTests(unittest.TestCase):
    def test_alignment_subset_missing_data_and_owned_snapshots(self):
        snapshot = frame()
        values = snapshot.aligned(np.array([10,77,15]))
        np.testing.assert_allclose(values,[[20.,.4],[np.nan,np.nan],[80.,.2]])
        values[0]=0
        self.assertEqual(snapshot.levels[1,0],20.)
        raw = np.array([[5.]])
        owned = NTSnapshot(0.,np.array([10]),(NTChannel("test","nM",0.,10.),),raw)
        raw[0,0]=8.
        self.assertEqual(owned.levels[0,0],5.)
        self.assertFalse(owned.levels.flags.writeable)
        self.assertFalse(owned.body_ids.flags.writeable)
        empty = NTSnapshot(0.,np.array([],dtype=int),owned.channels,np.empty((0,1)))
        self.assertTrue(np.isnan(empty.aligned(np.array([10]))).all())

    def test_invalid_snapshots_cannot_publish_ambiguous_data(self):
        channel = NTChannel("test","nM",0.,1.)
        for params in (("test","",0.,1.), ("test","nM",1.,1.), ("test","nM",0.,np.inf)):
            with self.assertRaises(ValueError): NTChannel(*params)
        cases = [(-1.,[1],(channel,),[[0.]]), (0.,[1,1],(channel,),[[0.],[1.]]),
                 (0.,[1.5],(channel,),[[0.]]), (0.,[1],(channel,channel),[[0.,1.]]),
                 (0.,[1],(channel,),[[np.inf]]), (0.,[1],(channel,),[0.])]
        for time,ids,channels,levels in cases:
            with self.assertRaises(ValueError): NTSnapshot(time,np.array(ids),channels,levels)

    def test_optional_source_is_on_demand_and_batch_explicit(self):
        # This graph has neither NT labels nor an optic module.
        source = SimpleNamespace(readout=Mock(return_value=frame()))
        fb = FlyBrain(graph(),device="cpu",batch=2,nt_source=source)
        self.assertIsNone(fb.optic)
        fb.step(10.)
        source.readout.assert_not_called()
        snapshot = fb.neurotransmitters(batch_index=1)
        source.readout.assert_called_once_with(batch_index=1)
        self.assertEqual(snapshot.aligned(fb.c.neurons.bodyId.to_numpy()).shape,(6,2))
        with self.assertRaises(IndexError): fb.neurotransmitters(batch_index=2)
        source.readout.return_value = np.zeros((6,2))
        with self.assertRaises(TypeError): fb.neurotransmitters()
        fb.nt_source = None
        self.assertIsNone(fb.neurotransmitters())

    def test_map_needs_no_nt_labels_optic_or_global_dataset(self):
        c = graph().subset([5,0,3])
        coords = {10:[0.,0.,0.],15:[2.,2.,2.],13:None}
        with patch("flyverse.brainmap.pd.read_feather",side_effect=AssertionError("no global dataset")):
            bmap = BrainMap(c,locations=coords,width=12,dorsal_h=8,lateral_h=6)
        np.testing.assert_array_equal(bmap.body_ids,[15,10,13])
        np.testing.assert_array_equal(bmap.idx,[0,1])
        activity = bmap.activity(SimpleNamespace(rate_np=lambda:np.array([40.,20.,0.])))
        np.testing.assert_array_equal(activity,[1.,.5,0.])
        levels = frame().aligned(bmap.body_ids)[:,0]
        images = bmap.render_field(levels,0.,100.)
        self.assertEqual([image.shape for image in images],[(8,12,3),(6,12,3)])
        for subset,locations in ((c.subset([]),{}),(c,{10:None,15:[np.nan,0,0]})):
            empty = BrainMap(subset,locations=locations,width=5,dorsal_h=3,lateral_h=3)
            self.assertEqual(len(empty.idx),0)
            self.assertTrue(np.isfinite(empty.render(np.zeros(subset.n))[0]).all())

    def test_field_averages_physical_values_before_clipping_and_preserves_missing(self):
        c = graph().subset([0,1,2])
        bmap = BrainMap(c,locations={10:[0,0,0],11:[0,0,0],12:[2,2,2]},width=3,dorsal_h=3,lateral_h=3)
        color = (100,200,240)
        levels = np.array([0.,4.,np.nan])
        image = bmap.render_field(levels,0.,2.,color)[0]
        np.testing.assert_array_equal(image[0,0],color)  # mean 2, not mean(clipped)=1
        self.assertTrue((image[2,2] == image[2,2,0]).all())  # unobserved: gray anatomy
        np.testing.assert_array_equal(levels,[0.,4.,np.nan])
        zero = bmap.render_field(np.zeros(3),0.,2.,color)[0]
        np.testing.assert_array_equal(zero[0,0],[21,37,43])
        self.assertFalse(np.array_equal(zero[2,2],image[2,2]))
        for values,lo,hi in (([1.],0.,1.),([0.,0.,np.inf],0.,1.),([0.,0.,0.],1.,1.)):
            with self.assertRaises(ValueError): bmap.render_field(values,lo,hi)


if __name__ == "__main__":
    unittest.main()
