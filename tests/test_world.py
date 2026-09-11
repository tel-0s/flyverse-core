"""Ray-capture contracts, including changing rays, scene edits and output ownership."""
import unittest

import torch

from flyverse.world import Box, Plane, Sphere, World, make_room


def rays(n=67, seed=0):
    generator = torch.Generator().manual_seed(seed)
    d = torch.randn(n, 3, generator=generator)
    d /= d.norm(dim=-1, keepdim=True)
    o = torch.tensor([-.5, .05, .7512]).expand_as(d)
    return o, d


class WorldTests(unittest.TestCase):
    def test_move_and_invalidate_eager_scene(self):
        w = World(spheres=[Sphere((1, 0, 0), (.2, .2, .2), "apple")],
                  device="cpu", detail=0, ambient=(1, 1, 1, 1), light_color=(0, 0, 0, 0))
        o, d = torch.zeros(1, 3), torch.tensor([[1., 0, 0]])
        torch.testing.assert_close(w.trace(o, d), torch.tensor([[.05, .08, .20, .85]]))
        w.move_sphere(0, (1, 1, 0))
        torch.testing.assert_close(w.trace(o, d), torch.zeros(1, 4), rtol=0, atol=0)
        w.spheres[0].center = (1, 0, 0)
        w.spheres[0].material = "lamp"
        w.invalidate()
        torch.testing.assert_close(w.trace(o, d), torch.tensor([[.6, 1., 1., 1.]]))

    def test_graph_request_requires_cuda(self):
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            World(device="cpu").trace(*rays(), cuda_graphs=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RayGraphTests(unittest.TestCase):
    def assert_capture_matches(self, w, o, d):
        expected = w.trace(o, d)
        actual = w.trace(o, d, cuda_graphs=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return actual

    def test_changing_rays_and_independent_output_storage(self):
        w, _ = make_room()
        w.device = "cuda"
        first = self.assert_capture_matches(w, *rays())
        snapshot = first.clone()
        # Noncontiguous inputs with the same shape must update the captured ray buffers.
        o, d = rays(134, seed=8)
        second = self.assert_capture_matches(w, o[::2] + .03, d[::2])
        self.assertFalse(torch.equal(second, first))
        # The display camera shares the scene but must not overwrite the sensory image.
        w.render_camera((0, 0, 1), (1, 0, 0), (0, 0, 1), width=12, height=8)
        torch.testing.assert_close(first, snapshot, rtol=0, atol=0)
        self.assertEqual(len(w._trace_graphs), 1)

    def test_moving_and_resizing_sphere_keeps_capture(self):
        w = World(spheres=[Sphere((1, 0, .5), (.3, .3, .3), "apple")],
                  planes=[Plane((0, 0, 0), (0, 0, 1), "floor")], device="cuda")
        o, d = rays()
        o = torch.tensor([[0., 0., .5]]).expand_as(d)
        d[0] = torch.tensor([1., 0., 0.])
        first = self.assert_capture_matches(w, o, d).clone()
        graph = next(iter(w._trace_graphs.values()))[0]
        for center, radii in (((1, .8, .5), (.3, .3, .3)), ((.5, 0, .5), (.4, .2, .1))):
            w.move_sphere(0, center, radii)
            actual = self.assert_capture_matches(w, o, d)
            self.assertIs(next(iter(w._trace_graphs.values()))[0], graph)
            self.assertFalse(torch.equal(first, actual))

    def test_topology_material_and_lighting_changes(self):
        w, _ = make_room()
        w.device = "cuda"
        o, d = rays()
        self.assert_capture_matches(w, o, d)
        w.boxes.append(Box((-.7, -.1, .8), (-.3, .3, 1.), "black"))
        self.assert_capture_matches(w, o, d)
        self.assertEqual(len(w._trace_graphs), 1)
        w.light_pos = (1, -1, 2)
        w.ambient = (.3, .2, .1, .4)
        w.spheres[0].material = "black"
        w.invalidate()
        self.assert_capture_matches(w, o, d)
        self.assertEqual(len(w._trace_graphs), 1)
        w.detail = 0  # a captured Python branch/scalar, included in the cache key
        self.assert_capture_matches(w, o, d)
        self.assertEqual(len(w._trace_graphs), 2)

    def test_bounded_cache_and_empty_scene_device_change(self):
        w = World(device="cuda")
        for n in (1, 2, 3):
            result = self.assert_capture_matches(w, *rays(n))
            torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
        self.assertEqual(len(w._trace_graphs), 2)
        w.device = "cpu"
        self.assertEqual(w.trace(*rays()).device.type, "cpu")
        self.assertEqual(len(w._trace_graphs), 0)

    def test_nondefault_stream(self):
        w, _ = make_room()
        w.device = "cuda"
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            actual = w.trace(*rays(), cuda_graphs=True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual, w.trace(*rays()), rtol=0, atol=0)
