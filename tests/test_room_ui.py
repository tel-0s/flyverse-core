"""Exercise the real pygame event loop and display caches without a connectome or GPU."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pygame
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import room_demo as demo
from flyverse.body import FlyState
from flyverse.room_ui import RoomUI, Layout, canvas_size, fit_transform
from flyverse.world import Sphere


class DisplaySim:
    """Deterministic display fixture; the application continues to draw real pygame widgets."""
    def __init__(self):
        self.c = SimpleNamespace(n=167106)
        self.brain = SimpleNamespace(t=0.,step_count=0,p=SimpleNamespace(dt=.5),mean_rate=lambda _:17.25)
        self.fly = FlyState(x=-.5,z=.75)
        self.info = {"table_top_z":.75}
        self.cam_scale = 4
        self.calls = []
        self.r = SimpleNamespace(n_columns=20,col_az_el=np.c_[np.linspace(-130,130,20),np.linspace(-60,60,20)])
        self.optic = SimpleNamespace(contrast=torch.zeros(1,100),delta_rate=torch.zeros(1,20))
        self.col_rad = torch.ones(20,4)*.2
        self.sc_idx = {s:[] for s in ("ol_intrinsic","visual_projection","cb_intrinsic","descending_neuron",
                                      "vnc_intrinsic","vnc_motor","cb_motor","vnc_sensory")}
        self.world = SimpleNamespace(spheres=[Sphere((0,0,1),(.1,.1,.1),"apple")],boxes=[],
                                     light_pos=(0,0,2),light_color=(1,1,1,1),ambient=(.1,)*4,detail=1)
        self.world.render_camera = Mock(side_effect=lambda pos,f,u,w,h,fov:torch.ones(h,w,4)*.2)
        self.trail,self.trail_seconds,self.spike_hist = [],20,[]
        self.metabolism = SimpleNamespace(energy=.6,state="ok",meals=0)
        self.air = SimpleNamespace(direction=0)
        self.cmd = dict(rates={f"Motor population {i}":i for i in range(25)},speed=0.,yaw=0.,proboscis=0.)
        self.wcmd = dict(gf=0,ttm=0,power=0)

    def step(self):
        self.brain.t += 10
        self.brain.step_count += 20
        self.spike_hist.append(15)

    def nearest_fruit(self): return "apple",.2
    def reset_fly(self): self.calls.append("reset")
    def teleport_to_fruit(self): self.calls.append("apple")
    def start_loom(self): self.calls.append("loom")
    def stimulate_gf(self): self.calls.append("escape")
    def stimulate_wing_dns(self): self.calls.append("flight")
    def save_state(self,path): self.calls.append("save")
    def load_state(self,path): self.calls.append("load")


class ObservatoryTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ,{"SDL_VIDEODRIVER":"dummy","SDL_AUDIODRIVER":"dummy"})
        self.environment.start()
        pygame.init()

    def tearDown(self):
        pygame.quit()
        self.environment.stop()

    def test_clicks_resize_scroll_and_modal_in_real_loop(self):
        sim = DisplaySim()
        bmap = SimpleNamespace(activity=lambda *_:np.zeros(20),
            render=lambda _: (np.zeros((60,100,3),np.uint8),np.zeros((40,100,3),np.uint8)),top_types=lambda _: {})
        frames, mouse = [0], [(0,0)]
        def key(value): return pygame.event.Event(pygame.KEYDOWN,key=value)
        def click(action):
            hit = next(h for h in sim._ui.hits if h.action==action)
            scale,x,y = fit_transform(sim._ui.layout.size,pygame.display.get_surface().get_size())
            mouse[0] = (round(hit.rect.centerx*scale+x),round(hit.rect.centery*scale+y))
            return pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=mouse[0])
        def events():
            frame = frames[0]; frames[0]+=1
            if frame == 0: return []
            if frame == 1:
                self.assertEqual(sim._ui.tab,"atlas")  # --brain-map opens the atlas at launch
                return [click("pause")]
            if frame == 2:
                self.assertEqual(sim.brain.t,10.)
                return [click("tab:motor")]
            if frame == 3:
                mouse[0] = sim._ui.layout.inspector.center
                return [pygame.event.Event(pygame.MOUSEWHEEL,y=-20,x=0)]
            if frame == 4:
                self.assertGreater(sim._ui.scroll["motor"],0)
                return [pygame.event.Event(pygame.VIDEORESIZE,w=900,h=620)]
            if frame == 5: return [click("follow")]
            if frame == 6: return [key(pygame.K_v)]
            if frame == 7:
                self.assertEqual(sim._ui.retina_mode,"contrast")
                return [click("help")]
            if frame == 8: return [key(pygame.K_r)]  # must be consumed by the guide
            if frame == 9: return [key(pygame.K_ESCAPE)]
            if frame == 10:
                self.assertEqual(sim.brain.t,10.)
                self.assertNotIn("reset",sim.calls)
                return [click("pause")]
            if frame == 11: return [key(pygame.K_h)]
            if frame == 12:
                self.assertEqual(sim.brain.t,20.)
                return [key(pygame.K_ESCAPE)]
            if frame == 13:
                self.assertEqual(sim.brain.t,30.)  # closing guide resumes a previously running brain
                return [click("save")]
            if frame == 14: return [click("load")]
            if frame == 15: return [click("loom")]
            if frame == 16: return [click("escape"),click("flight"),key(pygame.K_4)]
            self.assertEqual(sim._ui.tab,"atlas")
            return [pygame.event.Event(pygame.QUIT)]
        real_exists = os.path.exists
        with patch.object(demo,"Sim",return_value=sim), patch.object(sys,"argv",["room_demo.py","--headless","--brain-map"]), \
             patch.object(demo.brainmap,"BrainMap",return_value=bmap), \
             patch.object(pygame.event,"get",side_effect=events), patch.object(pygame.mouse,"get_pos",side_effect=lambda:mouse[0]), \
             patch.object(pygame.time,"Clock",return_value=SimpleNamespace(tick=lambda _:None)), \
             patch.object(os.path,"exists",side_effect=lambda path:path=="out/quicksave.pt" or real_exists(path)):
            demo.main()
        self.assertEqual(sim.calls,["save","load","loom","escape","flight"])
        self.assertEqual(sim.brain.t,70.)

    def test_readonly_redraw_and_moving_scene_cache(self):
        sim = DisplaySim()
        ui,orbit = RoomUI(),demo.OrbitCam((0,0,.75))
        surface = pygame.Surface((1440,960))
        ui.draw(sim,surface,orbit,paused=True)
        calls = sim.world.render_camera.call_count
        for _ in range(3): ui.draw(sim,surface,orbit,paused=True)
        self.assertEqual(sim.world.render_camera.call_count,calls)
        self.assertEqual(sim.brain.t,0.)
        self.assertEqual(sim.calls,[])
        sim.world.spheres[0].center = (1,0,1)
        ui.draw(sim,surface,orbit,paused=True)
        self.assertEqual(sim.world.render_camera.call_count,calls+2)

    def test_paused_atlas_keeps_activity_and_all_targets_fit(self):
        sim,ui,orbit = DisplaySim(),RoomUI(),demo.OrbitCam((0,0,.75))
        bmap = SimpleNamespace(activity=Mock(return_value=np.linspace(0,1,20)),
            render=lambda _: (np.zeros((60,100,3),np.uint8),np.zeros((40,100,3),np.uint8)),top_types=lambda _: {})
        for window in ((800,600),(1100,760),(1440,960),(1920,1080)):
            surface = pygame.Surface(canvas_size(window))
            ui.draw(sim,surface,orbit,paused=True,bmap=bmap)
            self.assertTrue(all(surface.get_rect().contains(hit.rect) for hit in ui.hits))
            for i,a in enumerate(ui.hits):
                self.assertFalse(any(a.rect.colliderect(b.rect) for b in ui.hits[i+1:]),a.action)
        self.assertEqual(bmap.activity.call_count,1)
        np.testing.assert_array_equal(ui._map_shown,np.linspace(0,1,20))
        ui.handle_action("tab:motor",sim)
        sim.step()
        ui.draw(sim,surface,orbit,bmap=bmap)
        self.assertEqual(ui.tab,"motor")  # startup flag must not force the map back on
        self.assertEqual(bmap.activity.call_count,1)  # hidden maps do no activity sampling
        ui.handle_action("tab:atlas",sim)
        ui.draw(sim,surface,orbit,bmap=bmap)
        self.assertEqual(ui.tab,"atlas")


if __name__ == "__main__":
    unittest.main()
