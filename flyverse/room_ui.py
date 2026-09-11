"""The room console: dense presentation and hit testing, independent of simulation dynamics.

The layout reflows at native window resolution above MIN_SIZE; smaller windows show a
uniformly scaled canvas. Variable text has width budgets and telemetry is scrollable.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pygame

from . import world

DEFAULT_SIZE = (1360, 820)
MIN_SIZE = (1100, 720)
BG = (13, 16, 18)
PANEL = (19, 24, 27)
INSET = (9, 12, 14)
LINE = (43, 53, 58)
TEXT = (213, 223, 219)
MUTED = (151, 171, 172)
DIM = (109, 135, 139)
SAGE = (185, 219, 126)
TEAL = (107, 201, 208)
AMBER = (232, 179, 104)
LILAC = (176, 157, 216)


def canvas_size(window_size):
    """Keep both axes readable; small windows preserve proportions via letterboxing."""
    w, h = window_size
    scale = max(1.0, MIN_SIZE[0] / max(w, 1), MIN_SIZE[1] / max(h, 1))
    return round(w * scale), round(h * scale)


def fit_transform(canvas, window):
    scale = min(window[0] / canvas[0], window[1] / canvas[1])
    return scale, (window[0] - canvas[0] * scale) / 2, (window[1] - canvas[1] * scale) / 2


@dataclass
class Layout:
    size: tuple[int, int]

    def __post_init__(self):
        w, h = self.size
        pad, gap, y = 12, 16, 96
        height = h - y - 34
        left = round((w - 2*pad - gap) * .55)
        camera_h = min(244, max(172, round(height * .30)))
        retina_h = min(226, max(174, round(height * .27)))
        eye_w = round((left - 12) * .59)
        self.eye = pygame.Rect(pad, y, eye_w, camera_h)
        self.scene = pygame.Rect(self.eye.right+12, y, left-eye_w-12, camera_h)
        self.scene_view = pygame.Rect(self.scene.x, y+26, self.scene.w, camera_h-45)
        self.retina = pygame.Rect(pad, self.eye.bottom+10, left, retina_h)
        self.trace = pygame.Rect(pad, self.retina.bottom+10, left, 96)
        self.senses = pygame.Rect(pad, self.trace.bottom+12, left, h-34-self.trace.bottom-12)
        self.inspector = pygame.Rect(pad+left+gap, y, w-2*pad-left-gap, height)
        self.inspector_view = pygame.Rect(self.inspector.x+5, y+39, self.inspector.w-12, height-49)
        self.footer = pygame.Rect(pad, h-25, w-pad*2, 20)


@dataclass
class Hit:
    rect: pygame.Rect
    action: str
    tip: str = ""


class RoomUI:
    def __init__(self):
        self.fonts = {}
        self.text_cache = {}
        self.hits = []
        self.mouse = (-1, -1)
        self.tab = "regions"
        self.retina_mode = "both"
        self.scroll = {"regions": 0, "motor": 0, "senses": 0, "atlas": 0}
        self.scroll_limit = 0
        self.help_open = False
        self.toast = None
        self.bmap = None
        self.layout = Layout(DEFAULT_SIZE)
        self._last_clock = None
        self._last_sim_time = None
        self.fps = self.rtf = None
        self._eye_key = self._scene_key = None
        self._map_time = None
        self._map_count = 0

    def font(self, size=14, weight="normal"):
        key = size, weight
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont("Cascadia Mono,Consolas,DejaVu Sans Mono,Liberation Mono",
                                                size, bold=weight == "bold")
        return self.fonts[key]

    def text(self, surface, value, xy, size=14, color=TEXT, weight="normal", width=None, right=False):
        value = str(value)
        font = self.font(size, weight)
        if width is not None:
            if width <= 0:
                return pygame.Rect(*xy, 0, 0)
            if font.size(value)[0] > width:
                if font.size("…")[0] > width:
                    return pygame.Rect(*xy,0,0)
                while value and font.size(value + "…")[0] > width:
                    value = value[:-1]
                value += "…"
        key = (value, size, color, weight)
        if key not in self.text_cache:
            if len(self.text_cache) > 900:
                self.text_cache.clear()
            self.text_cache[key] = font.render(value, True, color)
        image = self.text_cache[key]
        rect = image.get_rect(topright=xy) if right else image.get_rect(topleft=xy)
        surface.blit(image, rect)
        return rect

    def panel(self, surface, rect):
        pygame.draw.rect(surface, PANEL, rect)
        pygame.draw.rect(surface, LINE, rect, 1)

    def label(self, surface, text, pos, color=MUTED, width=None):
        return self.text(surface, text, pos, 11, color, "bold", width)

    def button(self, surface, rect, label, action, *, active=False, primary=False, tip="", key=None, padding=6):
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(self.mouse) and not self.help_open
        color = SAGE if primary or active else TEXT
        if hover or active:
            pygame.draw.rect(surface, (30, 42, 44), rect)
        if active:
            pygame.draw.line(surface, SAGE, (rect.x,rect.bottom-1), (rect.right-1,rect.bottom-1))
        reserve = self.font(11).size(key)[0]+10 if key else 0
        self.text(surface, label, (rect.x+padding,rect.centery-8), 12, color,
                  "normal", rect.w-2*padding-reserve)
        if key:
            self.text(surface, key, (rect.right-padding,rect.centery-7), 11, DIM, right=True)
        self.hits.append(Hit(rect, action, tip))

    def notify(self, text):
        self.toast = str(text), time.perf_counter() + 3.5

    def action_at(self, pos):
        if self.help_open:
            return "help"  # clicking anywhere closes the shortcut sheet; never activates behind it
        return next((hit.action for hit in reversed(self.hits) if hit.rect.collidepoint(pos)), None)

    def handle_action(self, action, sim):
        """Consume presentation actions. Simulation/camera actions are handled by the demo."""
        if action == "help":
            self.help_open = not self.help_open
        elif action.startswith("tab:"):
            tab = action.split(":", 1)[1]
            if tab == "atlas" and self.bmap is None:
                try:
                    from .brainmap import BrainMap
                    self.bmap = BrainMap(sim.c, sim.optic)
                except (OSError, ValueError, KeyError) as exc:
                    self.notify(f"Brain atlas unavailable: {exc}")
                    return True
            self.tab = tab
        elif action.startswith("retina:"):
            self.retina_mode = action.split(":", 1)[1]
        else:
            return False
        return True

    def wheel(self, pos, amount):
        if self.help_open:
            return True
        if self.layout.inspector.collidepoint(pos):
            self.scroll[self.tab] = max(0, min(self.scroll_limit, self.scroll[self.tab] - amount * 34))
            return True
        return False

    def _header(self, sim, surface, paused):
        w = surface.get_width()
        self.text(surface,"flyverse",(12,7),20,SAGE,"bold")
        self.text(surface,"// room",(123,12),13,MUTED)
        self.text(surface,"[PAUSED]" if paused else "[RUN]",(222,11),14,AMBER if paused else SAGE)
        self.text(surface,f"t {sim.brain.t/1000:010.3f} s",(330,9),17,TEXT)
        self.text(surface,f"step {sim.brain.step_count:,}",(538,13),12,DIM,width=190)
        self.text(surface,f"MaleCNS  n={sim.c.n:,}  columns={sim.r.n_columns:,}",(w-12,13),12,MUTED,right=True)
        pygame.draw.line(surface,LINE,(12,35),(w-12,35))
        x, y = 12, 39
        buttons = [(104,"run" if paused else "pause","pause","SPACE"), (77,"reset","reset","R"),
                   (83,"apple","apple","T"), (72,"loom","loom","L"),
                   (87,"escape","escape","F"), (85,"flight","flight","W")]
        tips = {"pause":"Pause or resume brain time", "reset":"Reset the fly's pose", "apple":"Teleport next to the apple",
                "loom":"Present an approaching black sphere", "escape":"Stimulate the giant fibre", "flight":"Stimulate flight descending neurons"}
        for i,(width,label,action,key) in enumerate(buttons):
            self.button(surface,(x,y,width,23),label,action,primary=i==0,key=key,tip=tips[action])
            x += width + 4
        for width,label,action,key in reversed([(82,"save","save","F5"),(82,"load","load","F9"),(72,"keys","help","?")]):
            w -= width
            self.button(surface,(w-12,y,width,23),label,action,key=key,tip=f"{label} · {key}")
            w -= 4
        brain = sim.brain
        device = str(getattr(getattr(sim,"fb",None),"device","cpu"))
        backend = "cuda kernels" if getattr(brain,"cuda",False) else "metal kernels" if getattr(brain,"metal",False) else "torch"
        sparse = "metal" if getattr(brain,"metal",False) else getattr(brain,"cuda_sparse","torch")
        events = "events" if getattr(brain,"event_driven",False) else "spmm"
        weights = str(getattr(brain.p,"weight_dtype","float32")).replace("torch.","")
        optic_dt = getattr(getattr(sim.optic,"p",None),"dt_ms",1.)
        graphs = "on" if getattr(getattr(sim,"fb",None),"cuda_graphs",False) else "off"
        config = f"{device} / {backend} / {events}:{sparse}   lif {brain.p.dt:g}ms   optic {optic_dt:g}ms   {weights}   graphs {graphs}"
        self.text(surface,config,(12,72),11,DIM,width=surface.get_width()-330)
        self.text(surface,"D. melanogaster / closed loop",(surface.get_width()-12,72),11,DIM,right=True)

    def _rule(self, surface, rect, title, color=MUTED):
        self.label(surface,title,(rect.x,rect.y),color,width=rect.w)
        pygame.draw.line(surface,LINE,(rect.x,rect.y+20),(rect.right,rect.y+20))

    def _image(self, surface, image, rect):
        if image.get_size() != rect.size:
            image = pygame.transform.smoothscale(image, rect.size)
        surface.blit(image, rect)

    def _scene(self, sim, surface, orbit):
        r, viewport = self.layout.scene, self.layout.scene_view
        self.label(surface,"ORBIT",r.topleft)
        self.button(surface,(r.right-139,r.y-4,80,23),"follow","follow",active=orbit.follow,key="C")
        self.button(surface,(r.right-54,r.y-4,54,23),"home","home",tip="Reset camera · Home")
        cam = orbit.camera(sim.fly, *viewport.size)
        # Background caching includes moving geometry, so the approaching loom stays visible.
        key = (orbit.key(), viewport.size, sim.cam_scale,
               tuple((s.center,s.radii,s.material) for s in sim.world.spheres),
               tuple((b.lo,b.hi,b.material) for b in sim.world.boxes),
               sim.world.light_pos, sim.world.light_color, sim.world.ambient, sim.world.detail)
        if key != self._scene_key:
            image = world.to_rgb8(cam.render(sim.world, scale=max(1,sim.cam_scale//2)), exposure=2.0)
            self._scene_image = pygame.surfarray.make_surface(np.transpose(image,(1,0,2)))
            self._scene_key = key
        self._image(surface,self._scene_image,viewport)
        old_clip = surface.get_clip(); surface.set_clip(viewport)
        now = sim.brain.t / 1000
        prev = None
        for ts,position in sim.trail:
            q = cam.project(position)
            if q is None or not (0 <= q[0] < viewport.w and 0 <= q[1] < viewport.h):
                prev = None
                continue
            alpha = max(0,1-(now-ts)/max(sim.trail_seconds,.001))
            color = tuple(round(40+(c-40)*alpha) for c in SAGE)
            pt = viewport.x+q[0], viewport.y+q[1]
            if prev:
                pygame.draw.line(surface,color,prev,pt,2)
            prev = pt
        q = cam.project(sim.fly.eye_pos)
        if q and 0 <= q[0] < viewport.w and 0 <= q[1] < viewport.h:
            point = viewport.x+q[0],viewport.y+q[1]
            pygame.draw.circle(surface,BG,point,5)
            pygame.draw.circle(surface,SAGE,point,5,1)
            pygame.draw.circle(surface,SAGE,point,2)
            forward = cam.project(sim.fly.eye_pos+.04*sim.fly.forward)
            if forward:
                pygame.draw.aaline(surface,SAGE,point,(viewport.x+forward[0],viewport.y+forward[1]))
        surface.set_clip(old_clip)
        self.text(surface,f"az {orbit.az:.0f}  el {orbit.el:.0f}  d {orbit.dist:.2f}m",
                  (r.x,r.bottom-15),11,DIM,width=r.w)

    def _eye(self, sim, surface):
        r = self.layout.eye
        self.label(surface,"BODY CAM / 110°",r.topleft,TEAL)
        self.text(surface,"human RGB",(r.right,r.y),11,DIM,right=True)
        viewport = pygame.Rect(r.x,r.y+26,r.w,r.h-45)
        fly = sim.fly
        key = (sim.brain.t,tuple(fly.eye_pos),tuple(fly.forward),tuple(fly.up),viewport.size,sim.cam_scale,
               tuple((s.center,s.radii) for s in sim.world.spheres))
        if key != self._eye_key:
            scale = max(1,sim.cam_scale)
            image = sim.world.render_camera(fly.eye_pos,fly.forward,fly.up,max(1,viewport.w//scale),max(1,viewport.h//scale),110)
            self._eye_image = pygame.surfarray.make_surface(np.transpose(world.to_rgb8(image,exposure=2.5),(1,0,2)))
            self._eye_key = key
        self._image(surface,self._eye_image,viewport)
        # A restrained reticle belongs to the display camera, not the sensory renderer.
        x,y = viewport.center
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            pygame.draw.line(surface,(213,223,214),(x+dx*4,y+dy*4),(x+dx*10,y+dy*10))
        self.text(surface,f"hdg {np.rad2deg(fly.heading)%360:05.1f}°  pitch {np.rad2deg(fly.pitch):+.0f}  roll {np.rad2deg(fly.roll):+.0f}",
                  (r.x,r.bottom-15),11,DIM,width=r.w)

    def _mosaic(self, sim, surface, viewport, colors):
        az, el = sim.r.col_az_el.T
        key = id(sim.r), tuple(viewport)
        cache = getattr(self, "_mosaics", {})
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            scale = min((viewport.w-20)/max(np.ptp(az),1), (viewport.h-8)/max(np.ptp(el),1))
            xy = np.c_[-(az-(az.max()+az.min())/2)*scale+viewport.w/2,
                        -(el-(el.max()+el.min())/2)*scale+viewport.h/2]
            radius = max(1., min(3.2, scale*1.35))
            angles = np.arange(6)*np.pi/3
            cache[key] = (np.round(xy[:,None,:]+radius*np.c_[np.cos(angles),np.sin(angles)]).astype(int)
                          + np.array(viewport.topleft)).tolist()
            self._mosaics = cache
        pygame.draw.rect(surface, INSET, viewport)
        old_clip = surface.get_clip(); surface.set_clip(viewport)
        for poly, color in zip(cache[key], colors.tolist()):
            pygame.draw.polygon(surface, color, poly)
        surface.set_clip(old_clip)

    def _retina(self, sim, surface):
        r = self.layout.retina
        self.label(surface, f"RETINA / {sim.r.n_columns:,} COLUMNS", r.topleft, TEAL)
        x = r.right-233
        for mode, label, width in (("both","both",52),("colour","UV/B/G",77),("contrast","ON/OFF",84)):
            self.button(surface,(x,r.y-4,width,23),label,"retina:"+mode,
                        active=self.retina_mode==mode,tip="V cycles both / colour / contrast")
            x += width+6
        modes = ("colour","contrast") if self.retina_mode=="both" else (self.retina_mode,)
        width = (r.w-12*(len(modes)-1))//len(modes)
        for i, mode in enumerate(modes):
            view = pygame.Rect(r.x+i*(width+12),r.y+26,width,r.h-45)
            if mode=="colour":
                colors = world.to_fly_false_color(sim.col_rad,exposure=2.5).astype(np.uint8)
            else:
                contrast = sim.optic.contrast[0].reshape(-1,5)[:,0].detach().cpu().numpy()
                gray = np.clip(128+127*contrast/.5,0,255).astype(np.uint8)
                colors = np.repeat(gray[:,None],3,axis=1)
            self._mosaic(sim,surface,view,colors)
            if mode=="colour":
                self.text(surface,"UV",(view.x,r.bottom-15),11,LILAC)
                self.text(surface,"B",(view.x+25,r.bottom-15),11,TEAL)
                self.text(surface,"G",(view.x+43,r.bottom-15),11,SAGE)
                self.text(surface,"false colour",(view.right,r.bottom-15),11,DIM,right=True)
            else:
                self.text(surface,"R1-R6 / contrast",(view.x,r.bottom-15),11,MUTED)
                self.text(surface,"OFF -/+ ON",(view.right,r.bottom-15),11,DIM,right=True)

    def _trace(self, sim, surface):
        r = self.layout.trace
        self.label(surface,"LIF SPIKES / STEP",r.topleft)
        value = sim.spike_hist[-1] if sim.spike_hist else 0
        self.text(surface,f"{value:,.0f}",(r.right,r.y-2),15,SAGE,right=True)
        plot = pygame.Rect(r.x,r.y+24,r.w,r.h-41)
        for i in range(5):
            x = plot.x+round(i*(plot.w-1)/4)
            pygame.draw.line(surface,LINE,(x,plot.y),(x,plot.bottom))
            self.text(surface,"now" if i==4 else f"-{4-i}s",(min(x,r.right-23),plot.bottom+2),10,DIM)
        history = np.asarray(sim.spike_hist[-400:])
        if history.size > 1:
            maximum = max(10,float(history.max()))
            x = np.linspace(plot.right-1-(len(history)-1)*(plot.w-1)/399,plot.right-1,len(history))
            y = plot.bottom-2-(plot.h-7)*history/maximum
            points = list(zip(x.round().astype(int),y.round().astype(int)))
            pygame.draw.lines(surface,SAGE,False,points)
            self.text(surface,f"peak {maximum:.0f}",(plot.x+7,plot.y),10,DIM)

    def _meter(self, surface, rect, label, value, unit="Hz", color=TEAL, limit=50):
        self.text(surface,label,(rect.x,rect.y),12,TEXT,width=rect.w-92)
        self.text(surface,f"{value:6.2f} {unit}",(rect.right,rect.y),12,color,width=92,right=True)
        bar = pygame.Rect(rect.right-78,rect.y+18,78,2)
        pygame.draw.rect(surface,LINE,bar)
        length = round(bar.w*np.clip(abs(value)/max(limit,.0001),0,1))
        if length:
            pygame.draw.rect(surface,color,(bar.x,bar.y,length,bar.h))

    def _row(self, surface, viewport, y, label, value, color=TEXT):
        self.text(surface,label,(viewport.x,y),12,MUTED,width=viewport.w*.39)
        self.text(surface,value,(viewport.right,y),12,color,width=viewport.w*.60,right=True)
        return y+21

    def _regions(self, sim, surface, viewport, y):
        self._rule(surface,pygame.Rect(viewport.x,y,viewport.w,22),"POPULATIONS / MEAN Hz",TEAL)
        y+=29
        names = [("ol_intrinsic","optic_lobe"),("visual_projection","visual_proj"),("cb_intrinsic","central_brain"),
                 ("descending_neuron","descending"),("vnc_intrinsic","vnc_intrinsic"),
                 ("vnc_motor","vnc_motor"),("cb_motor","cb_motor"),("vnc_sensory","vnc_sensory")]
        for name,label in names:
            graded = name=="ol_intrinsic"
            value = float(sim.optic.delta_rate.abs().mean())*100 if graded else sim.brain.mean_rate(sim.sc_idx[name])
            self._meter(surface,pygame.Rect(viewport.x,y,viewport.w,23),label,value,"%Δ" if graded else "Hz",LILAC if graded else TEAL)
            y+=23
        self.text(surface,"optic = mean |rate change|",(viewport.x,y+3),10,DIM,width=viewport.w)
        return y+26

    def _motor_values(self, sim, surface, viewport, y):
        if not hasattr(sim,"cmd"):
            self.text(surface,"Awaiting first frame.",(viewport.x,y),12,DIM)
            return y+24
        for title,values in (("WALK / MEAN Hz",sim.cmd["rates"]),("FLIGHT / MEAN Hz",sim.wcmd)):
            self._rule(surface,pygame.Rect(viewport.x,y,viewport.w,22),title,AMBER)
            y+=29
            for name,value in values.items():
                unit = "" if "gate" in name.lower() or "x10" in name else "Hz"
                self._meter(surface,pygame.Rect(viewport.x,y,viewport.w,23),name,float(value),unit,AMBER,80)
                y+=23
            y+=13
        return y

    def _body(self, sim, surface, viewport, y):
        fly = sim.fly
        self._rule(surface,pygame.Rect(viewport.x,y,viewport.w,22),"COMMAND -> BODY",SAGE)
        y+=29
        cmd = getattr(sim,"cmd",{})
        for label,value in (("v_cmd",f"{cmd.get('speed',0)*100:+.2f} cm/s"),
                            ("yaw_cmd",f"{np.rad2deg(cmd.get('yaw',0)):+.1f} deg/s"),
                            ("proboscis",f"{cmd.get('proboscis',0):.3f}")):
            y = self._row(surface,viewport,y,label,value,SAGE)
        y+=12
        face = fly.face.label if getattr(fly,"face",None) is not None else "surface"
        mode = "airborne" if fly.airborne else face
        rows = [("surface",mode),("x / y",f"{fly.x:+.3f} / {fly.y:+.3f} m"),("z",f"{fly.z:.3f} m"),
                ("v_actual",f"{fly.speed*100:+.2f} cm/s"),("heading",f"{np.rad2deg(fly.heading)%360:.1f} deg"),
                ("energy",f"{sim.metabolism.energy*100:.1f}% / {sim.metabolism.state}"),
                ("meals",str(sim.metabolism.meals)),
                ("taste/feed",f"{int(getattr(sim,'tasting',0)>0)} / {int(getattr(sim,'feeding',False))}")]
        if fly.airborne:
            rows += [("flight vx/y",f"{fly.vx*100:+.1f}/{fly.vy*100:+.1f} cm/s"),
                     ("flight vz",f"{fly.vz*100:+.1f} cm/s"),("air_time",f"{fly.air_time:.2f} s")]
        for label,value in rows:
            y = self._row(surface,viewport,y,label,value)
        return y

    def _overview(self, sim, surface, viewport, y):
        width = (viewport.w-20)//2
        left = pygame.Rect(viewport.x,y,width,viewport.h)
        right = pygame.Rect(left.right+20,y,viewport.right-left.right-20,viewport.h)
        end_left = self._regions(sim,surface,left,y)
        end_left = self._body(sim,surface,left,end_left+10)
        end_right = self._motor_values(sim,surface,right,y)
        self.text(surface,"readout names = model keys",(right.x,end_right+3),10,DIM,width=right.w)
        program = getattr(sim,"program",None)
        name = type(program).__name__ if program is not None else "none"
        end_right = self._row(surface,right,end_right+29,"program",name)
        return max(end_left,end_right)

    def _motors(self, sim, surface, viewport, y):
        y = self._motor_values(sim,surface,viewport,y)
        return self._body(sim,surface,viewport,y)

    def _odor_rows(self, sim):
        left,right = getattr(sim,"smell_values",({},{}))
        def val(values,name): return float(np.asarray(values.get(name,0)).reshape(-1)[0])
        names = sorted(set(left)|set(right),key=lambda n:-(val(left,n)+val(right,n)))
        return [(name,val(left,name),val(right,name)) for name in names]

    def _odor_table(self, surface, viewport, y, rows):
        self.text(surface,"glom",(viewport.x,y),11,DIM)
        for text,x in (("L",viewport.right-145),("R",viewport.right-72),("L-R",viewport.right)):
            self.text(surface,text,(x,y),11,DIM,right=True)
        y+=21
        for name,left,right in rows:
            self.text(surface,name,(viewport.x,y),12,TEXT,width=viewport.w-215)
            self.text(surface,f"{left:.3g}",(viewport.right-145,y),12,TEAL,width=69,right=True)
            self.text(surface,f"{right:.3g}",(viewport.right-72,y),12,AMBER,width=69,right=True)
            self.text(surface,f"{left-right:+.2g}",(viewport.right,y),12,MUTED,width=69,right=True)
            y+=20
        return y

    def _sensory_summary(self, sim, surface):
        r = self.layout.senses
        rows = self._odor_rows(sim)
        width = (r.w-22)//2
        count = max(0,(r.h-51)//20)
        self._rule(surface,r,"ANTENNAE / REL CONC",TEAL)
        wind = (np.rad2deg(sim.air.direction)+180)%360
        self.text(surface,f"top {min(len(rows),count*2)}/{len(rows)}  wind from {wind:.0f}°  [3] all",
                  (r.right,r.y),11,DIM,right=True)
        shown = rows[:count*2]
        split = (len(shown)+1)//2
        for i in range(2):
            view = pygame.Rect(r.x+i*(width+22),r.y+27,width,r.h-27)
            self._odor_table(surface,view,view.y,shown[i*split:(i+1)*split])
        if not rows:
            self.text(surface,"Awaiting odor sample.",(r.x,r.y+52),12,DIM)

    def _senses(self, sim, surface, viewport, y):
        self._rule(surface,pygame.Rect(viewport.x,y,viewport.w,22),"SENSORY INPUT / CURRENT FRAME",TEAL)
        y+=29
        for label,value in (("sugar contact",str(int(getattr(sim,"tasting",0)>0))),
                            ("wind from",f"{(np.rad2deg(sim.air.direction)+180)%360:.1f} deg")):
            y = self._row(surface,viewport,y,label,value,SAGE)
        y+=17
        self.text(surface,"Antennal concentration / strongest first",(viewport.x,y),12,MUTED,width=viewport.w)
        return self._odor_table(surface,viewport,y+26,self._odor_rows(sim))

    def _atlas(self, sim, surface, viewport, y):
        bmap = self.bmap
        if bmap is None:
            self.text(surface,"Select map to load soma positions.",(viewport.x,y),12,MUTED,width=viewport.w)
            return y+30
        # Rendering does not decay a paused map; hidden maps do not sample the brain.
        if self._map_time != sim.brain.t:
            every = max(1,int(getattr(sim,"map_every",4)))
            shown = getattr(self,"_map_shown",None)
            if shown is None or self._map_count%every == 0:
                activity = bmap.activity(sim.brain,sim.optic)
                shown = activity if shown is None or every==1 or getattr(sim,"map_no_blur",False) else np.maximum(shown,activity)
            elif not getattr(sim,"map_no_blur",False):
                shown = shown*float(np.exp(-1/(2*every)))
            self._map_shown = shown
            self._map_images = bmap.render(shown)
            self._map_time = sim.brain.t
            self._map_count += 1
        width = (viewport.w-16)//2
        image_bottom = y
        for i,(title,image) in enumerate(zip(("DORSAL / BRAIN -> VNC","LATERAL / DORSAL UP"),self._map_images)):
            x = viewport.x+i*(width+16)
            self.label(surface,title,(x,y),DIM,width=width)
            height = round(image.shape[0]*width/image.shape[1])
            self._image(surface,pygame.surfarray.make_surface(np.transpose(image,(1,0,2))),pygame.Rect(x,y+26,width,height))
            image_bottom = max(image_bottom,y+26+height)
        y = image_bottom+18
        self.text(surface,f"{int((self._map_shown>.1).sum()):,} cells with activity > 0.1",(viewport.x,y),12,AMBER)
        y+=33
        left = pygame.Rect(viewport.x,y,width,viewport.h)
        right = pygame.Rect(viewport.x+width+16,y,width,viewport.h)
        end_left = self._regions(sim,surface,left,y)
        self._rule(surface,right,"TOP TYPES / NORM ACTIVITY",AMBER)
        end_right = y+29
        for name,value in bmap.top_types(self._map_shown).items():
            self.text(surface,name,(right.x,end_right),12,TEXT,width=right.w-61)
            self.text(surface,f"{value:.3f}",(right.right,end_right),12,AMBER,right=True)
            end_right+=21
        self.text(surface,"LIF: Hz/40; optic: |Δr|×2",(right.x,end_right+15),10,DIM,width=right.w)
        return max(end_left,end_right+37)

    def _inspector(self, sim, surface):
        r = self.layout.inspector
        pygame.draw.line(surface,LINE,(r.x-8,r.y),(r.x-8,r.bottom))
        for i,(tab,title) in enumerate((("regions","live"),("motor","motor"),("senses","senses"),("atlas","map"))):
            width = (r.w-12)//4
            self.button(surface,(r.x+i*(width+4),r.y-4,width,27),title,"tab:"+tab,
                        active=self.tab==tab,key=str(i+1))
        viewport = self.layout.inspector_view
        old_clip = surface.get_clip(); surface.set_clip(viewport)
        start = viewport.y-self.scroll[self.tab]
        end = {"regions":self._overview,"motor":self._motors,"senses":self._senses,"atlas":self._atlas}[self.tab](sim,surface,viewport,start)
        surface.set_clip(old_clip)
        self.scroll_limit = max(0,end-start-viewport.h)
        self.scroll[self.tab] = min(self.scroll[self.tab],self.scroll_limit)
        if self.scroll_limit:
            track = pygame.Rect(r.right-2,viewport.y,2,viewport.h)
            pygame.draw.rect(surface,LINE,track)
            height = max(24,round(viewport.h*viewport.h/(end-start)))
            offset = round((viewport.h-height)*self.scroll[self.tab]/self.scroll_limit)
            pygame.draw.rect(surface,DIM,(track.x,track.y+offset,2,height))
            self.text(surface,"scroll",(r.right,r.bottom-10),10,DIM,right=True)

    def _footer(self, sim, surface, paused):
        r = self.layout.footer
        now = time.perf_counter()
        if self._last_clock is not None:
            dt = max(.001,now-self._last_clock)
            fps,rtf = 1/dt,max(0,sim.brain.t-self._last_sim_time)/1000/dt
            self.fps = fps if self.fps is None else .85*self.fps+.15*fps
            self.rtf = rtf if self.rtf is None else .85*self.rtf+.15*rtf
        self._last_clock,self._last_sim_time = now,sim.brain.t
        face = sim.fly.face.label if getattr(sim.fly,"face",None) is not None else "surface"
        place = "airborne" if sim.fly.airborne else face
        fruit,distance = sim.nearest_fruit()
        pygame.draw.line(surface,LINE,(r.x,r.y-6),(r.right,r.y-6))
        self.text(surface,f"{place} / nearest {fruit} {max(0,distance)*100:.1f}cm / ? keys",
                  r.topleft,11,MUTED,width=r.w//2)
        perf = "display initializing" if self.fps is None else f"draw {self.fps:.0f} fps / sim {0 if paused else self.rtf:.2f}x"
        self.text(surface,perf,(r.right,r.y),11,DIM,"mono",right=True)

    def _overlay(self, surface):
        w,h = surface.get_size()
        if self.help_open:
            shade = pygame.Surface((w,h),pygame.SRCALPHA); shade.fill((3,9,11,210)); surface.blit(shade,(0,0))
            card = pygame.Rect((w-740)//2,(h-460)//2,740,460)
            self.panel(surface,card)
            self.text(surface,"flyverse // key bindings",(card.x+20,card.y+18),18,SAGE)
            self.text(surface,"Simulation paused. Closing restores playback.",(card.x+20,card.y+51),12,MUTED)
            rows = [("Space","Pause / resume"),("R  /  T","Reset pose / move to apple"),("L","Approaching object (loom)"),
                    ("F  /  W","Giant fibre / flight DN stimulation"),("Drag  /  wheel","Orbit / zoom the room camera"),
                    ("Arrows  /  + −","Orbit / zoom with the keyboard"),("C  /  Home","Follow the fly / reset camera"),
                    ("1 / 2 / 3 / 4","Live / motor / senses / brain map"),("V","Retina: both / colour / contrast"),
                    ("F5  /  F9  /  S","Quick-save / quick-load / timestamped save"),("?  /  Esc","Close this guide / exit when guide is closed")]
            y = card.y+92
            for key,label in rows:
                self.text(surface,key,(card.x+20,y),13,SAGE,width=190)
                self.text(surface,label,(card.x+230,y),12,TEXT,width=card.w-250)
                y+=26
            self.text(surface,"Scroll telemetry for overflow. Click anywhere to return.",(card.x+20,card.bottom-40),12,MUTED)
        elif self.toast and self.toast[1] > time.perf_counter():
            message = self.toast[0]
            width = min(w-48,self.font(14).size(message)[0]+38)
            r = pygame.Rect((w-width)//2,h-74,width,34)
            pygame.draw.rect(surface,PANEL,r)
            pygame.draw.rect(surface,TEAL,r,1)
            self.text(surface,message,(r.x+18,r.y+7),13,TEXT,width=r.w-36)
        else:
            hit = next((hit for hit in self.hits if hit.tip and hit.rect.collidepoint(self.mouse)),None)
            if hit:
                width = min(w-32,self.font(12).size(hit.tip)[0]+22)
                x = min(max(16,hit.rect.x),w-width-16)
                r = pygame.Rect(x,hit.rect.bottom+8,width,29)
                pygame.draw.rect(surface,PANEL,r)
                pygame.draw.rect(surface,LINE,r,1)
                self.text(surface,hit.tip,(r.x+11,r.y+5),12,TEXT,width=r.w-22)

    def loading(self, surface):
        surface.fill(BG)
        self.text(surface,"flyverse // room",(20,18),20,SAGE,"bold")
        self.text(surface,"> loading connectome, neural backend and sensory circuits...",(20,58),13,MUTED)

    def draw(self, sim, surface, orbit, paused=False, bmap=None):
        self.layout = Layout(surface.get_size())
        if bmap is not None and self.bmap is None:
            self.bmap = bmap
            self.tab = "atlas"
        self.hits = []
        if not hasattr(sim,"col_rad"):
            sim.col_rad = sim.column_radiance()
        surface.fill(BG)
        self._header(sim,surface,paused)
        self._scene(sim,surface,orbit)
        self._eye(sim,surface)
        self._retina(sim,surface)
        self._trace(sim,surface)
        self._sensory_summary(sim,surface)
        self._inspector(sim,surface)
        self._footer(sim,surface,paused)
        self._overlay(surface)
