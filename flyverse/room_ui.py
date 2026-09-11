"""The room observatory: presentation and hit testing, independent of simulation dynamics.

The layout reflows at native window resolution above MIN_SIZE; smaller windows show a
uniformly scaled canvas. Every text field has a width budget and telemetry is scrollable.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np
import pygame

from . import world

DEFAULT_SIZE = (1440, 960)
MIN_SIZE = (1100, 760)
BG = (11, 19, 21)
PANEL = (19, 29, 32)
INSET = (14, 23, 26)
LINE = (43, 57, 59)
TEXT = (231, 236, 224)
MUTED = (151, 169, 168)
DIM = (105, 130, 130)
SAGE = (185, 220, 159)
TEAL = (119, 196, 183)
AMBER = (235, 185, 121)
LILAC = (183, 163, 222)


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
        pad, gap = (24 if w >= 1300 else 18), 16
        side = 340 if w >= 1360 else 306
        left = w - 2 * pad - side - gap
        y, height = 132, h - 132 - 48
        sense_h = min(278, max(204, round(height * .32)))
        trace_h = 116
        scene_h = height - sense_h - trace_h - 2 * gap
        self.scene = pygame.Rect(pad, y, left, scene_h)
        self.scene_view = pygame.Rect(pad + 12, y + 48, left - 24, scene_h - 82)
        sy = self.scene.bottom + gap
        eye_w = (left - gap) // 2
        self.eye = pygame.Rect(pad, sy, eye_w, sense_h)
        self.retina = pygame.Rect(self.eye.right + gap, sy, left - eye_w - gap, sense_h)
        self.trace = pygame.Rect(pad, self.eye.bottom + gap, left, trace_h)
        self.subject = pygame.Rect(self.scene.right + gap, y, side, 176)
        self.inspector = pygame.Rect(self.subject.x, self.subject.bottom + gap, side, height - 192)
        self.inspector_view = self.inspector.inflate(-32, 0)
        self.inspector_view.top += 93
        self.inspector_view.height = self.inspector.height - 108
        self.footer = pygame.Rect(pad, h - 33, w - pad * 2, 24)


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
        self.retina_mode = "colour"
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
            family = ("Cascadia Mono,Consolas,DejaVu Sans Mono" if weight == "mono" else
                      "Segoe UI,Inter,DejaVu Sans,Liberation Sans")
            # SDL's Windows font enumeration aliases Segoe UI to its Light face on
            # some machines. Resolve the intended weights explicitly when installed.
            face = "seguisb.ttf" if weight == "bold" else "segoeuil.ttf" if weight == "light" else "segoeui.ttf"
            path = Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts"/face
            self.fonts[key] = (pygame.font.Font(str(path),size) if weight != "mono" and path.is_file() else
                               pygame.font.SysFont(family,size,bold=weight == "bold"))
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
        pygame.draw.rect(surface, (7, 14, 16), rect.move(0, 3), border_radius=13)
        pygame.draw.rect(surface, PANEL, rect, border_radius=13)
        pygame.draw.rect(surface, LINE, rect, 1, border_radius=13)

    def label(self, surface, text, pos, color=MUTED, width=None):
        return self.text(surface, text, pos, 11, color, "bold", width)

    def button(self, surface, rect, label, action, *, active=False, primary=False, tip="", key=None, padding=12):
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(self.mouse) and not self.help_open
        fill = SAGE if primary else (39, 58, 57) if active or hover else (23, 35, 38)
        color = BG if primary else SAGE if active else TEXT
        pygame.draw.rect(surface, fill, rect, border_radius=7)
        if not primary:
            pygame.draw.rect(surface, TEAL if active else LINE, rect, 1, border_radius=7)
        reserve = self.font(10,"mono").size(key)[0]+12 if key else 0
        self.text(surface, label, (rect.x + padding, rect.centery - 10), 13, color,
                  "bold" if primary else "normal", rect.w - 2*padding - reserve)
        if key:
            self.text(surface, key, (rect.right - 10, rect.centery - 8), 10,
                      (57, 79, 55) if primary else DIM, "mono", right=True)
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

    def _fly_mark(self, surface, center, scale=1.0):
        """A small line-drawn fly, using native primitives rather than a bitmap asset."""
        x, y = center
        def point(dx, dy): return round(x + dx * scale), round(y + dy * scale)
        for side in (-1, 1):
            wing = [point(0,-4), point(side*8,-17), point(side*17,-19), point(side*20,-11),
                    point(side*13,1), point(side*3,5), point(0,-4)]
            pygame.draw.polygon(surface, (32, 57, 55), wing)
            pygame.draw.aalines(surface, SAGE, False, wing)
            pygame.draw.aaline(surface, DIM, point(0,0), point(side*17,-14))
            for dy in (-3, 3, 9):
                pygame.draw.aalines(surface, TEAL, False, [point(side*2,dy),point(side*8,dy+2),point(side*12,dy+7)])
        pygame.draw.ellipse(surface, SAGE, pygame.Rect(point(-3,-6), (max(2,round(6*scale)),round(22*scale))))
        pygame.draw.circle(surface, SAGE, point(0,-10), max(2,round(4*scale)))
        for dy in (4, 8, 12):
            pygame.draw.line(surface, PANEL, point(-2,dy), point(2,dy))

    def _header(self, sim, surface, paused):
        w, h = surface.get_size()
        pad = self.layout.scene.x
        self._fly_mark(surface, (pad + 25, 38), .95)
        self.text(surface, "flyverse", (pad + 62, 12), 32, TEXT,"light")
        self.label(surface, "THE NEURAL OBSERVATORY", (pad + 65, 51), DIM)
        self.text(surface, "A world through another mind.", (pad + 330, 29), 15, MUTED)
        status_x = w - pad - 192
        pygame.draw.circle(surface, AMBER if paused else SAGE, (status_x, 30), 4)
        self.label(surface, "PAUSED" if paused else "LIVE SIMULATION", (status_x + 13, 22), AMBER if paused else SAGE)
        self.text(surface, f"{sim.c.n:,} neurons · MaleCNS", (w - pad, 45), 11, DIM, right=True)
        pygame.draw.line(surface, LINE, (pad, 72), (w-pad,72))
        x, y = pad, 86
        buttons = [(112,"Play" if paused else "Pause","pause","SPACE"), (83,"Reset","reset","R"),
                   (105,"To apple","apple","T"), (82,"Loom","loom","L"),
                   (90,"Escape","escape","F"), (86,"Flight","flight","W")]
        tips = {"pause":"Pause or resume brain time", "reset":"Reset the fly's pose", "apple":"Teleport next to the apple",
                "loom":"Present an approaching black sphere", "escape":"Stimulate the giant fibre", "flight":"Stimulate flight descending neurons"}
        for i,(width,label,action,key) in enumerate(buttons):
            self.button(surface,(x,y,width,32),label,action,primary=i==0,key=key,tip=tips[action])
            x += width + 8
        for width,label,action,key in reversed([(78,"Save","save","F5"),(78,"Load","load","F9"),(91,"Controls","help","?")]):
            w -= width
            self.button(surface,(w-pad,y,width,32),label,action,key=key,tip=f"{label} · {key}")
            w -= 8

    def _image(self, surface, image, rect):
        if image.get_size() != rect.size:
            image = pygame.transform.smoothscale(image, rect.size)
        surface.blit(image, rect)

    def _scene(self, sim, surface, orbit):
        r, viewport = self.layout.scene, self.layout.scene_view
        self.panel(surface, r)
        self.label(surface,"01",(r.x+16,r.y+17),TEAL)
        self.text(surface,"Habitat",(r.x+46,r.y+11),19)
        self.text(surface,"ROOM VIEW",(r.x+132,r.y+19),10,DIM,"bold")
        self.button(surface,(r.right-207,r.y+10,105,28),"Follow fly","follow",active=orbit.follow,key="C")
        self.button(surface,(r.right-94,r.y+10,78,28),"Home","home",tip="Reset camera · Home")
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
            pygame.draw.circle(surface,BG,point,9)
            pygame.draw.circle(surface,SAGE,point,9,1)
            pygame.draw.circle(surface,SAGE,point,3)
            forward = cam.project(sim.fly.eye_pos+.04*sim.fly.forward)
            if forward:
                pygame.draw.aaline(surface,SAGE,point,(viewport.x+forward[0],viewport.y+forward[1]))
            label_x = min(max(point[0]+16,viewport.x+8),viewport.right-105)
            label_y = max(viewport.y+8,point[1]-27)
            pygame.draw.rect(surface,BG,(label_x-7,label_y-3,94,23),border_radius=5)
            self.label(surface,"SUBJECT 001",(label_x,label_y),SAGE)
        surface.set_clip(old_clip)
        self.text(surface,"Drag to orbit · scroll to zoom",(r.x+16,r.bottom-25),11,MUTED)
        wind_from = (np.rad2deg(sim.air.direction)+180)%360
        self.text(surface,f"Wind from {wind_from:.0f}°  ·  heading {np.rad2deg(sim.fly.heading)%360:.0f}°",
                  (r.right-16,r.bottom-25),11,MUTED,width=r.w//2,right=True)

    def _eye(self, sim, surface):
        r = self.layout.eye
        self.panel(surface,r)
        self.label(surface,"02",(r.x+16,r.y+17),TEAL)
        self.text(surface,"Fly's-eye view",(r.x+46,r.y+11),18)
        self.label(surface,"110°",(r.right-45,r.y+19),DIM)
        viewport = pygame.Rect(r.x+12,r.y+46,r.w-24,r.h-80)
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
        self.text(surface,"Human-visible colour",(r.x+16,r.bottom-25),11,MUTED)
        self.text(surface,f"pitch {np.rad2deg(fly.pitch):+.0f}°  roll {np.rad2deg(fly.roll):+.0f}°",
                  (r.right-16,r.bottom-25),11,DIM,width=r.w-174,right=True)

    def _retina(self, sim, surface):
        r = self.layout.retina
        self.panel(surface,r)
        self.label(surface,"03",(r.x+16,r.y+17),TEAL)
        self.text(surface,"Compound eyes",(r.x+46,r.y+11),18,width=r.w-218)
        self.button(surface,(r.right-167,r.y+11,68,27),"Colour","retina:colour",active=self.retina_mode=="colour")
        self.button(surface,(r.right-93,r.y+11,77,27),"Contrast","retina:contrast",active=self.retina_mode=="contrast")
        viewport = pygame.Rect(r.x+14,r.y+48,r.w-28,r.h-82)
        pygame.draw.rect(surface,INSET,viewport,border_radius=6)
        if self.retina_mode == "colour":
            colors = world.to_fly_false_color(sim.col_rad,exposure=2.5).astype(np.uint8)
        else:
            contrast = sim.optic.contrast[0].view(-1,5)[:,0].detach().cpu().numpy()
            gray = np.clip(128+127*contrast/.5,0,255).astype(np.uint8)
            colors = np.repeat(gray[:,None],3,axis=1)
        az,el = sim.r.col_az_el.T
        geometry_key = (id(sim.r),tuple(viewport))
        if getattr(self,"_retina_key",None) != geometry_key:
            scale = min((viewport.w-32)/max(np.ptp(az),1),(viewport.h-18)/max(np.ptp(el),1))
            xy = np.c_[-(az-(az.max()+az.min())/2)*scale+viewport.w/2,
                        -(el-(el.max()+el.min())/2)*scale+viewport.h/2]
            radius = max(1.1,min(3.2,scale*1.35))
            angles = np.arange(6)*np.pi/3
            self._retina_polys = (np.round(xy[:,None,:]+radius*np.c_[np.cos(angles),np.sin(angles)]).astype(int)
                                  + np.array(viewport.topleft)).tolist()
            self._retina_key = geometry_key
        old_clip = surface.get_clip(); surface.set_clip(viewport)
        for poly,color in zip(self._retina_polys,colors.tolist()):
            pygame.draw.polygon(surface,color,poly)
        surface.set_clip(old_clip)
        if self.retina_mode == "colour":
            for dx,label,color in ((0,"UV",LILAC),(53,"Blue",(137,177,223)),(114,"Green",SAGE)):
                pygame.draw.circle(surface,color,(r.x+20+dx,r.bottom-17),3)
                self.text(surface,label,(r.x+29+dx,r.bottom-25),11,MUTED)
            self.text(surface,"false colour",(r.right-16,r.bottom-25),11,DIM,right=True)
        else:
            self.text(surface,"R1–R6  ·  OFF / ON",(r.x+16,r.bottom-25),11,MUTED)
            for i in range(64):
                pygame.draw.line(surface,(i*4,i*4,i*4),(r.right-80+i,r.bottom-21),(r.right-80+i,r.bottom-15))

    def _trace(self, sim, surface):
        r = self.layout.trace
        self.panel(surface,r)
        self.label(surface,"POPULATION ACTIVITY",(r.x+16,r.y+12))
        value = sim.spike_hist[-1] if sim.spike_hist else 0
        self.text(surface,f"{value:,.0f}",(r.right-113,r.y+6),22,SAGE,"mono",right=True)
        self.text(surface,"spikes / step",(r.right-99,r.y+15),11,MUTED)
        plot = pygame.Rect(r.x+17,r.y+39,r.w-34,r.h-59)
        for i in range(5):
            x = plot.x+round(i*plot.w/4)
            pygame.draw.line(surface,LINE,(x,plot.y),(x,plot.bottom))
            self.text(surface,"now" if i==4 else f"−{4-i}s",(min(x,r.right-42),plot.bottom+3),10,DIM,"mono")
        history = np.asarray(sim.spike_hist[-400:])
        if history.size > 1:
            maximum = max(10,float(history.max()))
            x = np.linspace(plot.right-(len(history)-1)*plot.w/399,plot.right,len(history))
            y = plot.bottom-2-(plot.h-7)*history/maximum
            points = list(zip(x.round().astype(int),y.round().astype(int)))
            fill = pygame.Surface(plot.size,pygame.SRCALPHA)
            local = [(px-plot.x,py-plot.y) for px,py in points]
            pygame.draw.polygon(fill,(*SAGE,22),[(local[0][0],plot.h),*local,(local[-1][0],plot.h)])
            surface.blit(fill,plot)
            pygame.draw.aalines(surface,SAGE,False,points)
            pygame.draw.circle(surface,SAGE,points[-1],3)
            self.text(surface,f"peak {maximum:.0f}",(plot.x+8,plot.y),10,DIM,"mono")

    def _subject(self, sim, surface):
        r,fly = self.layout.subject,sim.fly
        self.panel(surface,r)
        self.label(surface,"SUBJECT 001",(r.x+16,r.y+14),TEAL)
        self.text(surface,"Drosophila melanogaster",(r.x+16,r.y+33),14,TEXT)
        seconds = sim.brain.t/1000
        minutes = int(seconds//60)
        stamp = f"{minutes:02d}:{seconds%60:05.2f}"
        size = 32
        while size > 20 and self.font(size,"mono").size(stamp)[0] > r.w-128:
            size -= 1
        self.text(surface,stamp,(r.x+16,r.y+56),size,TEXT,"mono",width=r.w-122)
        self.label(surface,"BRAIN TIME",(r.x+18,r.y+98),DIM)
        mode = "FLYING" if fly.airborne else "FEEDING" if getattr(sim,"feeding",False) else "WALKING" if abs(fly.speed)>.0005 else "STILL"
        self.text(surface,mode,(r.right-16,r.y+71),11,SAGE,"bold",right=True)
        self.text(surface,f"{fly.speed*100:.2f} cm/s",(r.right-16,r.y+91),12,MUTED,"mono",right=True)
        energy = float(np.clip(sim.metabolism.energy,0,1))
        self.label(surface,"ENERGY",(r.x+16,r.y+128),DIM)
        bar = pygame.Rect(r.x+82,r.y+132,r.w-142,6)
        pygame.draw.rect(surface,LINE,bar,border_radius=3)
        if energy > 0:
            pygame.draw.rect(surface,SAGE if energy>.3 else AMBER,(bar.x,bar.y,round(bar.w*energy),bar.h),border_radius=3)
        self.text(surface,f"{energy:.0%}",(r.right-16,r.y+123),14,TEXT,"mono",right=True)
        program = getattr(sim,"cmd",{}).get("mode","")
        state = f"{sim.metabolism.state}  ·  {sim.metabolism.meals} meals" + (f"  ·  {program}" if program else "")
        self.text(surface,state,(r.x+16,r.bottom-22),11,MUTED,width=r.w-32)

    def _meter(self, surface, rect, label, value, unit="Hz", color=TEAL, limit=50):
        self.text(surface,label,(rect.x,rect.y),13,TEXT,width=rect.w-90)
        self.text(surface,f"{value:.1f} {unit}",(rect.right,rect.y),12,color,"mono",width=88,right=True)
        bar = pygame.Rect(rect.x,rect.y+23,rect.w,3)
        pygame.draw.rect(surface,LINE,bar,border_radius=1)
        length = round(bar.w*np.clip(abs(value)/max(limit,.0001),0,1))
        if length:
            pygame.draw.rect(surface,color,(bar.x,bar.y,length,3),border_radius=1)

    def _regions(self, sim, surface, viewport, y):
        names = [("ol_intrinsic","Optic lobe"),("visual_projection","Visual projection"),("cb_intrinsic","Central brain"),
                 ("descending_neuron","Descending neurons"),("vnc_intrinsic","Ventral nerve cord"),
                 ("vnc_motor","VNC motor neurons"),("cb_motor","Brain motor neurons"),("vnc_sensory","VNC sensory neurons")]
        for name,label in names:
            graded = name == "ol_intrinsic"
            value = float(sim.optic.delta_rate.abs().mean())*100 if graded else sim.brain.mean_rate(sim.sc_idx[name])
            self._meter(surface,pygame.Rect(viewport.x,y,viewport.w-5,35),label,value,"% Δ" if graded else "Hz",LILAC if graded else TEAL)
            y += 42
        y += 12
        self.label(surface,"A COMPLETE NERVOUS SYSTEM",(viewport.x,y),DIM,width=viewport.w)
        y += 27
        self.text(surface,f"{sim.c.n:,}",(viewport.x,y),24,TEXT,"mono")
        self.text(surface,f"{sim.r.n_columns:,}",(viewport.right-5,y),24,TEXT,"mono",right=True)
        y += 33
        self.text(surface,"neurons",(viewport.x,y),11,MUTED)
        self.text(surface,"eye columns",(viewport.right-5,y),11,MUTED,right=True)
        y += 37
        self.text(surface,"Optic: mean absolute rate change.",(viewport.x,y),11,DIM,width=viewport.w)
        self.text(surface,"Spiking regions: mean firing rate in Hz.",(viewport.x,y+18),11,DIM,width=viewport.w)
        return y+34

    def _motors(self, sim, surface, viewport, y):
        if not hasattr(sim,"cmd"):
            self.text(surface,"Awaiting the first simulation step.",(viewport.x,y),12,MUTED,width=viewport.w)
            return y+30
        labels = {"fwdDN":"Forward drive","MDN":"Reverse drive","opto_L":"Optomotor · left","opto_R":"Optomotor · right",
                  "wind_L":"Wind · left","wind_R":"Wind · right","DNa02_L":"Turn · left","DNa02_R":"Turn · right",
                  "legMN_L":"Leg motor · left","legMN_R":"Leg motor · right","MN9":"Proboscis",
                  "gf":"Giant fibre","ttm":"Jump muscle","power":"Wing power","steer_L":"Wing steering · left",
                  "steer_R":"Wing steering · right","haltere":"Halteres","gf_threshold":"Escape threshold"}
        for title,values in (("WALKING & SENSING",sim.cmd["rates"]),("FLIGHT & ESCAPE",sim.wcmd)):
            self.label(surface,title,(viewport.x,y),DIM); y+=28
            for name,value in values.items():
                unit = "" if "gate" in name.lower() or "x10" in name else "Hz"
                self._meter(surface,pygame.Rect(viewport.x,y,viewport.w-5,35),labels.get(name,name),float(value),unit,AMBER,80)
                y += 39
            y+=15
        self.label(surface,"BODY COMMANDS",(viewport.x,y),DIM); y+=27
        for name,value,unit in (("Speed",sim.cmd['speed']*100,"cm/s"),("Yaw",np.rad2deg(sim.cmd['yaw']),"°/s"),
                                ("Proboscis",sim.cmd['proboscis'],"")):
            self.text(surface,name,(viewport.x,y),13)
            self.text(surface,f"{value:+.2f} {unit}",(viewport.right-5,y),12,AMBER,"mono",right=True)
            y+=28
        y+=15
        self.label(surface,"BODY STATE",(viewport.x,y),DIM); y+=27
        fly = sim.fly
        for name,value in (("Height",f"{fly.z:.3f} m"),("Flight vx / vy",f"{fly.vx*100:+.1f} / {fly.vy*100:+.1f} cm/s"),
                           ("Flight vz",f"{fly.vz*100:+.1f} cm/s"),("Time airborne",f"{fly.air_time:.2f} s")):
            self.text(surface,name,(viewport.x,y),12,TEXT,width=viewport.w//2)
            self.text(surface,value,(viewport.right-5,y),12,AMBER,"mono",width=viewport.w//2,right=True)
            y+=28
        return y

    def _atlas(self, sim, surface, viewport, y):
        bmap = self.bmap
        if bmap is None:
            self.text(surface,"Select Atlas to load soma positions.",(viewport.x,y),12,MUTED,width=viewport.w)
            return y+30
        # Sample on simulation advances, never fade a paused brain because of UI redraws.
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
        for title,image in zip(("DORSAL · BRAIN TO VNC","LATERAL · DORSAL UP"),self._map_images):
            self.label(surface,title,(viewport.x,y),DIM); y+=25
            height = round(image.shape[0]*(viewport.w-5)/image.shape[1])
            self._image(surface,pygame.surfarray.make_surface(np.transpose(image,(1,0,2))),pygame.Rect(viewport.x,y,viewport.w-5,height))
            y+=height+20
        self.label(surface,f"{int((self._map_shown>.1).sum()):,} ACTIVE CELLS",(viewport.x,y),AMBER); y+=28
        for name,value in bmap.top_types(self._map_shown).items():
            self.text(surface,name,(viewport.x,y),12,MUTED,width=viewport.w-55)
            self.text(surface,f"{value:.2f}",(viewport.right-5,y),12,AMBER,"mono",right=True)
            y+=23
        return y

    def _senses(self, sim, surface, viewport, y):
        self.label(surface,"CONTACT & AIRFLOW",(viewport.x,y),DIM); y+=27
        for label,value in (("Sugar contact","yes" if getattr(sim,"tasting",0)>0 else "no"),
                            ("Wind from",f"{(np.rad2deg(sim.air.direction)+180)%360:.0f}°")):
            self.text(surface,label,(viewport.x,y),13,TEXT)
            self.text(surface,value,(viewport.right-5,y),13,SAGE,"mono",right=True)
            y+=30
        y+=18
        self.label(surface,"ODOR AT THE ANTENNAE",(viewport.x,y),DIM); y+=23
        self.text(surface,"Relative concentration · strongest first",(viewport.x,y),11,MUTED,width=viewport.w); y+=31
        left,right = getattr(sim,"smell_values",({},{}))
        def val(values,name): return float(np.asarray(values.get(name,0)).reshape(-1)[0])
        names = sorted(set(left)|set(right),key=lambda n:-(val(left,n)+val(right,n)))
        self.label(surface,"GLOMERULUS",(viewport.x,y),DIM)
        self.text(surface,"LEFT",(viewport.right-90,y),10,TEAL,"bold",right=True)
        self.text(surface,"RIGHT",(viewport.right-5,y),10,AMBER,"bold",right=True)
        y+=28
        for name in names:
            self.text(surface,name,(viewport.x,y),13,TEXT,width=viewport.w-167)
            self.text(surface,f"{val(left,name):.3g}",(viewport.right-90,y),12,TEAL,"mono",width=70,right=True)
            self.text(surface,f"{val(right,name):.3g}",(viewport.right-5,y),12,AMBER,"mono",width=70,right=True)
            pygame.draw.line(surface,LINE,(viewport.x,y+23),(viewport.right-5,y+23))
            y+=32
        if not names:
            self.text(surface,"Awaiting an odor sample.",(viewport.x,y),12,MUTED)
            y+=28
        return y

    def _inspector(self, sim, surface):
        r = self.layout.inspector
        self.panel(surface,r)
        self.text(surface,"Inside the fly",(r.x+16,r.y+13),20)
        self.label(surface,"NEURAL TELEMETRY",(r.x+16,r.y+40),DIM)
        width = (r.w-44)//4
        for i,(tab,title) in enumerate((("regions","Regions"),("motor","Motor"),("senses","Senses"),("atlas","Atlas"))):
            self.button(surface,(r.x+16+i*(width+4),r.y+60,width,27),title,"tab:"+tab,active=self.tab==tab,padding=8)
        viewport = self.layout.inspector_view
        old_clip = surface.get_clip(); surface.set_clip(viewport)
        start = viewport.y-self.scroll[self.tab]
        end = {"regions":self._regions,"motor":self._motors,"senses":self._senses,"atlas":self._atlas}[self.tab](sim,surface,viewport,start)
        surface.set_clip(old_clip)
        self.scroll_limit = max(0,end-start-viewport.h)
        self.scroll[self.tab] = min(self.scroll[self.tab],self.scroll_limit)
        if self.scroll_limit:
            track = pygame.Rect(r.right-9,viewport.y,3,viewport.h)
            pygame.draw.rect(surface,LINE,track,border_radius=1)
            height = max(24,round(viewport.h*viewport.h/(end-start)))
            offset = round((viewport.h-height)*self.scroll[self.tab]/self.scroll_limit)
            pygame.draw.rect(surface,DIM,(track.x,track.y+offset,3,height),border_radius=1)
            self.text(surface,"Scroll to explore",(r.right-16,r.bottom-15),10,DIM,right=True)

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
        self.text(surface,f"{place}   /   ({sim.fly.x:+.2f}, {sim.fly.y:+.2f}) m   /   nearest {fruit}: {max(0,distance)*100:.1f} cm",
                  r.topleft,11,MUTED,width=r.w//2)
        perf = "Preparing display" if self.fps is None else f"{self.fps:.0f} display fps  ·  {0 if paused else self.rtf:.2f}× brain time"
        self.text(surface,perf,(r.right,r.y),11,DIM,"mono",right=True)

    def _overlay(self, surface):
        w,h = surface.get_size()
        if self.help_open:
            shade = pygame.Surface((w,h),pygame.SRCALPHA); shade.fill((3,9,11,210)); surface.blit(shade,(0,0))
            card = pygame.Rect((w-680)//2,(h-540)//2,680,540)
            self.panel(surface,card)
            self.label(surface,"FIELD GUIDE",(card.x+28,card.y+24),TEAL)
            self.text(surface,"Explore a living connectome",(card.x+28,card.y+47),28)
            self.text(surface,"Brain time is paused while this guide is open.",(card.x+28,card.y+90),14,MUTED)
            rows = [("Space","Pause / resume"),("R  /  T","Reset pose / move to apple"),("L","Approaching object (loom)"),
                    ("F  /  W","Giant fibre / flight DN stimulation"),("Drag  /  wheel","Orbit / zoom the room camera"),
                    ("Arrows  /  + −","Orbit / zoom with the keyboard"),("C  /  Home","Follow the fly / reset camera"),
                    ("1  /  2  /  3  /  4","Regions / motor / senses / brain atlas"),("V","Retinal colour / contrast"),
                    ("F5  /  F9  /  S","Quick-save / quick-load / timestamped save"),("?  /  Esc","Close this guide / exit when guide is closed")]
            y = card.y+137
            for key,label in rows:
                self.text(surface,key,(card.x+28,y),13,SAGE,"mono",width=180)
                self.text(surface,label,(card.x+225,y),14,TEXT,width=card.w-253)
                y+=29
            self.text(surface,"Scroll inside neural telemetry to inspect every row.",(card.x+28,card.bottom-49),12,MUTED)
            self.text(surface,"Click anywhere to return.",(card.x+28,card.bottom-28),11,DIM)
        elif self.toast and self.toast[1] > time.perf_counter():
            message = self.toast[0]
            width = min(w-48,self.font(14).size(message)[0]+38)
            r = pygame.Rect((w-width)//2,h-88,width,40)
            pygame.draw.rect(surface,(38,57,52),r,border_radius=8)
            pygame.draw.rect(surface,TEAL,r,1,border_radius=8)
            self.text(surface,message,(r.x+18,r.y+9),14,TEXT,width=r.w-36)
        else:
            hit = next((hit for hit in self.hits if hit.tip and hit.rect.collidepoint(self.mouse)),None)
            if hit:
                width = min(w-32,self.font(12).size(hit.tip)[0]+22)
                x = min(max(16,hit.rect.x),w-width-16)
                r = pygame.Rect(x,hit.rect.bottom+8,width,29)
                pygame.draw.rect(surface,(38,57,52),r,border_radius=5)
                self.text(surface,hit.tip,(r.x+11,r.y+5),12,TEXT,width=r.w-22)

    def loading(self, surface):
        surface.fill(BG)
        w,h = surface.get_size()
        self._fly_mark(surface,(w//2,h//2-94),2.4)
        word = self.font(48,"light").size("flyverse")[0]
        self.text(surface,"flyverse",((w-word)//2,h//2-35),48,TEXT,"light")
        caption = "Loading neural and sensory circuits"
        self.text(surface,caption,((w-self.font(15).size(caption)[0])//2,h//2+43),15,MUTED)
        pygame.draw.line(surface,LINE,(w//2-100,h//2+87),(w//2+100,h//2+87))
        pygame.draw.circle(surface,SAGE,(w//2,h//2+87),3)

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
        self._subject(sim,surface)
        self._inspector(sim,surface)
        self._footer(sim,surface,paused)
        self._overlay(surface)
