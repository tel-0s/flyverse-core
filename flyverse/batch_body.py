"""Batch the room body's continuous calculations without retuning its scalar model.

Public FlyState/Locomotion/Flight/Metabolism instances remain per fly. Parameters are
read from those instances. Edge transitions, takeoff and landing use the reference
methods; ordinary surface movement and free flight run as NumPy array operations.
"""
import numpy as np

from . import body
from .batch_surfaces import BatchSurfaces


def attr(objects, name, default=None):
    return np.asarray([getattr(obj,name,default) for obj in objects])


def frames(flies):
    """Eye positions and full forward/left/up axes, including surface-edge blending."""
    f = attr(flies,"fwd").copy(); up = attr(flies,"normal").copy()
    edge = attr(flies,"_edge_left")
    theta = attr(flies,"_edge_theta")
    enabled = (edge>0)&(theta!=0)
    angle = np.divide(-theta*edge,attr(flies,"edge_len"),out=np.zeros(len(flies)),where=enabled)
    k = attr(flies,"_edge_axis")
    c,s = np.cos(angle)[:,None],np.sin(angle)[:,None]
    def rotate(v): return v*c+np.cross(k,v)*s+k*(k*v).sum(-1)[:,None]*(1-c)
    f,up = rotate(f),rotate(up)
    left = np.cross(up,f)
    air = attr(flies,"airborne")
    heading,pitch,roll = attr(flies,"_heading"),attr(flies,"pitch"),attr(flies,"roll")
    fa = np.stack([np.cos(pitch)*np.cos(heading),np.cos(pitch)*np.sin(heading),np.sin(pitch)],axis=1)
    l0 = np.stack([-np.sin(heading),np.cos(heading),np.zeros(len(flies))],axis=1)
    la = np.cos(roll)[:,None]*l0+np.sin(roll)[:,None]*np.cross(fa,l0)
    f[air],left[air],up[air] = fa[air],la[air],np.cross(fa,la)[air]
    pos = np.stack([attr(flies,k) for k in ("x","y","z")],axis=1)
    return pos+attr(flies,"eye_height")[:,None]*up,f,left,up


class BatchBody:
    def __init__(self, flies, surfaces, fence=None):
        self.flies = list(flies)
        self.B = len(flies)
        self.locos = [body.Locomotion() for _ in flies]
        self.flights = [body.Flight() for _ in flies]
        self.metabolisms = [body.Metabolism() for _ in flies]
        self.surfaces = BatchSurfaces(surfaces)
        self.fence = fence  # (x0,x1,y0,y1,z), or None for full surface physics
        self.feeding = np.zeros(self.B,dtype=bool)

    def readout(self, motor, dt_s):
        # Scalar MotorRates.row promotes float32 neural samples to Python doubles.
        def m(name): return np.broadcast_to(np.asarray(getattr(motor,name),dtype=float),(self.B,))
        def p(name): return attr(self.locos,name)
        fwd,back,tL,tR,oL,oR,lL,lR = [m(k) for k in ("fwd_dn","back_dn","turn_L","turn_R","opto_L","opto_R","leg_L","leg_R")]
        asym = oL-oR
        a = np.exp(-dt_s/p("opto_hp_tau_s"))
        bias = np.array([getattr(l,"_opto_bias",asym[i]) for i,l in enumerate(self.locos)])
        bias = a*bias+(1-a)*asym
        for i,l in enumerate(self.locos): l._opto_bias = float(bias[i])
        speed = p("baseline_speed")+p("k_fwd")*fwd+p("k_leg")*.5*(lL+lR)-p("k_back")*np.maximum(back-p("mdn_threshold"),0.)
        yaw = p("k_opto")*(asym-bias)-p("k_turn")*(tR-tL)+p("k_leg_turn")*(lL-lR)
        speed = np.clip(speed,-p("max_speed"),p("max_speed"))
        yaw = np.clip(yaw,-p("max_yaw"),p("max_yaw"))
        prob = np.clip(m("proboscis")/30.,0,1)
        lh = np.maximum.reduce([np.broadcast_to(v,(self.B,)) for v in motor.lh_odour.values()]) if motor.lh_odour else np.zeros(self.B)
        rates = dict(zip(("fwdDN","MDN","opto_L","opto_R","wind_L","wind_R","LH odour","DNa02_L","DNa02_R","legMN_L","legMN_R","MN9"),
                         (fwd,back,oL,oR,m("wind_ipsi_L"),m("wind_ipsi_R"),lh,tL,tR,lL,lR,m("proboscis"))))
        wings = {k:m(k) for k in ("gf","ttm","power","steer_L","steer_R","haltere")}
        wings["gf_threshold"] = attr(self.flights,"gf_hz")
        commands = [dict(speed=float(speed[i]),yaw=float(yaw[i]),proboscis=float(prob[i]),rates={k:float(v[i]) for k,v in rates.items()}) for i in range(self.B)]
        wcommands = [{k:float(v[i]) for k,v in wings.items()} for i in range(self.B)]
        return commands,wcommands

    def step(self, commands, wings, tasting, dt_s):
        airborne = attr(self.flies,"airborne")
        gf,power = (np.array([w[k] for w in wings]) for k in ("gf","power"))
        threshold = np.array([w.get("gf_threshold",f.gf_hz) for w,f in zip(wings,self.flights)])
        ground_time = attr(self.flies,"ground_time")+dt_s
        escape = ~airborne&(gf>=threshold)&(ground_time>=attr(self.flights,"landing_refractory_s"))
        hold = attr(self.flights,"_power_hold",0.)
        hold = np.where(~airborne&~escape,np.where(power>=attr(self.flights,"takeoff_power_hz"),hold+dt_s,0.),hold)
        voluntary = ~airborne&~escape&(hold>=attr(self.flights,"takeoff_hold_s"))
        launch = escape|voluntary
        for i in np.flatnonzero(~airborne):
            self.flies[i].ground_time = float(ground_time[i])
            self.flights[i]._power_hold = 0. if voluntary[i] else float(hold[i])
        # Launch uses the sensed surface frame, including an in-progress edge rotation.
        for i in np.flatnonzero(launch): self.flights[i].launch(self.flies[i],bool(escape[i]))
        walk = ~airborne&~launch
        self._metabolism(tasting,airborne,~launch,dt_s)
        moving = [dict(c,speed=0.,yaw=0.) if self.feeding[i] else c for i,c in enumerate(commands)]
        self._walk(np.flatnonzero(walk),moving,dt_s)
        self._flight(np.flatnonzero(airborne),wings,dt_s)
        return self.feeding.copy()

    def _metabolism(self, tasting, airborne, active, dt):
        rows = np.flatnonzero(active)
        if not len(rows): return
        objects = [self.metabolisms[i] for i in rows]
        def p(k): return attr(objects,k)
        speed = np.where(airborne[rows],0.,attr(self.flies,"speed")[rows])
        energy = p("energy")-(p("drain_per_s")*dt+p("walk_drain_per_m")*np.abs(speed)*dt)
        sated = p("sated")&~(energy<p("resume_below"))
        feed = np.asarray(tasting,dtype=bool)[rows]&~airborne[rows]&~sated
        energy += np.where(feed,p("feed_per_s")*dt,0.)
        meals = p("meals")+(feed&~p("_feeding_prev"))
        full = feed&(energy>=p("satiety"))
        sated |= full; feed &= ~full
        energy = np.clip(energy,0.,1.)
        for j,i in enumerate(rows):
            m = self.metabolisms[i]
            m.energy,m.sated,m.meals,m._feeding_prev = float(energy[j]),bool(sated[j]),int(meals[j]),bool(feed[j])
        self.feeding[rows] = feed

    def _walk(self, rows, commands, dt):
        if not len(rows): return
        if self.fence is not None:
            # Manually placed vertical poses retain the scalar heading setter's semantics.
            unusual = [i for i in rows if abs(self.flies[i].normal[2])<=.5]
            for i in unusual: self.locos[i].step(self.flies[i],commands[i],dt,self.fence[:4])
            rows = np.array([i for i in rows if i not in unusual],dtype=int)
            if not len(rows): return
        flies = [self.flies[i] for i in rows]; locos = [self.locos[i] for i in rows]
        cmd = [commands[i] for i in rows]
        a = np.exp(-dt*1000/attr(locos,"tau_ms"))
        speed = a*attr(flies,"speed")+(1-a)*np.array([c["speed"] for c in cmd])
        yaw = a*attr(flies,"yaw_rate")+(1-a)*np.array([c["yaw"] for c in cmd])
        pos = np.stack([attr(flies,k) for k in ("x","y","z")],axis=1)
        if self.fence is None:
            before = attr(flies,"normal")
            q,f,n,faces = self.surfaces.walk(pos,attr(flies,"fwd"),before,[f.face for f in flies],speed*dt,yaw*dt)
            for j,fly in enumerate(flies):
                fly.set_pos(q[j]); fly.fwd,fly.normal,fly.face = f[j].copy(),n[j].copy(),faces[j]
                if np.dot(n[j],before[j])<.99: fly.begin_edge(before[j],n[j])
                elif fly._edge_left>0.: fly._edge_left = max(0.,fly._edge_left-abs(speed[j])*dt)
                if abs(f[j,2])<.999: fly._heading = float(np.arctan2(f[j,1],f[j,0]))
        else:
            x0,x1,y0,y1,_ = self.fence
            heading = attr(flies,"heading")+yaw*dt
            nx,ny = pos[:,0]+speed*dt*np.cos(heading),pos[:,1]+speed*dt*np.sin(heading)
            in_x,in_y = (nx>=x0)&(nx<=x1),(ny>=y0)&(ny<=y1)
            x,y = np.where(in_x,nx,pos[:,0]),np.where(in_y,ny,pos[:,1])
            outside = ~(in_x&in_y)
            x = np.where(outside,np.clip(x,x0,x1),x); y = np.where(outside,np.clip(y,y0,y1),y)
            centre = np.arctan2(.5*(y0+y1)-y,.5*(x0+x1)-x)
            err = (centre-heading+np.pi)%(2*np.pi)-np.pi
            heading += np.where(outside,np.sign(err)*np.minimum(np.abs(err),attr(locos,"edge_turn")*dt),0.)
            heading = (heading+np.pi)%(2*np.pi)-np.pi
            for j,fly in enumerate(flies):
                fly.x,fly.y = float(x[j]),float(y[j]); fly.heading = float(heading[j])
        for j,fly in enumerate(flies):
            fly.speed,fly.yaw_rate,fly.proboscis = float(speed[j]),float(yaw[j]),float(cmd[j]["proboscis"])

    def _flight(self, rows, wings, dt):
        if not len(rows): return
        flies = [self.flies[i] for i in rows]; flights = [self.flights[i] for i in rows]
        def p(k): return attr(flights,k)
        def w(k): return np.array([wings[i][k] for i in rows])
        a = np.exp(-dt*1000/p("tau_ms"))
        yaw = np.clip(-p("k_yaw")*(w("steer_R")-w("steer_L")),-p("max_yaw"),p("max_yaw"))
        yaw = a*attr(flies,"yaw_rate")+(1-a)*yaw
        heading = attr(flies,"_heading")+yaw*dt
        pitch,roll = attr(flies,"pitch"),attr(flies,"roll")
        f = np.stack([np.cos(pitch)*np.cos(heading),np.cos(pitch)*np.sin(heading),np.sin(pitch)],axis=1)
        velocity = np.stack([attr(flies,k) for k in ("vx","vy","vz")],axis=1)
        acceleration = (p("k_thrust")*w("power"))[:,None]*f-p("drag")[:,None]*velocity
        acceleration[:,2] = np.where(w("power")>1,p("k_lift")*(w("power")-p("hover_hz"))-p("gravity"),-p("gravity"))-p("drag")*velocity[:,2]
        velocity += acceleration*dt
        old = np.stack([attr(flies,k) for k in ("x","y","z")],axis=1)
        pos = old+velocity*dt
        air_time = attr(flies,"air_time")+dt
        if self.fence is None:
            _,index = self.surfaces.land(old,pos)
            exceptional = index>=0
            surface,room = self.surfaces.scalar,(-2,2,-2,2,2.6)
        else:
            x0,x1,y0,y1,z = self.fence
            exceptional = (pos[:,0]<=x0)|(pos[:,0]>=x1)|(pos[:,1]<=y0)|(pos[:,1]>=y1)|(pos[:,2]>2.59)|((pos[:,2]<=z)&(velocity[:,2]<=0)&(air_time>.05))
            surface,room = lambda x,y:z,(x0,x1,y0,y1,2.6)
        ao = np.exp(-dt*1000/p("orient_tau_ms"))
        target = np.clip(p("pitch_follow")*np.arctan2(velocity[:,2],np.maximum(np.hypot(velocity[:,0],velocity[:,1]),.05)),-p("max_pitch"),p("max_pitch"))
        pitch = ao*pitch+(1-ao)*target
        target = np.clip(-p("bank_per_yaw")*yaw+p("k_roll")*(w("steer_L")-w("steer_R")),-np.deg2rad(60),np.deg2rad(60))
        roll = ao*roll+(1-ao)*target
        for j,fly in enumerate(flies):
            if exceptional[j]:
                flights[j].step(fly,wings[rows[j]],dt,surface,room)
            else:
                fly.set_pos(pos[j]); fly.vx,fly.vy,fly.vz = map(float,velocity[j])
                fly._heading,fly.yaw_rate,fly.pitch,fly.roll,fly.air_time = map(float,(heading[j],yaw[j],pitch[j],roll[j],air_time[j]))
