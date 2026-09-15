"""Headless room rollouts: independent environments around one batched FlyBrain.

Uses the room demo's sensory resolution, frame order and body/program parameters.
Environment seeds select scenes and plume phases; ``seed`` selects the batched brain's
RNG stream. Changing batch size changes its random draw layout, as in FlyRoomEnv.
"""
from __future__ import annotations

import copy
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from . import air, body, brain, optic, programs, senses, surfaces, world
from .fly import FlyBrain
from .batch_air import BatchAir
from .batch_body import BatchBody, frames
from .batch_world import BatchWorld


def _program_state(program):
    if program is None: return None
    state = {}
    for k,v in vars(program).items():
        if k == "parts": state[k] = [_program_state(p) for p in v]
        elif k == "gate" and hasattr(v,"apply"): state[k] = _program_state(v)
        elif isinstance(v,np.random.Generator): state[k] = {"rng_state":copy.deepcopy(v.bit_generator.state)}
        else: state[k] = copy.deepcopy(v)
    return state


def _load_program(program, state):
    if program is None: return
    values = {}
    for k,v in state.items():
        if k == "parts":
            for part,part_state in zip(program.parts,v): _load_program(part,part_state)
            values[k] = program.parts
        elif k == "gate" and hasattr(getattr(program,k,None),"apply"):
            _load_program(program.gate,v); values[k] = program.gate
        elif isinstance(v,dict) and "rng_state" in v:
            rng = getattr(program,k); rng.bit_generator.state = copy.deepcopy(v["rng_state"]); values[k] = rng
        else: values[k] = copy.deepcopy(v)
    vars(program).clear(); vars(program).update(values)


def _motor_rows_cleared(motor, rows, batch):
    """A copy of a MotorRates snapshot with the selected rows' rates zeroed: what a reset row must read next frame.

    The snapshot is the opt-in proprioception transducer's only cross-frame input (``BatchBody.proprio_state``), and a
    reset row has no previous frame, so it must read zeros -- the same state a full reset produces by dropping the
    snapshot. ``time_ms`` (the shared clock) and ``pn_glom_cells`` (per-glomerulus cell counts) are not rates and are
    left alone; at ``batch == 1`` the fields are scalars, so the only row there is zeroed outright.
    """
    if motor is None: return None
    def zero(value):
        if isinstance(value,np.ndarray): value = value.copy(); value[rows] = 0; return value
        return 0. if batch == 1 else value                              # a B=1 snapshot is scalars; its one row is `rows`
    changed = {}
    for f in fields(motor):
        if f.name in ("time_ms","pn_glom_cells"): continue
        value = getattr(motor,f.name)
        changed[f.name] = {k:zero(v) for k,v in value.items()} if isinstance(value,dict) else zero(value)
    return replace(motor,**changed)                                     # MotorRates is frozen


class _RowStimulus:
    """Coalesce same-frame program pulses before touching the batched controller."""
    def __init__(self, owner, row): self.owner,self.row = owner,row

    def stimulate(self, selection, hz, ms):
        owner = self.owner
        key = repr(selection)
        if key not in owner._selections:
            owner._selections[key] = owner.fb.c.indices(selection)
        idx = owner._selections[key]
        key = key,float(ms)
        if key not in owner._program_pulses:
            owner._program_pulses[key] = idx,np.zeros((owner.B,len(idx)),dtype=np.float32),ms
        values = owner._program_pulses[key][1]
        values[self.row] = np.maximum(values[self.row],hz)


class BatchSim:
    FRAME_MS = 10.0

    def __init__(self, batch=1, *, seed=0, seeds=None, start=None, c=None, device=None,
                 brain_dt=.5, optic_dt=1., cuda_graphs=False, cuda_kernels=None,
                 event_driven=None, cuda_sparse="torch", cuda_compact=True, weight_dtype="float32",
                 sensory_cuda_graphs=None, dt_by_module=None, prune_frozen=True, modules=None,
                 wind_speed=.3, wind_dir=180., fruit_set="all", fence=False,
                 program="none", escape_gating=False, proprioception=None, preset="raw", instruments=None):
        if not isinstance(batch,(int,np.integer)) or batch<1:
            raise ValueError("batch must be a positive integer")
        self.B = int(batch)
        self.seeds = tuple(int(s) for s in (range(seed,seed+self.B) if seeds is None else seeds))
        if len(self.seeds) != self.B: raise ValueError("provide one environment seed per batch row")
        if not np.isfinite(brain_dt) or brain_dt<=0 or not np.isclose(self.FRAME_MS/brain_dt,round(self.FRAME_MS/brain_dt),rtol=0,atol=1e-8):
            raise ValueError("brain_dt must divide the 10 ms room frame")
        self.program_names = [program]*self.B if isinstance(program,str) or program is None else list(program)
        if len(self.program_names) != self.B: raise ValueError("provide one program per batch row")
        self.programs = [programs.make_program(p) for p in self.program_names]
        self.fruit_set,self.escape_gating = fruit_set,bool(escape_gating)
        self.gatings = [programs.EscapeGating() if escape_gating else None for _ in range(self.B)]
        self.fb = FlyBrain(c,seed=seed,batch=self.B,device=device,modules=modules,cuda_graphs=cuda_graphs,
                           cuda_kernels=cuda_kernels,cuda_sparse=cuda_sparse,cuda_compact=cuda_compact,
                           lif_params=brain.LIFParams(dt=brain_dt,weight_dtype=weight_dtype,event_driven=event_driven,
                                                      dt_by_module=dt_by_module,prune_frozen=prune_frozen),
                           optic_params=optic.OpticParams(dt_ms=optic_dt),preset=preset,instruments=instruments)
        self.c,self.r,self.optic,self.brain = self.fb.c,self.fb.retina,self.fb.optic,self.fb.brain
        # Opt-in proprioception (senses.Proprioception; default None = the shipped path, no sense attached). The
        # transducer reads the previous frame's motor readout and the body state in step().
        self.proprioception = None if proprioception in (None,False,"","off","none") else str(proprioception)
        self.motor = None
        if self.proprioception is not None:
            self.fb.proprioception_sense = senses.Proprioception(self.c,self.proprioception)
            # preset 'instrumented' (docs/PRESETS_SPEC.md): a stand-in that rides on the sense re-installs onto this one
            for inst in self.fb.instruments.values():
                if callable(getattr(inst,"install",None)): inst.install(self.fb)
        if self.optic is not None: self.optic.diagnostics = False
        self.sensory_cuda_graphs = cuda_graphs if sensory_cuda_graphs is None else sensory_cuda_graphs
        pairs = [world.make_room(s,fruit_set) for s in self.seeds]
        worlds,self.infos = [p[0] for p in pairs],[p[1] for p in pairs]
        for w in worlds: w.spheres.append(world.Sphere((9,9,9),(.03,.03,.03),"black"))
        self.loom_idx = len(worlds[0].spheres)-1
        self.world = BatchWorld(worlds,self.fb.device)
        # make_room's fruit geometry varies by seed; its walkable room/table faces do not.
        info = self.infos[0]
        self.surfaces = surfaces.Surfaces(info["room"],info["solids"],info["solid_labels"])
        self.fence = bool(fence)
        self.starts = np.broadcast_to(np.asarray(start if start is not None else (-.5,.05,info["table_top_z"]),float),(self.B,3)).copy()
        if not np.isfinite(self.starts).all(): raise ValueError("start positions must be finite")
        self.flies = [body.FlyState(x=x,y=y,z=z,heading=np.deg2rad(5)) for x,y,z in self.starts]
        self.body = BatchBody(self.flies,self.surfaces,(*info["table_extent"],info["table_top_z"]) if fence else None)
        self.flies = self.body.flies
        self.locos,self.flights,self.metabolisms = self.body.locos,self.body.flights,self.body.metabolisms
        wind_speed,wind_dir = (np.broadcast_to(np.asarray(v,float),(self.B,)) for v in (wind_speed,wind_dir))
        self.airs = [air.Air([(name,cen,rad/.02,rad) for name,cen,rad in data["fruit"]],
                            air.WindParams(speed=float(wind_speed[i]),direction_deg=float(wind_dir[i])),seed=self.seeds[i]) for i,data in enumerate(self.infos)]
        self.air = BatchAir(self.airs)
        self._air_initial = [copy.deepcopy(a.__dict__) for a in self.airs]
        self._program_initial = [_program_state(p) for p in self.programs]
        self._gating_initial = [_program_state(g) for g in self.gatings]
        self._fruit_centres = np.array([[cen for _,cen,_ in d["fruit"]] for d in self.infos])
        self._fruit_radii = np.array([[rad for _,_,rad in d["fruit"]] for d in self.infos])
        self._fruit_names = np.array([[name for name,_,_ in d["fruit"]] for d in self.infos])
        if self.r is not None:
            self.dirs_b,self.wts = self.r.ray_directions()
            self.wts_t = torch.as_tensor(self.wts,dtype=torch.float32,device=self.fb.device)
        self._selections,self._program_pulses = {},{}
        self._row_inputs = [_RowStimulus(self,i) for i in range(self.B)]
        self.tasting = np.zeros(self.B)
        self.smell_values = ({},{})
        self.commands,self.wcommands = [],[]
        self.episode_frames = np.zeros(self.B,dtype=np.int64)
        self.loom_t = np.full(self.B,-1.)
        self.loom_speed = np.ones(self.B); self.loom_final = np.full(self.B,.035); self.loom_start = np.full(self.B,.5)

    @property
    def feeding(self): return self.body.feeding

    @property
    def hops_escape(self):
        """Per-row count of take-offs by the GF escape route (batch_body.BatchBody.hops_escape); not part of the checkpoint."""
        return self.body.hops_escape

    @property
    def hops_voluntary(self):
        """Per-row count of take-offs by the voluntary wing-power route (batch_body.BatchBody.hops_voluntary)."""
        return self.body.hops_voluntary

    def _rows(self, rows):
        if rows is None: return np.arange(self.B)
        rows = np.atleast_1d(np.asarray(rows))
        if rows.ndim != 1 or not np.issubdtype(rows.dtype,np.integer) or np.any((rows<0)|(rows>=self.B)) or len(np.unique(rows)) != len(rows):
            raise ValueError("rows must be unique batch indices")
        return rows

    def nearest_fruit(self):
        pos = np.array([f.pos for f in self.flies])
        distances = np.linalg.norm(pos[:,None]-self._fruit_centres,axis=-1)-self._fruit_radii
        j = distances.argmin(1); rows = np.arange(self.B)
        return self._fruit_names[rows,j],distances[rows,j]

    def column_radiance(self, pose=None):
        if self.optic is None: return None
        eye,f,left,up = frames(self.flies) if pose is None else pose
        rotation = np.stack([f,left,up],axis=2)
        directions = self.dirs_b.reshape(1,-1,3)@rotation.transpose(0,2,1)
        origins = np.broadcast_to(eye[:,None],directions.shape)
        radiance = self.world.trace(torch.from_numpy(np.ascontiguousarray(origins,dtype=np.float32)),
                                    torch.from_numpy(np.ascontiguousarray(directions,dtype=np.float32)),
                                    cuda_graphs=self.sensory_cuda_graphs)
        return (radiance.reshape(self.B,*self.dirs_b.shape[:2],4)*self.wts_t[None,None,:,None]).sum(2)

    def step(self):
        dt = self.FRAME_MS/1000
        pose = frames(self.flies)
        if self.optic is not None:
            self.col_rad = self.column_radiance(pose)
            self.fb.vision(self.col_rad)
        self.tasting = ((self.nearest_fruit()[1]<.015)&~np.array([f.airborne for f in self.flies])).astype(float)
        self.air.step(dt)
        eye,f,left,_ = pose
        self.smell_values = self.air.antennae(eye,left,f)
        available = self.fb.available_senses
        if "smell" in available: self.fb.smell(*self.smell_values)
        if "wind" in available: self.fb.wind(*self.air.deflections(f,left))
        if "taste" in available: self.fb.taste(self.tasting)
        if "proprioception" in available:
            # the sided extras (leg-cycle state, side-split haltere MN rates) ride on the sense, since FlyBrain.proprioception's
            # signature is the round-2 one; take_body returns the five base arguments and holds the rest for rates()
            sense = self.fb.proprioception_sense
            self.fb.proprioception(**sense.take_body(self.body.proprio_state(self.motor,haltere_sides=sense.haltere_sides(self.brain))))
        self.fb.step(self.FRAME_MS)
        self.motor = self.fb.motor()
        self.commands,self.wcommands = self.body.readout(self.motor,dt)
        self._program_pulses.clear()
        for i,(program,gating) in enumerate(zip(self.programs,self.gatings)):
            if program is None and gating is None: continue
            motor = self.motor.row(i)
            cmd,wcmd = self.commands[i],self.wcommands[i]
            if program is not None:
                parts = program.parts if isinstance(program,programs.Composite) else [program]
                for part in parts:
                    if hasattr(part,"pfl_hz"):
                        info = part.apply(self._row_inputs[i],motor,self.flies[i],self.metabolisms[i],dt)
                        cmd = dict(cmd,mode=info["mode"],rates=dict(cmd["rates"],**{"odour Hz":info["odour_hz"],"gate x10":info["gate"]*10,"steer err x10":info["error"]*10}))
                    elif isinstance(part,programs.KlinotaxisProgram):
                        antennae = tuple({k:float(v[i]) for k,v in values.items()} for values in self.smell_values)
                        cmd = part.apply(motor,cmd,self.flies[i],self.metabolisms[i],dt,antennae=antennae)
                    else: cmd = part.apply(motor,cmd,self.flies[i],self.metabolisms[i],dt)
            if gating is not None: wcmd = gating.apply(motor,wcmd,self.flies[i],dt)
            self.commands[i],self.wcommands[i] = cmd,wcmd
        for idx,hz,ms in self._program_pulses.values(): self.fb.stimulate(idx,hz,ms)
        self.body.step(self.commands,self.wcommands,self.tasting,dt)
        self._update_loom(dt)
        self.episode_frames += 1
        return self.motor

    def start_loom(self, *, rows=None, speed=1., radius=.03, final=.035, start=.5):
        rows = self._rows(rows)
        if not np.isfinite([speed,radius,final,start]).all() or min(speed,radius,final)<=0 or start<final:
            raise ValueError("loom needs finite positive speed/radius/final and start >= final")
        self.loom_t[rows] = 0.
        self.loom_speed[rows],self.loom_final[rows],self.loom_start[rows] = speed,final,start
        self.world.move_sphere(self.loom_idx,(9,9,9),(radius,)*3,rows=rows)

    def _update_loom(self, dt):
        rows = np.flatnonzero(self.loom_t>=0)
        if not len(rows): return
        self.loom_t[rows] += dt
        done = self.loom_t[rows]>(self.loom_start[rows]-self.loom_final[rows])/self.loom_speed[rows]+.3
        dist = np.maximum(self.loom_start[rows]-self.loom_speed[rows]*self.loom_t[rows],self.loom_final[rows])
        centre = frames([self.flies[i] for i in rows])[0]+np.stack([np.zeros(len(rows)),dist,np.full(len(rows),.01)],axis=1)
        centre[done] = (9,9,9); self.loom_t[rows[done]] = -1.
        self.world.move_sphere(self.loom_idx,centre,rows=rows)

    def stimulate_gf(self, rows=None):
        hz = np.zeros((self.B,1)); hz[self._rows(rows)] = 200.
        self.fb.stimulate({"type":"DNp01"},hz,30.)

    def stimulate_wing_dns(self, ms=1000., rows=None):
        hz = np.zeros((self.B,1)); hz[self._rows(rows)] = 120.
        self.fb.stimulate({"type":["DNg02_a","DNa08"]},hz,ms)

    def reset(self, rows=None):
        """Reset selected episodes; partial resets retain the shared neural clock/RNG."""
        selected = self._rows(rows)
        if not len(selected): return
        self.fb.reset(None if rows is None else selected)
        for i in selected:
            x,y,z = self.starts[i]
            self.flies[i] = body.FlyState(x=x,y=y,z=z,heading=np.deg2rad(5))
            self.locos[i],self.flights[i],self.metabolisms[i] = body.Locomotion(),body.Flight(),body.Metabolism()
            self.airs[i].__dict__.clear(); self.airs[i].__dict__.update(copy.deepcopy(self._air_initial[i]))
            self.programs[i] = programs.make_program(self.program_names[i])
            _load_program(self.programs[i],self._program_initial[i])
            self.gatings[i] = programs.EscapeGating() if self._gating_initial[i] is not None else None
            _load_program(self.gatings[i],self._gating_initial[i])
        self.body.feeding[selected] = False; self.tasting[selected] = 0.
        self.body.hops_escape[selected] = 0; self.body.hops_voluntary[selected] = 0
        self.body.launched_escape[selected] = False; self.body.launched_voluntary[selected] = False
        self.episode_frames[selected] = 0; self.loom_t[selected] = -1.
        self.world.move_sphere(self.loom_idx,(9,9,9),(.03,)*3,rows=selected)
        self._program_pulses.clear()
        # The proprioception transducer (if attached) must start a reset row from zero MN rates again: drop the whole
        # snapshot on a full reset, zero only the selected rows on a partial one.
        self.motor = None if rows is None else _motor_rows_cleared(self.motor,selected,self.B)

    def state_dict(self):
        """Owned CPU snapshot, including every program RNG and independent environment."""
        names = ("flies","locos","flights","metabolisms","tasting","smell_values","commands","wcommands",
                 "episode_frames","loom_t","loom_speed","loom_final","loom_start","motor")
        state = {k:copy.deepcopy(getattr(self,k)) for k in names}
        state.update(version=1,batch=self.B,seeds=self.seeds,program_names=self.program_names.copy(),fence=self.fence,
                     fruit_set=self.fruit_set,escape_gating=self.escape_gating,proprioception=self.proprioception,
                     controller=self.fb.state_dict(),feeding=self.feeding.copy(),
                     airs=[copy.deepcopy(a.__dict__) for a in self.airs],
                     programs=[_program_state(p) for p in self.programs],gatings=[_program_state(g) for g in self.gatings],
                     scenes=[{k:copy.deepcopy(getattr(w,k)) for k in ("spheres","boxes","planes","light_pos","light_color","ambient","detail")} for w in self.world.worlds])
        return state

    def load_state_dict(self, state):
        if (state["version"],state["batch"],tuple(state["seeds"]),state["program_names"],state["fence"],state["fruit_set"],state["escape_gating"],state.get("proprioception")) != (1,self.B,self.seeds,self.program_names,self.fence,self.fruit_set,self.escape_gating,self.proprioception):
            raise ValueError("BatchSim checkpoint configuration does not match")
        self.fb.load_state_dict(state["controller"])
        for k in ("flies","locos","flights","metabolisms"):
            getattr(self,k)[:] = copy.deepcopy(state[k])
        for k in ("tasting","smell_values","commands","wcommands","episode_frames","loom_t","loom_speed","loom_final","loom_start"):
            setattr(self,k,copy.deepcopy(state[k]))
        # The previous frame's MotorRates: the opt-in proprioception transducer's only cross-frame input, so a resumed
        # run with the sense on must feed the same snapshot. Absent in checkpoints written before it was saved -> None.
        self.motor = copy.deepcopy(state.get("motor"))
        self.body.feeding[:] = state["feeding"]
        for i in range(self.B):
            self.airs[i].__dict__.clear(); self.airs[i].__dict__.update(copy.deepcopy(state["airs"][i]))
            _load_program(self.programs[i],state["programs"][i]); _load_program(self.gatings[i],state["gatings"][i])
            self.world.worlds[i].__dict__.update(copy.deepcopy(state["scenes"][i]))
        self.world.invalidate(); self.air = BatchAir(self.airs)
        self._program_pulses.clear()

    def save_state(self, path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        torch.save(self.state_dict(),path)

    def load_state(self, path):
        self.load_state_dict(torch.load(path,map_location="cpu",weights_only=False))
