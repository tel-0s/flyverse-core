"""Independent scenes traced together, with the same shading as :mod:`world`.

Rows have the same primitive counts, but separate geometry, materials and lights.
Rays never intersect another row's scene. CUDA capture covers the entire batch.
"""
from __future__ import annotations

import numpy as np
import torch

from .world import World, MATERIALS, INF, _NOISE_CORNERS


class BatchWorld(World):
    def __init__(self, worlds, device=None):
        self.worlds = list(worlds)
        if not self.worlds:
            raise ValueError("BatchWorld needs at least one scene")
        self.B = len(self.worlds)
        self.device = torch.device(device or self.worlds[0].device)
        self.detail = self.worlds[0].detail
        self._packed = False

    def _shape(self):
        shapes = [(len(w.spheres),len(w.boxes),len(w.planes)) for w in self.worlds]
        if any(s != shapes[0] for s in shapes):
            raise ValueError("batched scenes must have matching primitive counts")
        if any(w.detail != self.detail for w in self.worlds):
            raise ValueError("batched scenes must share the texture detail setting")
        return shapes[0]

    def _pack(self):
        self._scene_shape = self._shape()
        self.S,self.K,self.P = self._scene_shape
        self.stride = self.S+self.K+self.P+1
        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(np.asarray(value),dtype=dtype,device=self.device)
        for target,kind,attr,n in (("sc","spheres","center",self.S),("sr","spheres","radii",self.S),
                                   ("blo","boxes","lo",self.K),("bhi","boxes","hi",self.K),
                                   ("pp","planes","point",self.P),("pn","planes","normal",self.P)):
            data = [[getattr(p,attr) for p in getattr(w,kind)] for w in self.worlds]
            setattr(self,"_"+target,tensor(data).reshape(self.B,n,3))
        self._pn = self._pn/self._pn.norm(dim=-1,keepdim=True).clamp_min(1e-9)
        materials = []
        for w in self.worlds:
            materials.extend([MATERIALS[p.material] for p in w.spheres+w.boxes+w.planes]+[None])
        for attr,default,dtype in (("refl",(0,)*4,torch.float32),("refl2",(0,)*4,torch.float32),
                                   ("emit",(0,)*4,torch.float32),("pattern",0,torch.long),("scale",1.,torch.float32)):
            setattr(self,"_pscale" if attr=="scale" else "_"+attr,
                    tensor([getattr(m,attr) if m else default for m in materials],dtype))
        self._lp = tensor([w.light_pos for w in self.worlds])
        self._lc = tensor([w.light_color for w in self.worlds])
        self._amb = tensor([w.ambient for w in self.worlds])
        self._row = torch.arange(self.B,device=self.device)[:,None]
        self._noise_corners = torch.tensor(_NOISE_CORNERS,device=self.device)
        self._trace_graphs = {}
        self._packed = True

    def move_sphere(self, idx, center, radii=None, *, rows=None):
        rows = np.arange(self.B) if rows is None else np.asarray(rows,dtype=int)
        centers = np.broadcast_to(np.asarray(center,float),(len(rows),3))
        sizes = None if radii is None else np.broadcast_to(np.asarray(radii,float),(len(rows),3))
        if not 0 <= idx < len(self.worlds[0].spheres) or np.any((rows<0)|(rows>=self.B)):
            raise IndexError("sphere/scene index out of range")
        for j,row in enumerate(rows):
            self.worlds[row].move_sphere(idx,centers[j],None if sizes is None else sizes[j])
        if self._packed:
            self._sc[:,idx].copy_(torch.tensor([w.spheres[idx].center for w in self.worlds],dtype=torch.float32))
            if sizes is not None:
                self._sr[:,idx].copy_(torch.tensor([w.spheres[idx].radii for w in self.worlds],dtype=torch.float32))

    @torch.no_grad()
    def trace(self, origins, dirs, *, cuda_graphs=False):
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda",torch.cuda.current_device())
        elif device.type == "mps" and device.index is None:
            device = torch.device("mps",0)
        self.device = device
        if cuda_graphs and device.type != "cuda":
            raise ValueError("ray-tracing CUDA graphs require a CUDA device")
        if origins.shape != dirs.shape or origins.ndim != 3 or origins.shape[0] != self.B or origins.shape[2] != 3:
            raise ValueError("batched rays must have shape (batch, rays, 3)")
        if not self._packed or self._sc.device != device or self._scene_shape != self._shape():
            self._pack()
        o,d = origins.to(device,torch.float32),dirs.to(device,torch.float32)
        return self._graph_trace(o,d) if cuda_graphs else self._trace(o,d)

    def _intersect(self, o, d):
        shape = o.shape[:2]
        best_t = torch.full(shape,INF,device=o.device)
        best_n = torch.zeros_like(o)
        best_id = torch.full(shape,self.stride-1,dtype=torch.long,device=o.device)
        eps = 1e-4
        if self.S:
            oc = (o[:,:,None]-self._sc[:,None])/self._sr[:,None]
            dd = d[:,:,None]/self._sr[:,None]
            a = (dd*dd).sum(-1); b = 2*(oc*dd).sum(-1); c = (oc*oc).sum(-1)-1
            disc = b*b-4*a*c
            sq = torch.sqrt(disc.clamp_min(0))
            t0,t1 = (-b-sq)/(2*a),(-b+sq)/(2*a)
            t = torch.where(t0>eps,t0,t1)
            t = torch.where((disc>0)&(t>eps),t,torch.full_like(t,INF))
            tmin,si = t.min(dim=2)
            hit = tmin<best_t
            pt = o+d*tmin[...,None]
            n = (pt-self._sc[self._row,si])/(self._sr[self._row,si]**2)
            n = n/n.norm(dim=-1,keepdim=True).clamp_min(1e-9)
            best_n = torch.where(hit[...,None],n,best_n); best_t = torch.where(hit,tmin,best_t)
            best_id = torch.where(hit,si,best_id)
        if self.K:
            inv = 1./torch.where(d.abs()<1e-9,torch.full_like(d,1e-9),d)
            tlo = (self._blo[:,None]-o[:,:,None])*inv[:,:,None]
            thi = (self._bhi[:,None]-o[:,:,None])*inv[:,:,None]
            tn,tf = torch.minimum(tlo,thi),torch.maximum(tlo,thi)
            enter,ax = tn.max(dim=-1); leave = tf.min(dim=-1).values
            t = torch.where((leave>enter)&(enter>eps),enter,torch.full_like(enter,INF))
            tmin,bi = t.min(dim=2); hit = tmin<best_t
            axis = ax.gather(2,bi[...,None])[...,0]
            sign = -torch.sign(d.gather(2,axis[...,None])[...,0])
            n = torch.zeros_like(o); n.scatter_(2,axis[...,None],sign[...,None])
            best_n = torch.where(hit[...,None],n,best_n); best_t = torch.where(hit,tmin,best_t)
            best_id = torch.where(hit,bi+self.S,best_id)
        if self.P:
            denom = (d[:,:,None]*self._pn[:,None]).sum(-1)
            t = ((self._pp[:,None]-o[:,:,None])*self._pn[:,None]).sum(-1)/torch.where(denom.abs()<1e-9,torch.full_like(denom,1e-9),denom)
            t = torch.where((t>eps)&(denom.abs()>1e-9),t,torch.full_like(t,INF))
            tmin,pi = t.min(dim=2); hit = tmin<best_t
            n = self._pn[self._row,pi]
            n = torch.where((n*d).sum(-1,keepdim=True)>0,-n,n)
            best_n = torch.where(hit[...,None],n,best_n); best_t = torch.where(hit,tmin,best_t)
            best_id = torch.where(hit,pi+self.S+self.K,best_id)
        return best_t,best_n,best_id+self._row*self.stride

    def _trace(self, o, d):
        t,n,mid = self._intersect(o,d)
        p = o+d*t[...,None]
        to_l = self._lp[:,None]-p
        dist = to_l.norm(dim=-1,keepdim=True)
        light = to_l/dist.clamp_min(1e-9)
        lam = (n*light).sum(-1).clamp_min(0)
        ts,_,_ = self._intersect(p+n*1e-3,light)
        lit = (ts>=dist[...,0]).float()
        falloff = 4./(1.+dist[...,0]**2)
        refl = self._texture(p.reshape(-1,3),n.reshape(-1,3),mid.reshape(-1)).reshape(*t.shape,4)
        rad = refl*(self._amb[:,None]+self._lc[:,None]*(lam*lit*falloff)[...,None])+self._emit[mid]
        return torch.where((t<INF)[...,None],rad,torch.zeros_like(rad))
