"""Vectorised physical sensing across independent Air instances (one environment per row)."""
import numpy as np


class BatchAir:
    def __init__(self, airs):
        self.airs = list(airs)
        if not self.airs:
            raise ValueError("BatchAir needs at least one environment")
        self.B = len(airs)
        self.gloms = self.airs[0].gloms
        if any(a.gloms != self.gloms or len(a.sources) != len(self.airs[0].sources) for a in airs):
            raise ValueError("batched air requires matching source counts and glomeruli")
        self.src = np.array([[p for _,p,_ in a.sources] for a in airs]).reshape(self.B,-1,3)
        self.strength = np.array([[s for _,_,s in a.sources] for a in airs])
        self.radius = np.array([a.src_rad for a in airs])
        self.weights = np.array([a._odour_mat for a in airs])

    def step(self, dt_s):
        for a in self.airs:
            a.step(dt_s)

    def _param(self, kind, name):
        return np.array([getattr(getattr(a,kind),name) for a in self.airs])

    def vectors(self):
        directions = np.array([a.direction for a in self.airs])
        return self._param("wind","speed")[:,None]*np.stack([np.cos(directions),np.sin(directions),np.zeros(self.B)],axis=1)

    def concentration(self, pos):
        """(B, samples, 3) -> (B, samples, glomeruli)."""
        pos = np.asarray(pos,float)
        if pos.ndim != 3 or pos.shape[0] != self.B or pos.shape[-1] != 3:
            raise ValueError("positions must have shape (batch, samples, 3)")
        if not self.src.shape[1]:
            return np.zeros((*pos.shape[:2],len(self.gloms)))
        def p(name): return self._param("plume",name)[:,None,None]
        direction = self.vectors()/np.maximum(self._param("wind","speed"),1e-6)[:,None]
        delta = pos[:,:,None,:]-self.src[:,None,:,:]
        x = (delta*direction[:,None,None,:]).sum(-1)
        perp = delta-x[...,None]*direction[:,None,None,:]
        pz = np.maximum(np.abs(perp[...,2])-self.radius[:,None],0.)
        r2 = perp[...,0]**2+perp[...,1]**2+(2*pz)**2
        conc = p("near_gain")/(1.+(np.linalg.norm(delta,axis=-1)/p("near_d0"))**2)
        sig = np.maximum(p("sigma0"),self.radius[:,None])+p("spread")*np.maximum(x,0)
        time = np.array([a.t for a in self.airs])[:,None,None]
        phase = np.array([a.phase for a in self.airs])[:,None,:]
        puff = 1.+p("puff_depth")*np.sin(2*np.pi*time/p("puff_period_s")+phase)
        conc += np.where(x>0,p("plume_gain")*puff*(p("sigma0")/sig)**2*np.exp(-r2/(2*sig**2)),0.)
        return (conc*self.strength[:,None])@self.weights

    def antennae(self, eye, left, forward, antenna_sep=.001):
        positions = np.stack([eye+left*antenna_sep/2+forward*.0005,eye-left*antenna_sep/2+forward*.0005],axis=1)
        values = self.concentration(positions)
        return tuple({g:values[:,side,j] for j,g in enumerate(self.gloms)} for side in (0,1))

    def deflections(self, forward, left, full_speed=.5):
        vector = self.vectors()
        wx,wy = (forward*vector).sum(-1),(left*vector).sum(-1)
        c = np.cos(np.pi/4)
        return -(wx*c+wy*c)/full_speed,-(wx*c-wy*c)/full_speed
