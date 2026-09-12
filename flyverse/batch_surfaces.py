"""Batch surface queries; uncommon edge transitions use the scalar geometry authority."""
import numpy as np

from .surfaces import EPS


class BatchSurfaces:
    def __init__(self, surfaces):
        self.scalar = surfaces
        faces = surfaces.faces
        self.axis = np.array([f.axis for f in faces])
        self.sign = np.array([f.sign for f in faces])
        self.coord = np.array([f.coord for f in faces])
        self.lo = np.array([f.lo for f in faces]); self.hi = np.array([f.hi for f in faces])
        self.normal = np.array([f.normal for f in faces])
        self.ids = {id(f):i for i,f in enumerate(faces)}

    def crossing(self, p, q):
        delta = q[:,self.axis]-p[:,self.axis]
        valid = (np.abs(delta)>=EPS)&((p[:,self.axis]-self.coord)*self.sign>1e-7)&((q[:,self.axis]-self.coord)*self.sign<=1e-9)
        t = (self.coord-p[:,self.axis])/np.where(np.abs(delta)>=EPS,delta,1.)
        point = p[:,None,:]+t[...,None]*(q-p)[:,None,:]
        valid &= ((point>=self.lo-1e-4)&(point<=self.hi+1e-4)).all(-1)
        times = np.where(valid,t,np.inf)
        index = times.argmin(1)
        first = times[np.arange(len(p)),index]
        return first,np.where(np.isfinite(first),index,-1)

    def walk(self, p, fwd, normal, faces, ds, dyaw):
        indices = np.array([self.ids.get(id(f),-1) for f in faces])
        for i in np.flatnonzero(indices<0):
            f = self.scalar.face_at(p[i],normal[i]) or self.scalar.faces[0]
            indices[i] = self.ids[id(f)]
        n = self.normal[indices].copy()
        f = np.cos(dyaw)[:,None]*fwd+np.sin(dyaw)[:,None]*np.cross(n,fwd)
        f -= n*(f*n).sum(-1)[:,None]
        f /= np.maximum(np.linalg.norm(f,axis=1),EPS)[:,None]
        q = p+ds[:,None]*f
        _,cross = self.crossing(p+1e-5*n,q+1e-5*n)
        outside = ((q<self.lo[indices]-1e-9)|(q>self.hi[indices]+1e-9)).any(-1)
        moving = np.abs(ds)>=EPS
        exceptional = moving&(((cross>=0)&(cross!=indices))|outside)
        q[np.arange(len(p)),self.axis[indices]] = self.coord[indices]
        q[~moving] = p[~moving]
        out_faces = [self.scalar.faces[j] for j in indices]
        for i in np.flatnonzero(exceptional):
            q[i],f[i],n[i],out_faces[i] = self.scalar.walk(p[i],fwd[i],normal[i],out_faces[i],ds[i],dyaw[i])
        return q,f,n,out_faces

    def land(self, p, q):
        t,index = self.crossing(p,q)
        hit = index>=0
        points = q.copy()
        rows = np.flatnonzero(hit)
        points[rows] = p[rows]+t[rows,None]*(q[rows]-p[rows])
        points[rows,self.axis[index[rows]]] = self.coord[index[rows]]+self.sign[index[rows]]*1e-5
        solid = ((q<self.scalar.room[0]-1e-7)|(q>self.scalar.room[1]+1e-7)).any(-1)
        for lo,hi in self.scalar.solids:
            solid |= ((q>lo+1e-7)&(q<hi-1e-7)).all(-1)
        for i in np.flatnonzero(~hit&solid):
            points[i],face = self.scalar.snap(q[i])
            index[i] = self.ids[id(face)]
        return points,index
