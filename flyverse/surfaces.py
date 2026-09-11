"""Walkable surfaces: flies stick to what they stand on.

The walkable world is a set of axis-aligned rectangles (faces): the inside of the room box (floor,
walls, ceiling) and the outside of every solid box (the table slab, its legs). A walking fly carries a
position on a face, a forward vector in the face's plane and the face's outward normal; it turns about
the normal and advances along the forward vector. Two things happen at edges, both without any
decision by the fly:

* **convex edge** (walking off the table top): the fly continues onto the side face, rotated over the
  edge -- new normal = the side's outward normal (the direction it was walking), new forward = minus
  the old normal (down the side). Flies do not fall off tables; they walk around them.
* **concave corner** (walking into a wall or a table leg from the floor): the fly climbs -- new normal =
  the face it ran into, new forward = the old normal (up).

A flying fly lands on the first face its path crosses (the table top from above, a wall, the floor).
Everything else -- gravity, the size of a hop, what the fly sees -- is unchanged; the point is that
"fell off the table and starved on the floor" stops being a body-model artefact. Fruit are not
walkable (the fly's legs are on the substrate; contact with fruit is the taste test).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


@dataclass(frozen=True)
class Face:
    axis: int          # 0, 1, 2: the face is perpendicular to this axis
    sign: int          # outward normal = sign * e_axis (points into the air the fly stands in)
    coord: float       # position of the plane along `axis`
    lo: np.ndarray     # (3,) rectangle bounds (the `axis` entry equals coord)
    hi: np.ndarray
    label: str

    @property
    def normal(self) -> np.ndarray:
        n = np.zeros(3); n[self.axis] = self.sign; return n

    def contains(self, p: np.ndarray, tol: float = 1e-4) -> bool:
        return bool(np.all(p >= self.lo - tol) and np.all(p <= self.hi + tol))


class Surfaces:
    def __init__(self, room: tuple, solids: list[tuple], labels: list[str] | None = None):
        """room = (x0, x1, y0, y1, z0, z1): the fly lives inside; solids = [(x0, x1, y0, y1, z0, z1), ...]."""
        self.room = np.array(room, float).reshape(3, 2).T          # (2, 3): lo / hi
        self.solids = [np.array(s, float).reshape(3, 2).T for s in solids]
        self.faces: list[Face] = []
        room_labels = {(2, 1): "floor", (2, -1): "ceiling"}
        for axis in range(3):
            for sign, k in ((1, 0), (-1, 1)):                       # inward normal: +e at the low plane, -e at the high plane
                lo, hi = self.room[0].copy(), self.room[1].copy(); lo[axis] = hi[axis] = self.room[k, axis]
                self.faces.append(Face(axis, sign, self.room[k, axis], lo, hi, room_labels.get((axis, sign), "wall")))
        labels = labels or [f"solid{i}" for i in range(len(self.solids))]
        for box, lab in zip(self.solids, labels):
            for axis in range(3):
                for sign, k in ((-1, 0), (1, 1)):                   # outward normal: -e at the low plane, +e at the high plane
                    lo, hi = box[0].copy(), box[1].copy(); lo[axis] = hi[axis] = box[k, axis]
                    name = lab + (" top" if (axis, sign) == (2, 1) else " underside" if (axis, sign) == (2, -1) else " side")
                    self.faces.append(Face(axis, sign, box[k, axis], lo, hi, name))

    # ------------------------------------------------------------------ queries
    def in_solid(self, p: np.ndarray, tol: float = 1e-7) -> bool:
        if np.any(p < self.room[0] - tol) or np.any(p > self.room[1] + tol):
            return True                                              # outside the room counts as solid (walls)
        return any(np.all(p > b[0] + tol) and np.all(p < b[1] - tol) for b in self.solids)

    def face_at(self, p: np.ndarray, n: np.ndarray) -> Face | None:
        axis = int(np.argmax(np.abs(n))); sign = int(np.sign(n[axis]))
        best = None
        for f in self.faces:
            if f.axis == axis and f.sign == sign and abs(p[axis] - f.coord) < 1e-3 and f.contains(p, 2e-3):
                best = f; break
        return best

    def support(self, x: float, y: float) -> float:
        """z of the highest upward-facing surface below (x, y): the table top or the floor."""
        z = self.room[0, 2]
        for b in self.solids:
            if b[0, 0] <= x <= b[1, 0] and b[0, 1] <= y <= b[1, 1]:
                z = max(z, b[1, 2])
        return float(z)

    def _crossing(self, p: np.ndarray, q: np.ndarray):
        """First face plane the segment p -> q crosses into a solid (or out of the room); (t, face) or None."""
        best = None
        for f in self.faces:
            d = q[f.axis] - p[f.axis]
            if abs(d) < EPS:
                continue
            if (p[f.axis] - f.coord) * f.sign <= 1e-7:               # start must be strictly on the air side of the plane
                continue
            if (q[f.axis] - f.coord) * f.sign > 1e-9:                # end still on the air side: not entering
                continue
            if abs(q[f.axis] - p[f.axis]) < EPS:
                continue
            t = (f.coord - p[f.axis]) / d
            hit = p + t * (q - p)
            if f.contains(hit, 1e-4) and (best is None or t < best[0]):
                best = (t, f)
        return best

    # ------------------------------------------------------------------ walking
    def walk(self, p: np.ndarray, fwd: np.ndarray, n: np.ndarray, face: Face | None, ds: float, dyaw: float):
        """Advance ds along fwd after turning dyaw about n. Returns (p, fwd, n, face)."""
        face = face or self.face_at(p, n) or self.faces[0]
        n = face.normal
        left = np.cross(n, fwd)
        fwd = np.cos(dyaw) * fwd + np.sin(dyaw) * left
        fwd -= n * np.dot(fwd, n); fwd /= max(np.linalg.norm(fwd), EPS)
        if abs(ds) < EPS:
            return p, fwd, n, face
        q = p + ds * fwd
        # concave: the step runs into another solid (or the room boundary) -> climb it
        cross = self._crossing(p + 1e-5 * n, q + 1e-5 * n)
        if cross is not None and cross[1] is not face:
            t, hit = cross
            hp = p + max(t - 1e-6, 0.0) * (q - p)
            hp[hit.axis] = hit.coord + hit.sign * 1e-5
            new_fwd = n.copy()                                       # up the new face along the old normal
            return hp, new_fwd, hit.normal, hit
        # convex: the step leaves this face's rectangle -> over the edge onto the adjacent face
        if not face.contains(q, 1e-9):
            over = [(k, -1 if q[k] < face.lo[k] else 1) for k in range(3) if k != face.axis and (q[k] < face.lo[k] - 1e-9 or q[k] > face.hi[k] + 1e-9)]
            if over:
                k, s = over[0]
                edge = q.copy(); edge[k] = face.lo[k] if s < 0 else face.hi[k]; edge[face.axis] = face.coord
                target_n = np.zeros(3); target_n[k] = s
                nxt = self.face_at(edge + target_n * 1e-5 - n * 1e-5, target_n)
                if nxt is not None:
                    new_p = edge.copy(); new_p[k] = nxt.coord + s * 1e-5
                    new_fwd = -n.copy()                              # down the side
                    return new_p, new_fwd, nxt.normal, nxt
                q[k] = edge[k]                                       # no adjacent face (e.g. the room ceiling edge): slide
        q[face.axis] = face.coord
        return q, fwd, n, face

    def snap(self, p: np.ndarray):
        """Nearest face to a point that has ended up inside a solid / outside the room: (point on it, face)."""
        best = None
        for f in self.faces:
            q = np.clip(p, f.lo, f.hi); q[f.axis] = f.coord + f.sign * 1e-5
            d = float(np.linalg.norm(q - p))
            if best is None or d < best[0]:
                best = (d, q, f)
        return best[1], best[2]

    # ------------------------------------------------------------------ flight
    def land(self, p_prev: np.ndarray, p_new: np.ndarray):
        """Did the flight path cross into a face? -> (point, face) or None."""
        cross = self._crossing(p_prev, p_new)
        if cross is None:
            return None
        t, f = cross
        hit = p_prev + t * (p_new - p_prev); hit[f.axis] = f.coord + f.sign * 1e-5
        return hit, f
