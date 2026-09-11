"""A connectome controller with physical inputs and named neural outputs."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time

import numpy as np
import torch

from . import brain, connectome, optic, regions, retina, senses
from .motor import MotorRates, motor_groups, wing_groups, read_motor


@dataclass(frozen=True)
class StepResult:
    simulated_ms: float
    wall_ms: float
    steps: int

    @property
    def time_dilation(self):
        """Simulated time / elapsed wall time (1 is real time)."""
        return self.simulated_ms / self.wall_ms if self.wall_ms else 0.0


class FlyBrain:
    """Held sensory frames, simulation time in ms, and MotorRates in Hz.

    Smell takes dictionaries {glomerulus: scalar or (B,) concentration}; wind takes
    normalized antennal deflections; taste is sugar contact in [0,1]. Vision takes
    (columns,4) or (B,columns,4) radiance [UV,B,G,R] in this controller's retina order.
    Body IDs remain stable across modules; integer stimulus selectors are local row indices.
    """
    BRAIN_TENSORS = ("v", "g", "refrac", "drive", "poisson_p", "rate", "spikes", "adapt", "res", "spike_buf", "spike_counts")
    OPTIC_TENSORS = ("v", "adapt", "I_lp", "I_mean", "_fresh", "contrast", "delta_rate")

    def __init__(self, c=None, *, modules=None, batch=1, device=None, seed=0,
                 lif_params=None, optic_params=None, eye_geometry=None, cuda_graphs=False):
        self.c = regions.subset(c if c is not None else connectome.load(verbose=False), modules)
        if self.c.n == 0:
            raise ValueError("FlyBrain needs at least one neuron")
        self.brain = brain.Brain(self.c, lif_params, device=device, batch=batch, seed=seed)
        self.B, self.device = self.brain.B, self.brain.device
        if cuda_graphs and (self.device.type != "cuda" or self.brain.event_driven):
            raise ValueError("CUDA graphs require a CUDA device and the sparse-matmul backend")
        self.cuda_graphs = cuda_graphs
        self._graphs = {}
        self.brain.record_activity = True
        self.retina = self.optic = None
        if len(self.c.select(type=connectome.PHOTORECEPTOR_TYPES)) and len(self.c.select(superclass="ol_intrinsic")):
            self.retina = retina.build_retina(self.c, eye_geometry)
            if self.retina.n_columns:
                self.optic = optic.OpticLobe(self.c, self.retina, optic_params, device=self.device, batch=self.B)
                self.optic.relax()
                self.brain.freeze(self.optic.rate_idx)
        self.olfaction = senses.Smell(self.c) if len(self.c.select(**{"class": "olfactory"})) else None
        self.wind_sense = senses.Wind(self.c) if len(self.c.select(type="~^JO-[CE]")) else None
        taste = senses.Taste(self.c)
        self.taste_sense = taste if len(taste.sweet) else None
        self.groups, self.wings = motor_groups(self.c), wing_groups(self.c)
        self._radiance = None
        self._pending_ms = 0.0
        self._base_poisson = torch.zeros_like(self.brain.poisson_p)
        self._inputs_on = {}
        self._input_indices = {}
        self._pulses = []
        self._activity_ms = np.zeros(self.B)
        self._budget_estimate = None

    @property
    def t(self):
        return self.brain.t

    @property
    def available_senses(self):
        return tuple(name for name, value in (("vision", self.optic), ("smell", self.olfaction),
                     ("wind", self.wind_sense), ("taste", self.taste_sense)) if value is not None)

    def _require(self, name, sensor):
        if sensor is None:
            raise ValueError(f"{name} is absent from this connectome subset")

    def vision(self, radiance):
        self._require("vision", self.optic)
        value = torch.as_tensor(radiance, dtype=torch.float32, device=self.device)
        if value.ndim == 2:
            value = value[None].expand(self.B, -1, -1)
        if value.shape != (self.B, self.retina.n_columns, 4):
            raise ValueError(f"radiance must have shape ({self.B}, {self.retina.n_columns}, 4) or ({self.retina.n_columns}, 4)")
        if not bool(torch.isfinite(value).all() & (value >= 0).all()):
            raise ValueError("radiance must be finite and nonnegative")
        if self._radiance is None:
            self._radiance = value.clone()
        else:
            self._radiance.copy_(value)

    def _input(self, name, idx, hz):
        # Cache GPU selectors and avoid a host sync to rediscover whether inputs are on.
        if name not in self._input_indices:
            self._input_indices[name] = self.brain._idx(idx)
        ti = self._input_indices[name]
        prob = torch.as_tensor(np.asarray(hz, dtype=np.float32), device=self.device) * (self.brain.p.dt / 1000)
        self.brain.poisson_p[:, ti] = prob
        self._base_poisson[:, ti] = prob
        self._inputs_on[name] = bool(np.any(np.asarray(hz) > 0))
        if self._pulses:
            self._refresh_poisson()
        else:
            self.brain._poisson_on = any(self._inputs_on.values())

    def smell(self, cL, cR):
        self._require("smell", self.olfaction)
        self._input("smell", self.olfaction.orn_idx, self.olfaction.rates(cL, cR, self.B))

    def wind(self, dL, dR):
        self._require("wind", self.wind_sense)
        rE, rC = self.wind_sense.rates(dL, dR, self.B)
        self._input("wind_E", self.wind_sense.joE, rE)
        self._input("wind_C", self.wind_sense.joC, rC)

    def taste(self, sugar):
        self._require("taste", self.taste_sense)
        self._input("taste", self.taste_sense.sweet, self.taste_sense.rates(sugar, self.B))

    def stimulate(self, selection, hz, ms):
        """Force a pulse, combined with sensory drive by maximum; expire on a LIF boundary."""
        if not math.isfinite(ms) or ms <= 0:
            raise ValueError("pulse duration must be positive and finite")
        idx = self.c.indices(selection)
        value = np.broadcast_to(np.asarray(hz, dtype=np.float32), (self.B, len(idx))).copy()
        if not np.isfinite(value).all() or np.any((value < 0) | (value * self.brain.p.dt > 1000)):
            raise ValueError("pulse rates must be finite and between 0 and 1000/dt Hz")
        duration = math.ceil(ms / self.brain.p.dt) * self.brain.p.dt
        self._pulses.append((idx.copy(), value, self.t + duration))
        self._refresh_poisson()

    def _refresh_poisson(self):
        self.brain.poisson_p.copy_(self._base_poisson)
        self._pulses = [p for p in self._pulses if p[2] > self.t + 1e-9]
        for idx, hz, _ in self._pulses:
            ti = self.brain._idx(idx)
            prob = torch.as_tensor(hz, device=self.device) * (self.brain.p.dt / 1000)
            self.brain.poisson_p[:, ti] = torch.maximum(self.brain.poisson_p[:, ti], prob)
        self.brain._poisson_on = any(self._inputs_on.values()) or any(np.any(p[1] > 0) for p in self._pulses)

    def step(self, ms=10.0) -> float:
        """Advance a frame; fractional LIF steps carry over. Return actual simulated ms."""
        if not math.isfinite(ms) or ms < 0:
            raise ValueError("ms must be finite and nonnegative")
        total = self._pending_ms + ms
        steps = int(math.floor((total + 1e-9) / self.brain.p.dt))
        elapsed = steps * self.brain.p.dt
        self._pending_ms = max(0.0, total - elapsed)
        if not steps:
            return 0.0
        # A pulse ending inside a frame needs the eager path's exact boundary split.
        can_capture = self.cuda_graphs and not any(p[2] < self.t + elapsed - 1e-9 for p in self._pulses)
        if can_capture:
            self._graph_frame(steps)
            if self._pulses:
                self._refresh_poisson()
            self._activity_ms += elapsed
            return elapsed
        self._vision_frame(elapsed)
        remaining = steps
        while remaining:
            count = remaining
            if self._pulses:
                until = min(p[2] for p in self._pulses) - self.t
                count = min(count, max(1, int(round(until / self.brain.p.dt))))
            self.brain.step(count)
            remaining -= count
            if self._pulses:
                self._refresh_poisson()
        self._activity_ms += elapsed
        return elapsed

    def _vision_frame(self, elapsed):
        if self.optic is not None and self._radiance is not None:
            self.brain.drive.copy_(self.optic.step_frame(self._radiance, self.brain.rate, elapsed))

    def _frame(self, steps):
        self._vision_frame(steps * self.brain.p.dt)
        self.brain.step(steps)

    def _graph_frame(self, steps):
        b, o = self.brain, self.optic
        optic_pending = o._pending_ms if o is not None else 0.0
        key = (steps, b.buf_pos, b._poisson_on, self._radiance is not None, round(optic_pending, 9))
        if key not in self._graphs:
            # Bound memory when a caller supplies arbitrarily many frame lengths.
            if len(self._graphs) >= 8:
                self._frame(steps)
                return
            names = self.BRAIN_TENSORS
            saved = {name: getattr(b, name).clone() for name in names}
            optical = {name: getattr(o, name).clone() for name in self.OPTIC_TENSORS} if o is not None else {}
            rng = b.gen.get_state()
            scalars = b.t, b.step_count, b.buf_pos
            diagnostics = o.diagnostics if o is not None else False

            def restore():
                for name, value in saved.items():
                    getattr(b, name).copy_(value)
                for name, value in optical.items():
                    getattr(o, name).copy_(value)
                b.t, b.step_count, b.buf_pos = scalars
                if o is not None:
                    o._pending_ms = optic_pending
                b.gen.set_state(rng)

            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            graph = torch.cuda.CUDAGraph()
            graph.register_generator_state(b.gen)
            try:
                if o is not None:
                    o.diagnostics = False
                with torch.cuda.stream(stream):
                    self._frame(steps)  # warm allocator and sparse kernels on a side stream
                torch.cuda.current_stream(self.device).wait_stream(stream)
                restore()
                with torch.cuda.graph(graph, stream=stream):
                    self._frame(steps)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                restore()
                self._graphs[key] = graph
            except Exception:
                torch.cuda.current_stream(self.device).wait_stream(stream)
                restore()
                raise
            finally:
                if o is not None:
                    o.diagnostics = diagnostics
        self._graphs[key].replay()
        b.t += steps * b.p.dt
        b.step_count += steps
        b.buf_pos = (b.buf_pos + steps) % b.n_delay
        if o is not None and self._radiance is not None:
            total = optic_pending + steps * b.p.dt
            o._pending_ms = max(0.0, total - int(math.floor((total + 1e-9) / o.p.dt_ms)) * o.p.dt_ms)
        if o is not None and o.diagnostics and self._radiance is not None:
            o.last["contrast"] = o.contrast.view(self.B, -1, 5).cpu().numpy()

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def step_budget(self, wall_ms, *, frame_ms=10.0) -> StepResult:
        """Best-effort wall budget, including GPU completion and enabled senses.

        Start with a single LIF step to measure cost, then use the largest measured or
        estimated chunk that fits (up to frame_ms). This changes optic feedback cadence.
        A cold step, graph capture, or GPU contention can exceed the budget; there is no
        hard deadline guarantee. Use AsyncFlyBrain when the host must never wait for GPU.
        """
        if not math.isfinite(wall_ms) or wall_ms < 0 or not math.isfinite(frame_ms) or frame_ms < self.brain.p.dt:
            raise ValueError("wall_ms must be nonnegative and frame_ms at least one LIF step")
        start = time.perf_counter()
        simulated = 0.0
        count = 0
        max_steps = int(frame_ms / self.brain.p.dt)
        key = (max_steps, self.cuda_graphs)
        if self._budget_estimate is None or self._budget_estimate[0] != key:
            self._budget_estimate = (key, {})
        costs = self._budget_estimate[1]
        candidates = sorted(set([1, max_steps] + [2**i for i in range(max_steps.bit_length()) if 2**i <= max_steps]))
        while wall_ms > 0:
            remaining = wall_ms - (time.perf_counter() - start) * 1000
            if remaining <= 0:
                break
            steps = 1
            if costs:
                # Use the smallest measured chunk as a conservative per-step estimate.
                smallest = min(costs)
                per_step = costs[smallest] / smallest
                fits = [n for n in candidates if costs.get(n, per_step*n) * 1.1 <= remaining]
                if not fits:
                    break
                steps = max(fits)
            before = time.perf_counter()
            graphs_before = len(self._graphs)
            advanced = self.step(steps * self.brain.p.dt)
            self._synchronize()
            cost = (time.perf_counter() - before) * 1000
            # Capture is a one-off initialization cost, not an estimate of steady replay.
            if len(self._graphs) == graphs_before:
                costs[steps] = cost if steps not in costs else .5 * costs[steps] + .5 * cost
            simulated += advanced
            count += round(advanced / self.brain.p.dt)
        return StepResult(simulated, (time.perf_counter() - start) * 1000, count)

    def motor(self) -> MotorRates:
        return read_motor(self.brain, self.groups, self.wings)

    def activity_mask(self, min_hz=1.0):
        """Threshold mean LIF Hz since reset, (N,) or (B,N). Retain graded vision cells.

        Optic/photoreceptor units have no spike frequency; conservatively keeping them
        prevents a downstream activity-based parer from accidentally deleting vision.
        """
        if not math.isfinite(min_hz) or min_hz < 0:
            raise ValueError("min_hz must be nonnegative and finite")
        seconds = torch.as_tensor(np.maximum(self._activity_ms / 1000, 1e-12), device=self.device, dtype=torch.float32)
        rates = self.brain.spike_counts / seconds[:, None]
        result = (rates >= min_hz).cpu().numpy()
        if self.optic is not None:
            result[:, self.optic.rate_idx] = True
            result[:, self.optic.pr_idx] = True
        return result[0] if self.B == 1 else result

    def reset(self, rows=None):
        self.brain.reset(rows)
        if self.optic is not None:
            self.optic.reset(rows)
        sel = slice(None) if rows is None else self.brain._idx(rows)
        self._base_poisson[sel] = 0
        if rows is None:
            self._pulses.clear(); self._inputs_on.clear()
            self._pending_ms = 0.0
            self._activity_ms[:] = 0.0
            self._graphs.clear()
            self._budget_estimate = None
            self._radiance = None
            self.brain.t = 0.0; self.brain.step_count = 0; self.brain.buf_pos = 0
        else:
            self._activity_ms[np.asarray(rows)] = 0.0
            for _, hz, _ in self._pulses:
                hz[np.asarray(rows)] = 0
        self._refresh_poisson()

    def state_dict(self):
        b, o = self.brain, self.optic
        return {"version": 1, "body_ids": self.c.neurons.bodyId.to_numpy().copy(),
                "lif_params": asdict(b.p), "optic_params": asdict(o.p) if o is not None else None,
                "brain": {k: getattr(b, k).detach().cpu().clone() for k in self.BRAIN_TENSORS},
                "brain_scalars": {k: getattr(b, k) for k in ("buf_pos", "t", "step_count", "_poisson_on")},
                "rng": b.gen.get_state().cpu().clone(),
                "optic": {k: getattr(o, k).detach().cpu().clone() for k in self.OPTIC_TENSORS} if o is not None else None,
                "optic_pending_ms": o._pending_ms if o is not None else 0.0,
                "radiance": self._radiance.cpu().clone() if self._radiance is not None else None,
                "base_poisson": self._base_poisson.cpu().clone(), "inputs_on": dict(self._inputs_on),
                "pulses": [(idx.copy(), hz.copy(), end) for idx, hz, end in self._pulses],
                "pending_ms": self._pending_ms, "activity_ms": self._activity_ms.copy()}

    def load_state_dict(self, state):
        if state["version"] != 1 or not np.array_equal(state["body_ids"], self.c.neurons.bodyId.to_numpy()):
            raise ValueError("state version/connectome does not match")
        if state["lif_params"] != asdict(self.brain.p) or state["optic_params"] != (asdict(self.optic.p) if self.optic is not None else None):
            raise ValueError("state simulation parameters do not match")
        for name, tensor in state["brain"].items():
            if tensor.shape != getattr(self.brain, name).shape:
                raise ValueError(f"state shape mismatch for {name}")
        self._graphs.clear()
        self._budget_estimate = None
        for name, tensor in state["brain"].items():
            getattr(self.brain, name).copy_(tensor.to(self.device))
        for name, value in state["brain_scalars"].items():
            setattr(self.brain, name, value)
        self.brain.gen.set_state(state["rng"].cpu())
        self.brain._rate_np_key = None
        if self.optic is not None:
            for name, tensor in state["optic"].items():
                getattr(self.optic, name).copy_(tensor.to(self.device))
            self.optic._pending_ms = state.get("optic_pending_ms", 0.0)
            self.optic.last["dr"] = self.optic.delta_rate
            if self.optic.diagnostics:
                self.optic.last["contrast"] = self.optic.contrast.view(self.B, -1, 5).cpu().numpy()
        self._radiance = state["radiance"].to(self.device).clone() if state["radiance"] is not None else None
        self._base_poisson.copy_(state["base_poisson"].to(self.device))
        self._inputs_on = dict(state["inputs_on"])
        self._pulses = [(idx.copy(), hz.copy(), end) for idx, hz, end in state["pulses"]]
        self._pending_ms, self._activity_ms = state["pending_ms"], state["activity_ms"].copy()
