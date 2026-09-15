"""A connectome controller with physical inputs and named neural outputs."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time
import copy

import numpy as np
import torch

from . import brain, connectome, regions, retina, senses
from . import optic as optic_module
from .motor import MotorRates, motor_groups, wing_groups, read_motor
from .nt_readout import NTSource, NTSnapshot


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
    BRAIN_TENSORS = ("v", "g", "g_slow", "g_slow_cls", "refrac", "drive", "poisson_p", "rate", "spikes", "adapt", "res", "spike_buf", "spike_counts")
    OPTIC_TENSORS = ("v", "adapt", "I_lp", "I_mean", "_fresh", "contrast", "delta_rate", "g_slow", "g_slow_cls", "_slow_in_s")

    def __init__(self, c=None, *, modules=None, batch=1, device=None, seed=0,
                 lif_params=None, optic_params=None, eye_geometry=None, cuda_graphs=False, cuda_kernels=None,
                 cuda_sparse="torch", cuda_compact=True, nt_source: NTSource | None = None,
                 dataset=None, optic="auto", preset="raw", instruments=None):
        """`preset` (docs/PRESETS_SPEC.md): 'raw' -- the default -- is the connectome, the LIF, the receptor table, the
        senses and the motor readout exactly as shipped, byte-identical to a call without the keyword
        (tests/test_bit_identity.py); 'instrumented' is 'raw' plus the named `instruments` (flyverse.instruments
        objects: a labelled stand-in, a held edge, a relabel), each installed through the surface that already exists
        for its kind and each recorded by describe() in provenance()['instruments'] beside 'preset'. Nothing about the
        default changes: preset='raw' with instruments raises, and the constructor never builds an instrument on its own."""
        if optic not in (None, "auto"):
            raise ValueError("optic must be 'auto' or None")
        from .instruments import PRESETS, _check_instrument
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS}, got {preset!r}")
        instruments = [instruments] if isinstance(instruments, str) else list(instruments or [])
        if preset == "raw" and instruments:
            raise ValueError("preset 'raw' attaches no instrument (docs/PRESETS_SPEC.md 1); use preset='instrumented'")
        self.preset = preset
        self.instruments = {}
        if c is not None and dataset is not None and c.dataset != dataset:
            raise ValueError("dataset disagrees with the supplied connectome")
        self.c = regions.subset(c if c is not None else connectome.load(dataset=dataset, verbose=False), modules)
        if any(isinstance(inst, str) for inst in instruments):
            from .instruments import make_instrument
            instruments = [make_instrument(self.c, inst) if isinstance(inst, str) else inst for inst in instruments]
        names = [_check_instrument(inst) for inst in instruments]
        if len(set(names)) != len(names):
            raise ValueError(f"instrument names must be unique: {names}")
        if self.c.n == 0:
            raise ValueError("FlyBrain needs at least one neuron")
        lp = lif_params or brain.LIFParams()
        if lp.surrogate_grad and cuda_graphs:
            raise ValueError("surrogate_grad cannot use CUDA graphs")
        # The optional receptor model (LIFParams.receptor_model): one per-edge lookup shared by the LIF and the optic lobe;
        # the slow term's resolved spec (None unless 'full' with a non-zero class scale) is shared the same way.
        self.slow = brain._slow_spec(lp)
        self.receptor = brain._receptor(self.c, lp, with_counts=self.slow is not None)
        self.brain = brain.Brain(self.c, lp, device=device, batch=batch, seed=seed, cuda_kernels=cuda_kernels,
                                 cuda_sparse=cuda_sparse, cuda_compact=cuda_compact, receptor=self.receptor)
        self.B, self.device = self.brain.B, self.brain.device
        if cuda_graphs and (self.device.type != "cuda" or (self.brain.event_driven and not self.brain.cuda)):
            raise ValueError("CUDA graphs require CUDA and either sparse matmul or native CUDA events")
        self.cuda_graphs = cuda_graphs
        self._graphs = {}
        self.brain.record_activity = True
        self.retina = self.optic = None
        if optic is not None and self.c.has_optic_columns and len(self.c.select(type=connectome.PHOTORECEPTOR_TYPES)) and len(self.c.select(superclass="ol_intrinsic")):
            self.retina = retina.build_retina(self.c, eye_geometry)
            if self.retina.n_columns:
                self.optic = optic_module.OpticLobe(self.c, self.retina, optic_params, device=self.device, batch=self.B,
                                           cuda_kernels=cuda_kernels, cuda_sparse=cuda_sparse,
                                           receptor=self.receptor, receptor_gain=brain._receptor_gain(lp), slow=self.slow,
                                           surrogate_grad=lp.surrogate_grad)
                self.optic.relax()
                self.brain.freeze(self.optic.rate_idx)
                if self.brain.p.prune_frozen:
                    self.brain.prune(self.optic.rate_idx)
        if self.brain.p.dt_by_module:
            lab = regions.labels(self.c); k = np.ones(self.c.n, np.int64); dt = self.brain.p.dt
            for mod, dt_mod in self.brain.p.dt_by_module.items():
                if mod not in set(regions.MODULES) | set(lab):
                    raise ValueError(f"unknown module {mod!r} in dt_by_module")
                m = int(round(dt_mod / dt))
                if abs(m * dt - dt_mod) > 1e-9 or m < 1:
                    raise ValueError(f"dt_by_module[{mod!r}] = {dt_mod} is not a positive multiple of dt = {dt}")
                k[lab == mod] = m
            self.brain.set_clocks(k)
        self.olfaction = senses.Smell(self.c) if len(self.c.select(**{"class": "olfactory"})) else None
        self.wind_sense = senses.Wind(self.c) if len(self.c.select(type="~^JO-[CE]")) else None
        taste = senses.Taste(self.c)
        self.taste_sense = taste if len(taste.sweet) else None
        self.groups, self.wings = motor_groups(self.c, allow_missing_vnc=True), wing_groups(self.c, allow_missing_vnc=True)
        self._radiance = None
        self._pending_ms = 0.0
        self._base_poisson = torch.zeros_like(self.brain.poisson_p)
        self._inputs_on = {}
        self._input_indices = {}
        self._pulses = []
        self._activity_ms = np.zeros(self.B)
        self._budget_estimate = None
        self.nt_source = nt_source
        self._extensions = None
        # Whether an optic frame has already overwritten brain.drive: it decides how ExtensionRuntime splits the
        # drive into the caller's held part and the per-frame sensory part. Nothing else is tracked per set_drive --
        # the shipped path must not clone on every call nor grow a record trimmed only by a vision frame (B1 of
        # docs/audits/extensibility_review.md).
        self._vision_has_advanced = False
        # preset 'instrumented': install each instrument through the surface its kind already has (a stand-in grows
        # the proprioception sense's channel; a hold / relabel record verifies the gain list / the cache carries it)
        for inst in instruments:
            install = getattr(inst, "install", None)
            if callable(install):
                install(self)
            self.instruments[inst.name] = inst

    def instrument_records(self):
        """[describe() of each instrument], JSON-ready -- provenance()['instruments']; [] under preset 'raw'."""
        self._register_sense_instrument()
        from .instruments import records
        return records(self.instruments.values())

    def _register_sense_instrument(self):
        """A named sense token must carry the same provenance and preset guard as a constructor instrument."""
        inst = getattr(getattr(self, "proprioception_sense", None), "turn_afferent", None)
        if inst is None:
            return
        if self.preset != "instrumented":
            raise ValueError("turn_afferent requires preset='instrumented'")
        old = self.instruments.get(inst.name)
        if old is not None and old is not inst:
            raise ValueError("turn_afferent differs from the recorded instrument; reinstall the named instrument")
        if old is None:
            from .instruments import _check_instrument
            _check_instrument(inst)
            self.instruments[inst.name] = inst

    def _extension_runtime(self):
        if self._extensions is None:
            from .modules import ExtensionRuntime
            self._extensions = ExtensionRuntime(self)
            self._graphs.clear()
        return self._extensions

    @property
    def hooks(self):
        return self._extensions.hook_records() if self._extensions else []

    @property
    def attached_modules(self):
        """Named live modules, distinct from the anatomical ``modules=`` subset selector."""
        return dict(self._extensions.modules) if self._extensions else {}

    def module_records(self):
        return self._extensions.records() if self._extensions else []

    def module_inputs(self):
        """Held extension input classes for recording, in mV or Hz (read-only copies)."""
        if self._extensions is None:
            return {}
        return {name: value.clone() for name, value in self._extensions.applied_inputs.items()}

    def add_hook(self, fn, *, when="pre", name):
        if not callable(fn) or when not in ("pre", "post") or not isinstance(name, str) or not name.strip():
            raise ValueError("hook requires a callable, a nonempty name and when='pre' or 'post'")
        runtime = self._extension_runtime()
        if name in runtime.hooks:
            raise ValueError(f"duplicate hook {name!r}")
        runtime.hooks[name] = (fn, when)
        self._graphs.clear()        # a hook runs eagerly around the frame; drop any captured CUDA graph, as attach does

    def remove_hook(self, name):
        if self._extensions is None:
            raise KeyError(name)
        self._extensions.remove(name, hook=True)
        self._release_extensions()

    def attach(self, module):
        validate = getattr(module, "validate_parent", None)
        if callable(validate):
            validate(self)
        is_instrument = getattr(module, "required_preset", None) is not None
        if is_instrument:
            from .instruments import _check_instrument
            _check_instrument(module)
            if self.preset != module.required_preset or module.name in self.instruments:
                raise ValueError("module instrument requires its named preset and a unique instrument name")
        try:
            self._extension_runtime().attach(module)
        except Exception:
            self._release_extensions()
            raise
        self._graphs.clear()
        if is_instrument:
            self.instruments[module.name] = module
        return module

    def detach(self, name):
        if self._extensions is None:
            raise KeyError(name)
        module = self._extensions.modules[name]
        self._extensions.remove(name)
        if self.instruments.get(name) is module:
            del self.instruments[name]
        self._release_extensions()
        return module

    def _release_extensions(self):
        runtime = self._extensions
        self._graphs.clear()
        if runtime and not runtime.hooks and not runtime.modules:
            self.brain._input_target = None
            self.brain.drive.copy_(runtime.base_drive + runtime.sensory_drive)
            self.brain.poisson_p.copy_(torch.maximum(self._base_poisson, runtime.external_poisson))
            self.brain._poisson_on = bool((self.brain.poisson_p > 0).any())
            self._extensions = None
            if self._pulses:
                self._refresh_poisson()

    @property
    def t(self):
        return self.brain.t

    @property
    def available_senses(self):
        # The explicit compass instrument can receive angular motion without enabling other afferents.
        # Neither a proprioceptive sense nor an instrument is built by default.
        motion = any(callable(getattr(inst, "observe_turn", None)) for inst in self.instruments.values())
        return tuple(name for name, value in (("vision", self.optic), ("smell", self.olfaction),
                     ("wind", self.wind_sense), ("taste", self.taste_sense),
                     ("proprioception", getattr(self, "proprioception_sense", None) or (True if motion else None))) if value is not None)

    def neurotransmitters(self, batch_index=0) -> NTSnapshot | None:
        """Optional live NT levels, sampled on demand through a read-only adapter.

        The NT module's owner advances and checkpoints it. This hook does not infer
        levels from transmitter labels, register dynamics, or run during ``step``.
        Sources may be attached/detached through ``nt_source`` at runtime.
        """
        if not isinstance(batch_index, (int, np.integer)) or not 0 <= batch_index < self.B:
            raise IndexError(batch_index)
        if self.nt_source is None:
            return None
        readout = getattr(self.nt_source,"readout",None)
        if not callable(readout):
            raise TypeError("NTSource must implement readout(batch_index=...)")
        snapshot = readout(batch_index=batch_index)
        if snapshot is not None and not isinstance(snapshot, NTSnapshot):
            raise TypeError("NTSource.readout must return NTSnapshot or None")
        return snapshot

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
        if self._radiance is None or self.brain.p.surrogate_grad:
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
        if self._pulses or self._extensions is not None:
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

    def proprioception(self, leg_L, leg_R, haltere, airborne, yaw_rate=0.0):
        """Opt-in (senses.Proprioception attached as `proprioception_sense`): the leg MN rates per side, the haltere
        MN rate and the airborne flag of the body -> afferent Hz per channel, injected like wind. `yaw_rate` is read
        by the labelled Coriolis term, signed afferent, or an explicit angular-motion instrument."""
        sense = getattr(self, "proprioception_sense", None)
        receivers = [inst for inst in self.instruments.values() if callable(getattr(inst, "observe_turn", None))]
        self._require("proprioception", sense if sense is not None else (receivers or None))
        for inst in receivers:
            inst.observe_turn(yaw_rate)
        self._register_sense_instrument()
        if sense is not None:
            for name, idx, hz in sense.rates(leg_L, leg_R, haltere, airborne, yaw_rate, self.B):
                self._input(f"proprioception_{name}", idx, hz)

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
        if self._extensions is not None:
            return self._extensions.refresh_poisson()
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
        if self._extensions is not None:
            return self._step_extensions(ms)
        return self._step_plain(ms)

    def _module_frames_needed(self):
        """True when an attached module can change what the LIF integrates, and only then is step(ms) split into
        <= 10 ms module frames (B3). A module that writes a neural channel, or a vision encoder that supplies the
        photoreceptor intensity of the optic frame, has to be sampled per frame -- its output is the input of the
        next frame. A reads-only module (a ReadoutModule, any kind='analysis' observer, a motor boundary decoder
        evaluated by the body) writes nothing into the brain, so it must be numerically neutral: it is stepped once
        per step(ms) call and step(50) stays bit-identical to the same call with nothing attached
        (tests/test_modules.py::test_readonly_module_is_numerically_neutral)."""
        runtime = self._extensions
        if runtime is None:
            return False
        return any(m.writes or getattr(m, "boundary", None) == "vision" for m in runtime.modules.values())

    def _step_extensions(self, ms):
        runtime = self._extensions
        runtime.run_hooks("pre")
        total = self._pending_ms + ms
        steps = int(math.floor((total + 1e-9) / self.brain.p.dt))
        elapsed = steps * self.brain.p.dt
        self._pending_ms = max(0., total - elapsed)
        frame_steps = max(1, int(10. / self.brain.p.dt)) if self._module_frames_needed() else max(steps, 1)
        remaining = steps
        while remaining:
            count = min(remaining, frame_steps)
            dt_ms = count * self.brain.p.dt
            if self.cuda_graphs and runtime.can_capture_frame() and not self._pulses:
                self._graph_frame(count, modules=True)
                self._activity_ms += dt_ms
                remaining -= count
                continue
            runtime.run_modules(dt_ms)
            before = self.brain.spike_counts.clone()
            # _step_plain handles pulses and capture; preserve the caller's fractional remainder.
            pending = self._pending_ms
            self._pending_ms = 0.
            self._step_plain(dt_ms)
            self._pending_ms = pending
            runtime.previous_spikes = self.brain.spike_counts - before
            remaining -= count
        runtime.run_hooks("post")
        return elapsed

    def _step_plain(self, ms):
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
        if self.brain.p.surrogate_grad:
            if self.optic is not None and self._radiance is not None:
                intensity = self._extensions.vision_intensity if self._extensions else None
                self.brain.drive = self.optic.step_frame(self._radiance, self.brain.rate, elapsed, intensity=intensity)
                self._vision_has_advanced = True
                if self._extensions:
                    self._extensions.sensory_drive = self.brain.drive
                    self.brain.drive = self.brain.drive + self._extensions.drive
            elif self._extensions:
                self.brain.drive = self._extensions.drive
            return
        if self.optic is not None and self._radiance is not None:
            intensity = self._extensions.vision_intensity if self._extensions else None
            kwargs = {} if intensity is None else {"intensity": intensity}
            self.brain.drive.copy_(self.optic.step_frame(self._radiance, self.brain.rate, elapsed, **kwargs))
            self._vision_has_advanced = True
            if self._extensions is not None:
                self._extensions.sensory_drive.copy_(self.brain.drive)
                self.brain.drive.add_(self._extensions.drive)
        elif self._extensions is not None:
            self.brain.drive.copy_(self._extensions.drive)

    def _frame(self, steps, modules=False):
        if modules:
            self._extensions.run_modules(steps * self.brain.p.dt)
            before = self.brain.spike_counts.clone()
        self._vision_frame(steps * self.brain.p.dt)
        self.brain.step(steps)
        if modules:
            self._extensions.previous_spikes = self.brain.spike_counts - before

    def _graph_frame(self, steps, modules=False):
        b, o = self.brain, self.optic
        if modules:
            b._poisson_on = True  # can_capture_frame proves the next full output keeps RNG active
        if getattr(self, "_graph_weights_version", None) != b._weights_version:
            self._graphs.clear()
            self._graph_weights_version = b._weights_version
        optic_pending = o._pending_ms if o is not None else 0.0
        key = (steps, b.buf_pos, b.step_count % b.K, b._poisson_on, b.record_activity, b.cuda_compact,
               self._radiance is not None, round(optic_pending, 9), modules)
        if steps % b.K:
            self._frame(steps, modules)  # a frame that is not a whole number of clock periods is not capturable
            return
        if key not in self._graphs:
            # Bound memory when a caller supplies arbitrarily many frame lengths.
            if len(self._graphs) >= 8:
                self._frame(steps, modules)
                return
            names = self.BRAIN_TENSORS
            saved = {name: getattr(b, name).clone() for name in names}
            acc_saved = {kk: t.clone() for kk, t in b._acc.items()}
            optical = {name: getattr(o, name).clone() for name in self.OPTIC_TENSORS} if o is not None else {}
            rng = b.gen.get_state()
            scalars = b.t, b.step_count, b.buf_pos
            diagnostics = o.diagnostics if o is not None else False
            module_states = {name: m.state_dict() for name, m in self._extensions.modules.items()} if modules else {}
            extension_states = ({name: getattr(self._extensions, name).clone()
                                 for name in ('previous_spikes', 'drive', 'sensory_drive')} if modules else {})

            def restore():
                for name, value in saved.items():
                    getattr(b, name).copy_(value)
                for kk, value in acc_saved.items():
                    b._acc[kk].copy_(value)
                for name, value in optical.items():
                    getattr(o, name).copy_(value)
                b.t, b.step_count, b.buf_pos = scalars
                if o is not None:
                    o._pending_ms = optic_pending
                b.gen.set_state(rng)
                if modules:
                    for name, state in module_states.items():
                        self._extensions.modules[name].load_state_dict(state)
                    for name, value in extension_states.items():
                        getattr(self._extensions, name).copy_(value)

            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            graph = torch.cuda.CUDAGraph()
            graph.register_generator_state(b.gen)
            try:
                if o is not None:
                    o.diagnostics = False
                with torch.cuda.stream(stream):
                    self._frame(steps, modules)  # warm allocator and sparse kernels on a side stream
                torch.cuda.current_stream(self.device).wait_stream(stream)
                restore()
                # The input ownership clone reads the warmup field, which lives outside the graph's
                # private pool. Keep that storage alive for every replay, even after eager frames
                # replace the runtime dictionary. Graphs do not retain arbitrary input tensors.
                captured_inputs = dict(self._extensions.poisson) if modules else None
                with torch.cuda.graph(graph, stream=stream):
                    self._frame(steps, modules)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                restore()
                if modules:
                    # Each graph owns its output buffers. An eager pulse frame or a different frame
                    # length may replace the runtime dictionaries; rebind the right buffers on replay.
                    bound = {name: (dict(getattr(self._extensions, name)) if name in ('poisson', 'applied_inputs')
                                    else getattr(self._extensions, name))
                             for name in ('poisson', 'applied_inputs', 'previous_spikes')}
                    self._graphs[key] = (graph, bound, captured_inputs)
                else:
                    self._graphs[key] = graph
            except Exception:
                torch.cuda.current_stream(self.device).wait_stream(stream)
                restore()
                raise
            finally:
                if o is not None:
                    o.diagnostics = diagnostics
        cached = self._graphs[key]
        if modules:
            graph, bound, _ = cached
            for name, value in bound.items():
                setattr(self._extensions, name, dict(value) if isinstance(value, dict) else value)
            graph.replay()
        else:
            cached.replay()
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
        self.c.require("vnc")
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
        if self._extensions is not None:
            if rows is not None:
                for name, m in self._extensions.modules.items():
                    if not callable(getattr(m, "reset_rows", None)):
                        raise ValueError(f"module {name!r} must implement reset_rows for partial batch resets")
            self._extensions.reset(rows)
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
            self._vision_has_advanced = False
            self.brain.t = 0.0; self.brain.step_count = 0; self.brain.buf_pos = 0
        else:
            self._activity_ms[np.asarray(rows)] = 0.0
            for _, hz, _ in self._pulses:
                hz[np.asarray(rows)] = 0
        self._refresh_poisson()

    def detach_state(self):
        """Truncate the optional autograd history while retaining neural/module state."""
        self.brain.detach_state()
        if self.optic is not None:
            self.optic.detach_state()
        if self._radiance is not None:
            self._radiance = self._radiance.detach().clone()
        if self._extensions is not None:
            from .modules import snapshot
            for name in ("base_drive", "drive", "drives", "poisson", "previous_spikes", "vision_intensity", "sensory_drive", "applied_inputs"):
                setattr(self._extensions, name, snapshot(getattr(self._extensions, name), self.device))

    def state_dict(self):
        b, o = self.brain, self.optic
        return {"version": 1, "body_ids": self.c.neurons.bodyId.to_numpy().copy(),
                "preset": self.preset, "instruments": self.instrument_records(),
                "lif_params": asdict(b.p), "optic_params": asdict(o.p) if o is not None else None,
                "brain": {k: getattr(b, k).detach().cpu().clone() for k in self.BRAIN_TENSORS},
                "clock_multipliers": b._kvec.copy() if b._kvec is not None else None,
                "clock_accumulators": {k:v.detach().cpu().clone() for k,v in b._acc.items()},
                "brain_scalars": {k: getattr(b, k) for k in ("buf_pos", "t", "step_count", "_poisson_on")},
                "rng": b.gen.get_state().cpu().clone(),
                "optic": {k: getattr(o, k).detach().cpu().clone() for k in self.OPTIC_TENSORS} if o is not None else None,
                "optic_pending_ms": o._pending_ms if o is not None else 0.0,
                "radiance": self._radiance.detach().cpu().clone() if self._radiance is not None else None,
                "base_poisson": self._base_poisson.cpu().clone(), "inputs_on": dict(self._inputs_on),
                "pulses": [(idx.copy(), hz.copy(), end) for idx, hz, end in self._pulses],
                "pending_ms": self._pending_ms, "activity_ms": self._activity_ms.copy(),
                "hooks": self.hooks, "modules": self.module_records(),
                "vision_has_advanced": self._vision_has_advanced,
                "connectome_extension": copy.deepcopy(self.c._extension),
                "extensions": self._extensions.state_dict() if self._extensions else None}

    def load_state_dict(self, state):
        if state.get("preset", "raw") != self.preset or state.get("instruments", []) != self.instrument_records():
            raise ValueError("state preset/instruments do not match")
        if state["version"] != 1 or not np.array_equal(state["body_ids"], self.c.neurons.bodyId.to_numpy()):
            raise ValueError("state version/connectome does not match")
        if state.get("connectome_extension") != self.c._extension:
            raise ValueError("state synthetic graph extension does not match")
        lif_state = {"surrogate_grad": False, **state["lif_params"]}
        if lif_state != asdict(self.brain.p) or state["optic_params"] != (asdict(self.optic.p) if self.optic is not None else None):
            raise ValueError("state simulation parameters do not match")
        extension_state = state.get("extensions")
        if (extension_state is None) != (self._extensions is None):
            raise ValueError("checkpoint extensions differ; attach matching code before loading")
        if self._extensions is not None:
            self._extensions.validate_state(extension_state)
        for name, tensor in state["brain"].items():
            if tensor.shape != getattr(self.brain, name).shape:
                raise ValueError(f"state shape mismatch for {name}")
        if not np.array_equal(state.get("clock_multipliers"), self.brain._kvec):
            raise ValueError("state integration clocks do not match (older clocked checkpoints lack this state)")
        accumulators = state.get("clock_accumulators", {})
        if accumulators.keys() != self.brain._acc.keys() or any(
                v.shape != self.brain._acc[k].shape for k,v in accumulators.items()):
            raise ValueError("state clock accumulators do not match")
        self._graphs.clear()
        self._budget_estimate = None
        if self.brain.p.surrogate_grad:
            self.detach_state()
        for name, tensor in state["brain"].items():
            getattr(self.brain, name).copy_(tensor.to(self.device))
        for k,v in accumulators.items():
            self.brain._acc[k].copy_(v.to(self.device))
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
        self._vision_has_advanced = state.get("vision_has_advanced", self.optic is not None and self._radiance is not None)
        if self._extensions is not None:
            self._extensions.load_state_dict(extension_state)
