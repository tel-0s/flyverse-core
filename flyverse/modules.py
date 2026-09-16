"""Opt-in, named extensions at the neural boundary. All live values are (B, cells).

Selection uses ``interp.common.resolve``. Neural outputs are held for a frame:
drive is additive; Poisson forcing combines with senses and pulses by maximum.
No module receives the parent's weights or mutable views of its state.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Protocol

import numpy as np
import torch

Selection = Any
KINDS = {"stop-gap", "mechanism", "sensor", "decoder", "analysis"}
QUANTITIES = {"rate_hz", "drive_mv", "spike_count", "optic_rate"}
CHANNELS = {"poisson_hz", "drive_mv"}
KEEP_IDS_MAX = 10_000      # the interp convention (decompose / trace / ledger / atlas): past this, the count only


def id_list(ids):
    """The bodyIds of a selection for provenance and checkpoints, or [] past KEEP_IDS_MAX cells.

    A module may legitimately read a whole optic lobe (~70k cells); its describe() lands in every Result JSON and
    every checkpoint, so the identities are dropped past the same cap the interp Results use. The count is kept
    beside the list (`n_reads` / `n_writes` in ExtensionRuntime.records()), so nothing becomes ambiguous.
    """
    ids = np.asarray(ids)
    return ids.tolist() if len(ids) <= KEEP_IDS_MAX else []


def identifier(fn):
    return f"{fn.__module__}:{getattr(fn, '__qualname__', type(fn).__qualname__)}"


def snapshot(value, device="cpu"):
    """Copy checkpoint data, never Python callables or a live autograd graph."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {k: snapshot(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(v, device) for v in value)
    return copy.deepcopy(value)


def checkpoint_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            h.update(str((t.dtype, tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


class Module(Protocol):
    name: str
    reads: dict[str, Selection]
    writes: dict[str, Selection]
    quantity_in: str = "rate_hz"
    channel_out: str = "poisson_hz"

    def reset(self, B: int, device) -> None: ...
    def step(self, dt_ms: float, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict) -> None: ...
    def describe(self) -> dict: ...


class FunctionModule:
    """A stateless ``fn(dt_ms, inputs) -> outputs``. Stateful callables implement Module."""
    def __init__(self, reads, writes, fn, *, name=None, quantity_in="rate_hz",
                 channel_out="poisson_hz", kind="stop-gap", parameters=None):
        self.name = name or getattr(fn, "__name__", type(fn).__name__)
        self.reads, self.writes, self.fn = dict(reads), dict(writes), fn
        self.quantity_in, self.channel_out, self.kind = quantity_in, channel_out, kind
        self.parameters = dict(parameters or {})

    def reset(self, B, device):
        self.B, self.device = B, torch.device(device)

    def reset_rows(self, rows):
        pass

    def step(self, dt_ms, inputs):
        return self.fn(dt_ms, inputs)

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        if d:
            raise ValueError("FunctionModule is stateless")

    def describe(self):
        return {"name": self.name, "class": identifier(type(self)), "callable": identifier(self.fn),
                "parameters": self.parameters, "trainable": False, "checkpoint_hash": None, "kind": self.kind}


class TorchModule(FunctionModule):
    """Concatenate inputs in reads order; split a torch network's output in writes order.

    ``forward`` also supports boundary encoders/decoders outside the neural loop.
    Such modules have no neural writes and specify ``output_sizes`` instead.
    """
    def __init__(self, reads, writes, net, channel_out="poisson_hz", *, name="torch",
                 quantity_in="rate_hz", kind="stop-gap", output_sizes=None, parameters=None, boundary=None):
        super().__init__(reads, writes, net, name=name, quantity_in=quantity_in,
                         channel_out=channel_out, kind=kind, parameters=parameters)
        self.net = net
        self.output_sizes = dict(output_sizes or {})
        self.outputs = {}
        if boundary not in (None, "vision", "motor") or (boundary is not None and (reads or writes)):
            raise ValueError("boundary modules use boundary='vision'/'motor' and empty neural reads/writes")
        self.boundary = boundary

    def reset(self, B, device):
        super().reset(B, device)
        self.net.to(self.device)
        self.outputs = {}

    def reset_rows(self, rows):
        for k, value in self.outputs.items():
            value = value.clone(); value[rows] = 0
            self.outputs[k] = value

    def forward(self, inputs, *, remember=True):
        x = inputs if isinstance(inputs, torch.Tensor) else (
            torch.cat([inputs[k] for k in self.reads], dim=1) if self.reads else torch.empty(self.B, 0, device=self.device))
        y = self.net(x)
        sizes = self.output_sizes
        if y.shape != (x.shape[0], sum(sizes.values())):
            raise ValueError(f"network output must have shape {(x.shape[0], sum(sizes.values()))}")
        outputs = dict(zip(sizes, y.split(list(sizes.values()), dim=1)))
        if remember:
            self.outputs = outputs
        return outputs

    def step(self, dt_ms, inputs):
        outputs = self.forward(inputs)
        return outputs if self.writes else {}

    def state_dict(self):
        return {"net": snapshot(self.net.state_dict()), "outputs": snapshot(self.outputs), "training": self.net.training}

    def load_state_dict(self, d):
        self.net.load_state_dict(d["net"])
        self.net.train(d.get("training", self.net.training))
        self.outputs = snapshot(d.get("outputs", {}), self.device)

    def describe(self):
        return {"name": self.name, "class": identifier(type(self)), "network": repr(self.net),
                "parameters": self.parameters, "kind": self.kind,
                "boundary": self.boundary, "output_sizes": self.output_sizes,
                "training": self.net.training,
                "trainable": any(p.requires_grad for p in self.net.parameters()),
                "checkpoint_hash": checkpoint_hash(self.net.state_dict())}


class MotorDecoder(TorchModule):
    """Learned MotorRates -> normalized forward/yaw; also a body program.

    Training reads ``features(motor)`` and calls ``forward`` without stepping the
    brain. ``apply`` replaces speed/yaw while preserving other body commands.
    """
    def __init__(self, net, features=("fwd_dn", "back_dn", "turn_L", "turn_R"), *,
                 name="motor_decoder", rate_scale=50., max_speed=.02, max_yaw=np.deg2rad(200)):
        super().__init__({}, {}, net, name=name, kind="decoder", boundary="motor", output_sizes={"command": 2},
                         parameters={"features": list(features), "rate_scale": rate_scale,
                                     "max_speed": max_speed, "max_yaw": max_yaw})
        self.feature_names = tuple(features)
        self.rate_scale, self.max_speed, self.max_yaw = rate_scale, max_speed, max_yaw

    def features(self, motor, *, batch=None):
        B = self.B if batch is None else batch
        values = [torch.as_tensor(getattr(motor, k), device=self.device, dtype=torch.float32).reshape(-1).expand(B)
                  for k in self.feature_names]
        return torch.stack(values, dim=1) / self.rate_scale

    def action(self, motor):
        return self.forward(self.features(motor))["command"].clamp(-1, 1)

    def apply(self, motor, cmd, fly, metabolism, dt_s):
        values = self.forward(self.features(motor, batch=1), remember=False)["command"].clamp(-1, 1).detach().cpu().numpy()
        return {**cmd, "speed": float(values[0, 0]) * self.max_speed, "yaw": float(values[0, 1]) * self.max_yaw}

    def state(self):
        return self.state_dict()

    def load_state(self, state):
        self.load_state_dict(state)


class SNNModule(FunctionModule):
    """Independent synthetic Brain. Labels map parent reads/writes to input/output cells.

    ``input_cells`` and ``output_cells`` use the same selection grammar on ``c``.
    Inputs feed the child through ``input_channel``; child rates feed the parent.
    """
    def __init__(self, reads, writes, c, input_cells, output_cells, *, name="snn", params=None,
                 seed=0, quantity_in="rate_hz", channel_out="poisson_hz", input_channel="poisson_hz",
                 kind="mechanism"):
        super().__init__(reads, writes, self.step, name=name, quantity_in=quantity_in, channel_out=channel_out, kind=kind)
        if "dataset" not in c.neurons or not c.neurons.dataset.eq("synthetic").all():
            raise ValueError("SNNModule requires a synthetic connectome")
        if input_channel not in CHANNELS:
            raise ValueError("unknown SNN input channel")
        from .interp.common import resolve
        self.c, self.params, self.seed, self.input_channel = c, params, seed, input_channel
        self.input_cells = {k: resolve(c, v) for k, v in input_cells.items()}
        self.output_cells = {k: resolve(c, v) for k, v in output_cells.items()}
        if self.input_cells.keys() != self.reads.keys() or self.output_cells.keys() != self.writes.keys():
            raise ValueError("SNN input/output labels must match reads/writes")

    def reset(self, B, device):
        from .brain import Brain
        super().reset(B, device)
        self.brain = Brain(self.c, self.params, batch=B, device=device, seed=self.seed)
        self._pending_ms = 0.0

    def reset_rows(self, rows):
        self.brain.reset(rows)

    def step(self, dt_ms, inputs):
        for k, value in inputs.items():
            if value.shape[1] != len(self.input_cells[k]):
                raise ValueError(f"SNN input {k!r} has the wrong number of cells")
            setter = self.brain.set_drive if self.input_channel == "drive_mv" else self.brain.set_poisson
            setter(self.input_cells[k], value if self.input_channel == "drive_mv" else value.clamp_min(0))
        total = self._pending_ms + dt_ms
        steps = int((total + 1e-9) // self.brain.p.dt)
        self._pending_ms = max(0., total - steps * self.brain.p.dt)
        self.brain.step(steps)
        return {k: self.brain.rate[:, self.brain._idx(idx)] for k, idx in self.output_cells.items()}

    def state_dict(self):
        from .fly import FlyBrain
        b = self.brain
        return {"tensors": snapshot({k: getattr(b, k) for k in FlyBrain.BRAIN_TENSORS}),
                "scalars": {k: getattr(b, k) for k in ("t", "step_count", "buf_pos", "_poisson_on")},
                "rng": b.gen.get_state().cpu().clone(), "pending_ms": self._pending_ms}

    def load_state_dict(self, d):
        for k, v in d["tensors"].items():
            setattr(self.brain, k, snapshot(v, self.device))
        for k, v in d["scalars"].items():
            setattr(self.brain, k, v)
        self.brain.gen.set_state(d["rng"].cpu())
        self.brain._rate_np_key = None
        self._pending_ms = d["pending_ms"]

    def describe(self):
        from dataclasses import asdict
        from .brain import LIFParams
        from .interp.common import connectome_fingerprint
        return {"name": self.name, "class": identifier(type(self)), "kind": self.kind, "trainable": False,
                "parameters": asdict(self.params or LIFParams()), "seed": self.seed,
                "input_cells": {k: id_list(self.c.neurons.bodyId.to_numpy()[idx]) for k, idx in self.input_cells.items()},
                "output_cells": {k: id_list(self.c.neurons.bodyId.to_numpy()[idx]) for k, idx in self.output_cells.items()},
                "n_input_cells": {k: int(len(idx)) for k, idx in self.input_cells.items()},
                "n_output_cells": {k: int(len(idx)) for k, idx in self.output_cells.items()},
                "input_channel": self.input_channel, "graph": connectome_fingerprint(self.c), "checkpoint_hash": None}


class ReadoutModule(FunctionModule):
    """Computed per-cell channels, sampled on demand through the NTSource interface.

    ``fn(dt_ms, inputs)`` returns {channel name: (B, n_union_of_reads)}. Channels
    explicitly declare units/ranges; readouts are listed separately from NT chemistry.
    """
    def __init__(self, reads, fn, *, channels=None, name="readout", quantity_in="rate_hz", kind="analysis"):
        super().__init__(reads, {}, fn, name=name, quantity_in=quantity_in, kind=kind)
        self._declared_channels = tuple(channels) if channels is not None else None
        self.channels = self._declared_channels or ()
        self.levels = None
        # attach() resolves the reads and fills this in; an unattached instance stepped by hand must raise the
        # readout's own shape error, not AttributeError
        self.body_ids = np.zeros(0, np.int64)

    def reset(self, B, device):
        super().reset(B, device)
        self.levels, self.time_ms = None, 0.
        self.channels = self._declared_channels or ()

    def reset_rows(self, rows):
        if self.levels is not None:
            self.levels = self.levels.clone(); self.levels[rows] = float("nan")

    def step(self, dt_ms, inputs):
        values = self.fn(dt_ms, inputs)
        if self._declared_channels is None and not self.channels:
            from .nt_readout import NTChannel
            self.channels = tuple(NTChannel(k, "a.u.", 0., 1.) for k in values)
        if set(values) != {c.name for c in self.channels}:
            raise ValueError("readout outputs must match channel names")
        self.levels = torch.stack([values[c.name] for c in self.channels], dim=-1) if self.channels else torch.empty(self.B, len(self.body_ids), 0, device=self.device)
        if self.levels.shape != (self.B, len(self.body_ids), len(self.channels)):
            raise ValueError("readout values must cover the union of reads")
        self.time_ms += dt_ms
        return {}

    def readout(self, *, batch_index=0):
        from .nt_readout import NTSnapshot
        if not 0 <= batch_index < self.B:
            raise IndexError(batch_index)
        if self.levels is None:
            return None
        return NTSnapshot(self.time_ms, self.body_ids, self.channels, self.levels[batch_index].detach().cpu().numpy())

    def state_dict(self):
        from dataclasses import asdict
        return {"levels": snapshot(self.levels), "time_ms": self.time_ms, "channels": [asdict(c) for c in self.channels]}

    def load_state_dict(self, d):
        from .nt_readout import NTChannel
        self.levels, self.time_ms = snapshot(d["levels"], self.device), d["time_ms"]
        self.channels = tuple(NTChannel(**c) for c in d["channels"])

    def describe(self):
        from dataclasses import asdict
        return {**super().describe(), "channels": [asdict(c) for c in self._declared_channels] if self._declared_channels is not None else None}


class ExtensionRuntime:
    """Internal input ownership and frame scheduler; allocated only when opted in."""
    def __init__(self, fb):
        self.fb = fb
        self.hooks, self.modules, self.bindings = {}, {}, {}
        self.device_bindings = {}  # fixed selections uploaded once, never in a captured frame
        self.origin = None
        self.drives, self.poisson = {}, {}
        self._positive_poisson = set()  # ephemeral activity proofs; rebuilt by full module writes
        # brain.drive splits into the part the caller injected (base_drive: held, and additive from here on) and the
        # part the sensory frame owns (sensory_drive: overwritten by every optic frame). Before the first optic frame
        # the whole drive is the caller's; after one, the optic frame has overwritten the whole tensor
        # (fly.FlyBrain._vision_frame copies, it does not accumulate), so none of it is the caller's -- a set_drive
        # issued after the last vision frame and before the first attach is therefore treated as sensory, i.e. the
        # next vision frame overwrites it, exactly as it would with no extensions attached at all.
        self.base_drive = fb.brain.drive.clone()
        if getattr(fb, "_vision_has_advanced", False):
            self.base_drive = torch.zeros_like(self.base_drive)
        self.external_poisson = fb.brain.poisson_p.clone()
        for idx in fb._input_indices.values():
            self.external_poisson[:, idx] = 0
        for idx, _, _ in fb._pulses:
            self.external_poisson[:, fb.brain._idx(idx)] = 0
        self.previous_spikes = torch.zeros_like(fb.brain.spike_counts)
        self.drive = torch.zeros_like(fb.brain.drive)
        self.vision_intensity = None
        self.sensory_drive = fb.brain.drive.clone() - self.base_drive
        self.applied_inputs = {}
        fb.brain._input_target = self.input

    def input(self, channel, idx, value):
        b = self.fb.brain
        idx = b._idx(idx)
        v = torch.as_tensor(value, dtype=torch.float32, device=b.device)
        if self.origin is None:
            if channel == "drive_mv":
                self.base_drive = self.base_drive.clone(); self.base_drive[:, idx] = v
                b.drive[:, idx] = v
                self.fb._graphs.clear()  # a captured module frame binds the previous base tensor
            else:
                self.external_poisson[:, idx] = v * (b.p.dt / 1000)
                self.refresh_poisson()
            return
        fields = self.drives if channel == "drive_mv" else self.poisson
        previous = fields.get(self.origin)
        field = torch.zeros_like(b.drive) if previous is None else previous.clone()
        field[:, idx] = v if channel == "drive_mv" else v.clamp_min(0)
        fields[self.origin] = field

    def run_hooks(self, when):
        for name, (fn, phase) in tuple(self.hooks.items()):
            if phase != when:
                continue
            self.origin = "hook:" + name
            try:
                fn(self.fb, self.fb.t)
            except Exception as e:
                raise RuntimeError(f"hook {name!r} ({when}) failed: {e}") from e
            finally:
                self.origin = None

    def hook_records(self):
        return [{"name": k, "when": w, "identifier": identifier(fn)} for k, (fn, w) in self.hooks.items()]

    def attach(self, module):
        from .interp.common import resolve
        name = module.name
        if not isinstance(name, str) or not name.strip() or name in self.modules:
            raise ValueError(f"module name must be nonempty and unique: {name!r}")
        if module.quantity_in not in QUANTITIES or module.channel_out not in CHANNELS:
            raise ValueError(f"module {name!r}: unknown quantity/channel")
        if module.describe().get("kind") not in KINDS:
            raise ValueError(f"module {name!r}: describe() must declare kind in {sorted(KINDS)}")
        boundary = getattr(module, "boundary", None)
        if boundary == "vision":
            if self.fb.retina is None:
                raise ValueError("a vision encoder needs a retina")
            if module.output_sizes != {"intensity": len(self.fb.retina.pr_index)}:
                raise ValueError("vision encoder output_sizes must be {'intensity': n_photoreceptors}")
            if any(getattr(m, "boundary", None) == "vision" for m in self.modules.values()):
                raise ValueError("only one vision encoder may replace photoreceptor intensity")
        reads = {k: resolve(self.fb.c, v) for k, v in module.reads.items()}
        writes = {k: resolve(self.fb.c, v) for k, v in module.writes.items()}
        claimed = set()
        for idx in writes.values():
            if claimed.intersection(idx):
                raise ValueError(f"module {name!r}: overlapping write labels")
            claimed.update(idx)
        for other, (_, ow) in self.bindings.items():
            if self.modules[other].channel_out == module.channel_out and any(claimed.intersection(idx) for idx in ow.values()):
                raise ValueError(f"module {name!r} overlaps {other!r} on {module.channel_out}")
        if module.quantity_in == "optic_rate":
            optic = self.fb.optic
            if optic is None or any(not np.isin(idx, optic.rate_idx).all() for idx in reads.values()):
                raise ValueError(f"module {name!r}: optic_rate reads must select optic rate cells")
        if isinstance(module, TorchModule) and writes:
            module.output_sizes = {k: len(idx) for k, idx in writes.items()}
        if isinstance(module, ReadoutModule):
            idx = np.unique(np.concatenate(list(reads.values()))) if reads else np.array([], dtype=np.int64)
            module.body_ids = self.fb.c.neurons.bodyId.to_numpy()[idx].copy()
        module.reset(self.fb.B, self.fb.device)
        if isinstance(module, ReadoutModule):
            module.time_ms = self.fb.t
        device_bindings = tuple({k:self.fb.brain._idx(idx) for k,idx in group.items()} for group in (reads,writes))
        self.modules[name], self.bindings[name] = module, (reads, writes)
        self.device_bindings[name] = device_bindings

    def records(self):
        ids = self.fb.c.neurons.bodyId.to_numpy()
        return [{**snapshot(m.describe()), "quantity_in": m.quantity_in, "channel_out": m.channel_out,
                 "read_order": list(self.bindings[name][0]), "write_order": list(self.bindings[name][1]),
                 "reads": {k: id_list(ids[v]) for k, v in self.bindings[name][0].items()},
                 "writes": {k: id_list(ids[v]) for k, v in self.bindings[name][1].items()},
                 "n_reads": {k: int(len(v)) for k, v in self.bindings[name][0].items()},
                 "n_writes": {k: int(len(v)) for k, v in self.bindings[name][1].items()}}
                for name, m in self.modules.items()]

    def can_capture_frame(self):
        # Explicit built-ins may gather neural inputs in the graph. At least one generator must
        # prove that Poisson RNG remains active on every frame. Hooks/boundaries keep the eager path.
        if self.hooks or not self.modules:
            return False
        return (any(getattr(m, 'poisson_always_on', False) for m in self.modules.values())
                and all(getattr(m, 'cuda_graph_safe', False) and getattr(m, 'cuda_async_validation', False)
                        # previous_spikes is replaced after a frame; capturing that input would bind stale storage.
                        and m.channel_out == 'poisson_hz' and m.quantity_in in ('rate_hz','drive_mv')
                        and getattr(m, 'boundary', None) is None for m in self.modules.values()))

    def run_modules(self, dt_ms):
        b = self.fb.brain
        # Gather EVERY input first, so attachment order cannot make a zero-delay loop.
        inputs = {}
        for name, m in self.modules.items():
            if getattr(m, "boundary", None) is not None:
                continue
            reads, _ = self.bindings[name]
            source = {"rate_hz": b.rate, "drive_mv": b.drive, "spike_count": self.previous_spikes}.get(m.quantity_in)
            if m.quantity_in == "optic_rate":
                o = self.fb.optic
                pos = {int(v): i for i, v in enumerate(o.rate_idx)}
                inputs[name] = {k: o.rates()[:, [pos[int(i)] for i in idx]].clone() for k, idx in reads.items()}
            else:
                inputs[name] = {k: source[:, idx].clone() for k, idx in self.device_bindings[name][0].items()}
        for name, m in self.modules.items():
            try:
                boundary = getattr(m, "boundary", None)
                if boundary == "motor":
                    continue  # evaluated by the body/environment through the program interface
                if boundary == "vision":
                    if self.fb._radiance is not None:
                        value = m.forward(self.fb._radiance.flatten(1))["intensity"]
                        if value.device != b.drive.device or not bool(torch.isfinite(value).all() & (value >= 0).all()):
                            raise ValueError("vision encoder intensity must be finite, nonnegative and on the brain device")
                        if b.p.surrogate_grad or self.vision_intensity is None:
                            self.vision_intensity = value if b.p.surrogate_grad else value.detach().clone()
                        else:
                            with torch.no_grad():
                                self.vision_intensity.copy_(value)
                    continue
                outputs = m.step(dt_ms, inputs[name])
                writes = self.bindings[name][1]
                if not isinstance(outputs, dict) or outputs.keys() != writes.keys():
                    raise ValueError("output labels must match writes")
                values = {}
                for k, idx in writes.items():
                    v = outputs[k]
                    if not isinstance(v, torch.Tensor) or v.device != b.drive.device or v.shape != (b.B, len(idx)):
                        raise ValueError(f"output {k!r} must be a tensor on {b.device} with shape {(b.B, len(idx))}")
                    finite = torch.isfinite(v).all()
                    if b.device.type == 'cuda' and getattr(m, 'cuda_async_validation', False):
                        # Explicit built-in invariant check, not regular user-input validation. A failure
                        # is asynchronous and aborts the CUDA context; arbitrary modules keep the old check.
                        torch._assert_async(finite, f"module {name!r} output {k!r} must be finite")
                    elif not bool(finite):
                        raise ValueError(f"output {k!r} must be finite")
                    values[k] = v
                self.origin = "module:" + name
                for k, idx in writes.items():
                    self.input(m.channel_out, self.device_bindings[name][1][k], values[k])
                if m.channel_out == 'poisson_hz' and getattr(m, 'poisson_always_on', False):
                    self._positive_poisson.add(self.origin)
                else:
                    self._positive_poisson.discard(self.origin)
            except Exception as e:
                raise RuntimeError(f"module {name!r} failed: {e}") from e
            finally:
                self.origin = None
        combined = self.base_drive
        for value in self.drives.values():
            combined = combined + value
        if b.p.surrogate_grad:
            self.drive = combined
        else:
            with torch.no_grad():
                self.drive.copy_(combined)
        self.refresh_poisson()
        # Input tensors are replaced, not mutated, by subsequent hook/module writes.
        # Keep the frame actually advanced distinct from post-hook inputs for the next frame.
        self.applied_inputs = {f"{name}:{channel}": value
                               for channel, fields in (("drive_mv", self.drives), ("poisson_hz", self.poisson))
                               for name, value in fields.items()}

    def refresh_poisson(self):
        fb, b = self.fb, self.fb.brain
        p = torch.maximum(fb._base_poisson, self.external_poisson)
        fb._pulses = [pulse for pulse in fb._pulses if pulse[2] > fb.t + 1e-9]
        for idx, hz, _ in fb._pulses:
            ti = b._idx(idx)
            p[:, ti] = torch.maximum(p[:, ti], torch.as_tensor(hz, device=b.device) * (b.p.dt / 1000))
        for value in self.poisson.values():
            p = torch.maximum(p, value * (b.p.dt / 1000))
        with torch.no_grad():
            b.poisson_p.copy_(p)
        # An opt-in module may prove a strictly positive probability for its held output, including dt.
        # Only use the proof while that module's field exists: before first output, reset, and detach
        # must still inspect the actual combined tensor. This preserves RNG activation at zero input.
        always_on = bool(self._positive_poisson.intersection(self.poisson))
        b._poisson_on = True if always_on else bool((p > 0).any())

    def remove(self, name, *, hook=False):
        collection = self.hooks if hook else self.modules
        if not hook and getattr(collection[name], "boundary", None) == "vision":
            self.vision_intensity = None
        del collection[name]
        if not hook:
            del self.bindings[name]
            del self.device_bindings[name]
        key = ("hook:" if hook else "module:") + name
        self._positive_poisson.discard(key)
        self.drives.pop(key, None); self.poisson.pop(key, None)
        self.refresh_poisson()

    def reset(self, rows=None):
        self._positive_poisson.clear()  # partial resets can also clear every row of a held field
        if rows is None:
            for m in self.modules.values():
                m.reset(self.fb.B, self.fb.device)
            self.drives.clear(); self.poisson.clear()
            self.base_drive = torch.zeros_like(self.base_drive)
            self.external_poisson.zero_(); self.previous_spikes.zero_(); self.drive = torch.zeros_like(self.drive)
            self.vision_intensity = None
            self.sensory_drive = torch.zeros_like(self.drive)
            self.applied_inputs = {}
        else:
            for m in self.modules.values():
                m.reset_rows(rows)
            for field in (self.base_drive, self.external_poisson, self.previous_spikes, self.drive, self.sensory_drive,
                          *self.drives.values(), *self.poisson.values()):
                field[rows] = 0
            if self.vision_intensity is not None:
                self.vision_intensity[rows] = 0
            for value in self.applied_inputs.values():
                value[rows] = 0

    def state_dict(self):
        return {"hooks": self.hook_records(), "modules": self.records(),
                "states": {k: snapshot(m.state_dict()) for k, m in self.modules.items()},
                "inputs": snapshot({k: getattr(self, k) for k in
                                    ("base_drive", "external_poisson", "previous_spikes", "drive", "drives", "poisson", "vision_intensity", "sensory_drive", "applied_inputs")})}

    def validate_state(self, d):
        if d["hooks"] != self.hook_records():
            raise ValueError("checkpoint hooks differ; reattach the named code before loading")
        def structural(records):
            from .interp.common import to_jsonable
            return to_jsonable([{k: v for k, v in r.items() if k not in ("checkpoint_hash", "training")} for r in records])
        if structural(d["modules"]) != structural(self.records()):
            raise ValueError("checkpoint modules differ; attach matching modules before loading")

    def load_state_dict(self, d):
        self._positive_poisson.clear()  # restored fields are checked normally until the next module output
        for k, m in self.modules.items():
            m.load_state_dict(snapshot(d["states"][k], self.fb.device))
        for k, v in d["inputs"].items():
            setattr(self, k, snapshot(v, self.fb.device))
