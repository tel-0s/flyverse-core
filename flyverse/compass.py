"""An imposed heading representation, NOT a recovered biological compass.

CompassDriver integrates the body's signed yaw velocity and continuously supplies an EPG Poisson
bump through the ordinary module interface. The program supplies the angular memory and the input;
the connectome still determines spikes, propagation and motor readout. No edge, receptor, membrane
state, motor command or world heading is overwritten. Intended only for preset='instrumented'.

The 16-wedge mapping follows the instance annotations and the interleaved map checked by Wang:
https://github.com/pwang724/fly-circuit-exploration/blob/80b94e6608cf927ca2c9f2bbce0577c5a998684c/docs/compass-recurrence.md
The functional motivation is Turner-Evans et al. 2017, eLife 23496 (angular-velocity integration).
The exact kinematic law, Gaussian width and imposed rate are engineering choices, UNVERIFIED.
This is neither Wang's rate-model implementation nor evidence for a receptor/feedback mechanism.
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch


def epg_columns(c):
    """Local EPG indices and interleaved EB wedges; require all 16 annotated wedges."""
    idx = np.flatnonzero(c.neurons.type.fillna('').to_numpy() == 'EPG')
    columns = []
    for instance in c.neurons.iloc[idx].instance.fillna(''):
        match = re.search(r'_([LR])([1-8])(?:$|_)', instance)
        if match is None:
            raise ValueError(f'compass requires EPG PB glomeruli L1-L8/R1-R8: {instance!r}')
        side, number = match[1], int(match[2])
        columns.append(2 * (number - 1) if side == 'L' else 2 * (8 - number) + 1)
    if set(columns) != set(range(16)):
        raise ValueError('compass requires all 16 annotated EPG wedges; this dataset/subset is unavailable')
    return idx, np.asarray(columns, dtype=np.int64)


class CompassDriver:
    """Batched kinematic heading memory with a continuous Poisson input to the biological EPGs.

    Feed ``fb.proprioception(..., yaw_rate=rad_per_s)`` before stepping, or ``observe_turn`` directly.
    Positive yaw advances the interleaved wedge coordinate. This is a declared convention; absolute
    phase is arbitrary, and there is no visual landmark anchoring or tilt compensation. It neither
    reads world heading/fruit locations nor computes steering. Stale motion input remains held, like
    the other sensory inputs. Reset clears it. O(B*n_EPG) work per module frame; no brain readback.
    """
    name = 'compass'
    kind = 'stop-gap'
    quantity_in = 'rate_hz'
    channel_out = 'poisson_hz'
    required_preset = 'instrumented'
    incompatible = ('compass_ring',)
    # User inputs are checked on CPU. A nonfinite output is an internal invariant failure;
    # on CUDA it may abort the CUDA context at a later kernel launch, without a per-frame host wait.
    cuda_async_validation = True
    cuda_graph_safe = True  # fixed Torch operations; held input tensors; state restored around capture

    def __init__(self, c, *, peak_hz=50., width_deg=35., initial_phase_deg=0., velocity_gain=1.):
        values = (peak_hz, width_deg, initial_phase_deg, velocity_gain)
        if not all(math.isfinite(x) for x in values) or peak_hz <= 0 or not 0 < width_deg < 180:
            raise ValueError('compass parameters must be finite; peak positive and width in (0,180) degrees')
        if velocity_gain == 0:
            raise ValueError('velocity_gain must be nonzero (negative is an explicit sign-control arm)')
        if peak_hz > np.finfo(np.float32).max or abs(velocity_gain) > np.finfo(np.float32).max:
            raise ValueError('compass peak and gain must be representable in float32')
        if math.radians(width_deg) < np.finfo(np.float32).tiny:
            raise ValueError('compass width must be representable in float32 radians')
        self.idx, self.columns = epg_columns(c)
        self.source_body_ids = c.neurons.bodyId.to_numpy().copy()
        self.body_ids = self.source_body_ids[self.idx].copy()
        self.peak_hz, self.width_deg = float(peak_hz), float(width_deg)
        self.initial_phase_deg, self.velocity_gain = float(initial_phase_deg), float(velocity_gain)
        self.reads, self.writes = {}, {'epg': self.idx}

    def validate_parent(self, fb):
        if fb.preset != self.required_preset:
            raise ValueError("compass requires preset='instrumented'")
        if not np.array_equal(self.source_body_ids, fb.c.neurons.bodyId.to_numpy()):
            raise ValueError('compass was built for a different connectome row order')
        if self.peak_hz * fb.brain.p.dt > 1000:
            raise ValueError('compass peak exceeds the Poisson probability bound at this dt')
        # Even the farthest target is at most pi from phase. A strictly positive representable
        # probability at that distance proves that this held Poisson field keeps RNG enabled.
        distance = math.pi/math.radians(self.width_deg)
        floor = self.peak_hz * math.exp(-.5*distance**2) if distance < 40 else 0.
        self.poisson_always_on = floor * fb.brain.p.dt / 1000 > np.finfo(np.float32).tiny

    def install(self, fb):
        self.validate_parent(fb)
        if fb.attached_modules.get(self.name) is self:
            return
        fb.attach(self)

    def reset(self, B, device):
        self.B, self.device = B, torch.device(device)
        self.phase = torch.full((B, 1), math.radians(self.initial_phase_deg) % (2 * math.pi), device=device)
        self.yaw_rate = torch.zeros((B, 1), device=device)
        self.angles = torch.as_tensor(self.columns * (2 * math.pi / 16), device=device, dtype=torch.float32)[None]

    def reset_rows(self, rows):
        self.phase[rows] = math.radians(self.initial_phase_deg) % (2 * math.pi)
        self.yaw_rate[rows] = 0

    def observe_turn(self, yaw_rate):
        """Held angular-velocity input from the body, scalar or (B,), in rad/s; no angle oracle."""
        value = np.asarray(yaw_rate, dtype=np.float32)
        if value.ndim == 0:
            value = np.full(self.B, value, dtype=np.float32)
        if value.shape != (self.B,) or not np.isfinite(value).all():
            raise ValueError(f'compass yaw_rate must be finite and scalar or shape ({self.B},)')
        if np.any(np.abs(value.astype(np.float64)*self.velocity_gain)>np.finfo(np.float32).max):
            raise ValueError('compass yaw_rate * velocity_gain overflows float32')
        self.yaw_rate.copy_(torch.from_numpy(value[:, None]), non_blocking=self.device.type=='cuda')

    def step(self, dt_ms, inputs):
        if not math.isfinite(dt_ms) or not 0 <= dt_ms <= 10.000001:
            raise ValueError('compass expects a finite module frame of 0-10 ms')
        self.phase.add_(self.yaw_rate * (self.velocity_gain * dt_ms / 1000)).remainder_(2 * math.pi)
        error = torch.remainder(self.angles - self.phase + math.pi, 2 * math.pi) - math.pi
        return {'epg': self.peak_hz * torch.exp(-.5 * (error / math.radians(self.width_deg)).square())}

    def state_dict(self):
        return {'phase': self.phase.clone(), 'yaw_rate': self.yaw_rate.clone()}

    def load_state_dict(self, state):
        values = {k: torch.as_tensor(state[k], device=self.device, dtype=torch.float32) for k in ('phase', 'yaw_rate')}
        if any(v.shape != (self.B, 1) or not bool(torch.isfinite(v).all()) for v in values.values()):
            raise ValueError('invalid compass checkpoint shape or values')
        if not bool(((values['phase']>=0)&(values['phase']<=2*math.pi)).all()):
            raise ValueError('compass checkpoint phase must be wrapped to [0, 2*pi]')
        if bool((values['yaw_rate'].double().abs()*abs(self.velocity_gain)>np.finfo(np.float32).max).any()):
            raise ValueError('compass checkpoint yaw_rate * velocity_gain overflows float32')
        self.phase.copy_(values['phase']); self.yaw_rate.copy_(values['yaw_rate'])

    def describe(self):
        return dict(name=self.name, kind=self.kind, law='unverified',
                    **{'class': 'flyverse.compass:CompassDriver'}, trainable=False, checkpoint_hash=None,
                    parameters=dict(peak_hz=self.peak_hz, width_deg=self.width_deg,
                                    initial_phase_deg=self.initial_phase_deg, velocity_gain=self.velocity_gain),
                    law_text='phase += velocity_gain*yaw_rad_s*dt_s; EPG Poisson Hz = peak*exp(-0.5*(wrapped angle error/width)^2)',
                    gap='the plain EPG ring does not maintain and update a confined heading bump (compass rounds 6-7)',
                    removal='a native circuit that maintains and integrates a heading bump under the same turn/reversal/dark tests',
                    audits=['docs/audits/compass_standin.md', 'docs/audits/compass_velocity_route.md'],
                    sources=['https://elifesciences.org/articles/23496',
                             'https://github.com/pwang724/fly-circuit-exploration/blob/80b94e6608cf927ca2c9f2bbce0577c5a998684c/docs/compass-recurrence.md'],
                    input='body yaw_rate through proprioception, held until updated; initial phase arbitrary; no absolute heading or goal input',
                    output='continuous artificial Poisson drive on biological EPG cells; parent spikes and edges untouched',
                    body_ids=self.body_ids.tolist(), wedges=self.columns.tolist(),
                    limitations='imposed angular memory; no visual anchoring, 3D attitude compensation, goal memory or steering policy')
