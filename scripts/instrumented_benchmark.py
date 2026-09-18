"""Bridge the benchmark's legacy Brain API to the real FlyBrain module scheduler.

Used only by benchmark.Context.new_brain when an explicit instrument is requested. It preserves
the externally computed optic current used by the legacy walk/motion probes. Raw benchmarks keep
constructing Brain directly. This adapter owns no new model law and is not a simulation backend.
"""
import numpy as np

from flyverse.fly import FlyBrain


class InstrumentedBenchmarkBrain:
    def __init__(self, c, p, *, instruments, preset, receptor=None, **kwargs):
        self.fb = FlyBrain(c, lif_params=p, optic=None, preset=preset, instruments=instruments, **kwargs)
        if receptor is not None:
            np.testing.assert_array_equal(self.fb.brain.receptor.fast_sign, receptor.fast_sign)

    def __getattr__(self, name):
        return getattr(self.fb.brain, name)

    @property
    def drive(self):
        return self.fb.brain.drive

    @drive.setter
    def drive(self, value):
        # The legacy probe computes the optic current outside FlyBrain, before each 10 ms frame.
        self.fb.brain.drive = value
        # The extension scheduler exists only when a module is attached (`compass`, the navigation instruments); a
        # sense-side transducer (`sided_turn_afferent`) or a configuration record (`ring_dc_hold`, `glno_sign`) attaches
        # none, and then the brain's own drive is the whole story (round 8, docs/audits/instrumented_suite.md).
        if self.fb._extensions is not None:
            self.fb._extensions.base_drive = value.clone()

    def step(self, n_steps):
        return self.fb.step(n_steps * self.fb.brain.p.dt)

    def run_ms(self, ms):
        return self.fb.step(ms)

    def reset(self, rows=None):
        return self.fb.reset(rows=rows)
