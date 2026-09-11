"""An independently advancing controller with latest-input and latest-output mailboxes."""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import copy
import math
import threading
import time

import numpy as np
import torch

from .fly import FlyBrain


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("AsyncFlyBrain.submit accepts CPU inputs; GPU copies belong to the worker")
        return value.detach().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot(v) for v in value)
    return copy.deepcopy(value)


class AsyncFlyBrain:
    """Own a FlyBrain exclusively until close(). Host reads never wait for GPU work.

    Construction publishes an initial snapshot and starts a worker. The worker waits for
    the first input, then advances continuously at frame_ms with held sensory values.
    submit() coalesces sensory updates; stimulus pulses are queued and never coalesced.
    Don't access the supplied controller concurrently. Use as a context manager to stop it.
    """
    def __init__(self, controller: FlyBrain, *, frame_ms=10.0):
        if not math.isfinite(frame_ms) or frame_ms < controller.brain.p.dt:
            raise ValueError("frame_ms must be finite and at least one LIF step")
        self._controller, self.frame_ms = controller, frame_ms
        self._condition = threading.Condition()
        self._inputs = {}
        self._stimuli = deque()
        self._started = self._closed = False
        self._error = None
        self._latest = controller.motor()
        self._stream = self._ready = None
        if controller.device.type == "cuda":
            self._stream = torch.cuda.Stream(device=controller.device)
            self._ready = torch.cuda.Event()
            self._ready.record(torch.cuda.current_stream(controller.device))
        self._thread = threading.Thread(target=self._run, name="FlyBrain", daemon=True)
        self._thread.start()

    def _check(self):
        if self._error is not None:
            raise RuntimeError("AsyncFlyBrain worker failed") from self._error
        if self._closed:
            raise RuntimeError("AsyncFlyBrain is closed")

    def submit(self, **sensors):
        """Atomically publish any of vision=rad, smell=(cL,cR), wind=(dL,dR), taste=contact."""
        unknown = set(sensors) - set(self._controller.available_senses)
        if unknown:
            raise ValueError(f"unavailable or unknown senses: {sorted(unknown)}")
        values = _snapshot(sensors)
        with self._condition:
            self._check()
            self._inputs.update(values)
            self._started = True
            self._condition.notify()

    def stimulate(self, selection, hz, ms):
        value = _snapshot((selection, hz, ms))
        with self._condition:
            self._check()
            if len(self._stimuli) >= 256:
                raise BufferError("stimulus queue is full")
            self._stimuli.append(value)
            self._started = True
            self._condition.notify()

    def motor(self):
        """Return an independent CPU snapshot immediately; time_ms reports its simulation age."""
        with self._condition:
            self._check()
            latest = self._latest
        return copy.deepcopy(latest)

    def wait_for_update(self, after_ms, timeout=5.0):
        """Optional blocking helper for offline consumers and tests, never used by motor()."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest.time_ms <= after_ms:
                self._check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no brain update before timeout")
                self._condition.wait(remaining)
            self._check()
            return copy.deepcopy(self._latest)

    def _run(self):
        try:
            context = torch.cuda.stream(self._stream) if self._stream is not None else nullcontext()
            with context:
                if self._stream is not None:
                    self._stream.wait_event(self._ready)
                while True:
                    with self._condition:
                        while not self._started and not self._closed:
                            self._condition.wait()
                        if self._closed:
                            return
                        inputs, self._inputs = self._inputs, {}
                        pulses, self._stimuli = self._stimuli, deque()
                    for name, value in inputs.items():
                        method = getattr(self._controller, name)
                        method(*value) if name in ("smell", "wind") else method(value)
                    for selection, hz, ms in pulses:
                        self._controller.stimulate(selection, hz, ms)
                    self._controller.step(self.frame_ms)
                    latest = self._controller.motor()
                    with self._condition:
                        self._latest = latest
                        self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
        finally:
            if self._stream is not None:
                self._stream.synchronize()

    def close(self, timeout=5.0):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("brain worker is still finishing a frame")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
