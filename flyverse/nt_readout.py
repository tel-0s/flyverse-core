"""Optional live neurotransmitter readouts. No dynamics, renderer, or device dependency.

An NT module publishes CPU snapshots on demand. The adapter declares units and display
ranges; the map never estimates concentrations from neuron labels or firing rates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class NTChannel:
    name: str
    unit: str
    display_min: float
    display_max: float

    def __post_init__(self):
        if not isinstance(self.name,str) or not isinstance(self.unit,str) or not self.name.strip() or not self.unit.strip():
            raise ValueError("NT channels require a name and an explicit unit")
        if not np.isfinite([self.display_min, self.display_max]).all() or self.display_max <= self.display_min:
            raise ValueError("NT display range must be finite and increasing")


@dataclass(frozen=True, eq=False)
class NTSnapshot:
    """Live NT levels at neurons, for one batch member at ``time_ms`` brain time.

    ``levels`` has shape (len(body_ids), len(channels)). Body IDs may be reordered
    or cover a different subset; NaN means unobserved, not zero. Values outside a
    channel's display range remain intact (only the map colour saturates).
    Owned, read-only copies prevent producer updates from changing a published frame.
    """
    time_ms: float
    body_ids: np.ndarray
    channels: tuple[NTChannel, ...]
    levels: np.ndarray
    _order: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        ids = np.array(self.body_ids, copy=True)
        channels = tuple(self.channels)
        levels = np.array(self.levels, dtype=np.float32, copy=True)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("NT body_ids must be a one-dimensional integer array")
        if not np.isfinite(self.time_ms) or self.time_ms < 0:
            raise ValueError("NT snapshot time must be finite and nonnegative")
        if not all(isinstance(channel, NTChannel) for channel in channels):
            raise TypeError("NT channels must be NTChannel instances")
        if len({channel.name for channel in channels}) != len(channels):
            raise ValueError("NT channel names must be unique")
        if levels.shape != (len(ids), len(channels)) or np.isinf(levels).any():
            raise ValueError("NT levels must have shape (body IDs, channels), with finite values or NaN")
        order = np.argsort(ids)
        if np.any(ids[order][1:] == ids[order][:-1]):
            raise ValueError("NT body_ids must be unique")
        for array in (ids, levels, order):
            array.flags.writeable = False
        object.__setattr__(self, "body_ids", ids)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "_order", order)

    def aligned(self, body_ids):
        """An owned (N, channels) array in a consumer's row order; absent IDs are NaN."""
        ids = np.asarray(body_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("requested body_ids must be a one-dimensional integer array")
        result = np.full((len(ids),len(self.channels)), np.nan, dtype=np.float32)
        if len(self.body_ids):
            ordered = self.body_ids[self._order]
            pos = np.searchsorted(ordered, ids)
            valid = pos < len(ordered)
            valid[valid] &= ordered[pos[valid]] == ids[valid]
            result[valid] = self.levels[self._order[pos[valid]]]
        return result


class NTSource(Protocol):
    """Read-only adapter implemented by an optional NT simulation module.

    Called only when a consumer requests NT data, never from the stepping hot path.
    Batch selection, device-to-CPU transfer and any field sampling belong to the source.
    Return None while unavailable. The source's owner handles stepping/checkpointing.
    """
    def readout(self, *, batch_index: int = 0) -> NTSnapshot | None: ...
