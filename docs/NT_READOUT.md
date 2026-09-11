# Live neurotransmitter readouts

The room map can display **live levels supplied by an optional NT module**. It does
not infer concentrations from transmitter labels, firing rates or synaptic signs.
There is currently no NT dynamics module in this branch. Until one is attached, the
NT view reports that its readout is unavailable; neural activity remains available.

Press **N**, or select **NT levels** within the map. Click a channel to display it.
`--brain-map-mode nt` opens that view at launch and does not enable NT dynamics.
`--brain-map` retains the neural activity view by default.

## Interface for a simulation module

`flyverse.nt_readout` depends only on NumPy and the Python standard library. Its three
types are also exported from `flyverse`:

| Type | Contract |
|---|---|
| `NTChannel(name, unit, display_min, display_max)` | Unique channel name, explicit unit, fixed finite display range. These bounds calibrate the display; they do not affect the simulation. |
| `NTSnapshot(time_ms, body_ids, channels, levels)` | One batch member's current levels at neurons. Shape `(N, K)` for N body IDs and K channels; timestamp in brain milliseconds. |
| `NTSource.readout(*, batch_index=0)` | Return an `NTSnapshot`, or `None` when no snapshot is available. |

Attach a source with `FlyBrain(..., nt_source=source)` or `fb.nt_source = source`.
Consumers call `fb.neurotransmitters(batch_index=0)`. With no source, this returns
`None`. The getter checks batch bounds and the returned snapshot type.

An adapter can wrap an existing module's sampling function without exposing that
module's internals to the UI:

```python
from flyverse import NTSnapshot

class ModuleNTReadout:
    def __init__(self, sample, channels):
        self.sample = sample
        self.channels = tuple(channels)

    def readout(self, *, batch_index=0):
        # The real module supplies its time, neuron IDs and CPU levels.
        result = self.sample(batch_index)
        if result is None:
            return None
        time_ms, body_ids, levels = result
        return NTSnapshot(time_ms, body_ids, self.channels, levels)

# sample_live_nt and channels come from the optional simulation module/adapter.
fb.nt_source = ModuleNTReadout(sample_live_nt, channels)
```

Sources own batch selection, device-to-CPU transfer and thread synchronization. A
spatial NT model can sample its field at neurons before publishing the snapshot.
The hook is a readout attachment only: **the module's owner must advance, reset and
checkpoint its dynamics**. Neither attaching a source nor opening the map registers
new simulation steps, edits weights, or adds state to existing checkpoints.

## Alignment and missing data

Snapshots contain owned, read-only CPU copies. Body IDs are unique integers, not
connectome row indices. `snapshot.aligned(body_ids)` returns an owned array in the
requested order. This preserves alignment through connectome subsets and reordering;
extra source neurons are ignored, and missing neurons become NaN.

NaN means no sample; zero is a real measured/modelled zero. Invalid dimensions,
duplicate IDs/names, infinite values, absent units and invalid display ranges are
rejected. NT annotations are neither required nor used by the readout or projection.
`BrainMap(c, optic=None, locations=...)` also works without an optic module or the
global dataset. Locations may be supplied by body ID, or in `c.neurons.somaLocation`.

## Display semantics

One channel is projected at a time. Colours use its declared, fixed linear range;
values sharing a pixel are averaged in their original units before clipping to that
range. Gray anatomy indicates no sampled value, not zero. The legend shows range
and units, snapshot time, and sampled/located counts. Channel mean/max statistics
include finite samples in the current connectome, including neurons without somata.
Out-of-range values stay intact in the statistics even when their colours saturate.

Readouts are requested only while the NT map is visible, at `--map-every` cadence.
Paused redraws and channel changes reuse a snapshot. Switching back to NT, explicitly
reselecting NT levels, or replacing/removing the public source attachment refreshes
the display. There is no visual decay of NT levels between samples. Source errors
clear old NT data and display the error instead of leaving stale levels on screen.

Validation covers optional/absent sources, batch selection, no readout calls during
stepping, immutable snapshots, reordered/subset body IDs, missing samples, physical
pixel means, fixed display ranges, no-optic maps, empty geometry, paused sampling,
channel selection, source removal/errors, and launch/toggle behaviour without a module.
