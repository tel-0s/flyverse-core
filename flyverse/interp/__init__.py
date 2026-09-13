"""flyverse.interp -- the interpretability toolkit: localize a behavioural deficit to populations, synapses and links.

Eight tools, one result schema, one rule: the toolkit READS the model (brain.py / optic.py / the cache) and never
changes it. Design, schema, composition, GPU / CPU split, validation targets and CLI examples: docs/INTERP.md.

    decompose  per-frame signed input to a cell set by presynaptic type / transmitter / receptor tier
    trace      stimulus vs null per type, ordered by synaptic depth from a sensory population; the first stage lost
    paths      effective k-step signed gains A -> B through the shaped weights (k <= 3), silent links flagged
    lesion     a manifest of lesion sets x the suite checks / probes, one batched job, check x lesion delta matrix
    atlas      stimulate each population briefly and record every motor readout, with a null
    health     per-type operating-point statistics of a rollout, also an observatory readout source (NTSource)
    ledger     expected responses (flyverse/data/expected_responses.csv) scored against every probe's per-type output
    export     the read-only Neurome probe export (docs/NEUROME_INTERFACE.md) from any Result

Each tool is a function in `flyverse/interp/<tool>.py` with the signature fixed in `stubs` below and a CLI wrapper
`scripts/interp_<tool>.py`. Until an implementer's module lands, `flyverse.interp.<tool>` resolves to the stub (which
raises NotImplementedError naming the module to create); once `flyverse/interp/<tool>.py` exists and defines the
function, the same attribute returns it -- so `from flyverse.interp import trace` is the stable import path. The
shared machinery (selection, shaped weights, Recorder / Recording, null helpers, Result, provenance) is in
`flyverse.interp.common` and is real code, not a stub.
"""
from __future__ import annotations

import importlib

from . import common
from .common import (Population, Recorder, Recording, Result, EffectiveWeights, TOOLS, VALIDATION, SCHEMA,
                     resolve, population, populations, effective_weights, links, silent_flags, compare, provenance)

__all__ = ["common", "stubs", "HealthReadoutStub", "Population", "Recorder", "Recording", "Result", "EffectiveWeights",
           "TOOLS", "VALIDATION", "SCHEMA", "resolve", "population", "populations", "effective_weights", "links",
           "silent_flags", "compare", "provenance", *TOOLS, "HealthReadout"]


def _todo(name: str):
    raise NotImplementedError(f"flyverse.interp.{name} is a stub: implement it in flyverse/interp/{name}.py with this "
                              f"signature (docs/INTERP.md, section 4) and it will be picked up automatically")


class stubs:
    """The contract: signatures and docstrings of the eight tools. Implementations keep every parameter named here
    (they may add keyword-only parameters with defaults). All return a `common.Result` unless stated."""

    @staticmethod
    def decompose(c, target, *, recording=None, params=None, optic_params=None, receptor=None, by=("type",),
                  tiers=True, window=None, kind="current", null_recording=None, fb=None, top=40):
        """Signed input to the cell set `target` decomposed by presynaptic population.

        Static (recording None): per (target type, pre type) the effective input a volley of the pre type would
        deliver, `EffectiveWeights.type_matrix` (mV per post cell per pre volley) with raw counts alongside -- the
        structural decomposition of docs/audits/cx_glno.md section 1. Dynamic (recording given): per frame the
        rate-weighted input  I_i(t) = sum_j A[i, j] r_j(t)  (mV/s of synaptic input per post cell; kind 'current')
        split by the presynaptic grouping `by` -- any of 'type', 'transmitter', 'tier' (receptor tier: nt_sign /
        receptor:<tier>), 'sign', 'module', 'side' -- over `window` (start_s, end_s) and, when `null_recording` is
        given, the same under the matched control with the arm comparison of `common.compare`. The recording must
        contain rate_hz of every presynaptic cell with a non-zero A entry onto the target (Recorder over
        `links(...).pre_index` -- `scripts/interp_decompose.py record` does this) and, for graded targets, optic_dr
        of the presynaptic rate units (decomposed through the OpticLobe's W_rr / W_rp / W_rs instead of A). `tiers`
        adds the receptor-tier split; `top` limits the printed rows.

        Tables: 'per_type' (target_type, pre_group, value, kind, n_pre, n_post, sign, raw_count, share), 'contributions'
        (Neurome fields; body-level rows for the `top` groups), 'readout_per_body'. Summary: total E / I per target
        type, the largest cancelling pair. Validation: VALIDATION['decompose'].
        """
        _todo("decompose")

    @staticmethod
    def trace(c, source, *, stimulus, control, null=None, params=None, optic_params=None, stat="figure_z", depth_max=6,
              stage_table=None, decompose_at="first_lost", quantity=None, min_cells=3, fb=None):
        """Where along the synaptic depth from `source` is a stimulus lost?

        `stimulus`, `control`, `null` are lists of Recordings (>= 3 independent runs each: stimulus runs, matched
        control runs, and control-vs-control runs for the null). Per type, `stat`: 'figure_z' (probe_figure_stages'
        signed figure z when the recording carries columns in meta) or 'dprime' (screen.rank's d' between the stimulus
        and control condition over the window) and, always, the arm comparison of `common.compare` (z against the
        none-vs-none null, Welch, U, p, verdict). Depth is the shortest path length from `source` over |A| > 0 in
        `effective_weights` (0 = the source itself; graded and spiking cells alike), or the `stage_table` stages
        (the Nern 2025 figure groups of probe_figure_stages.stage_table when given). Types are listed in depth
        order; 'first_lost' is the smallest depth at which no type of that depth has verdict 'result' while some
        type at the depth before has, and the tool then calls `decompose` (static and, over `stimulus[0]`, dynamic)
        on the types at that depth. Validation: VALIDATION['trace'] (Mi4 / Mi1 / Tm3 above the null, T2 / T3 / Tm5Y /
        TmY21 / LC11 / LC10a at it; the LH apple channel at depth 2 from the ORNs).

        Tables: 'per_type' (type, depth, stage, n_cells, stim_mean, ctrl_mean, null_mean, z, welch, U, p, verdict,
        figure_z per run), 'readout_per_body', and the decomposition tables of the lost stage prefixed 'lost_'.
        """
        _todo("trace")

    @staticmethod
    def paths(c, a, b, *, params=None, receptor=None, k_max=3, top=20, min_abs_mv=0.0, recording=None, frozen=None,
              level="type", exclude=(), rates_min_hz=common.NEVER_FIRING_HZ):
        """Effective k-step signed gains from population `a` to population `b` (k <= k_max), ranked, silent links flagged.

        Level 'type': the type-level matrix M[post, pre] = mean over post cells of the summed effective input from all
        pre cells (mV per post cell per pre volley; `EffectiveWeights.type_matrix`), restricted to the types on any
        path of length <= k_max between a and b; a k-step path a -> t1 -> ... -> b has gain prod M along it (mV^k per
        volley) and the tool lists the `top` paths by |gain| per k with every link's sign, raw count and silence flags:
        'sign0' (the presynaptic type carries no sign: its entries are explicit zeros, the raw count from
        cache/sign0_counts.npz), 'frozen' / 'pruned' (a rate unit of the optic lobe -- pass `frozen` = fb.optic.rate_idx),
        'never_firing' (max rate over `recording` below `rates_min_hz`). Level 'cell' keeps cells (for the ring:
        cx_wedge's wedge matrices are the cell level aggregated by wedge). `exclude` drops types from the intermediate
        set. Validation: VALIDATION['paths'] (GLNO -> PEN sign 0 at 16,371 raw synapses / 16.5 mV per pair if signed;
        EPG -> PEN +5.05 mV per pair; the ExR4/5/6 -> EPG two-step loop).

        Tables: 'paths' (k, path, gain, signs, silent_links, raw_counts), 'links' (common.links of every edge on a listed
        path), 'contributions' (the same in Neurome fields). Summary: the strongest silent link per k.
        """
        _todo("paths")

    @staticmethod
    def lesion(manifest, *, out_dir, mode="plan", replicates=common.MIN_REPLICATES, checks="all", probes=(), seeds=None,
               cluster=False, minutes=60, baseline="baseline"):
        """A manifest of lesion sets x the benchmark checks and probes -> a check x lesion delta matrix with scatter.

        `manifest` (dict or a JSON / YAML path): {'lesions': [{'id', 'kind', 'spec', ...}], 'sections': 'rest,taste,...',
        'probes': [{'id', 'cmd': 'python scripts/probe_x.py ... --out {out}', 'read': 'json path -> value'}]}. Kinds:
        'types' / 'population' (a common.resolve spec; silenced by zeroing the presynaptic columns of the shaped
        weights in-process, the same effect as screen.ablate: no output), 'module' (regions.MODULES), 'transmitter'
        (every entry with that presynaptic nt), 'receptor_tier' (entries the receptor lookup decided at that tier
        held at NT_SIGN), 'hold_table' (a receptors_by_type.csv from scripts/build_hold_tables.py: --receptor-table),
        'lif' / 'optic' (LIFParams / OpticParams overrides), 'pair_gain' (an optic pair-gain factor). Mode 'plan'
        writes the job commands (one per lesion x replicate, plus the baseline) as a cluster_run.py batch line and
        the manifest with resolved bodies; 'run' executes one job in-process (`scripts/interp_lesion.py --one`);
        'analyse' reads the finished JSONs in `out_dir` and builds the matrix: per (check, lesion) baseline, value,
        delta, replicate values / sd / n, plus the double dissociations -- pairs of lesions (L1, L2) and checks
        (c1, c2) with c1 moved by L1 and not L2 and c2 by L2 and not L1 (bit-identity when the section is
        deterministic, else beyond the replicate scatter). Validation: VALIDATION['lesion'].

        Tables: 'sensitivity' (Neurome fields), 'matrix' (check x lesion), 'dissociations'. Files: the batch line.
        """
        _todo("lesion")

    @staticmethod
    def atlas(c, populations, *, hz=150.0, ms=400.0, settle_ms=200.0, batch=64, readouts=("motor",), pattern=None,
              by_side=True, null=True, replicates=common.MIN_REPLICATES, params=None, optic_params=None, device=None,
              context=None, seed=0):
        """Stimulate each population in `populations` briefly and record every motor readout, with a null.

        One FlyBrain(batch=`batch`); row r stimulates population r (FlyBrain.stimulate at `hz` for `ms`, both sides
        unless the spec names one) after `settle_ms`; `null` rows receive no pulse in the same batch; `replicates`
        independent runs (seeds seed, seed+1, ...) give the scatter. Readouts: 'motor' = every MotorRates field
        (fwd_dn, turn_L/R, leg_L/R, gf, power, steer_L/R, wind_*, lh_odour.<channel> ...) and, when `pattern` is
        given, every population it matches through screen.TypeRecorder (by_side keys). `context` optionally names a
        sensory context ('blueberry_site', 'walking', or a callable that drives the FlyBrain's senses each frame);
        default none (the screen_dns.py protocol). Per (population, readout): pulse-window mean minus the null rows'
        mean with common.compare over replicates. Validation: VALIDATION['atlas'].

        Tables: 'atlas' (population, readout, stim_mean, null_mean, diff, z, verdict per replicate), 'readout_per_body'
        for the readout cells. Summary: the top readout per population and the top population per readout.
        """
        _todo("atlas")

    @staticmethod
    def health(recording, *, c=None, params=None, window=None, by="type", ew=None, thresholds=None, fb=None):
        """Per-type operating-point statistics of a rollout.

        From a Recording with rate_hz and, when present, v_mv / adapt_mv / refrac / spike_count (Recorder quantities)
        over `window`: silent fraction (max rate < NEVER_FIRING_HZ), at-threshold fraction (mean v within 1 mV of
        v_th while not refractory), refractory load (rate x t_ref: the fraction of time in the refractory period;
        'refractory-limited' above 0.4 -- the 200-260 Hz bump is 0.44-0.57 at t_ref 2.2 ms), E/I input balance (from
        `ew` and the recorded presynaptic rates: sum of positive vs negative rate-weighted input, as decompose's
        'current'), fan-in scale (EffectiveWeights.scale, min / median / max per type), adaptation load (mean adapt
        / (v_th - v_rest)), and the sign-0 / frozen / pruned shares of each type's input and output (silent_flags).
        `thresholds` overrides the classification bounds. Validation: VALIDATION['health'].

        Tables: 'per_type' (type, n_cells, unit_kind, rate_mean, rate_max, silent_frac, at_threshold_frac,
        refractory_load, ei_balance, fanin_scale_med, adapt_load, sign0_in_share, sign0_out_share), 'readout_per_body'.
        """
        _todo("health")

    @staticmethod
    def ledger(results, *, table=None, tolerance=0.0, strict=False):
        """Score every probe's per-type output against flyverse/data/expected_responses.csv.

        `results`: Result objects or paths (any tool whose tables carry 'per_type' or 'readout_per_body', and the
        benchmark suite's JSON via its 'checks' list). The table (columns: population, stimulus, quantity, expected,
        op (>, <, >=, <=, sign, range, abs>=), unit, source, model_reference, notes) is read from `table` or the
        shipped file; every row whose (population, stimulus) matches a result's populations and protocol gets a
        status PASS / FAIL / MISSING with the measured value, the replicate scatter and the citation. `strict` makes
        a MISSING row a failure. Validation: VALIDATION['ledger'] (the T4/T5 DSI, loom GF, HSN / DNp20 / HSE d',
        sugar / bitter MN9 rows reproduce the benchmark's statuses).

        Tables: 'ledger' (row fields + status, measured, sd, n). Summary: counts per status.
        """
        _todo("ledger")

    @staticmethod
    def export(result, *, out_root="out/export", run_id=None, retina=None, parquet_rows=1_000_000, control_ids=None):
        """The read-only Neurome probe export (docs/NEUROME_INTERFACE.md section 1) from a Result -- a serializer.

        Writes out/export/<run_id>/manifest.json (run_id; flyverse_commit + dirty; dataset_release with the four MaleCNS
        file hashes; compiled_connectome fingerprint; model = LIFParams / OpticParams / body thresholds; execution;
        stimulus; retina with the column -> photoreceptor-body map from `retina` (a path or dict) when given; the units
        table; the table list with row counts, columns, units and SHA-256) and the tables readout_per_body.csv /
        contributions.csv / sensitivity.csv (Parquet above `parquet_rows`) from `Result.tables` of the same names,
        keyed (dataset, release, bodyId) with bodyId as a decimal string, model_index and type alongside; LC11 and
        LC10a rows are kept per quantity (upstream_drive_mV, output_Hz), never pooled. Refuses a Result whose
        `check()` is non-empty. Returns the run directory. Validation: VALIDATION['export'].
        """
        _todo("export")


class HealthReadoutStub:
    """The observatory readout source of the health tool (docs/NT_READOUT.md): an NTSource whose channels are per-neuron
    operating-point quantities sampled live from a FlyBrain -- rate_hz, v_margin_mv (v_th - v), adapt_mv, refractory
    (0 / 1), fanin_scale -- as an NTSnapshot keyed by bodyId (NaN on graded units for the LIF quantities). Attach with
    `fb.nt_source = HealthReadout(fb)`; no UI file changes. Implemented as `HealthReadout` in flyverse/interp/health.py;
    `flyverse.interp.HealthReadout` resolves to it once that module exists, to this stub until then."""

    def __init__(self, fb, channels=("rate_hz", "v_margin_mv", "adapt_mv", "refractory", "fanin_scale")):
        _todo("health")

    def readout(self, *, batch_index: int = 0):
        _todo("health")


def _implementation(name: str, attr: str, fallback):
    """The implementer's `attr` from flyverse/interp/<name>.py when the module exists, else `fallback`."""
    try:
        mod = importlib.import_module(f".{name}", __name__)
    except ImportError as e:
        if e.name not in (f"{__name__}.{name}", name):       # a real import error inside the implementer's module
            raise
        return fallback
    obj = getattr(mod, attr, None)
    if obj is None:
        raise AttributeError(f"flyverse/interp/{name}.py exists but defines no `{attr}`")
    return obj


def __getattr__(name: str):
    if name in TOOLS:
        return _implementation(name, name, getattr(stubs, name))
    if name == "HealthReadout":
        return _implementation("health", "HealthReadout", HealthReadoutStub)
    raise AttributeError(name)
