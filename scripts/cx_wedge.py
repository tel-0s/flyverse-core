"""Does the connectome's central-complex wiring support a ring attractor under this model's synaptic rules?

Structural audit (no simulation needed; `python scripts/cx_wedge.py`, results in docs/audits/cx_wedge.json + PNGs):
  * wedge identity of every EPG / PEN / PEG / Delta7 cell from the instance names (PB glomerulus L1-L9 / R1-R9;
    the Delta7 instance carries its OUTPUT glomeruli, e.g. Delta7(PB15)_L1L9R8_R);
  * the 16-wedge ring order of the glomeruli, read off the direct EPG -> EPG matrix (L_i neighbours R_(9-i) and
    R_(8-i)): L1 R8 L2 R7 L3 R6 L4 R5 L5 R4 L6 R3 L7 R2 L8 R1; L9 wraps onto L1, R9 onto R1;
  * effective weights A[post, pre] in mV per presynaptic spike = w_syn * fan-in scale[post] * _shaped_weights
    (connection cap 60, path gains, same-type damping 0.1) -- exactly what Brain installs;
  * EPG x EPG two-step matrices through PEN, PEG, Delta7 and the ring / extrinsic-ring neurons (ER*, ExR*: the
    untuned EPG -> ER/ExR -> EPG, PEN feedback that turns out to dominate), A[EPG, X] @ A[X, EPG] in mV^2 per spike,
    cell-level ordered by wedge and aggregated to 16 wedges / 8 tiles ("input to a typical cell of wedge i if
    every cell of wedge k fires once, relayed through X");
  * the ring-attractor window in the linear estimate: for a bump of k wedges, net two-step input inside versus
    outside as a function of rho = gD / gE^2 (gD = Delta7 -> EPG gain, gE = EPG <-> PEN gain on both links), and
    the same with the flat ER/ExR term G added (feasible gD per gE).

Threshold-linear rate model (`--rate-grid gE,... gD,...`): mean-field fixed point r = f(tau_syn A r) of the compass
cells (+ ER/ExR unless --no-ring) with the LIF f-I curve, under the same background / pulse protocol as the LIF.

Simulation cross-check (`--sim gE:gD ...`): FlyBrain on the full connectome, no world, compass adaptation off
(adapt_by_type {'^(EPG|PEN|PEG|Delta7)': 0}), 10 Hz Poisson background on every EPG (--background), `--width`
contiguous wedges (default 4 = one 90 deg tile pair) driven at +40 Hz for 2 s, then 5 s free; reports EPG cells
above 22 Hz inside / outside the driven wedges, PEN / Delta7 / ER-ExR rates and the wedge profile at 0.5 / 1 / 2 /
3 / 5 s after the pulse. `--no-delta7-pen` applies gD to Delta7 -> EPG only (Delta7 -> PEN stays x1; with the gain
on Delta7 -> PEN as well, the NOTES-session-8 convention, Delta7 clamps PEN and no bump survives). `--ring-gain`
scales ER/ExR -> EPG / PEN / PEG. Rows append to --sim-out (JSON); `--plot-sim` draws the wedge profiles from it.
Round 2 of the receptor integration (docs/audits/cx_glno.md, driver scripts/cx_glno.py): `--nt-override TYPE=nt`
(repeatable) compiles the connectome with connectome.TYPE_NT_OVERRIDE extended into the scratch cache
out/cache_<hash>/ (`--scratch-cache` does so without an override), `--receptor-model` / `--receptor-net-rule` thread
LIFParams.receptor_model, and every row records the rate of GLNO (4 cells, 19 % of PEN's raw input, NT unknown).

Round 6A (docs/audits/compass_dc_balance.md): `--hold-edges PRE_REGEX:POST_REGEX` (repeatable, default None) holds one
class of edges at 0 through the existing `type_path_gain` stage -- an `edges`-kind LABELLED COUNTERFACTUAL
(docs/INTERP.md 10.1 step 5), not a candidate default; with the flag absent the installed gain list is entry-for-entry
the previous one. Every row records the hold and the cells / entries / synapses it silenced.

Round 6B (docs/audits/compass_local_recurrence.md): `--edge-gain PRE_REGEX:POST_REGEX:FACTOR` (repeatable, default None)
multiplies one class of edges by FACTOR through the same `type_path_gain` stage -- a LABELLED INSTRUMENT (a per-type
gain, never a default). Because `brain._shaped_weights` applies `type_path_gain` BEFORE `same_type_gain`, the entry
`^EPG$:^EPG$:10` composes with the shipped x0.1 to exactly x1.0 on the 842 EPG -> EPG pairs (10.0f x 0.1f rounds back
to the integer synapse count in float32; tests/test_cx_wedge_hold.py pins it bit for bit) while every other same-type
clique stays damped: the per-type undamping 5A / 6A asked for, with no brain.py change. Every ledger run also records
per-type ring rates (ExR6 / ER6 / ER4m, plus EPGt) and the per-cell maximum rate of the small compass groups, so the
rate model's ring rates and INTERP.md 10's `silent` (max rate per cell < 0.5 Hz) are measured, not inferred.

Round 7 (docs/audits/compass_velocity_route.md, docs/PRESETS_SPEC.md, docs/INSTRUMENTS.md): `--preset raw|instrumented`
(default raw: byte-identical to a call without it) and `--instrument NAME[:k=0.5][:sign=-1][:cells=variant]`
(repeatable; implies `--preset instrumented`) attach a flyverse.instruments stand-in to the FlyBrain --
`sided_turn_afferent`, Poisson spikes on PS196_b's ascending afferents at k * max(0, +-yaw_deg_s) on the side the
graph implies. Under `--preset instrumented` the 6A hold (`ring_dc_hold`), the GLNO relabel (`glno_sign`) and a 6B
edge gain are ALSO recorded as instruments (an `edges` / `relabel` record each, PRESETS_SPEC 2.4) beside the flags
that install them, so the JSON's `provenance.instruments` lists everything hand-written in the run; without the
preset flag those flags behave and record exactly as in 6A / 6B. `--turn DEG_S` (ledger runs only; default None =
nothing fed) imposes a signed yaw rate -- positive = a left turn, the body's convention -- over `--turn-window
START:END` seconds after the pulse end; cx_wedge has no body, so the turn is a PRESCRIBED protocol parameter (the
analogue of the 90 deg/s imposed visual rotation of deficit_rotation.md), fed to the afferent instrument through
FlyBrain.proprioception(yaw_rate=) when one is attached and merely recorded when none is. Every ledger run records
per-side rates of PEN / GLNO / DNa02 / PS196_b per frame and the round-7 measures: `<G>_LR_hz` (mean L - R over the
turn window), `bump_follow_wedges_per_s` (slope of the unwrapped bump centre over the turn window, times the sign of
the turn; ideal 16 / 360 * |turn|) and the afferents' own per-side rates.

Findings (docs/audits/cx_wedge.md): the tuned structure is there (PEN excitation local, Delta7 inhibition
cosine-shaped with own-wedge / opposite = 0.10); the loop is shut at gain x1 by the untuned EPG -> ExR6 / ExR4 /
ER6 / ER4m -> EPG, PEN feedback, not by Delta7; a persistent, confined bump exists in the LIF for gE 1.75-2 with
gD 15-40 on Delta7 -> EPG only (rho = gD / gE^2 of 4-10), or for gE 1-1.25 with the ER/ExR feedback damped x0.3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flyverse import brain, connectome  # noqa: E402

AUDIT_DIR = Path(__file__).resolve().parent.parent / "docs" / "audits"
SCRATCH_ROOT = Path(__file__).resolve().parent.parent / "out"      # scratch connectome caches: out/cache_<hash>/
RING16 =["L1", "R8", "L2", "R7", "L3", "R6", "L4", "R5", "L5", "R4", "L6", "R3", "L7", "R2", "L8", "R1"]
POS16 = {g: i for i, g in enumerate(RING16)}
POS16["L9"] = 0     # L9 wraps onto L1 (Delta7_L1L9R8), R9 onto R1 (Delta7_L8R1R9)
POS16["R9"] = 15
COMPASS_RE = r"^(EPG|PEN|PEG|Delta7)"
RING_RE = r"^(ER|ExR)"      # ring neurons and extrinsic ring neurons: the EPG -> ExR / ER -> EPG, PEN global feedback


def parse_nt_override(items) -> dict:
    """--nt-override TYPE=nt (repeatable) -> {type: nt}; nt must be a NT_SIGN key."""
    ov = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--nt-override expects TYPE=nt, got {it!r}")
        t, nt = it.split("=", 1)
        nt = nt.strip().lower()
        if nt not in connectome.NT_SIGN:
            raise SystemExit(f"--nt-override {it!r}: transmitter must be one of {sorted(connectome.NT_SIGN)}")
        ov[t.strip()] = nt
    return ov


def parse_hold_edges(items) -> list:
    """--hold-edges PRE_REGEX:POST_REGEX (repeatable) -> [(pre_re, post_re, 0.0)], an `edges`-kind HOLD in the sense of
    docs/INTERP.md 2 / 10.1 step 5: every edge from a cell whose type matches PRE_REGEX onto a cell whose type matches
    POST_REGEX is multiplied by 0 in `brain._shaped_weights` (the existing `LIFParams.type_path_gain` machinery, the
    same stage that carries gE / gD here), so the class is silenced in the weight matrix the LIF installs. The split is
    on the FIRST ':' (neither regex may contain one). A LABELLED COUNTERFACTUAL, never a default: with the flag absent
    the type_path_gain list is the one the previous code built, entry for entry.

    e.g. --hold-edges '^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)'  (thread 6A, docs/audits/compass_dc_balance.md)"""
    out = []
    for it in items or []:
        if ":" not in it:
            raise SystemExit(f"--hold-edges expects PRE_REGEX:POST_REGEX, got {it!r}")
        pre, post = it.split(":", 1)
        if not pre or not post:
            raise SystemExit(f"--hold-edges expects PRE_REGEX:POST_REGEX, got {it!r}")
        for r in (pre, post):
            try:
                re.compile(r)
            except re.error as e:
                raise SystemExit(f"--hold-edges {it!r}: bad regex {r!r} ({e})")
        out.append((pre, post, 0.0))
    return out


def parse_edge_gains(items) -> list:
    """--edge-gain PRE_REGEX:POST_REGEX:FACTOR (repeatable) -> [(pre_re, post_re, factor)]: one class of edges multiplied
    by FACTOR in `brain._shaped_weights` through `LIFParams.type_path_gain` (the stage that carries gE / gD and the 6A
    hold). The FACTOR is split off at the LAST ':' and the regexes at the FIRST (neither regex may contain one). A
    LABELLED INSTRUMENT (a per-type gain), never a default: with the flag absent the gain list is unchanged.

    Stage order matters and is the point: type_path_gain is applied BEFORE same_type_gain, so `^EPG$:^EPG$:10` under
    the shipped same_type_gain 0.1 leaves the EPG -> EPG pairs at exactly x1.0 (their connectome weight) while
    PEN_a -> PEN_a, PEN_b -> PEN_b, Delta7 -> Delta7 and PEG -> PEG stay x0.1 (thread 6B, arm H3E)."""
    out = []
    for it in items or []:
        if it.count(":") < 2:
            raise SystemExit(f"--edge-gain expects PRE_REGEX:POST_REGEX:FACTOR, got {it!r}")
        head, fac = it.rsplit(":", 1)
        pre, post = head.split(":", 1)
        if not pre or not post:
            raise SystemExit(f"--edge-gain expects PRE_REGEX:POST_REGEX:FACTOR, got {it!r}")
        try:
            f = float(fac)
        except ValueError:
            raise SystemExit(f"--edge-gain {it!r}: FACTOR must be a number, got {fac!r}")
        if not np.isfinite(f) or f < 0:
            raise SystemExit(f"--edge-gain {it!r}: FACTOR must be finite and >= 0")
        for r in (pre, post):
            try:
                re.compile(r)
            except re.error as e:
                raise SystemExit(f"--edge-gain {it!r}: bad regex {r!r} ({e})")
        out.append((pre, post, f))
    return out


# ---------------------------------------------------------------------------------------------- round 7: presets / instruments
RING_DC_HOLD = (r"^(ExR6|ER6|ER4m)$", r"^(PEN_|EPG$)")      # the 6A hold, named `ring_dc_hold` when recorded as an instrument
RING_DC_HOLD_PEN = (r"^(ExR6|ER6|ER4m)$", r"^PEN_")         # the PEN-side hold only (round 7 arm HGVp): `ring_dc_hold_pen`
PRESETS = ("raw", "instrumented")


def resolve_preset(preset, instrument_specs) -> str:
    """--preset / --instrument -> the preset: None with no instrument is 'raw' (the default, byte-identical); an
    instrument implies 'instrumented'; 'raw' with an instrument is refused (docs/PRESETS_SPEC.md 1)."""
    specs = list(instrument_specs or [])
    if preset is not None and preset not in PRESETS:
        raise SystemExit(f"--preset must be one of {PRESETS}, got {preset!r}")
    if specs and preset == "raw":
        raise SystemExit("--preset raw attaches no instrument; drop --instrument or use --preset instrumented")
    return preset or ("instrumented" if specs else "raw")


def parse_turn_window(s) -> tuple:
    """--turn-window START:END (seconds after the pulse end) -> (start, end); default 0.5:3.5."""
    if s is None:
        return (0.5, 3.5)
    try:
        a, b = (float(x) for x in str(s).split(":", 1))
    except ValueError:
        raise SystemExit(f"--turn-window expects START:END seconds, got {s!r}")
    if not (np.isfinite(a) and np.isfinite(b)) or a < 0 or b <= a:
        raise SystemExit(f"--turn-window needs 0 <= START < END, got {s!r}")
    return (a, b)


def build_instruments(c, instrument_specs, preset, hold_edges=None, nt_override=None, edge_gains=None, same_type_gain=None) -> list:
    """The flyverse.instruments list a run attaches: the --instrument stand-ins, and -- under preset 'instrumented'
    only -- an `edges` record per --hold-edges (the 6A hold is `ring_dc_hold`), a `relabel` record per --nt-override
    (GLNO is `glno_sign`) and an `edges` record per --edge-gain, each with the resolved counts. Under 'raw' the list
    is empty whatever the flags (the flags still record themselves in the row as in 6A / 6B)."""
    from flyverse import instruments as fi
    if preset == "raw":
        return []
    out = []
    for spec in instrument_specs or []:
        try:
            out.append(fi.parse_instrument(spec, c))
        except ValueError as e:
            raise SystemExit(f"--instrument {spec!r}: {e}")
    holds = list(hold_edges or [])
    resolved = hold_edge_counts(c, holds) if holds else []
    for i, ((pre, post, f), rec) in enumerate(zip(holds, resolved)):
        name = ("ring_dc_hold" if (pre, post) == RING_DC_HOLD and f == 0.0 else
                "ring_dc_hold_pen" if (pre, post) == RING_DC_HOLD_PEN and f == 0.0 else f"edge_hold_{i}")
        description = None
        if (pre, post, f) == (r"^GLNO$", r"^PEN_", 0.0):
            name = "glno_pen_hold"
            description = dict(gap="the transfer of a GLNO side signal into PEN in round 7",
                               source="an explicit pathway-removal control, not a physiological receptor model",
                               removal="retire after the diagnostic; this hold is not an adoption candidate",
                               audits=["docs/audits/compass_velocity_route.md"])
        out.append(fi.EdgeHold(pre, post, f, name=name, resolved=rec, description=description))
    for t, nt in (nt_override or {}).items():
        out.append(fi.TypeRelabel(t, nt))
    gains = list(edge_gains or [])
    resolved = hold_edge_counts(c, gains, same_type_gain=same_type_gain) if gains else []
    for i, ((pre, post, f), rec) in enumerate(zip(gains, resolved)):
        out.append(fi.EdgeGain(pre, post, f, name=f"edge_gain_{i}", resolved=rec))
    return out


def side_groups(c, types) -> dict:
    """{'<label>_L': idx, '<label>_R': idx} of the cells of `types` ({label: type or (types)}) by somaSide -- the per-side
    groups the ledger records (PEN, GLNO, DNa02, PS196_b). Types are looked up in the connectome, never assumed."""
    ty = c.neurons.type.fillna("").to_numpy().astype(str)
    soma = c.neurons.somaSide.fillna("").to_numpy().astype(str)
    out = {}
    for label, t in types.items():
        m = np.isin(ty, list(t) if isinstance(t, (list, tuple)) else [t])
        out[f"{label}_L"], out[f"{label}_R"] = np.flatnonzero(m & (soma == "L")), np.flatnonzero(m & (soma == "R"))
    return out


def bump_follow(centre, confined, tt, t_on, t_off, turn_deg_s):
    """The round-7 primary measure: slope (wedges / s) of the UNWRAPPED bump centre over the turn window, times the sign
    of the turn, so a bump that follows the turn in ring order is positive whatever the direction; NaN without a turn or
    with fewer than 3 finite frames. Also the confined fraction over the window and the ideal 16 / 360 * |turn|."""
    if turn_deg_s is None:
        return {"bump_follow_wedges_per_s": float("nan"), "bump_follow_confined_frac": float("nan"), "bump_follow_ideal_wedges_per_s": float("nan"), "bump_follow_n_frames": 0}
    m = (tt >= t_on - 1e-9) & (tt < t_off - 1e-9)
    cen = np.asarray(centre, float)[m]
    ok = np.isfinite(cen)
    out = {"bump_follow_confined_frac": float(np.mean(np.asarray(confined)[m])) if m.any() else float("nan"),
           "bump_follow_ideal_wedges_per_s": float(abs(turn_deg_s) * 16.0 / 360.0), "bump_follow_n_frames": int(ok.sum())}
    if ok.sum() < 3:
        out["bump_follow_wedges_per_s"] = float("nan")
        return out
    unwrapped = np.unwrap(cen[ok] * (2 * np.pi / 16.0)) * (16.0 / (2 * np.pi))
    slope = float(np.polyfit(tt[m][ok], unwrapped, 1)[0])
    out["bump_follow_wedges_per_s"] = slope * float(np.sign(turn_deg_s))
    return out


def prepare_neural_stimuli(c, events, total_s):
    """Resolve explicit assay pulses for the ledger loop; nothing installed when events is None.

    This is a stimulus protocol, not a physiological stand-in. Each event names indices on the supplied
    connectome, an unverified or sourced law, Hz, and frame-aligned onset/duration in recording seconds.
    The existing FlyBrain.stimulate surface owns application and expiry. No parent synapse is modified.
    """
    out = []
    for e in events or []:
        allowed = {"name", "idx", "poisson_hz", "start_s", "duration_s", "law"}
        if set(e) != allowed or not e["name"] or not e["law"]:
            raise ValueError("neural stimulus needs name, idx, poisson_hz, start_s, duration_s and law")
        idx = np.asarray(e["idx"])
        if (idx.ndim != 1 or not len(idx) or idx.dtype.kind not in "iu" or
                np.any(idx < 0) or np.any(idx >= c.n) or len(np.unique(idx)) != len(idx)):
            raise ValueError("neural stimulus indices must be unique cells in this connectome")
        hz, start, duration = (float(e[k]) for k in ("poisson_hz", "start_s", "duration_s"))
        if (not np.isfinite([hz, start, duration]).all() or hz < 0 or start < 0 or duration <= 0 or
                start + duration > total_s + 1e-9):
            raise ValueError("invalid neural stimulus rate or recording window")
        if any(abs(t/.01 - round(t/.01)) > 1e-7 for t in (start, duration)):
            raise ValueError("neural stimulus timing must align with 10 ms ledger frames")
        out.append(dict(name=e["name"], idx=idx.tolist(), body_ids=c.neurons.bodyId.iloc[idx].astype(str).tolist(),
                        poisson_hz=hz, start_s=start, duration_s=duration, law=e["law"]))
    return out


def hold_edge_counts(c, holds, same_type_gain=None) -> list:
    """Per hold (pre_re, post_re, factor): the cells it matches and the W entries / raw synapses it silences -- recorded
    in the row so a run's own JSON proves the hold was installed."""
    ty = c.neurons.type.fillna("").to_numpy()
    W = c.W.tocsr()
    rows = []
    for pre_re, post_re, factor in holds or []:
        pre = np.flatnonzero([bool(re.match(pre_re, t)) for t in ty])
        post = np.flatnonzero([bool(re.match(post_re, t)) for t in ty])
        B = W[post][:, pre] if len(pre) and len(post) else None
        row = dict(pre=pre_re, post=post_re, factor=float(factor), n_pre_cells=int(len(pre)), n_post_cells=int(len(post)),
                   pre_types=sorted(set(ty[pre].tolist())), post_types=sorted(set(ty[post].tolist())),
                   n_entries=int(B.nnz) if B is not None else 0,
                   synapses=float(np.abs(B.data).sum()) if B is not None and B.nnz else 0.0)
        if same_type_gain is not None and B is not None and B.nnz:
            # 6B: how many of the matched entries are same-type (they ALSO carry same_type_gain, applied after this stage)
            Bc = B.tocoo()
            same = ty[post][Bc.row] == ty[pre][Bc.col]
            row["n_entries_same_type"] = int(same.sum())
            row["effective_factor_same_type"] = float(np.float32(factor) * np.float32(same_type_gain))
            row["effective_factor_cross_type"] = float(factor)
        rows.append(row)
    return rows


def override_table(extra: dict | None) -> dict:
    """connectome.TYPE_NT_OVERRIDE (when it is the default) extended by `extra`."""
    table = dict(connectome.TYPE_NT_OVERRIDE if connectome.TYPE_NT_OVERRIDE_DEFAULT else {})
    table.update(extra or {})
    return table


def override_cache_dir(table: dict, root: Path = SCRATCH_ROOT) -> Path:
    h = hashlib.sha1(json.dumps(sorted(table.items())).encode()).hexdigest()[:8]
    return Path(root) / f"cache_{h}"


def load_connectome(extra: dict | None = None, scratch: bool = False, verbose: bool = False, root: Path = SCRATCH_ROOT):
    """The default cache, unless a type-NT override is given (or `scratch`): then the connectome is compiled from the
    raw MaleCNS files with TYPE_NT_OVERRIDE extended by `extra` into out/cache_<hash>/ (hash of the full table;
    TYPE_NT_OVERRIDE.json alongside), built in a temporary directory and renamed into place so that concurrent jobs
    building the same table cannot see a half-written cache. Returns (connectome, cache_dir or None, table)."""
    table = override_table(extra)
    if not extra and not scratch:
        return connectome.load(verbose=verbose), None, table
    cache_dir = override_cache_dir(table, root)
    if (cache_dir / "W_post_pre.npz").exists():
        return connectome.load(cache_dir=cache_dir, verbose=verbose), cache_dir, table
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=cache_dir.name + ".tmp-", dir=cache_dir.parent))
    c = connectome.load(cache_dir=tmp, rebuild=True, verbose=verbose, type_nt_override=table)
    (tmp / "TYPE_NT_OVERRIDE.json").write_text(json.dumps(table, indent=1))
    try:
        os.rename(tmp, cache_dir)                      # fails when another job has already put its copy there
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    return c, cache_dir, table


def effective_weights(c, p: brain.LIFParams):
    """A[post, pre] in mV per presynaptic spike, as Brain installs it (before the optic prune, which does not
    touch these cells)."""
    W = brain._shaped_weights(c, p)
    tot = np.asarray(abs(W).sum(axis=1)).ravel()
    scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0).astype(np.float32)
    A = (sp.diags(scale) @ W).tocsr()
    A.data *= np.float32(p.w_syn)
    return A, scale, tot


def glomerulus(instance: str) -> str | None:
    m = re.search(r"_([LR]\d)(?:$|_)", instance)
    return m.group(1) if m else None


def delta7_outputs(instance: str) -> list[str]:
    return re.findall(r"[LR]\d", instance.split("_")[1])


def compass_cells(c):
    n = c.neurons
    ty = n.type.fillna("").to_numpy()
    inst = n.instance.fillna("").to_numpy()
    cells = {}
    for name, pat in [("EPG", r"^EPG$"), ("EPGt", r"^EPGt$"), ("PEN", r"^PEN_"), ("PEN_a", r"^PEN_a"), ("PEN_b", r"^PEN_b"),
                      ("PEG", r"^PEG$"), ("Delta7", r"^Delta7$"), ("Ring", RING_RE)]:
        idx = np.flatnonzero([bool(re.match(pat, t)) for t in ty])
        if name == "Ring":
            pos = np.full(len(idx), np.nan); lab = list(ty[idx])
        elif name == "Delta7":
            outs = [delta7_outputs(inst[i]) for i in idx]
            pos = np.array([np.mean([POS16[g] for g in o if g in POS16]) for o in outs])   # mean output position
            # adjacent output wedges (e.g. L1=0, R8=1); L8R1R9 -> (14, 15); mean is the tile centre
            lab = ["".join(o) for o in outs]
        else:
            gl = [glomerulus(inst[i]) for i in idx]
            pos = np.array([POS16[g] for g in gl], dtype=float)
            lab = gl
        order = np.lexsort((idx, pos))
        cells[name] = dict(idx=idx[order], pos=pos[order], label=[lab[i] for i in order],
                           body=n.bodyId.to_numpy()[idx[order]])
    return cells


def ring_dist(a, b, n=16):
    d = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :]) % n
    return np.minimum(d, n - d)


def aggregate(K, pos_post, pos_pre, n_bins, bin_of):
    """M[i, k] = mean over post cells in bin i of the sum over pre cells in bin k of K[post, pre]."""
    bp, bq = bin_of(pos_post), bin_of(pos_pre)
    M = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        rows = bp == i
        if not rows.any():
            continue
        for k in range(n_bins):
            cols = bq == k
            M[i, k] = K[rows][:, cols].sum(axis=1).mean() if cols.any() else 0.0
    return M


def bump_window(E, I, k):
    """Bump = k contiguous bins starting at every offset (the ring is not perfectly uniform); per bin the two-step
    excitation E and inhibition I (negative) it receives from the bump; window of rho = gD / gE^2 such that every
    inside bin is net-excited and every outside bin net-inhibited. Returns the offset-averaged profile and the
    window (rho_low, rho_high) worst-case over offsets, plus the per-offset windows."""
    n = E.shape[0]
    rows = []
    for s in range(n):
        inb = np.zeros(n, bool)
        inb[[(s + j) % n for j in range(k)]] = True
        e = E[:, inb].sum(axis=1)
        i = I[:, inb].sum(axis=1)
        ratio = np.where(np.abs(i) > 1e-9, e / np.maximum(np.abs(i), 1e-9), np.inf)
        rho_low = ratio[~inb].max() if (~inb).any() else 0.0   # outside must be net inhibited: rho > E/|I|
        rho_high = ratio[inb].min()                            # inside must stay net excited: rho < E/|I|
        rows.append(dict(offset=s, e_in=e[inb].mean(), e_out=e[~inb].mean() if (~inb).any() else 0.0,
                         i_in=i[inb].mean(), i_out=i[~inb].mean() if (~inb).any() else 0.0,
                         e_in_min=e[inb].min(), e_out_max=e[~inb].max() if (~inb).any() else 0.0,
                         rho_low=float(rho_low), rho_high=float(rho_high)))
    df = pd.DataFrame(rows)
    return df


def heatmap(M, labels, title, path, cmap="RdBu_r", symmetric=True, fmt=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = np.abs(M).max() if M.size else 1.0
    fig, ax = plt.subplots(figsize=(max(4, 0.42 * len(labels) + 1.5), max(3.5, 0.42 * len(labels) + 1.2)))
    im = ax.imshow(M, cmap=cmap, vmin=-v if symmetric else 0, vmax=v, aspect="equal")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("presynaptic EPG wedge (ring order)"); ax.set_ylabel("postsynaptic EPG wedge")
    ax.set_title(title, fontsize=9)
    if fmt and len(labels) <= 16:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt % M[i, j], ha="center", va="center", fontsize=5.5,
                        color="white" if abs(M[i, j]) > 0.6 * v else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def structure(out_dir: Path, gamma_nominal=6.0, verbose=True, c=None):
    log = print if verbose else (lambda *a, **k: None)
    c = connectome.load(verbose=False) if c is None else c
    p = brain.LIFParams()
    A, scale, tot = effective_weights(c, p)
    cells = compass_cells(c)
    epg, pen, peg, d7, epgt, ring = (cells[k] for k in ("EPG", "PEN", "PEG", "Delta7", "EPGt", "Ring"))
    res = {"n_cells": {k: int(len(v["idx"])) for k, v in cells.items()},
           "ring16": RING16, "cells_per_wedge": {g: int((np.array(epg["label"]) == g).sum()) for g in RING16},
           "fan_in_scale": {k: [float(scale[v["idx"]].min()), float(scale[v["idx"]].max())] for k, v in cells.items()},
           "fan_in_total": {k: [float(tot[v["idx"]].min()), float(tot[v["idx"]].max())] for k, v in cells.items()}}
    log("cells:", res["n_cells"], "\nEPG per wedge:", res["cells_per_wedge"])
    log("Delta7 output labels:", sorted(set(d7["label"])))

    def block(post, pre):
        return A[post["idx"]][:, pre["idx"]].toarray().astype(np.float64)

    # ---- one-step pair statistics (mV per presynaptic spike per pair)
    one = {}
    for name, (pre, post) in {"EPG->PEN": (epg, pen), "PEN->EPG": (pen, epg), "EPG->PEG": (epg, peg), "PEG->EPG": (peg, epg),
                              "EPG->Delta7": (epg, d7), "Delta7->EPG": (d7, epg), "Delta7->PEN": (d7, pen), "Delta7->PEG": (d7, peg),
                              "Delta7->Delta7": (d7, d7), "EPG->EPG": (epg, epg), "PEN->PEN": (pen, pen), "PEG->PEN": (peg, pen),
                              "EPGt->Delta7": (epgt, d7), "Delta7->EPGt": (d7, epgt),
                              "EPG->Ring": (epg, ring), "Ring->EPG": (ring, epg), "Ring->PEN": (ring, pen), "Ring->Ring": (ring, ring)}.items():
        B = block(post, pre); nz = B != 0
        one[name] = dict(pairs=int(nz.sum()), mean_per_pair=float(B[nz].mean()) if nz.any() else 0.0,
                         total_per_post=float(B.sum(axis=1).mean()))
    res["one_step"] = one
    for k, v in one.items():
        log(f"  {k:>15}: {v['pairs']:5d} pairs, {v['mean_per_pair']:+6.2f} mV/pair, {v['total_per_post']:+7.1f} mV per post if all pre fire once")

    # ---- Delta7 own-wedge structure at the one-step level
    dpos = d7["pos"]
    d_in = block(d7, epg)     # [d7, epg]
    d_out = block(epg, d7)    # [epg, d7]
    dist_in = ring_dist(dpos, epg["pos"])      # distance from the Delta7's output tile to the EPG wedge
    dist_out = ring_dist(epg["pos"], dpos).T   # same, for the output block (transposed to [d7, epg])
    prof_in = np.array([d_in[dist_in <= 0.5].sum() / max((dist_in <= 0.5).sum(), 1)] +
                       [d_in[(dist_in > k - 1) & (dist_in <= k)].sum() / max(((dist_in > k - 1) & (dist_in <= k)).sum(), 1) for k in range(1, 9)])
    prof_out = np.array([d_out.T[dist_out <= 0.5].sum() / max((dist_out <= 0.5).sum(), 1)] +
                        [d_out.T[(dist_out > k - 1) & (dist_out <= k)].sum() / max(((dist_out > k - 1) & (dist_out <= k)).sum(), 1) for k in range(1, 9)])
    res["delta7_profile_by_distance"] = dict(distance_wedges=list(range(9)), epg_to_delta7_mean_pair_mV=prof_in.tolist(),
                                             delta7_to_epg_mean_pair_mV=prof_out.tolist())
    log("Delta7 one-step profile vs ring distance (wedges, 0 = the Delta7's own output tile), mean mV per cell pair:")
    log("   EPG -> Delta7:", np.round(prof_in, 2).tolist())
    log("   Delta7 -> EPG:", np.round(prof_out, 2).tolist())

    # ---- two-step EPG x EPG matrices (mV^2 per spike; one step each way through X)
    K = {"PEN": block(epg, pen) @ block(pen, epg), "PEG": block(epg, peg) @ block(peg, epg),
         "Delta7": block(epg, d7) @ block(d7, epg), "PEN_a": block(epg, cells["PEN_a"]) @ block(cells["PEN_a"], epg),
         "PEN_b": block(epg, cells["PEN_b"]) @ block(cells["PEN_b"], epg)}
    K["direct"] = block(epg, epg)   # mV per spike, no intermediate
    K["Ring"] = block(epg, ring) @ block(ring, epg)      # the ER / ExR global feedback, two steps
    # which ring / ExR types carry it (EPG -> type -> EPG, summed over the type's cells, mean per EPG pair)
    rt = np.array(ring["label"]); ring_types = {}
    for t in sorted(set(rt)):
        sel = rt == t
        Kt = block(epg, ring)[:, sel] @ block(ring, epg)[sel]
        if np.abs(Kt).sum() > 0:
            Kp = block(pen, ring)[:, sel] @ block(ring, epg)[sel]
            ring_types[t] = dict(cells=int(sel.sum()), nt=str(c.neurons.nt.to_numpy()[ring["idx"][sel]][0]),
                                 epg_to_type_mV_per_pair=float(block(ring, epg)[sel][block(ring, epg)[sel] != 0].mean()) if (block(ring, epg)[sel] != 0).any() else 0.0,
                                 type_to_epg_mV_per_pair=float(block(epg, ring)[:, sel][block(epg, ring)[:, sel] != 0].mean()) if (block(epg, ring)[:, sel] != 0).any() else 0.0,
                                 type_to_pen_mV_per_pair=float(block(pen, ring)[:, sel][block(pen, ring)[:, sel] != 0].mean()) if (block(pen, ring)[:, sel] != 0).any() else 0.0,
                                 two_step_epg_total_mV2=float(Kt.sum(axis=1).mean()), two_step_pen_total_mV2=float(Kp.sum(axis=1).mean()))
    res["ring_types"] = ring_types
    log("ring / ExR feedback types (EPG -> type -> EPG two-step total per EPG cell, mV^2; and onto PEN):")
    for t, v in sorted(ring_types.items(), key=lambda kv: kv[1]["two_step_epg_total_mV2"]):
        log(f"   {t:>8} ({v['cells']:2d} cells, {v['nt']:>9}): EPG->{t} {v['epg_to_type_mV_per_pair']:+.2f}/pair, {t}->EPG {v['type_to_epg_mV_per_pair']:+.2f}/pair, "
            f"{t}->PEN {v['type_to_pen_mV_per_pair']:+.2f}/pair; two-step onto EPG {v['two_step_epg_total_mV2']:+8.0f}, onto PEN {v['two_step_pen_total_mV2']:+8.0f}")
    # three-step: EPG -> Delta7 -> PEN -> EPG (the Delta7 clamp on PEN), mV^3
    K["Delta7_PEN"] = block(epg, pen) @ block(pen, d7) @ block(d7, epg)
    bin16 = lambda x: np.asarray(np.round(x), int) % 16
    bin8 = lambda x: (np.asarray(np.round(x), int) % 16) // 2
    tiles = [f"{RING16[2 * i]}/{RING16[2 * i + 1]}" for i in range(8)]
    M16 = {k: aggregate(v, epg["pos"], epg["pos"], 16, bin16) for k, v in K.items()}
    M8 = {k: aggregate(v, epg["pos"], epg["pos"], 8, bin8) for k, v in K.items()}
    res["M16"] = {k: np.round(v, 2).tolist() for k, v in M16.items()}
    res["M8"] = {k: np.round(v, 2).tolist() for k, v in M8.items()}
    res["tiles"] = tiles
    labels46 = [f"{g}" for g in epg["label"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in ("PEN", "PEG", "Delta7", "direct", "Delta7_PEN", "Ring"):
        unit = {"direct": "mV/spike", "Delta7_PEN": "mV^3 (3 steps)"}.get(k, "mV^2 (2 steps)")
        heatmap(K[k], labels46, f"EPG x EPG through {k}, per cell pair [{unit}]", out_dir / f"cx_wedge_{k}_cells.png")
        heatmap(M16[k], RING16, f"EPG x EPG through {k}, 16 wedges (sum over pre wedge, mean over post) [{unit}]",
                out_dir / f"cx_wedge_{k}_16.png", fmt="%.0f")
        heatmap(M8[k], tiles, f"EPG x EPG through {k}, 8 tiles [{unit}]", out_dir / f"cx_wedge_{k}_8.png", fmt="%.0f")
    pd.set_option("display.width", 250)
    for k in ("PEN", "PEG", "Delta7", "direct", "Ring"):
        log(f"\n{k}: 16-wedge matrix (rows post, cols pre; ring order)")
        log(pd.DataFrame(np.round(M16[k], 1), index=RING16, columns=RING16).to_string())
        log(f"{k}: 8-tile matrix")
        log(pd.DataFrame(np.round(M8[k], 1), index=tiles, columns=tiles).to_string())

    # ---- profile of each path against ring distance (16-wedge level, averaged over the diagonal bands)
    prof = {}
    for k in ("PEN", "PEG", "Delta7", "direct", "Delta7_PEN", "Ring"):
        M = M16[k]
        d = ring_dist(np.arange(16), np.arange(16))
        prof[k] = [float(M[d == j].mean()) for j in range(9)]
    res["profile16_by_distance"] = dict(distance_wedges=list(range(9)), **prof)
    log("\nring-distance profiles (16-wedge level, mean over bands; distance in wedges of 22.5 deg):")
    for k, v in prof.items():
        log(f"   {k:>10}: " + " ".join(f"{x:8.1f}" for x in v))
    # locality of PEN excitation: share of the total row mass within +-1 tile (+-2 wedges) and the +-1 wedge
    d = ring_dist(np.arange(16), np.arange(16))
    for k in ("PEN", "PEG"):
        M = M16[k]; total = M.sum()
        res[f"{k}_locality"] = dict(within_1_wedge=float(M[d <= 1].sum() / total), within_2_wedges=float(M[d <= 2].sum() / total),
                                    within_4_wedges=float(M[d <= 4].sum() / total), opposite_half=float(M[d >= 5].sum() / total),
                                    diag_over_mean=float(np.mean(np.diag(M)) / (total / 256)))
        log(f"{k} locality: {res[f'{k}_locality']}")
    Md = M16["Delta7"]
    own = np.mean(np.diag(Md)); nb = Md[d == 1].mean(); far = Md[d >= 5].mean(); other = Md[d >= 1].mean(); opp = Md[d == 8].mean()
    res["Delta7_inhibition"] = dict(own_wedge=float(own), neighbour_wedge=float(nb), other_wedges_mean=float(other),
                                    far_half_mean=float(far), opposite_wedge=float(opp),
                                    ratio_own_over_other=float(own / other), ratio_own_over_opposite=float(own / opp),
                                    ratio_own_over_far_half=float(own / far))
    log("Delta7 two-step inhibition (16-wedge):", {k: round(v, 3) for k, v in res["Delta7_inhibition"].items()})

    # ---- ring-attractor window as a function of rho = gD / gE^2, bump width k, at 16 and 8 resolution
    windows = {}
    for level, M, n in (("16", M16, 16), ("8", M8, 8)):
        E = M["PEN"] + M["PEG"]; I = M["Delta7"]
        for k in range(1, (n // 2) + 1):
            df = bump_window(E, I, k)
            w = dict(k=k, e_in=float(df.e_in.mean()), e_out=float(df.e_out.mean()), i_in=float(df.i_in.mean()), i_out=float(df.i_out.mean()),
                     locality_ratio=float((df.e_in.mean() / df.e_out.mean()) / (df.i_in.mean() / df.i_out.mean())),
                     rho_low_worst=float(df.rho_low.max()), rho_high_worst=float(df.rho_high.min()),
                     rho_low_mean=float(df.rho_low.mean()), rho_high_mean=float(df.rho_high.mean()),
                     offsets_with_window=int((df.rho_high > df.rho_low).sum()), n_offsets=int(len(df)))
            windows[f"{level}:{k}"] = w
    res["windows"] = windows
    # with the untuned ring / ExR feedback G (gain 1): u = gE^2 E + gD I + G; feasible gD per gE, bump of k bins
    windows_g = {}
    for level, M, n in (("16", M16, 16), ("8", M8, 8)):
        E = M["PEN"] + M["PEG"]; I = M["Delta7"]; G = M["Ring"]
        for k in ([2, 3, 4, 5, 6] if n == 16 else [1, 2, 3]):
            for gE in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
                lows, highs, ok = [], [], 0
                for s0 in range(n):
                    inb = np.zeros(n, bool); inb[[(s0 + j) % n for j in range(k)]] = True
                    e = E[:, inb].sum(axis=1); i = I[:, inb].sum(axis=1); g = G[:, inb].sum(axis=1)
                    num = gE * gE * e + g
                    # outside: gD |i| > num  -> gD > num/|i| (auto if num <= 0); inside: gD < num/|i| (impossible if num <= 0)
                    lo = max([num[j] / abs(i[j]) for j in range(n) if not inb[j]] + [0.0])
                    hi = min([num[j] / abs(i[j]) for j in range(n) if inb[j]])
                    lows.append(lo); highs.append(hi); ok += int(hi > lo and hi > 0)
                windows_g[f"{level}:{k}:{gE}"] = dict(level=level, k=k, gE=gE, gD_low_worst=float(max(lows)), gD_high_worst=float(min(highs)),
                                                     gD_low_mean=float(np.mean(lows)), gD_high_mean=float(np.mean(highs)), offsets_ok=ok, n_offsets=n)
    res["windows_with_global"] = windows_g
    log("\nwith the ring / ExR global feedback G at gain 1 (u = gE^2 E + gD I + G): feasible Delta7 gain gD per EPG<->PEN gain gE")
    log("  level k  gE    gD_low (worst / mean)   gD_high (worst / mean)   offsets with a window")
    for key, w in windows_g.items():
        log(f"  {w['level']:>4} {w['k']:>2} {w['gE']:<5} {w['gD_low_worst']:8.2f} / {w['gD_low_mean']:6.2f}     {w['gD_high_worst']:8.2f} / {w['gD_high_mean']:6.2f}        {w['offsets_ok']}/{w['n_offsets']}")
    log("\nring-attractor window (two-step, linear): rho = gD / gE^2; bump of k bins; E = PEN+PEG, I = Delta7 (mean per post cell, sum over bump)")
    log("  level k   E_in    E_out    I_in    I_out  E_in/E_out  |I_in|/|I_out|  rho_low(max out E/|I|)  rho_high(min in E/|I|)  offsets with window")
    for key, w in windows.items():
        lvl, k = key.split(":")
        log(f"  {lvl:>4} {k:>2} {w['e_in']:7.0f} {w['e_out']:7.0f} {w['i_in']:8.0f} {w['i_out']:8.0f}   {w['e_in']/w['e_out']:6.2f}      "
            f"{w['i_in']/w['i_out']:6.2f}        {w['rho_low_worst']:7.3f} (mean {w['rho_low_mean']:.3f})       {w['rho_high_worst']:7.3f} (mean {w['rho_high_mean']:.3f})     {w['offsets_with_window']}/{w['n_offsets']}")

    # ---- absolute scale: what the bump wedge receives at nominal intermediate gain gamma (Hz per mV), tau_syn
    tau = p.tau_syn / 1000.0
    res["gamma_nominal_hz_per_mV"] = gamma_nominal
    scale_2 = gamma_nominal * tau * tau   # mV of mean depolarisation per Hz of presynaptic rate per mV^2 of two-step weight
    res["mV_per_Hz_per_mV2"] = scale_2
    abs_rows = {}
    for key in ("16:2", "16:3", "16:4", "8:1", "8:2"):
        w = windows[key]
        abs_rows[key] = dict(exc_mV_per_Hz_in=w["e_in"] * scale_2, inh_mV_per_Hz_in=w["i_in"] * scale_2,
                             exc_mV_per_Hz_out=w["e_out"] * scale_2, inh_mV_per_Hz_out=w["i_out"] * scale_2,
                             direct_mV_per_Hz_in=float(np.mean([M16["direct"][:, :].sum(axis=1).mean()])) * tau)
    res["absolute_at_gamma"] = abs_rows
    log(f"\nabsolute (gamma = {gamma_nominal} Hz/mV, tau_syn {p.tau_syn} ms): mean depolarisation of an EPG per Hz of bump rate (two-step, gains x1)")
    for key, r in abs_rows.items():
        log(f"  {key}: inside exc {r['exc_mV_per_Hz_in']:+.3f} inh {r['inh_mV_per_Hz_in']:+.3f} mV/Hz; outside exc {r['exc_mV_per_Hz_out']:+.3f} inh {r['inh_mV_per_Hz_out']:+.3f} mV/Hz")

    # Delta7 -> PEN clamp: input to a PEN from the bump directly vs through Delta7, per Hz of bump rate at gamma
    Apen_e = block(pen, epg); Apen_d = block(pen, d7); Ad_e = block(d7, epg)
    direct_pen = aggregate(Apen_e, pen["pos"], epg["pos"], 16, bin16)          # mV per spike, [pen wedge, epg wedge]
    via_d7 = aggregate(Apen_d @ Ad_e, pen["pos"], epg["pos"], 16, bin16)        # mV^2
    via_ring = aggregate(block(pen, ring) @ block(ring, epg), pen["pos"], epg["pos"], 16, bin16)
    dd = ring_dist(np.arange(16), np.arange(16))
    res["PEN_drive_profile"] = dict(distance_wedges=list(range(9)),
                                    direct_EPG_to_PEN_mV=[float(direct_pen[dd == j].mean()) for j in range(9)],
                                    via_Delta7_mV2=[float(via_d7[dd == j].mean()) for j in range(9)],
                                    via_Delta7_at_gamma_mV=[float(via_d7[dd == j].mean() * gamma_nominal * tau) for j in range(9)],
                                    via_Ring_mV2=[float(via_ring[dd == j].mean()) for j in range(9)],
                                    via_Ring_at_gamma_mV=[float(via_ring[dd == j].mean() * gamma_nominal * tau) for j in range(9)])
    log("PEN drive from an EPG wedge vs distance (PEN wedge = its glomerulus): direct (mV/spike) and via Delta7 (mV, at gamma):")
    log("   direct     :", np.round(res["PEN_drive_profile"]["direct_EPG_to_PEN_mV"], 1).tolist())
    log("   via Delta7 :", np.round(res["PEN_drive_profile"]["via_Delta7_at_gamma_mV"], 1).tolist())
    log("   via ER/ExR :", np.round(res["PEN_drive_profile"]["via_Ring_at_gamma_mV"], 1).tolist())
    with open(out_dir / "cx_wedge.json", "w") as f:
        json.dump(res, f, indent=1)
    np.savez(out_dir / "cx_wedge_matrices.npz", **{f"K_{k}": v for k, v in K.items()}, **{f"M16_{k}": v for k, v in M16.items()},
             **{f"M8_{k}": v for k, v in M8.items()}, epg_pos=epg["pos"], epg_body=epg["body"])
    return res, cells, c


def gained_blocks(c, cells, gE, gD, delta7_pen=True, gR=1.0, with_ring=True):
    """Effective matrix among the compass cells (mV per spike) with the gains applied as Brain would apply
    type_path_gain (before the fan-in scale, which is 0.9806-1.00 for these cells shipped (one EPG above the 5,000 reference), 0.71-1.00 under the receptor tier and 0.68-1.00 with the damping off (skeptic pass on compass_dc_balance.md) up to gains of ~x1.7 on EPG)."""
    tpg = list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", gE), (r"^PEN_", r"^EPG$", gE),
                                                (r"^EPG$", r"^PEG$", gE), (r"^PEG$", r"^EPG$", gE),
                                                (r"^Delta7$", r"^(EPG$|PEN_)" if delta7_pen else r"^EPG$", gD),
                                                (RING_RE, r"^(EPG$|PEN_|PEG$)", gR)]
    p = brain.LIFParams(type_path_gain=tpg)
    A, scale, tot = effective_weights(c, p)
    idx = np.concatenate([cells[k]["idx"] for k in (("EPG", "PEN", "PEG", "Delta7", "Ring") if with_ring else ("EPG", "PEN", "PEG", "Delta7"))])
    return A[idx][:, idx].toarray().astype(np.float64), idx, p, scale[idx]


def lif_fi(u, p=None, sigma=2.0):
    """Mean firing rate (Hz) of the LIF at mean input u (mV above rest), smoothed over Gaussian input fluctuations
    of sigma mV (Poisson input); deterministic f-I: 1 / (t_ref + tau_m ln(u / (u - theta)))."""
    p = p or brain.LIFParams()
    theta = p.v_th - p.v_rest
    xs, ws = np.polynomial.hermite_e.hermegauss(15)
    ws = ws / ws.sum()
    out = np.zeros_like(u, dtype=float)
    for x, w in zip(xs, ws):
        uu = u + sigma * x
        m = uu > theta + 1e-6
        f = np.zeros_like(u, dtype=float)
        f[m] = 1000.0 / (p.t_ref + p.tau_m * np.log(uu[m] / (uu[m] - theta)))
        out += w * f
    return out


def rate_model(c, cells, gE, gD, delta7_pen=True, background_hz=10.0, pulse_hz=40.0, start_wedge=0, width=4,
               sigma=2.0, iters=4000, alpha=0.05, gR=1.0, with_ring=True):
    """Threshold-linear mean-field fixed point of the compass circuit: r = f(tau_syn * A r) with the EPG's forced
    Poisson background / pulse added to the intrinsic rate. Returns the state after the pulse and after release."""
    A, idx, p, scale = gained_blocks(c, cells, gE, gD, delta7_pen, gR, with_ring)
    nE = len(cells["EPG"]["idx"]); n = len(idx)
    nP, nG, nD = len(cells["PEN"]["idx"]), len(cells["PEG"]["idx"]), len(cells["Delta7"]["idx"])
    tau = p.tau_syn / 1000.0
    wedge_of = np.asarray(np.round(cells["EPG"]["pos"]), int) % 16
    inside = np.isin(wedge_of, [(start_wedge + j) % 16 for j in range(width)])
    forced = np.zeros(n); forced[:nE] = background_hz
    r = forced.copy()

    def relax(forced, r):
        for _ in range(iters):
            u = tau * (A @ r)
            target = lif_fi(u, p, sigma) + forced
            r = (1 - alpha) * r + alpha * target
        return r

    r_bg = relax(forced, r)
    f_pulse = forced.copy(); f_pulse[:nE][inside] += pulse_hz
    r_pulse = relax(f_pulse, r_bg)
    r_after = relax(forced, r_pulse)

    def summ(r):
        e = r[:nE]
        return dict(epg_in=float(e[inside].mean()), epg_out=float(e[~inside].mean()),
                    epg_in_min=float(e[inside].min()), epg_out_max=float(e[~inside].max()),
                    pen=float(r[nE:nE + nP].mean()), peg=float(r[nE + nP:nE + nP + nG].mean()),
                    delta7=float(r[nE + nP + nG:nE + nP + nG + nD].mean()),
                    ring=float(r[nE + nP + nG + nD:].mean()) if with_ring else 0.0,
                    profile=[float(e[wedge_of == w].mean()) for w in range(16)])
    return dict(gE=gE, gD=gD, gR=gR, delta7_pen=delta7_pen, with_ring=with_ring,
                background=summ(r_bg), pulse=summ(r_pulse), after=summ(r_after))


def rate_grid(c, cells, gEs, gDs, delta7_pen, log=print, **kw):
    rows = []
    bg = kw.get("background_hz", 10.0)
    log(f"threshold-linear rate model (Delta7 -> PEN {'x gD' if delta7_pen else 'x1'}; ER/ExR {'included, gain ' + str(kw.get('gR', 1.0)) if kw.get('with_ring', True) else 'excluded'}): EPG in / out after release "
        f"(background-state ring / PEN / Delta7 in brackets; BUMP = in > 2 x out and in > bg + 5 Hz)")
    for gE in gEs:
        for gD in gDs:
            rm = rate_model(c, cells, gE, gD, delta7_pen, **kw)
            a, b = rm["after"], rm["background"]
            bump = a["epg_in"] > 2 * a["epg_out"] and a["epg_in"] > bg + 5
            rm["bump"] = bool(bump)
            rows.append(rm)
            log(f"  gE {gE:<4} gD {gD:<4}: after in {a['epg_in']:6.1f} out {a['epg_out']:6.1f} (max {a['epg_out_max']:6.1f}) "
                f"PEN {a['pen']:5.1f} D7 {a['delta7']:5.1f} ER/ExR {a['ring']:5.1f}  [bg ring {b['epg_in']:5.1f} PEN {b['pen']:5.1f} D7 {b['delta7']:5.1f}]  {'BUMP' if bump else ''}")
    return rows


def simulate(c, cells, gains, seconds=5.0, pulse_s=2.0, background_hz=10.0, pulse_hz=40.0, start_wedge=0, width=4, seed=0,
             thresh_hz=22.0, cuda_graphs=True, delta7_pen=True, gR=1.0, verbose=True,
             receptor_model=None, receptor_net_rule="class", nt_override=None,
             lif_overrides=None, ledger=False, arm=None, block=None, device=None, ledger_npz=None, hold_edges=None,
             edge_gains=None, preset="raw", instrument_specs=None, turn_deg_s=None, turn_window=(0.5, 3.5), settle_s=1.0,
             neural_stimuli=None):
    """FlyBrain on the full connectome; drive `width` contiguous wedges (of 16) of the EPG ring from `start_wedge`;
    report persistence and confinement after the pulse. receptor_model / receptor_net_rule thread
    LIFParams.receptor_model (the optional receptor-expression sign stage); nt_override is recorded in the row (the
    connectome `c` must already carry it, see load_connectome). GLNO (the 4 LAL-NO1 cells, 19 % of PEN's raw input,
    transmitter unknown) is reported alongside PEN / Delta7.

    Thread 5A additions (docs/audits/compass_ring_mechanism.md; every default keeps the previous behaviour):
    lif_overrides = extra LIFParams fields ({'conn_cap': 0, 'same_type_gain': 1, ...}; `--lif KEY=VALUE`), recorded in
    the row; ledger = record the EPG per 10 ms frame (probe_unitary's loop) and score it with
    probe_compass_room.bump_frames into the ledger rows compass.EPG.bump_survival_s / bump_rate_hz / bump_width_wedges
    (+ frac_confined_post), with `common.provenance` and the realised device in the row and the per-frame record in
    `ledger_npz` (survival = end of the last confined post-pulse frame minus the pulse end, so a bump confined through
    the whole free period scores exactly `seconds`); arm / block are labels carried into the row; device is passed to
    FlyBrain.

    Thread 6A (docs/audits/compass_dc_balance.md): hold_edges = [(pre_re, post_re, factor), ...] from
    `parse_hold_edges` -- an `edges`-kind HOLD appended to type_path_gain, a LABELLED COUNTERFACTUAL. Default None
    leaves the gain list exactly as before (bit-identical shipped path).

    Thread 6B (docs/audits/compass_local_recurrence.md): edge_gains = [(pre_re, post_re, factor), ...] from
    `parse_edge_gains`, appended to type_path_gain after the holds -- a per-type gain, a LABELLED INSTRUMENT. Default
    None. The ledger record additionally carries the per-type ring rates (ExR6 / ER6 / ER4m) and EPGt as `g__*`
    groups, and the per-cell rates of the small compass groups (`cells__*`) with their per-cell maximum in `metrics`
    (`<group>_cell_max_pre/during/post`), so `silent` in the INTERP.md 10 sense is decidable from the run.

    Round 7 (docs/audits/compass_velocity_route.md): preset = 'raw' (default; FlyBrain(preset='raw'), byte-identical)
    or 'instrumented'; instrument_specs = the `--instrument` strings (flyverse.instruments.parse_instrument); under
    'instrumented' the holds / relabel / edge gains are ALSO recorded as instruments (build_instruments). turn_deg_s
    (ledger only, default None) = a prescribed signed yaw rate over turn_window = (start, end) seconds after the pulse
    end, fed through FlyBrain.proprioception(yaw_rate=) when an afferent instrument is attached, recorded either way.
    settle_s (default 1.0, the 6A value) is the background settle before the pulse. The ledger record grows the
    per-side groups PEN / GLNO / DNa02 / PS196_b (+ the afferents when attached) and `metrics` the round-7 measures
    `<G>_LR_hz` (mean L - R over the turn window; over the post window without a turn), `<G>_LR_hz_rest` (after the
    turn) and `bump_follow_*`."""
    from flyverse.fly import FlyBrain
    log = print if verbose else (lambda *a, **k: None)
    if turn_deg_s is not None and not ledger:
        raise ValueError("--turn needs --ledger (the per-frame loop feeds the yaw)")
    if neural_stimuli is not None and not ledger:
        raise ValueError("scheduled neural stimuli need the ledger loop")
    if turn_deg_s is not None and (not np.isfinite(turn_deg_s)):
        raise ValueError("--turn must be finite (deg/s; positive = a left turn)")
    if not np.isfinite(settle_s) or settle_s < 0:
        raise ValueError("settle_s must be finite and >= 0")
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}")
    stimuli = prepare_neural_stimuli(c, neural_stimuli, settle_s + pulse_s + seconds)
    epg = cells["EPG"]
    wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    inside = np.isin(wedge_of, [(start_wedge + j) % 16 for j in range(width)])
    idx_epg = epg["idx"]
    pen_idx, d7_idx, peg_idx, ring_idx = cells["PEN"]["idx"], cells["Delta7"]["idx"], cells["PEG"]["idx"], cells["Ring"]["idx"]
    ty_all = c.neurons.type.fillna("").to_numpy()
    glno_idx = np.flatnonzero(ty_all == "GLNO")
    epgt_idx = cells["EPGt"]["idx"]
    ring_type_idx = {t: np.flatnonzero(ty_all == t) for t in ("ExR6", "ER6", "ER4m")}     # 6B: the per-type ring rates
    others = np.setdiff1d(np.arange(c.n), np.concatenate([idx_epg, pen_idx, d7_idx, peg_idx, cells["EPGt"]["idx"]]))
    out = []
    for gE, gD in gains:
        tpg = list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", gE), (r"^PEN_", r"^EPG$", gE),
                                                    (r"^EPG$", r"^PEG$", gE), (r"^PEG$", r"^EPG$", gE),
                                                    (r"^Delta7$", r"^(EPG$|PEN_)" if delta7_pen else r"^EPG$", gD),
                                                    (RING_RE, r"^(EPG$|PEN_|PEG$)", gR)]
        tpg += [(pre, post, float(f)) for pre, post, f in (hold_edges or [])]   # 6A: the `edges`-kind hold, last (a factor 0 is order-free)
        tpg += [(pre, post, float(f)) for pre, post, f in (edge_gains or [])]   # 6B: per-type gains (before same_type_gain by stage order)
        params = brain.LIFParams(adapt_by_type={COMPASS_RE: 0.0}, type_path_gain=tpg,
                                 receptor_model=receptor_model, receptor_net_rule=receptor_net_rule, **(lif_overrides or {}))
        t0 = time.time()
        # round 7: the instrument list (empty under 'raw', whatever the flags) rides on the constructor and into provenance
        insts = build_instruments(c, instrument_specs, preset, hold_edges=hold_edges, nt_override=nt_override,
                                  edge_gains=edge_gains, same_type_gain=params.same_type_gain)
        fb = FlyBrain(c, lif_params=params, seed=seed, cuda_graphs=cuda_graphs, device=device, preset=preset, instruments=insts)
        afferent = next((i for i in insts if getattr(i, "name", None) == "sided_turn_afferent"), None)
        # background: FlyBrain.stimulate pulses expire, so hold the background as a long pulse (as the grid did)
        total_ms = (settle_s + pulse_s + seconds) * 1000
        fb.stimulate(idx_epg, background_hz, total_ms + 100)

        def sample(tag):
            r = fb.brain.rates(idx_epg).copy()
            d = {f"{tag}_in_mean": float(r[inside].mean()), f"{tag}_out_mean": float(r[~inside].mean()),
                 f"{tag}_in_above": int((r[inside] > thresh_hz).sum()), f"{tag}_out_above": int((r[~inside] > thresh_hz).sum()),
                 f"{tag}_pen": float(fb.brain.mean_rate(pen_idx)), f"{tag}_delta7": float(fb.brain.mean_rate(d7_idx)),
                 f"{tag}_peg": float(fb.brain.mean_rate(peg_idx)), f"{tag}_rest": float(fb.brain.mean_rate(others)),
                 f"{tag}_ring": float(fb.brain.mean_rate(ring_idx)),
                 f"{tag}_glno": float(fb.brain.mean_rate(glno_idx)),
                 f"{tag}_glno_cells": [float(x) for x in fb.brain.rates(glno_idx)],
                 f"{tag}_wedge_profile": [float(r[wedge_of == w].mean()) for w in range(16)]}
            ang = 2 * np.pi * wedge_of / 16
            z = np.sum(r * np.exp(1j * ang)) / max(r.sum(), 1e-9)
            d[f"{tag}_vector_strength"] = float(np.abs(z)); d[f"{tag}_centre_wedge"] = float((np.angle(z) % (2 * np.pi)) / (2 * np.pi) * 16)
            return d

        row = dict(gE=gE, gD=gD, gR=gR, delta7_pen=delta7_pen, background_hz=background_hz, pulse_hz=pulse_hz, start_wedge=start_wedge,
                   width=width, seed=seed, n_in=int(inside.sum()), n_out=int((~inside).sum()),
                   receptor_model=receptor_model, receptor_net_rule=receptor_net_rule if receptor_model else None,
                   nt_override=dict(nt_override or {}), n_glno=int(len(glno_idx)),
                   glno_nt=sorted(set(c.neurons.nt.to_numpy()[glno_idx].tolist())),
                   lif_overrides=dict(lif_overrides or {}), arm=arm, block=block, device=str(fb.brain.device),
                   hold_edges=[[p_, q_, float(f_)] for p_, q_, f_ in (hold_edges or [])],
                   hold_edges_resolved=hold_edge_counts(c, hold_edges) if hold_edges else [],
                   edge_gains=[[p_, q_, float(f_)] for p_, q_, f_ in (edge_gains or [])],
                   edge_gains_resolved=hold_edge_counts(c, edge_gains, same_type_gain=params.same_type_gain) if edge_gains else [],
                   # round 7: the preset and the instrument list (names here; every describe() in provenance.instruments)
                   preset=preset, instruments=[i.name for i in insts], instrument_specs=list(instrument_specs or []),
                   turn_deg_s=(float(turn_deg_s) if turn_deg_s is not None else None),
                   turn_window_s=[float(turn_window[0]), float(turn_window[1])], settle_s=float(settle_s))
        marks = (0.5, 1.0, 2.0, 3.0, 5.0)
        if not ledger:
            fb.step(settle_s * 1000)                                    # 1 s settle on background (settle_s default 1.0)
            row.update(sample("pre"))
            fb.stimulate(idx_epg[inside], background_hz + pulse_hz, pulse_s * 1000)
            fb.step(pulse_s * 1000)
            row.update(sample("during"))
            t = 0.0
            for mark in marks:
                fb.step((mark - t) * 1000); t = mark
                row.update(sample(f"t{mark}"))
        else:
            # per-10-ms-frame record of the EPG (probe_unitary's loop), the same samples at the same instants
            import probe_compass_room as pcr
            from flyverse.interp import common
            frames = int(round(total_ms / 10.0))
            rec_epg = np.zeros((frames, len(idx_epg)), np.float32)
            grp = {"PEN": pen_idx, "Delta7": d7_idx, "PEG": peg_idx, "Ring": ring_idx, "GLNO": glno_idx, "rest": others,
                   # 6B: per-type ring rates and EPGt (recording more groups does not touch the simulation)
                   "EPGt": epgt_idx, "ExR6": ring_type_idx["ExR6"], "ER6": ring_type_idx["ER6"], "ER4m": ring_type_idx["ER4m"]}
            # round 7: per-side groups (somaSide) of the velocity route -- the L-R measures; types looked up, never assumed
            sides = side_groups(c, {"PEN": ("PEN_a(PEN1)", "PEN_b(PEN2)"), "GLNO": "GLNO", "DNa02": "DNa02", "PS196b": "PS196_b"})
            if afferent is not None:
                sides["AFF_L"], sides["AFF_R"] = afferent.idx[afferent.side > 0], afferent.idx[afferent.side < 0]
            grp.update(sides)
            rec_grp = {k: np.zeros(frames, np.float32) for k in grp}
            # the prescribed turn (round 7): on over [t_on, t_off) after the pulse end; fed only when an afferent is attached
            t_on = settle_s + pulse_s + turn_window[0]; t_off = settle_s + pulse_s + turn_window[1]
            yaw_deg = np.zeros(frames, np.float32)
            feed_turn = turn_deg_s is not None and getattr(fb, "proprioception_sense", None) is not None
            # 6B: per-cell rates of the small groups (127 cells), for the per-cell maximum INTERP.md 10's `silent` needs
            cell_grp = {k: grp[k] for k in ("PEN", "Delta7", "PEG", "GLNO", "EPGt", "ExR6", "ER6", "ER4m")}
            rec_cells = {k: np.zeros((frames, len(v)), np.float32) for k, v in cell_grp.items()}
            tt = np.arange(frames) * 0.01                                 # frame start; rates are read at the frame end
            t_end = tt + 0.01
            sample_at = {int(round(settle_s / 0.01)) - 1: "pre", int(round((settle_s + pulse_s) / 0.01)) - 1: "during"}
            for mark in marks:
                k = int(round((settle_s + pulse_s + mark) / 0.01)) - 1
                if 0 <= k < frames:
                    sample_at[k] = f"t{mark}"
            pulsed = False
            stimuli_applied = []
            stimulus_hz = np.zeros((frames, len(stimuli)), np.float32)
            for k in range(frames):
                if not pulsed and tt[k] >= settle_s - 1e-9:
                    fb.stimulate(idx_epg[inside], background_hz + pulse_hz, pulse_s * 1000)
                    pulsed = True
                for j, event in enumerate(stimuli):
                    if k == int(round(event["start_s"]/.01)):
                        fb.stimulate(np.asarray(event["idx"], dtype=int), event["poisson_hz"], event["duration_s"]*1000)
                        stimuli_applied.append(dict(name=event["name"], start_s=float(tt[k])))
                    if event["start_s"] - 1e-9 <= tt[k] < event["start_s"] + event["duration_s"] - 1e-9:
                        stimulus_hz[k, j] = event["poisson_hz"]
                if turn_deg_s is not None:
                    yaw_deg[k] = turn_deg_s if (tt[k] >= t_on - 1e-9 and tt[k] < t_off - 1e-9) else 0.0
                    if feed_turn and (k == 0 or yaw_deg[k] != yaw_deg[k - 1]):
                        # the body channel that already exists: leg / haltere MN rates 0 (no body), airborne False, the yaw
                        fb.proprioception(0.0, 0.0, 0.0, False, yaw_rate=float(np.deg2rad(yaw_deg[k])))
                fb.step(10.0)
                rec_epg[k] = fb.brain.rates(idx_epg)
                for g, gi in grp.items():
                    rec_grp[g][k] = fb.brain.mean_rate(gi)
                for g, gi in cell_grp.items():
                    if len(gi):
                        rec_cells[g][k] = fb.brain.rates(gi)
                if k in sample_at:
                    row.update(sample(sample_at[k]))
            for mark in marks:                                            # a free period shorter than 5 s: fill the missing marks
                for key in ("in_mean", "out_mean", "in_above", "out_above", "pen", "delta7", "peg", "rest", "ring", "glno", "glno_cells",
                            "wedge_profile", "vector_strength", "centre_wedge"):
                    row.setdefault(f"t{mark}_{key}", None)
            b = pcr.bump_frames(rec_epg, wedge_of)
            pre = tt < settle_s; during = (tt >= settle_s) & (tt < settle_s + pulse_s); post = tt >= settle_s + pulse_s
            conf = b["confined"]

            def mean_if(x, m):
                mm = m & np.isfinite(x)
                return float(np.mean(x[mm])) if mm.any() else float("nan")
            last = np.flatnonzero(conf & post)
            survival = float(t_end[last[-1]] - (settle_s + pulse_s)) if len(last) else 0.0
            m = {"frac_confined_pre": float(conf[pre].mean()), "frac_confined_during": float(conf[during].mean()),
                 "frac_confined_post": float(conf[post].mean()), "survival_s": survival,
                 "bump_hz_post": mean_if(b["bump_hz"], post & conf), "out_hz_post": mean_if(b["out_hz"], post & conf),
                 "width_half_post": mean_if(b["width_half"], post & conf), "width_22_post": mean_if(b["width_22"], post & conf),
                 "vs_post_all": mean_if(b["vs"], post), "vs_post_confined": mean_if(b["vs"], post & conf),
                 "epg_in_mean_post": float(rec_epg[post][:, inside].mean()), "epg_out_mean_post": float(rec_epg[post][:, ~inside].mean()),
                 "epg_in_mean_during": float(rec_epg[during][:, inside].mean()), "epg_out_mean_during": float(rec_epg[during][:, ~inside].mean()),
                 "epg_max_post": float(rec_epg[post].max()), "epg_mean_pre": float(rec_epg[pre].mean()),
                 **{f"{g}_mean_post": float(v[post].mean()) for g, v in rec_grp.items()},
                 **{f"{g}_mean_during": float(v[during].mean()) for g, v in rec_grp.items()},
                 **{f"{g}_mean_pre": float(v[pre].mean()) for g, v in rec_grp.items()},
                 # 6B: the per-cell maximum of the window-mean rate (INTERP.md 10: silent = max rate per cell < 0.5 Hz)
                 **{f"{g}_cell_max_{w}": (float(v[msk].mean(axis=0).max()) if v.shape[1] else float("nan"))
                    for g, v in rec_cells.items() for w, msk in (("pre", pre), ("during", during), ("post", post))},
                 **{f"{g}_n_cells": int(v.shape[1]) for g, v in rec_cells.items()},
                 "in_above_end": int((rec_epg[-1][inside] > thresh_hz).sum()), "out_above_end": int((rec_epg[-1][~inside] > thresh_hz).sum()),
                 "frames": int(frames), "frame_s": 0.01}
            # round 7: the L-R measures over the turn window (the post window without a turn) and after the turn
            turn_m = ((tt >= t_on - 1e-9) & (tt < t_off - 1e-9)) if turn_deg_s is not None else post
            rest_m = (tt >= t_off - 1e-9) if turn_deg_s is not None else np.zeros(frames, bool)
            for g in ("PEN", "GLNO", "DNa02", "PS196b", "AFF"):
                if f"{g}_L" not in rec_grp:
                    continue
                L, R = rec_grp[f"{g}_L"], rec_grp[f"{g}_R"]
                m[f"{g}_L_hz_turn"], m[f"{g}_R_hz_turn"] = mean_if(L, turn_m), mean_if(R, turn_m)
                m[f"{g}_LR_hz"] = mean_if(L - R, turn_m)
                m[f"{g}_LR_hz_rest"] = mean_if(L - R, rest_m)
                m[f"{g}_n_L"], m[f"{g}_n_R"] = int(len(grp[f"{g}_L"])), int(len(grp[f"{g}_R"]))
            m.update(bump_follow(b["centre"], conf, tt, t_on, t_off, turn_deg_s))
            m["turn_deg_s"] = float(turn_deg_s) if turn_deg_s is not None else None
            m["turn_on_s"], m["turn_off_s"] = (float(t_on), float(t_off)) if turn_deg_s is not None else (None, None)
            m["turn_fed"] = bool(feed_turn)
            led = {"compass.EPG.bump_survival_s": {"value": survival, "op": ">=", "bound": 5.0, "status": "PASS" if survival >= 5.0 else "FAIL"}}
            for key, val, lo, hi in (("compass.EPG.bump_rate_hz", m["bump_hz_post"], 5.0, 60.0), ("compass.EPG.bump_width_wedges", m["width_half_post"], 2.5, 5.0)):
                st = "NOT_APPLICABLE" if survival < 5.0 else ("PASS" if (np.isfinite(val) and lo <= val <= hi) else "FAIL")
                led[key] = {"value": val, "op": "range", "bound": [lo, hi], "status": st, "requires": "compass.EPG.bump_survival_s"}
            row["metrics"] = m; row["ledger"] = led
            row["wedge_profile_post"] = [float(rec_epg[post][:, wedge_of == w].mean()) for w in range(16)]
            row["wedge_profile_end"] = [float(rec_epg[-1][wedge_of == w].mean()) for w in range(16)]
            row["provenance"] = common.to_jsonable(common.provenance(
                c, fb.brain.p, getattr(fb.optic, "p", None), fb=fb, device=device, seeds=[seed],
                stimulus={"protocol": "cx_wedge.simulate (ledger)", "params": {"gE": gE, "gD": gD, "gR": gR, "delta7_pen": delta7_pen,
                          "background_hz": background_hz, "pulse_hz": pulse_hz, "pulse_s": pulse_s, "seconds_after": seconds, "width": width,
                          "start_wedge": start_wedge, "settle_s": settle_s, "nt_override": dict(nt_override or {}),
                          "lif_overrides": dict(lif_overrides or {}), "receptor_model": receptor_model, "receptor_net_rule": receptor_net_rule,
                          "hold_edges": row["hold_edges"], "hold_edges_resolved": row["hold_edges_resolved"],
                          "edge_gains": row["edge_gains"], "edge_gains_resolved": row["edge_gains_resolved"],
                          "preset": preset, "instruments": row["instruments"], "instrument_specs": row["instrument_specs"],
                          "turn_deg_s": row["turn_deg_s"], "turn_window_s": row["turn_window_s"], "turn_fed": bool(feed_turn)},
                          "control": "arm shipped (gE 1 / gD 1, LIFParams() on the shipped cache)"}))
            if neural_stimuli is not None:
                row["neural_stimuli"], row["neural_stimuli_applied"] = stimuli, stimuli_applied
                row["provenance"]["stimulus"]["params"]["neural_stimuli"] = stimuli
            if ledger_npz:
                Path(ledger_npz).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(ledger_npz, t=tt, epg=rec_epg, wedge_of=wedge_of, inside=inside, yaw_deg_s=yaw_deg,
                                    **{f"g__{k}": v for k, v in rec_grp.items()},
                                    **{f"cells__{k}": v for k, v in rec_cells.items()},
                                    **({"neural_stimulus_hz": stimulus_hz} if neural_stimuli is not None else {}))
                row["ledger_npz"] = str(ledger_npz)
            log(f"ledger: confined pre {m['frac_confined_pre']:.2f} post {m['frac_confined_post']:.2f}, survival {survival:.2f} s, bump {m['bump_hz_post']:.1f} Hz, "
                f"width {m['width_half_post']:.1f}; {[(k, v['status']) for k, v in led.items()]}; device {row['device']}")
            log(f"round 7: preset {preset}, instruments {row['instruments']}, turn {row['turn_deg_s']} deg/s over {row['turn_window_s']} s after the pulse "
                f"(fed {feed_turn}); L-R Hz over the turn window: " + ", ".join(f"{g} {m[f'{g}_LR_hz']:+.3f}" for g in ("PEN", "GLNO", "DNa02", "PS196b", "AFF") if f"{g}_LR_hz" in m)
                + f"; bump_follow {m['bump_follow_wedges_per_s']:+.3f} w/s (ideal {m['bump_follow_ideal_wedges_per_s']:.2f}, confined {m['bump_follow_confined_frac']:.2f})")
        row["wall_s"] = round(time.time() - t0, 1)

        def fmt(tag):
            return (f"in {row[f'{tag}_in_mean']:.1f} ({row[f'{tag}_in_above']}/{row['n_in']}) out {row[f'{tag}_out_mean']:.1f} "
                    f"({row[f'{tag}_out_above']}/{row['n_out']}) PEN {row[f'{tag}_pen']:.1f} D7 {row[f'{tag}_delta7']:.1f} GLNO {row[f'{tag}_glno']:.1f} R {row[f'{tag}_ring']:.1f} vs {row[f'{tag}_vector_strength']:.2f}")
        have = [m for m in marks if row.get(f"t{m}_in_mean") is not None]
        hold_txt = "; ".join("{} -> {} x{:g} ({} entries, {:.0f} syn)".format(h["pre"], h["post"], h["factor"], h["n_entries"], h["synapses"])
                             for h in row["hold_edges_resolved"]) or "none"
        hold_txt += "; edge gains " + ("; ".join("{} -> {} x{:g} ({} entries, {} same-type at x{:g})".format(
            h["pre"], h["post"], h["factor"], h["n_entries"], h.get("n_entries_same_type", 0), h.get("effective_factor_same_type", h["factor"]))
            for h in row["edge_gains_resolved"]) or "none")
        log(f"gE {gE} gD {gD} gR {gR} (D7->PEN {'x gD' if delta7_pen else 'x1'}, bg {background_hz} Hz, width {width}, seed {seed}, "
            f"receptor {receptor_model or 'off'}{'/' + receptor_net_rule if receptor_model else ''}, GLNO nt {row['glno_nt']}, "
            f"lif {row['lif_overrides']}, hold {hold_txt}, arm {arm}): pre {fmt('pre')}; "
            f"during {fmt('during')}; " + "; ".join(f"{m}s {fmt(f't{m}')}" for m in have)
            + (f"; PEG {row['t5.0_peg']:.1f} rest {row['t5.0_rest']:.2f} Hz" if row.get("t5.0_peg") is not None else "") + f"; {row['wall_s']} s")
        out.append(row)
        del fb
        import torch; torch.cuda.empty_cache()
    return out


def plot_sim(sim_json: Path, out_png: Path, keys=None):
    """Wedge profiles (EPG Hz per wedge, ring order) before, during and after the pulse for every row in the JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.load(open(sim_json))
    tags = ["pre", "during", "t0.5", "t1.0", "t2.0", "t3.0", "t5.0"]
    rows = [r for r in rows if all(f"{t}_wedge_profile" in r for t in tags) and (keys is None or keys(r))]
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.6 * n + 1), sharex=True, squeeze=False)
    for ax, r in zip(axes[:, 0], rows):
        M = np.array([r[f"{t}_wedge_profile"] for t in tags])
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=max(60, M.max()))
        ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=6)
        w0, wd = r.get("start_wedge", 0), r.get("width", 4)
        ax.axvline(w0 - 0.5, color="cyan", lw=0.8); ax.axvline(w0 + wd - 0.5, color="cyan", lw=0.8)
        ax.set_title(f"gE {r['gE']} gD {r['gD']} gR {r.get('gR', 1.0)} D7->PEN {'x gD' if r.get('delta7_pen', True) else 'x1'} "
                     f"bg {r.get('background_hz', 10)} Hz width {wd} seed {r.get('seed', 0)}: 5 s after -> in {r['t5.0_in_mean']:.0f} Hz "
                     f"({r['t5.0_in_above']}/{r['n_in']} > 22 Hz), out {r['t5.0_out_mean']:.0f} Hz ({r['t5.0_out_above']}/{r['n_out']})", fontsize=7)
        ax.set_xticks(range(16)); ax.set_xticklabels(RING16, fontsize=6)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4, label="EPG Hz")
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(AUDIT_DIR))
    ap.add_argument("--sim", nargs="*", default=None, help="gE:gD pairs, e.g. 1:1 1.5:2")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--background", type=float, default=10.0)
    ap.add_argument("--pulse-hz", type=float, default=40.0)
    ap.add_argument("--start-wedge", type=int, default=0)
    ap.add_argument("--width", type=int, default=4, help="driven wedges (of 16); 4 = two tiles = 90 deg")
    ap.add_argument("--no-structure", action="store_true", help="skip the PNG / JSON structural outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-delta7-pen", action="store_true", help="apply gD to Delta7 -> EPG only (not Delta7 -> PEN)")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--sim-out", default=None, help="JSON file for the simulation rows")
    ap.add_argument("--rate-grid", nargs=2, default=None, metavar=("GE", "GD"),
                    help="threshold-linear rate-model grid, comma-separated gains, e.g. --rate-grid 0.8,1,1.5 1,4,15")
    ap.add_argument("--rate-out", default=None)
    ap.add_argument("--ring-gain", type=float, default=1.0, help="gain on ER/ExR -> EPG/PEN/PEG (rate model and simulation)")
    ap.add_argument("--no-ring", action="store_true", help="rate model without the ER/ExR cells")
    ap.add_argument("--plot-sim", default=None, help="draw wedge profiles from this simulation JSON (no structure / sim run)")
    ap.add_argument("--nt-override", action="append", default=None, metavar="TYPE=nt",
                    help="repeatable; compile the connectome with connectome.TYPE_NT_OVERRIDE extended by TYPE=nt (applied to the "
                         "type's sign-0 cells) into the scratch cache out/cache_<hash>/, e.g. --nt-override GLNO=glutamate")
    ap.add_argument("--scratch-cache", action="store_true",
                    help="compile into out/cache_<hash>/ even without --nt-override (a cluster whose shared cache predates TYPE_NT_OVERRIDE)")
    ap.add_argument("--receptor-model", default="off", choices=["off", "sign", "sign+gain", "full", "shipped"],
                    help="LIFParams.receptor_model for the simulation (default off = the presynaptic NT_SIGN rule; 'shipped' = LIFParams()'s "
                         "receptor_model AND receptor_net_rule, i.e. the shipped default)")
    ap.add_argument("--receptor-net-rule", default="class", choices=list(connectome.RECEPTOR_NET_RULES))
    # thread 5A (docs/audits/compass_ring_mechanism.md): new flags only, every default unchanged
    ap.add_argument("--lif", action="append", default=None, metavar="KEY=VALUE",
                    help="repeatable LIFParams override for the simulation (JSON / literal values), e.g. --lif conn_cap=0 --lif same_type_gain=1")
    ap.add_argument("--ledger", action="store_true",
                    help="record the EPG per 10 ms frame and score the ledger rows compass.EPG.* (probe_compass_room.bump_frames) with provenance")
    ap.add_argument("--ledger-npz", default=None, help="--ledger: per-frame record path (default <sim-out stem>_gE<gE>_gD<gD>_s<seed>.npz)")
    ap.add_argument("--arm", default=None, help="label recorded in every row")
    ap.add_argument("--block", default=None, help="scheduling-block token recorded in every row (cluster_run --arm-block fam reads fam_<x> from the command)")
    ap.add_argument("--device", default=None, help="FlyBrain device (default: auto)")
    ap.add_argument("--pulse-s", type=float, default=2.0, help="pulse duration (s)")
    # thread 6A (docs/audits/compass_dc_balance.md): one new flag, default None, the shipped path bit-identical with it absent
    ap.add_argument("--hold-edges", action="append", default=None, metavar="PRE_REGEX:POST_REGEX",
                    help="repeatable; hold one class of edges at 0 in the weight matrix (an `edges`-kind LABELLED COUNTERFACTUAL, "
                         "docs/INTERP.md 10.1 step 5), e.g. --hold-edges '^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)'")
    # thread 6B (docs/audits/compass_local_recurrence.md): one new flag, default None, a per-type gain through the same stage
    ap.add_argument("--preset", default=None, choices=list(PRESETS),
                    help="docs/PRESETS_SPEC.md: 'raw' (default, byte-identical to no flag) or 'instrumented' (the --instrument stand-ins, "
                         "and the --hold-edges / --nt-override / --edge-gain records, listed in provenance.instruments); --instrument implies it")
    from flyverse.instruments import add_cli_arguments
    add_cli_arguments(ap)
    ap.add_argument("--turn", type=float, default=None, metavar="DEG_S",
                    help="--ledger only: a prescribed signed yaw rate (deg/s, positive = a left turn) over --turn-window, fed to the afferent "
                         "instrument when attached and recorded either way (cx_wedge has no body: docs/audits/compass_velocity_route.md)")
    ap.add_argument("--turn-window", default=None, metavar="START:END", help="seconds after the pulse end the turn is on (default 0.5:3.5)")
    ap.add_argument("--settle-s", type=float, default=1.0, help="background settle before the pulse (s; default 1.0, the 6A value)")
    ap.add_argument("--edge-gain", action="append", default=None, metavar="PRE_REGEX:POST_REGEX:FACTOR",
                    help="repeatable; multiply one class of edges by FACTOR in the weight matrix through type_path_gain (applied BEFORE "
                         "same_type_gain, so '^EPG$:^EPG$:10' undamps exactly the EPG -> EPG pairs under the shipped x0.1) -- a LABELLED "
                         "INSTRUMENT, never a default")
    a = ap.parse_args()
    out_dir = Path(a.out)
    if (a.nt_override or (a.receptor_model and a.receptor_model != 'off')) and Path(a.out).resolve() == AUDIT_DIR.resolve() and not a.no_structure:
        a.no_structure = True                                   # docs/audits/cx_wedge.* describe the default connectome only
        print('note: --nt-override / --receptor-model with the default --out: structural outputs skipped (pass --out <dir> to write them)')
    if a.plot_sim:
        plot_sim(Path(a.plot_sim), out_dir / "cx_wedge_sim_profiles.png")
        return
    nt_override = parse_nt_override(a.nt_override)
    hold_edges = parse_hold_edges(a.hold_edges)
    edge_gains = parse_edge_gains(a.edge_gain)
    preset = resolve_preset(a.preset, a.instrument)             # round 7: 'raw' unless asked (or an instrument is named)
    from flyverse.instruments import validate_cli
    validate_cli(ap, a.instrument)
    turn_window = parse_turn_window(a.turn_window)
    if a.turn is not None and not a.ledger:
        raise SystemExit("--turn needs --ledger")
    c, cache_dir, table = load_connectome(nt_override, scratch=a.scratch_cache)
    if cache_dir is not None:
        print(f"connectome from scratch cache {cache_dir} (TYPE_NT_OVERRIDE = {table})")
    for t in nt_override:
        m = c.neurons.type.fillna("") == t
        print(f"  {t}: {int(m.sum())} cells, nt {c.neurons.nt[m].value_counts().to_dict()}, sign {c.neurons.sign[m].value_counts().to_dict()}")
    receptor_model = None if a.receptor_model == "off" else a.receptor_model
    receptor_net_rule = a.receptor_net_rule
    if a.receptor_model == "shipped":
        receptor_model, receptor_net_rule = brain.LIFParams().receptor_model, brain.LIFParams().receptor_net_rule
    lif_overrides = None
    if a.lif:
        from flyverse.interp import common
        lif_overrides = common.parse_kv(a.lif)
        bad = sorted(set(lif_overrides) - set(brain.LIFParams.__dataclass_fields__))
        if bad:
            raise SystemExit(f"--lif: unknown LIFParams fields {bad}")
    if a.no_structure:
        cells = compass_cells(c)
    else:
        res, cells, c = structure(out_dir, c=c)
    if a.rate_grid is not None:
        gEs = [float(x) for x in a.rate_grid[0].split(",")]; gDs = [float(x) for x in a.rate_grid[1].split(",")]
        rows = rate_grid(c, cells, gEs, gDs, not a.no_delta7_pen, background_hz=a.background, pulse_hz=a.pulse_hz,
                         start_wedge=a.start_wedge, width=a.width, gR=a.ring_gain, with_ring=not a.no_ring)
        if a.rate_out:
            with open(a.rate_out, "w") as f:
                json.dump(rows, f, indent=1)
    if a.sim is not None:
        gains = [tuple(float(x) for x in g.split(":")) for g in a.sim] or [(1.0, 1.0)]
        rows = []
        for gE, gD in gains:
            npz = None
            if a.ledger:
                npz = a.ledger_npz or (str(Path(a.sim_out).with_suffix("")) + f"_gE{gE:g}_gD{gD:g}_s{a.seed}.npz" if a.sim_out else None)
            rows += simulate(c, cells, [(gE, gD)], seconds=a.seconds, pulse_s=a.pulse_s, background_hz=a.background, pulse_hz=a.pulse_hz,
                             start_wedge=a.start_wedge, width=a.width, seed=a.seed, cuda_graphs=not a.no_graphs, delta7_pen=not a.no_delta7_pen,
                             gR=a.ring_gain, receptor_model=receptor_model, receptor_net_rule=receptor_net_rule, nt_override=nt_override,
                             lif_overrides=lif_overrides, ledger=a.ledger, arm=a.arm, block=a.block, device=a.device, ledger_npz=npz,
                             hold_edges=hold_edges, edge_gains=edge_gains, preset=preset, instrument_specs=a.instrument or [],
                             turn_deg_s=a.turn, turn_window=turn_window, settle_s=a.settle_s)
        if a.sim_out:
            path = Path(a.sim_out)
            old = json.load(open(path)) if path.exists() else []
            with open(path, "w") as f:
                json.dump(old + rows, f, indent=1)


if __name__ == "__main__":
    main()
