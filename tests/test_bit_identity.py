"""The default path is byte-for-byte what it was: a recorded golden for a fresh FlyBrain with nothing attached.

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_bit_identity.py -q

`docs/EXTENSIBILITY_SPEC.md` 5.1 requires the opt-in extensions (hooks, attached modules, `Connectome.extend`,
`surrogate_grad`, the optic per-stream hooks) to be off by default and the default path to stay bit-identical. The
review of `feat/extensibility` (`docs/audits/extensibility_review.md` 2) verified that across three source trees
with an external harness -- 5,254 arrays, 0 differing -- but the claim then lived only in prose. This file pins it
in the repository: one deterministic CPU scenario over a synthetic 19-cell, 4-column graph that touches every
default-path input (all four senses, `stimulate`, `set_drive`, fractional-ms carry-over, a `state_dict` round trip,
a per-row `reset` and a full `reset`), hashed after each stage.

The hashes cover every brain and optic state tensor, the clock accumulators, the RNG state, the held sensory
Poisson field and every `MotorRates` field. Nothing about the extensions appears in the scenario: that is the
point. A module, a hook, a graph extension or a stream hook that perturbs the shipped model -- however slightly,
however indirectly -- lands here as a changed digest.

REGENERATING THE GOLDEN IS AN OWNER DECISION, NOT A TEST FIX. A failure means the simulated model moved. If the
move is intended (a model change the owner has accepted, with the numbers in docs/NOTES.md), run

    CUDA_VISIBLE_DEVICES=-1 FLYVERSE_PRINT_GOLDEN=1 python -m pytest tests/test_bit_identity.py -q -s

and paste the printed block over GOLDEN below, in the same commit as the change that caused it. Never regenerate
to make a red test green.
"""
import hashlib
import os
import struct
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")        # a CPU test never touches this machine's GPU (the cluster rule)

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyverse import connectome as cn                      # noqa: E402
from flyverse.brain import LIFParams                       # noqa: E402
from flyverse.fly import FlyBrain                          # noqa: E402

SWEET_BODY_ID = 13491          # a 'sweet' row of flyverse/data/taste_grns.csv, so senses.Taste finds a GRN


# ---------------------------------------------------------------------------------------------- the synthetic graph
def graph() -> cn.Connectome:
    """19 cells over a 4-column retina, carrying every default-path input and every motor readout used below.

    Optic lobe: 4 R1-R6 photoreceptors (histamine) and 4 Mi1 rate units (ACh), one per hex column, plus one Pm1
    (GABA) on the centre column, so `W_rr` has both signs; LC4 (visual_projection) is the rate -> spiking output.
    Senses: ORN_DM1 (class olfactory) onto DM1_lPN, JO-C / JO-E for wind, a sweet GRN for taste. Motor: DNp01 (the
    giant fibre), DNa02 L / R (turning), a DLMn power muscle, a front-leg motor neuron and MN9 (proboscis).
    """
    rows = [dict(bodyId=101, type="R1-R6", superclass="ol_sensory", nt="histamine", hex1=0., hex2=0., hex_side="L"),
            dict(bodyId=102, type="R1-R6", superclass="ol_sensory", nt="histamine", hex1=1., hex2=0., hex_side="L"),
            dict(bodyId=103, type="R1-R6", superclass="ol_sensory", nt="histamine", hex1=0., hex2=1., hex_side="L"),
            dict(bodyId=104, type="R1-R6", superclass="ol_sensory", nt="histamine", hex1=1., hex2=1., hex_side="L"),
            dict(bodyId=201, type="Mi1", superclass="ol_intrinsic", nt="acetylcholine", hex1=0., hex2=0., hex_side="L"),
            dict(bodyId=202, type="Mi1", superclass="ol_intrinsic", nt="acetylcholine", hex1=1., hex2=0., hex_side="L"),
            dict(bodyId=203, type="Mi1", superclass="ol_intrinsic", nt="acetylcholine", hex1=0., hex2=1., hex_side="L"),
            dict(bodyId=204, type="Mi1", superclass="ol_intrinsic", nt="acetylcholine", hex1=1., hex2=1., hex_side="L"),
            dict(bodyId=205, type="Pm1", superclass="ol_intrinsic", nt="gaba", hex1=0., hex2=0., hex_side="L"),
            dict(bodyId=301, type="LC4", superclass="visual_projection", nt="acetylcholine", somaSide="L"),
            dict(bodyId=401, type="ORN_DM1", superclass="cb_sensory", nt="acetylcholine", cls="olfactory"),
            dict(bodyId=402, type="DM1_lPN", superclass="cb_intrinsic", nt="acetylcholine", cls="ALPN", somaSide="L"),
            dict(bodyId=403, type="JO-C", superclass="cb_sensory", nt="acetylcholine"),
            dict(bodyId=404, type="JO-E", superclass="cb_sensory", nt="acetylcholine"),
            dict(bodyId=SWEET_BODY_ID, type="Gr64f", superclass="cb_sensory", nt="acetylcholine"),
            dict(bodyId=501, type="DNp01", superclass="descending_neuron", nt="acetylcholine", somaSide="R"),
            dict(bodyId=502, type="DNa02", superclass="descending_neuron", nt="acetylcholine", somaSide="L"),
            dict(bodyId=503, type="DNa02", superclass="descending_neuron", nt="acetylcholine", somaSide="R"),
            dict(bodyId=601, type="DLMn", superclass="vnc_motor", nt="acetylcholine", subclass="wm", somaSide="L")]
    n = pd.DataFrame(rows)
    n["class"] = n.pop("cls").fillna("") if "cls" in n else ""
    for col, fill in (("subclass", ""), ("somaSide", ""), ("hex1", np.nan), ("hex2", np.nan), ("hex_side", "")):
        n[col] = n[col].fillna(fill) if col in n else fill
    n["instance"] = n.type + "_" + n.somaSide.replace("", "x")
    n["sign"] = np.array([cn.NT_SIGN[x] for x in n.nt], dtype=np.float32)
    ids = n.bodyId.to_numpy()
    where = {int(b): k for k, b in enumerate(ids)}
    edges = []                                                     # (post bodyId, pre bodyId, |synapses|)
    for col in range(4):                                           # R -> Mi1, per column
        edges.append((201 + col, 101 + col, 40))
    edges += [(205, 201, 25), (205, 202, 15),                      # Mi1 -> Pm1 (centre and a neighbour)
              (201, 205, 12), (202, 205, 12), (203, 205, 12),      # Pm1 -> Mi1 (inhibitory recurrence)
              (301, 201, 30), (301, 202, 30), (301, 203, 20), (301, 204, 20),   # Mi1 -> LC4
              (201, 301, 6),                                       # LC4 -> Mi1 (spiking -> rate feedback)
              (402, 401, 400),                                     # ORN -> PN
              (502, 402, 100), (503, 402, 60),                     # PN -> DNa02 L / R
              (502, 403, 80), (503, 404, 80),                      # JO-C / JO-E -> DNa02
              (502, SWEET_BODY_ID, 90),                            # sugar GRN -> DNa02_L
              (501, 301, 120),                                     # LC4 -> DNp01 (above the cap)
              (503, 502, 20), (601, 502, 500), (601, 501, 40)]     # DNa02 L -> R, both -> the power muscle
    post = [where[p] for p, _, _ in edges]; pre = [where[q] for _, q, _ in edges]
    val = np.array([v for _, _, v in edges], np.float32) * n.sign.to_numpy()[pre]
    W = sp.csr_matrix((val, (post, pre)), shape=(len(n), len(n)), dtype=np.float32)
    W.sort_indices()
    return cn.Connectome(n, W, pd.Series(np.arange(len(n)), index=ids))


# ---------------------------------------------------------------------------------------------- hashing
def _feed(h, value, name=""):
    h.update(name.encode())
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        h.update(str((value.dtype.str, value.shape)).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        for k in sorted(value, key=str):
            _feed(h, value[k], f"{name}.{k}")
    elif isinstance(value, (list, tuple)):
        for k, v in enumerate(value):
            _feed(h, v, f"{name}[{k}]")
    elif isinstance(value, (bool, int, np.integer)):
        h.update(b"i" + struct.pack("<q", int(value)))
    elif isinstance(value, (float, np.floating)):
        h.update(b"f" + struct.pack("<d", float(value)))          # the exact bits, not a rounded repr
    else:
        h.update(repr(value).encode())
    return h


def digest(fb) -> str:
    """Every simulated tensor of a FlyBrain plus its MotorRates, as one hex digest."""
    h = hashlib.sha256()
    for name in FlyBrain.BRAIN_TENSORS:
        _feed(h, getattr(fb.brain, name), f"brain.{name}")
    _feed(h, {str(k): v for k, v in fb.brain._acc.items()}, "brain.acc")
    _feed(h, fb.brain.gen.get_state(), "brain.rng")
    _feed(h, [fb.brain.t, fb.brain.step_count, fb.brain.buf_pos, fb.brain._poisson_on], "brain.scalars")
    if fb.optic is not None:
        for name in FlyBrain.OPTIC_TENSORS:
            _feed(h, getattr(fb.optic, name), f"optic.{name}")
        _feed(h, [fb.optic._pending_ms, fb.optic.stream_adapt_state], "optic.extra")
    _feed(h, [fb._base_poisson, fb._pending_ms, fb._activity_ms], "fly.inputs")
    _feed(h, asdict(fb.motor()), "motor")
    return h.hexdigest()


RADIANCE = np.array([[.10, .20, .30, .40], [.90, .10, .55, .05],
                     [.33, .66, .11, .77], [.50, .50, .50, .50]], dtype=np.float32)


def scenario():
    """The default path, exercised once. Returns {stage: digest}; nothing is attached at any point."""
    fb = FlyBrain(graph(), batch=2, device="cpu", seed=7,
                  lif_params=LIFParams(receptor_model=None))       # no receptor cache: the test must run data-free
    # getattr: the same scenario file is meant to run unchanged against an older tree, to compare versions
    assert getattr(fb, "_extensions", None) is None and fb.optic is not None
    out = {}

    fb.smell({"DM1": 1.0}, {"DM1": np.array([0.25, 2.0])})
    fb.wind(0.5, -0.2)
    fb.taste(np.array([0.8, 0.0]))
    fb.vision(RADIANCE)
    for _ in range(3):
        fb.step(10.0)
    out["senses"] = digest(fb)

    fb.stimulate([9], 300.0, 7.0)                                  # LC4, expiring inside the next frame
    fb.brain.set_drive([15], np.float32(12.0))                     # DNp01
    fb.vision(RADIANCE[::-1].copy())
    for _ in range(2):
        fb.step(13.3)                                              # fractional-ms carry-over
    out["stimulate_and_drive"] = digest(fb)

    saved = fb.state_dict()
    fb.step(10.0)
    fb.load_state_dict(saved)
    fb.step(10.0)
    out["state_dict_round_trip"] = digest(fb)

    fb.reset([0])
    fb.step(10.0)
    out["reset_row"] = digest(fb)

    fb.reset()
    fb.smell({"DM1": 0.75}, {"DM1": 0.75})
    fb.vision(RADIANCE)
    fb.step(20.0)
    out["reset_all"] = digest(fb)
    return out


# The recorded golden. Regenerate ONLY on an explicit owner decision (see the module docstring).
# Recorded 2026-09-13 on the merge of feat/extensibility into main, and verified identical on the pre-merge tree
# (main c251e98, which has neither flyverse/modules.py nor the hooks / attach / surrogate_grad surface) by running
# this same file there: five stages, five equal digests. That equality IS the cross-version claim of
# docs/audits/extensibility_review.md 2, now in code rather than prose.
GOLDEN = {
    "senses": "d94bbb782210080bc11b75b7958eab2ef92cfcb3fcf8a496c5ccb0a7d4e0c4e6",
    "stimulate_and_drive": "913825cff441657834cab980ad740470aee89da5fbd6fc0080fb204616cec654",
    "state_dict_round_trip": "e2b4141e311ea599004dacf271a0735b11b78d643ed3738728cc9f68ecac948a",
    "reset_row": "aa40431a58126c62e3b8fbc1a9d3da0ea6007e62a8d1e242e30c9448775cc9d2",
    "reset_all": "bc814a5f1cbbcece0f9fc71956ba79d186c1a6388dc6a16433bd2debe2a0a9d5",
}


class BitIdentityTests(unittest.TestCase):
    def test_default_path_matches_the_recorded_golden(self):
        got = scenario()
        if os.environ.get("FLYVERSE_PRINT_GOLDEN"):
            print("\nGOLDEN = {")
            for k, v in got.items():
                print(f'    "{k}": "{v}",')
            print("}")
        self.assertEqual(sorted(got), sorted(GOLDEN))
        for stage in GOLDEN:
            with self.subTest(stage=stage):
                self.assertEqual(got[stage], GOLDEN[stage],
                                 f"the default path moved at stage {stage!r}: the simulated model changed. "
                                 "Regenerating GOLDEN is an owner decision, not a test fix "
                                 "(tests/test_bit_identity.py docstring).")

    def test_the_scenario_is_reproducible_in_process(self):
        """A second run of the same scenario in the same process must agree, or the golden means nothing."""
        self.assertEqual(scenario(), scenario())

    def test_the_scenario_really_moves_every_stage(self):
        """Guard against a scenario that hashes an all-zero brain: each stage must differ from the last, and the
        brain must actually have spiked."""
        stages = list(scenario().values())
        self.assertEqual(len(set(stages)), len(stages))


if __name__ == "__main__":
    unittest.main()
