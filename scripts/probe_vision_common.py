"""Shared opt-in loading and Result export for the motion/loom acceptance probes."""

from pathlib import Path

import numpy as np

from flyverse import connectome, retina
from flyverse.banc_vision import validate_candidate_cache
from flyverse.interp.common import Result, provenance


def add_arguments(ap):
    ap.add_argument("--dataset", choices=["malecns", "fafb", "banc"], default="malecns")
    ap.add_argument("--vision", choices=["candidate"])
    ap.add_argument(
        "--vision-cache",
        type=Path,
        help="explicit persisted candidate graph; not the biological cache",
    )
    ap.add_argument("--eye", choices=["both", "right"], default="both")
    ap.add_argument("--device", choices=["cpu", "cuda"])
    ap.add_argument("--json", type=Path)


def select_eye(r, eye):
    """Keep the full eye's directions and only remap the selected column indices."""
    if eye == "both":
        return r
    columns = np.flatnonzero(r.col_side == "R")
    mapping = np.full(r.n_columns, -1, dtype=np.int64)
    mapping[columns] = np.arange(len(columns))
    keep = mapping[r.pr_column] >= 0
    return retina.Retina(
        r.pr_index[keep],
        mapping[r.pr_column[keep]],
        r.pr_sens[keep],
        r.col_side[columns],
        r.col_hex[columns],
        r.col_dir[columns],
        r.col_az_el[columns],
        r.geometry,
    )


def load(args):
    if args.vision_cache is not None and args.vision != "candidate":
        raise ValueError("--vision-cache requires --vision candidate")
    if (
        args.vision_cache is not None
        and (args.vision_cache / "extension.json").exists()
    ):
        c = connectome.load(args.vision_cache, dataset=args.dataset, verbose=False)
        validate_candidate_cache(c)
    else:
        c = connectome.load(
            dataset=args.dataset,
            vision=args.vision,
            vision_cache_dir=args.vision_cache,
            verbose=False,
        )
    if c.vision is not None:
        print(c.vision["qualification"], flush=True)
    return c, select_eye(retina.build_retina(c), args.eye)


def result(args, c, r, b, op, protocol, *, summary, rows, status, criterion):
    p = provenance(
        c,
        lif=b.p,
        optic=op,
        device=b.device,
        seeds=[getattr(args, "seed", 0)],
        batch=1,
        stimulus={
            "protocol": protocol,
            "params": vars(args),
            "control": "matched MaleCNS right eye; cross-dataset comparison",
        },
        retina={
            "eye": args.eye,
            "coverage": r.coverage(),
            "geometry": r.geometry,
            "columns": r.col_hex,
            "directions": r.col_az_el,
        },
    )
    out = Result.new(
        "ledger",
        p,
        summary=summary,
        tables={"per_type": rows},
        validation={
            "name": protocol,
            "reference": {},
            "source": "owner functional experiment gate",
            "status": status,
            "measured": summary,
            "criterion": criterion,
        },
        replicates={
            "n": 1,
            "unit": "run",
            "runs": [getattr(args, "seed", 0)],
            "null": None,
        },
    )
    if args.json:
        out.save(args.json)
    return out
