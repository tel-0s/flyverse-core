"""Cross-release anatomy through Connectome, including silent sign-0 contacts.

python scripts/cross_connectome.py --out out/connectome_backends/anatomy
Raw counts are never causal effect estimates. Scaling uses each release's total
DNa02 input relative to MaleCNS and is descriptive, not a model parameter.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse import connectome as cn
from flyverse.interp import common

EDGES = [(p, "DNa02") for p in ("PS049", "PS059", "VES051", "LAL126", "AOTU019", "AN04B003", "LT51", "IN12B014",
                                    "AN06A026", "AN07B035", "IN19A003", "GNG562", "PFL3")]
EDGES += [(p, "PS059") for p in ("PS196_a", "PS196_b", "LAL074")]
EDGES += [("GLNO", "PS196_b"), ("DNa02", "AN04B003"), ("GLNO", "PEN_a(PEN1)"),
          ("EPG", "PEN_a(PEN1)"), ("Delta7", "PEN_a(PEN1)")]
EDGES += [(p, t) for t in ("LC11", "LC10a") for p in ("T2", "T3", "Tm5Y", "TmY21", "Mi1", "Mi4", "Tm3", "Tm21", "Tm2")]
HALTERE_TARGETS = ("AN08B010", "IN08B008", "w-cHIN", "IN08B093", "IN06B017", "PS196_a", "PS196_b", "PS059")
BUDGET_TARGETS = ("DNa02", "PS059", "LC11", "LC10a")
NT_CONFLICT_TYPES = ("PFL3", "PFL2", "Delta7", "LAL074", "PS059", "Mi19", "TmY14", "aMe8")


def edge_rows(c, raw, pre_idx, post_idx, pre_name, post_name, scale):
    n = c.neurons
    if not len(pre_idx) or not len(post_idx):
        return [dict(dataset=c.dataset, release=c.release, pre=pre_name, post=post_name,
                     pre_side=None, post_side=None, n_pre=len(pre_idx), n_post=len(post_idx),
                     synapses=None, signed_synapses=None, post_input_share=None, scaled_synapses=None,
                     status="population absent")]
    rows = []
    for ps in sorted(n.iloc[pre_idx].somaSide.fillna("?").unique()):
        pre = pre_idx[n.iloc[pre_idx].somaSide.fillna("?").to_numpy() == ps]
        for qs in sorted(n.iloc[post_idx].somaSide.fillna("?").unique()):
            post = post_idx[n.iloc[post_idx].somaSide.fillna("?").to_numpy() == qs]
            count = float(raw[post][:, pre].sum()); total = float(raw[post].sum())
            rows.append(dict(dataset=c.dataset, release=c.release, pre=pre_name, post=post_name,
                pre_side=ps, post_side=qs, n_pre=len(pre), n_post=len(post), synapses=count,
                signed_synapses=float(c.W[post][:, pre].sum(dtype=np.float64)),
                post_input_share=count / total if total else None, scaled_synapses=count / scale,
                status="present" if count else "no retained edge"))
    return rows


def generate(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    graphs = {d: cn.load(dataset=d, verbose=False) for d in ("malecns", "fafb", "banc")}
    male = graphs["malecns"].neurons
    edge_table, budgets, top, conflicts, coverage, presence, silenced = [], [], [], [], [], [], []
    provenance, scales = {}, {}
    for dataset, c in graphs.items():
        n = c.neurons
        raw, available = common.raw_counts(c, dtype=np.float64, build=False)
        if not available:
            raise ValueError(f"{dataset} lacks sign-0 counts; refusing a partial anatomy table")
        anchor = float(raw[c.select(type="DNa02")].sum())
        if dataset == "malecns":
            male_anchor = anchor
        scales[dataset] = dict(anchor="total raw input to DNa02", raw_input=anchor, scale=anchor / male_anchor,
                               pair_threshold=c._manifest.get("pair_threshold", 1), edges=c._manifest.get("edges", "threshold"))
        scale = scales[dataset]["scale"]
        for pre, post in EDGES:
            edge_table += edge_rows(c, raw, c.select(type=pre), c.select(type=post), pre, post, scale)
        haltere = c.select(**{"class": "mechanosensory_proprioceptive", "subclass": "haltere"}) if c.has_vnc else np.array([], dtype=int)
        for target in HALTERE_TARGETS:
            edge_table += edge_rows(c, raw, haltere, c.select(type=target), "haltere_afferents", target, scale)
        for target in BUDGET_TARGETS:
            post = c.select(type=target)
            inputs = np.asarray(raw[post].sum(axis=0)).ravel()
            df = pd.DataFrame(dict(type=n.type.fillna("<untyped>"), nt=n.nt, synapses=inputs))
            for nt, count in df.groupby("nt").synapses.sum().items():
                budgets.append(dict(dataset=dataset, release=c.release, target=target, nt=nt,
                    synapses=float(count), input_share=float(count / inputs.sum()) if inputs.sum() else None))
            for ty, count in df.groupby("type").synapses.sum().nlargest(25).items():
                top.append(dict(dataset=dataset, release=c.release, target=target, pre=ty, synapses=float(count),
                                input_share=float(count / inputs.sum()), scaled_synapses=float(count / scale)))
        for ty in NT_CONFLICT_TYPES:
            cells = n[n.type == ty]
            conflicts.append(dict(dataset=dataset, release=c.release, type=ty, n=len(cells),
                nt_counts=cells.nt.value_counts().to_dict(),
                verified_counts=cells.nt_verified.dropna().value_counts().to_dict() if "nt_verified" in cells else {}))
        for old_nt in ("unknown", "dopamine", "serotonin", "octopamine"):
            old = male[male.nt == old_nt]
            matched = n[n.type.isin(old.type.dropna())]
            silenced.append(dict(dataset=dataset, malecns_nt=old_nt, malecns_cells=len(old),
                malecns_cells_with_type_present=int(old.type.isin(n.type.dropna()).sum()),
                female_or_reference_cells_by_nt=matched.nt.value_counts().to_dict(),
                interpretation="type-level overlap; not one-to-one neuron matches"))
        coverage.append(dict(dataset=dataset, malecns_cells=len(male),
            before_alias=int(male.type.isin(n.flywireType.dropna() if dataset != "malecns" else n.type.dropna()).sum()),
            after_alias=int(male.type.isin(n.type.dropna()).sum())))
        for ty, count in n.type.value_counts().items():
            presence.append(dict(dataset=dataset, type=ty, n=int(count)))
        provenance[dataset] = common.provenance(c, device="cpu", stimulus={"protocol": "cross_connectome_anatomy", "params": scales[dataset],
            "control": "descriptive cross-release comparison; thresholds and synapse yields differ"})
    for name, rows in (("edges", edge_table), ("budgets", budgets), ("top_inputs", top), ("alias_coverage", coverage)):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    counts = pd.DataFrame(presence).pivot(index="type", columns="dataset", values="n").fillna(0).astype(int)
    counts.to_csv(out / "type_presence.csv")
    absence = counts[(counts == 0).any(axis=1)]
    absence.to_csv(out / "type_absence.csv")
    report = dict(provenance=provenance, scales=scales, nt_conflicts=conflicts, sign0_type_overlap=silenced, alias_coverage=coverage,
        caveats=["Absence of a matching type is not proof of sex specificity (fru/dsx circuits require independent annotation).",
                 "BANC optic lobes are incomplete; FAFB has no VNC.", "Scaled counts use one anchor, not a universal synapse calibration.",
                 "BANC verified co-transmitters remain in nt_verified; nt selects the first classical transmitter."])
    (out / "report.json").write_text(json.dumps(common.to_jsonable(report), indent=2), encoding="utf-8")
    print(pd.DataFrame(coverage).to_string(index=False))
    print("DNa02 input scales:", scales)
    print(f"wrote {len(edge_table)} edge rows, {len(budgets)} budgets, {len(top)} ranked inputs -> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/connectome_backends/anatomy")
    generate(ap.parse_args().out)
