"""Audit explicit MaleCNS sex annotations separately from anatomical API queries.

The legacy compiled neuron table is unchanged. Its source-only dimorphism/fruDsx
columns are consulted here, with their source hash, never inferred from type names.
Female names absent from MaleCNS are candidates, not proven female-specific types.
"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse import connectome as cn
from flyverse.backends.common import sha256
from flyverse.interp import common


def generate(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    graphs = {k: cn.load(dataset=k, verbose=False) for k in ("malecns", "fafb", "banc")}
    source = cn.DATA_DIR / cn.ANNOT_FILE
    labels = pd.read_feather(source, columns=["bodyId", "type", "dimorphism", "fruDsx"])
    labels = labels[labels.bodyId.isin(graphs["malecns"].neurons.bodyId)]
    counts = pd.DataFrame({k: c.neurons.type.value_counts() for k, c in graphs.items()}).fillna(0).astype(int)
    counts.index.name = "type"
    known = labels[labels.dimorphism == "male-specific"].groupby("type").size()
    table = counts.copy()
    table["male_specific_annotated_cells"] = known.reindex(table.index, fill_value=0)
    fru = labels.dropna(subset="fruDsx").groupby("type").fruDsx.agg(lambda s: ";".join(sorted(set(s))))
    table["malecns_fruDsx"] = fru.reindex(table.index).fillna("")
    male = table[table.male_specific_annotated_cells > 0].copy()
    male["evidence"] = "MaleCNS source dimorphism=male-specific; annotation applies to these cells"
    female = table[(table.malecns == 0) & ((table.fafb > 0) | (table.banc > 0))].copy()
    female["evidence"] = "female name unmatched in MaleCNS; sex specificity unestablished"
    combined = pd.concat([male, female]).sort_index()
    combined.to_csv(out / "sex_matches.csv")
    if args.audit_table:
        combined.to_csv(args.audit_table)
    absent = male[(male.fafb == 0) & (male.banc == 0)]
    report = dict(provenance={k: common.provenance(c, device="cpu", stimulus={"protocol": "sex_annotation_audit",
        "params": {}, "control": "source annotations and name matches; not a biological homology claim"}) for k, c in graphs.items()},
        annotation_source=dict(file=source.name, sha256=sha256(source)),
        source_dimorphism_counts=labels.dimorphism.value_counts().to_dict(),
        male_specific_cells=int(known.sum()), male_specific_types=len(male),
        male_specific_names_absent_both=absent.index.tolist(),
        male_specific_names_with_female_match=male.loc[~male.index.isin(absent.index)].reset_index().to_dict("records"),
        female_names_absent_malecns=len(female),
        caveats=["MaleCNS dimorphism and fruDsx are explicit source annotations; matched female names do not establish sex-shared circuitry.",
                 "Female releases lack the corresponding structured dimorphism/fruDsx fields. The reverse list records name absence only; female specificity requires independent evidence.",
                 "FAFB lacks VNC and BANC has incomplete optic reconstruction. These coverage gaps are not sex differences."])
    (out / "report.json").write_text(json.dumps(common.to_jsonable(report), indent=2), encoding="utf-8")
    print(f"{len(male)} source-labelled male-specific types ({int(known.sum())} cells): {len(absent)} absent from both female name sets")
    print(male.loc[~male.index.isin(absent.index)].to_string())
    print(f"{len(female)} female names unmatched in MaleCNS; sex specificity unestablished")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/connectome_backends/sex_labels")
    ap.add_argument("--audit-table", help="optional public CSV path to reproduce the reviewed audit table")
    generate(ap.parse_args())
