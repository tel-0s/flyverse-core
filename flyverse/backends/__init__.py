"""Release adapters. Import these through connectome.load(dataset=...)."""

RELEASES = {"malecns": "v1.0", "fafb": "v783", "banc": "v888"}
CAPABILITIES = {
    "malecns": frozenset({"vnc", "optic_columns"}),
    "fafb": frozenset({"optic_columns"}),
    "banc": frozenset({"vnc"}),
}


class NotAvailable(ValueError):
    """The requested anatomical capability is absent from this dataset."""


def capabilities(dataset):
    """Release capabilities, retained when selecting an induced subgraph."""
    if not isinstance(dataset, str) or dataset not in CAPABILITIES:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {tuple(RELEASES)}")
    return CAPABILITIES[dataset]


def backend(dataset):
    from importlib import import_module
    capabilities(dataset)
    return import_module(f"{__name__}.{dataset}")
