"""Release adapters. Import these through connectome.load(dataset=...)."""

RELEASES = {"malecns": "v1.0", "fafb": "v783", "banc": "v888"}


class NotAvailable(ValueError):
    """The requested anatomical capability is absent from this dataset."""


def backend(dataset):
    from importlib import import_module
    if dataset not in RELEASES:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {tuple(RELEASES)}")
    return import_module(f"{__name__}.{dataset}")
