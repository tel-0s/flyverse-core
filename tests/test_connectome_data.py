"""Optional acceptance against compiled female release caches."""
import hashlib
import numpy as np
import pytest

from flyverse import connectome as cn, regions
from flyverse.brain import Brain
from flyverse.retina import build_retina

# The `data` mark is applied by file name in tests/conftest.py's FILE_MARKS, the stated convention of this suite
# (connectome_backends_review nit 13) -- not by a pytestmark here.


@pytest.mark.parametrize("dataset", ["fafb", "banc"])
def test_female_cpu_smoke_and_raw_counts(dataset):
    path = cn.default_cache_directory(dataset)
    if not (path / "W_post_pre.npz").exists():
        pytest.skip(f"compile {dataset} first")
    c = cn.load(dataset=dataset, verbose=False)
    assert c.neurons.bodyId.dtype == np.int64 and c.neurons.bodyId.is_unique
    assert len(regions.labels(c)) == c.n
    post = c.select(type="DNa02")[0]; row = c.W[post]
    sub = c.subset(np.unique(np.r_[post, row.indices[np.argsort(-abs(row.data))[:199]]]))
    brain = Brain(sub, device="cpu"); brain.step(100)
    assert sub.n == 200 and np.isfinite(brain.v.cpu().numpy()).all()
    counts = cn.sign0_counts(c)
    assert (counts[c.W.data != 0] == 0).all() and counts.sum() > 0
    if dataset == "fafb":
        coverage = build_retina(c).coverage()
        assert coverage["n_columns"] == 1581
        assert coverage["without_photoreceptors"] == 51
    else:
        with pytest.raises(cn.NotAvailable):
            build_retina(c)


def test_loading_malecns_never_rewrites_cache():
    if not (cn.CACHE_DIR / "W_post_pre.npz").exists():
        pytest.skip("MaleCNS cache absent")
    paths = [p for p in cn.CACHE_DIR.iterdir() if p.is_file()]
    old = [hashlib.md5(p.read_bytes()).digest() for p in paths]
    c = cn.load(verbose=False)
    assert (c.dataset, c.release) == ("malecns", "v1.0")
    assert c.cache_dir == cn.CACHE_DIR            # the default MaleCNS graph is the repository cache, B2
    assert old == [hashlib.md5(p.read_bytes()).digest() for p in paths]


def test_wing_motor_groups_are_populated_on_every_vnc_release():
    """B1: BANC names the wing MNs without the MaleCNS ' MN' / 'DLMn' notation; the alias table maps them, so the
    steering and power groups carry the same cells as MaleCNS instead of coming back silently empty."""
    from flyverse.motor import wing_groups
    for dataset, expect in (("malecns", (16, 16, 24)), ("banc", (16, 16, 24))):
        path = cn.default_cache_directory(dataset)
        if not (path / "W_post_pre.npz").exists():
            pytest.skip(f"compile {dataset} first")
        wg = wing_groups(cn.load(dataset=dataset, verbose=False))
        assert (len(wg.steer_L), len(wg.steer_R), len(wg.power)) == expect


def test_banc_haltere_names_normalize_without_changing_selected_cells():
    from flyverse.motor import wing_groups
    if not (cn.default_cache_directory("banc") / "W_post_pre.npz").exists():
        pytest.skip("compile banc first")
    c = cn.load(dataset="banc", verbose=False)
    h = c.neurons.iloc[wing_groups(c).haltere]
    assert len(h) == 25
    for raw, canonical, count in (("hi1", "hi1 MN", 4), ("hi2", "hi2 MN", 3),
                                  ("hDVM", "hDVM MN", 2), ("hiii2", "hiii2 MN", 2)):
        selected = h[h.flywireType == raw]
        assert len(selected) == count and selected.type.eq(canonical).all()
