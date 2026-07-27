"""Tests for the bundled colormaps (skyplothelper.colormaps)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

import skyplothelper as sph
from skyplothelper._colormaps_lut import ANCHORS, LUTS
from skyplothelper.colormaps import PREFIX


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_all_maps_registered_forward_and_reverse():
    for name in sph.list_colormaps():
        assert name in mpl.colormaps
        assert name + "_r" in mpl.colormaps


def test_list_colormaps_order_and_prefix():
    names = sph.list_colormaps()
    assert len(names) == len(LUTS) + len(ANCHORS)
    assert all(n.startswith(PREFIX) for n in names)
    # curated order: LUT maps first (deepsky …), diffs last
    assert names[0] == "sph.deepsky"
    assert names[-1] == "sph.diff_greenorange"


def test_list_colormaps_include_reverse():
    fwd = sph.list_colormaps()
    both = sph.list_colormaps(include_reverse=True)
    assert len(both) == 2 * len(fwd)
    assert both[0] == "sph.deepsky"
    assert both[1] == "sph.deepsky_r"


def test_get_colormap_accepts_bare_prefixed_reversed():
    bare = sph.get_colormap("deepsky")
    pref = sph.get_colormap("sph.deepsky")
    rev = sph.get_colormap("sph.deepsky_r")
    assert bare.name == pref.name == "sph.deepsky"
    assert rev.name == "sph.deepsky_r"


def test_get_colormap_unknown_raises():
    with pytest.raises(KeyError):
        sph.get_colormap("not_a_real_map")


def test_get_colormap_falls_back_when_unregistered():
    """get_colormap resolves from the registration-independent _CMAPS table
    even if the matplotlib registry entry is missing (registration skipped or
    removed), so bundled maps never hard-fail on registry state."""
    import matplotlib as mpl

    from skyplothelper.colormaps import _CMAPS
    fwd = mpl.colormaps["sph.deepsky"]
    rev = mpl.colormaps["sph.deepsky_r"]
    mpl.colormaps.unregister("sph.deepsky")
    mpl.colormaps.unregister("sph.deepsky_r")
    try:
        cm = sph.get_colormap("deepsky")
        assert cm.name == "sph.deepsky"
        assert np.allclose(cm(0.0)[:3], _CMAPS["deepsky"](0.0)[:3])
        # reversed variant also falls back (built from the same table)
        assert sph.get_colormap("sph.deepsky_r").name == "sph.deepsky_r"
    finally:
        mpl.colormaps.register(fwd, name="sph.deepsky")
        mpl.colormaps.register(rev, name="sph.deepsky_r")


def test_reverse_is_actually_reversed():
    fwd = sph.get_colormap("sph.lagoon")(np.linspace(0, 1, 256))
    rev = sph.get_colormap("sph.lagoon_r")(np.linspace(0, 1, 256))
    assert np.allclose(fwd, rev[::-1])


def test_lut_map_has_256_entries_and_black_start():
    # deepsky starts at black (its baked LUT row 0 is 0,0,0)
    cm = sph.get_colormap("deepsky")
    assert np.allclose(cm(0.0)[:3], LUTS["deepsky"][0], atol=1e-3)
    assert len(LUTS["deepsky"]) == 256


def test_anchor_map_endpoints_match_source_anchors():
    for name, spec in ANCHORS.items():
        cm = sph.get_colormap(name)
        assert np.allclose(cm(0.0)[:3], spec["colors"][0], atol=2e-3)
        assert np.allclose(cm(1.0)[:3], spec["colors"][-1], atol=2e-3)


def test_anchor_map_nonuniform_nodes_preserved():
    # diff_desert has non-uniform nodes; the mid anchor color should appear
    # near its node position, not at the midpoint.
    spec = ANCHORS["diff_purplegreen"]  # strongly non-uniform nodes
    cm = sph.get_colormap("diff_purplegreen")
    # atol covers 256-LUT quantization at sharp kinks (~0.01); a positions-
    # ignored bug would place these colors far from their nodes (>> atol).
    for node, color in zip(spec["nodes"], spec["colors"]):
        assert np.allclose(cm(node)[:3], color, atol=0.015)


def test_usable_as_cmap_string():
    fig, ax = plt.subplots()
    ax.imshow(np.random.rand(8, 8), cmap="sph.penumbra")


def test_show_colormaps_returns_figure():
    fig = sph.show_colormaps()
    assert len(fig.axes) == len(sph.list_colormaps())


def test_show_colormaps_subset():
    fig = sph.show_colormaps(["sph.deepsky", "mesa"])
    assert len(fig.axes) == 2


def test_attribute_access():
    from skyplothelper import colormaps as cm
    assert cm.deepsky.name == "sph.deepsky"
    assert cm.deepsky_r.name == "sph.deepsky_r"
    assert cm.diff_tealorange.name == "sph.diff_tealorange"
    # also via the package: sph.colormaps.<name>
    assert sph.colormaps.lagoon.name == "sph.lagoon"


def test_attribute_access_unknown_raises_attributeerror():
    from skyplothelper import colormaps as cm
    with pytest.raises(AttributeError):
        cm.not_a_real_map  # noqa: B018


def test_attribute_names_in_dir_for_completion():
    from skyplothelper import colormaps as cm
    d = dir(cm)
    assert "deepsky" in d and "diff_purplegreen" in d
    assert "deepsky_r" in d
    # real members still present
    assert "get_colormap" in d


def test_public_exports():
    for name in ("list_colormaps", "get_colormap", "show_colormaps"):
        assert hasattr(sph, name)
        assert name in sph.__all__
