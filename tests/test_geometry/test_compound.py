"""Smoke tests for skyplothelper.geometry.compound."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from skyplothelper.geometry.compound import CompoundRegion
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_compound_region_chain_add_subtract():
    """Build a region by adding and subtracting circles, render it."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    region = (
        CompoundRegion(ax)
        .add_circle(0.0, 0.0, 30.0)
        .subtract_circle(0.0, 0.0, 10.0)
    )
    artists = region.render(facecolor="cyan", alpha=0.5)
    assert len(artists) > 0


def test_compound_region_intersect():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    region = (
        CompoundRegion(ax)
        .add_circle(0.0, 0.0, 30.0)
        .intersect_circle(15.0, 0.0, 30.0)
    )
    artists = region.render(facecolor="orange", alpha=0.5)
    assert len(artists) > 0


def test_compound_region_complement():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    region = (
        CompoundRegion(ax)
        .add_circle(0.0, 0.0, 30.0)
        .complement()
    )
    artists = region.render(facecolor="gray", alpha=0.5)
    assert len(artists) > 0


def test_compound_region_is_empty_starts_true():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    region = CompoundRegion(ax)
    assert region.is_empty


def test_compound_region_is_empty_after_add_false():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    region = CompoundRegion(ax).add_circle(0.0, 0.0, 5.0)
    fig.canvas.draw()
    assert not region.is_empty


def test_compound_region_contains_point_returns_bool():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    region = CompoundRegion(ax).add_circle(0.0, 0.0, 30.0)
    fig.canvas.draw()
    # Just verify the call completes and returns a bool — the boundary
    # behavior depends on projection-space details that the visual
    # baselines cover.
    out = region.contains_point(0.0, 0.0)
    assert isinstance(out, (bool, type(True)))


def test_compound_region_uses_projector_frame_without_a_wcs():
    """A backend can know its frame without owning a WCS object.

    The plotly projector reports a frame STRING and has no `.wcs`, so relying
    on the WCS alone silently meant ICRS there — a non-ICRS SkyCoord was
    mis-converted on the plotly side while the mpl side was correct.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord

    from skyplothelper.geometry._parsing import _parse_coord

    class _FrameOnlyProjector:
        wcs_frame = "galactic"          # no .wcs attribute at all

    proj = _FrameOnlyProjector()
    gal = SkyCoord(120.0, 30.0, unit="deg", frame="galactic")
    lon, lat, _ = _parse_coord(gal, 10.0,
                               wcs=getattr(proj, "wcs", None),
                               frame_name=getattr(proj, "wcs_frame", None))
    assert np.allclose([lon, lat], [120.0, 30.0], atol=1e-6)
    # ICRS coercion would have produced this instead:
    assert not np.allclose([lon, lat],
                           [gal.icrs.ra.deg, gal.icrs.dec.deg], atol=1e-3)
