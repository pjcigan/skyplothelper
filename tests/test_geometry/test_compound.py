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


# ---- location / label introspection (representative_point, label, annotate) ----

import numpy as np  # noqa: E402

import skyplothelper as sph  # noqa: E402


def _fits_field(projection="TAN", center=(0.0, 0.0), fov_deg=12):
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, center=center,
                        fov_deg=fov_deg, fig=fig)
    fig.canvas.draw()
    return ax


def test_representative_point_is_inside_when_centroid_is_not():
    """On an annulus the area centroid sits in the central hole (outside the
    region); representative_point must land on the ring (inside)."""
    ax = _fits_field()
    reg = CompoundRegion(ax).add_circle(0, 0, radius_deg=4).subtract_circle(
        0, 0, radius_deg=1.5)
    assert not reg.contains_point(*reg.centroid)
    assert reg.contains_point(*reg.representative_point())


def test_representative_point_empty_is_nan():
    ax = _fits_field()
    lon, lat = CompoundRegion(ax).representative_point()
    assert np.isnan(lon) and np.isnan(lat)


def test_label_default_none_and_settable():
    ax = _fits_field()
    reg = CompoundRegion(ax).add_circle(0, 0, radius_deg=3)
    assert reg.label is None
    reg.label = "Footprint"
    assert reg.label == "Footprint"


def test_annotate_places_label_inside_region():
    ax = _fits_field()
    reg = CompoundRegion(ax).add_circle(2, 1, radius_deg=3)
    reg.label = "Blob"
    txt = reg.annotate(ax, color="white")
    assert txt.get_text() == "Blob"
    # the anchor lies inside the region
    x, y = txt.get_position()
    assert reg.contains_point(x, y)


def test_annotate_text_override_and_empty_noop():
    ax = _fits_field()
    reg = CompoundRegion(ax).add_circle(0, 0, radius_deg=3)
    assert reg.annotate(ax) is None            # no label, no text -> nothing
    assert reg.annotate(ax, text="X").get_text() == "X"


def test_annotate_without_axes_raises():
    ax = _fits_field()
    reg = CompoundRegion(ax).add_circle(0, 0, radius_deg=3)
    reg.label = "x"
    reg.ax = None            # a region with no mpl axes (e.g. the plotly backend)
    with pytest.raises(TypeError):
        reg.annotate()


def test_zoom_to_accepts_compound_region():
    ax = _fits_field(projection="CAR", center=(0, 0), fov_deg=60)
    reg = CompoundRegion(ax).add_circle(15, 10, radius_deg=5)
    sph.zoom_to(ax, reg, pad=2)
    # the region's interior point is in view; a far point is not
    rp = reg.representative_point()
    assert reg.contains_point(*rp)
    disp = ax.get_transform("world").transform([[rp[0], rp[1]]])
    px = ax.transData.inverted().transform(disp)[0]
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    assert x0 <= px[0] <= x1 and y0 <= px[1] <= y1


def test_zoom_to_empty_region_raises():
    ax = _fits_field()
    with pytest.raises(ValueError):
        sph.zoom_to(ax, CompoundRegion(ax))
