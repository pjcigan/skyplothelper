"""Smoke tests for skyplothelper.geometry.bands."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from skyplothelper.geometry.bands import (
    add_frame_band,
    add_great_circle_band,
    add_latitude_band,
    add_longitude_band,
    add_lonlat_box,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_add_latitude_band_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    add_latitude_band(ax, lat_min=-10, lat_max=10)
    fig.canvas.draw()


def test_add_longitude_band_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    add_longitude_band(ax, lon_min=20, lon_max=60)
    fig.canvas.draw()


def test_add_great_circle_band_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    add_great_circle_band(ax, ra_pole=192.859, dec_pole=27.128, half_width=10.0)
    fig.canvas.draw()


@pytest.mark.parametrize("backend", ["patch", "contour"])
def test_add_frame_band_galactic_smoke(backend):
    """Cross-frame band (galactic) on an ICRS plot — the headline v6/v7 feature."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", frame="ICRS", fig=fig)
    add_frame_band(
        ax, lat_min=-10, lat_max=10, frame="galactic", backend=backend,
        facecolor="orange", alpha=0.3,
    )
    fig.canvas.draw()


def test_add_lonlat_box_galactic_on_icrs_smoke():
    """Cross-frame box (galactic) on an ICRS plot — the 12.14 headline
    feature. Delegates the eROSITA-style cross-frame rect rendering
    that surveys.py used to materialize inline."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", frame="ICRS", fig=fig)
    patches = add_lonlat_box(ax, lat_min=-90, lat_max=90,
                              lon_min=180, lon_max=360, frame='galactic',
                              facecolor='red', alpha=0.3)
    assert patches, "expected at least one rendered patch"
    fig.canvas.draw()


def test_add_lonlat_box_polar_touching_galactic():
    """Polar-touching box (lat_max=90) renders without crashing — the
    north edge collapses to a point and is omitted from the outline
    walk."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", frame="ICRS", fig=fig)
    patches = add_lonlat_box(ax, lat_min=60, lat_max=90,
                              lon_min=0, lon_max=90, frame='galactic',
                              facecolor='blue', alpha=0.3)
    assert patches
    fig.canvas.draw()


def test_add_lonlat_box_longitude_wrap():
    """Wrap-around box (lon_max < lon_min) interprets as crossing the
    source-frame antimeridian, not as the complement slice."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", frame="ICRS", fig=fig)
    patches = add_lonlat_box(ax, lat_min=-10, lat_max=10,
                              lon_min=350, lon_max=30, frame='galactic',
                              facecolor='green', alpha=0.3)
    assert patches
    fig.canvas.draw()


def test_add_lonlat_box_complement_kwarg():
    """complement=True fills everything OUTSIDE the box."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", frame="ICRS", fig=fig)
    patches = add_lonlat_box(ax, lat_min=-10, lat_max=10,
                              lon_min=0, lon_max=90, frame='galactic',
                              complement=True,
                              facecolor='gray', alpha=0.3)
    assert patches
    fig.canvas.draw()


def test_add_lonlat_box_rejects_inverted_lat_range():
    """lat_min must be less than lat_max — inverted ranges raise."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", frame="ICRS", fig=fig)
    with pytest.raises(ValueError, match="lat_min must be less than lat_max"):
        add_lonlat_box(ax, lat_min=10, lat_max=-10,
                       lon_min=0, lon_max=90, frame='galactic')


def test_latitude_band_on_tan_zoom_doesnt_fill_frame():
    """Regression for the ``_project_shape`` over-fire on tight-FOV
    projections. ``add_latitude_band`` on a small-cdelt TAN frame used
    to fill the entire panel instead of rendering as a horizontal
    stripe: TAN projects off-FOV input samples to finite-but-absurd
    pixel coords (millions of pixels from the visible region), and
    including those as polygon vertices made the band's edges sweep
    across the whole frame. ``_project_shape`` now filters projected
    coords to a generous bbox around the frame so only
    meaningfully-near samples enter the polygon."""
    from skyplothelper import CompoundRegion
    fig = plt.figure(figsize=(6, 6))
    ax = make_wcs_frame(111, projection='TAN', center=(150.12, 2.21),
                        cdelt=0.01, fig=fig)
    fig.canvas.draw()

    r = CompoundRegion(ax)
    r.add_latitude_band(1.51, 2.91)   # 1.4° band on the TAN center

    # Band should cover well under half the frame (it's a thin stripe).
    frame_area = r._frame_poly.area
    assert r._geom is not None
    assert r._geom.area / frame_area < 0.5, (
        f"latitude band on TAN zoom should be a stripe, got "
        f"area_ratio={r._geom.area / frame_area:.3f}")

    # The stripe is horizontal, so y-extent is smaller than x-extent.
    xmin, ymin, xmax, ymax = r._geom.bounds
    assert (ymax - ymin) < (xmax - xmin), (
        f"latitude band should be wider than tall, got "
        f"bounds=({xmin}, {ymin}, {xmax}, {ymax})")
    plt.close(fig)
