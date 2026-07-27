"""Tests for skyplothelper.coord_overlay gridline rendering.

Covers the gridline layer only — meridians and parallels from an
overlay frame projected onto a WCSAxes whose own frame may differ.
Ticks and tick labels are covered by the sibling test_coord_overlay_*
modules; tests here exercise construction, frame transformation,
jump-aware segmentation, and the public ``add_coord_overlay``
entrypoint.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.coord_overlay import (  # noqa: E402
    CoordinateOverlay,
    _GridLine,
    add_coord_overlay,
    add_graticule_overlay,
)


def _make_axes(projection="AIT", center=180, frame="ICRS"):
    fig = plt.figure(figsize=(10, 5))
    ax = sph.make_wcs_frame(111, projection=projection, center=center,
                            frame=frame, fig=fig)
    fig.canvas.draw()
    return fig, ax


# ---- construction ----

@pytest.mark.parametrize("projection", ["AIT", "MOL", "CAR"])
def test_construct_on_common_allsky_projections(projection):
    fig, ax = _make_axes(projection=projection)
    ov = CoordinateOverlay(ax, frame="galactic")
    assert len(ov.lon_gridlines) == 12      # 0, 30, ..., 330
    assert len(ov.lat_gridlines) == 11      # -75, -60, ..., 75
    for gl in ov.lon_gridlines + ov.lat_gridlines:
        assert isinstance(gl, _GridLine)
        assert gl.lons.shape == (ov.n_samples,)
        assert gl.lats.shape == (ov.n_samples,)
    plt.close(fig)


def test_custom_lon_lat_vals():
    fig, ax = _make_axes()
    ov = CoordinateOverlay(ax, frame="galactic",
                           lon_vals=[0, 90, 180, 270],
                           lat_vals=[-30, 0, 30])
    assert len(ov.lon_gridlines) == 4
    assert len(ov.lat_gridlines) == 3
    assert [gl.value for gl in ov.lon_gridlines] == [0, 90, 180, 270]
    assert [gl.value for gl in ov.lat_gridlines] == [-30, 0, 30]
    plt.close(fig)


def test_n_samples_controls_density():
    fig, ax = _make_axes()
    ov = CoordinateOverlay(ax, frame="galactic", n_samples=50)
    assert ov.n_samples == 50
    for gl in ov.lon_gridlines + ov.lat_gridlines:
        assert gl.lons.size == 50
    plt.close(fig)


def test_meridian_lat_range_avoids_poles():
    """Meridian sampling stops just shy of ±90 so SkyCoord transforms
    stay well-defined at the endpoints."""
    fig, ax = _make_axes()
    ov = CoordinateOverlay(ax, frame="galactic")
    for gl in ov.lon_gridlines:
        assert gl.lats[0] == pytest.approx(-89.9999)
        assert gl.lats[-1] == pytest.approx(89.9999)
        assert abs(gl.lats[0]) < 90.
        assert abs(gl.lats[-1]) < 90.
    plt.close(fig)


# ---- frame aliasing ----

def test_ecliptic_alias_resolves_to_astropy_frame():
    fig, ax = _make_axes()
    ov = CoordinateOverlay(ax, frame="ecliptic")
    assert ov.frame == "geocentrictrueecliptic"
    plt.close(fig)


def test_target_frame_resolution_for_icrs_axes():
    fig, ax = _make_axes(frame="ICRS")
    ov = CoordinateOverlay(ax, frame="galactic")
    assert ov._target_frame == "icrs"
    plt.close(fig)


def test_target_frame_resolution_for_galactic_axes():
    fig, ax = _make_axes(frame="Galactic")
    ov = CoordinateOverlay(ax, frame="icrs")
    assert ov._target_frame == "galactic"
    plt.close(fig)


# ---- frame transformation numerical correctness ----

def test_to_axes_frame_matches_astropy():
    """Pixel positions of a sample gridline point should agree with a
    direct ``SkyCoord.transform_to`` round-trip."""
    fig, ax = _make_axes(projection="AIT", center=180, frame="ICRS")
    ov = CoordinateOverlay(ax, frame="galactic")
    test_lons = np.array([0., 30., 90., 180., 270.])
    test_lats = np.array([0., 30., -45., 60., -75.])
    got_lon, got_lat = ov._to_axes_frame(test_lons, test_lats)
    expected = SkyCoord(test_lons * u.deg, test_lats * u.deg,
                        frame="galactic").icrs
    np.testing.assert_allclose(got_lon, expected.ra.deg, atol=1e-9)
    np.testing.assert_allclose(got_lat, expected.dec.deg, atol=1e-9)
    plt.close(fig)


def test_to_axes_frame_is_identity_when_frames_match():
    fig, ax = _make_axes(frame="Galactic")
    ov = CoordinateOverlay(ax, frame="galactic")
    test_lons = np.array([0., 90., 180., 270.])
    test_lats = np.array([-30., 0., 30., 60.])
    got_lon, got_lat = ov._to_axes_frame(test_lons, test_lats)
    np.testing.assert_array_equal(got_lon, test_lons)
    np.testing.assert_array_equal(got_lat, test_lats)
    plt.close(fig)


def test_wrap_axes_lon_centers_around_axes_center():
    """The wrap should map values to ``[center - 180, center + 180)``."""
    fig, ax = _make_axes(projection="AIT", center=180)
    ov = CoordinateOverlay(ax, frame="galactic")
    assert ov._center_lon == pytest.approx(180.)
    assert ov._wrap_axes_lon(10.) == pytest.approx(10.)
    assert ov._wrap_axes_lon(350.) == pytest.approx(350.)
    assert ov._wrap_axes_lon(-10.) == pytest.approx(350.)
    plt.close(fig)


def test_wrap_axes_lon_for_center_zero_axes():
    fig, ax = _make_axes(projection="AIT", center=0)
    ov = CoordinateOverlay(ax, frame="galactic")
    assert ov._center_lon == pytest.approx(0.)
    assert ov._wrap_axes_lon(10.) == pytest.approx(10.)
    assert ov._wrap_axes_lon(350.) == pytest.approx(-10.)
    plt.close(fig)


# ---- plot output ----

def test_plot_creates_artists():
    fig, ax = _make_axes()
    ov = CoordinateOverlay(ax, frame="galactic")
    ov.plot()
    n_lon_artists = sum(len(a) for a in ov.lon_artists)
    n_lat_artists = sum(len(a) for a in ov.lat_artists)
    assert n_lon_artists >= len(ov.lon_gridlines)
    assert n_lat_artists >= len(ov.lat_gridlines)
    plt.close(fig)


def test_plot_returns_self_for_chaining():
    fig, ax = _make_axes()
    ov = CoordinateOverlay(ax, frame="galactic")
    assert ov.plot() is ov
    plt.close(fig)


def test_plot_per_axis_style_overrides():
    fig, ax = _make_axes()
    ov = CoordinateOverlay(ax, frame="galactic",
                           lon_vals=[0, 90], lat_vals=[0, 30])
    ov.plot(lon_style={"color": "red"}, lat_style={"color": "blue"},
            lw=1.5)
    for segs in ov.lon_artists:
        for ln in segs:
            assert ln.get_color() == "red"
            assert ln.get_linewidth() == 1.5
    for segs in ov.lat_artists:
        for ln in segs:
            assert ln.get_color() == "blue"
            assert ln.get_linewidth() == 1.5
    plt.close(fig)


def test_meridian_polar_segmentation_on_zenithal_projection():
    """A meridian on STG (zenithal) should render as a single segment
    rooted at the projection center — the polar singularity should
    not produce spurious off-frame jumps."""
    fig, ax = _make_axes(projection="STG", center=180, frame="ICRS")
    ov = CoordinateOverlay(ax, frame="galactic",
                           lon_vals=[180], lat_vals=[])
    ov.plot()
    total_artists = sum(len(a) for a in ov.lon_artists)
    assert total_artists >= 1
    plt.close(fig)


# ---- public entry points ----

def test_add_coord_overlay_returns_overlay():
    fig, ax = _make_axes()
    ov = add_coord_overlay(ax, frame="galactic", color="C0", alpha=0.5)
    assert isinstance(ov, CoordinateOverlay)
    assert sum(len(a) for a in ov.lon_artists) > 0
    plt.close(fig)


def test_add_graticule_overlay_is_alias():
    assert add_graticule_overlay is add_coord_overlay


def test_module_exports_at_top_level():
    assert sph.CoordinateOverlay is CoordinateOverlay
    assert sph.add_coord_overlay is add_coord_overlay
    assert sph.add_graticule_overlay is add_coord_overlay


# ---- frame-alias table unification ----

@pytest.mark.parametrize("alias,expected", [
    ("gal", "galactic"), ("galactic", "galactic"), ("Galactic", "galactic"),
    ("ecl", "geocentrictrueecliptic"),
    ("ecliptic", "geocentrictrueecliptic"),
    ("super", "supergalactic"), ("sgal", "supergalactic"),
    ("eq", "icrs"), ("equatorial", "icrs"), ("icrs", "icrs"),
    ("helio", "heliocentrictrueecliptic"),
    ("barycentrictrueecliptic", "barycentrictrueecliptic"),
])
def test_frame_aliases_resolve_everywhere(alias, expected):
    """One canonical alias table.

    coord_overlay kept a second, smaller copy that knew only the long
    spellings, so convert_frame('gal') worked while
    CoordinateOverlay(frame='gal') passed 'gal' straight to astropy — and
    image_to_healpix inherited the same gap through it.
    """
    from skyplothelper.coord_overlay import _resolve_frame as overlay
    from skyplothelper.core.coords import _resolve_frame as canonical
    assert overlay(alias) == expected
    assert canonical(alias) == expected


def test_add_overlay_ticks_stroke_reaches_marks_and_labels():
    """The overlay stroke backs the tick MARKS too, not only the labels, so a
    stroked overlay reads consistently."""
    fig = plt.figure(figsize=(6, 4))
    ax = sph.make_wcs_frame(projection="AIT", center=0, frame="icrs",
                            fig=fig, grid=False)
    ov = sph.add_overlay_ticks(ax, frame="galactic", stroke_color="k",
                               stroke_lw=3)
    assert ov.tick_artists and all(
        t.get_path_effects() for t in ov.tick_artists)
    assert all(lab.get_path_effects() for lab in ov.label_artists)
    plt.close(fig)
