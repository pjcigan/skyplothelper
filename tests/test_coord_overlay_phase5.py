"""Tests for ``add_overlay_ticks`` on projection boundaries.

The public helper ``add_overlay_ticks`` places
overlay ticks + labels on a projection's natural visible boundary
(circular for SIN / ZEA / STG / AIR / ARC / AZP / SZP, elliptical
for AIT / MOL, sinusoidal for SFL, parabolic for PAR, and various
custom curves for the non-FITS pseudo-cylindricals; rectangular
fall-back for default frames).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper._compat import coord_ticklabels  # noqa: E402
from skyplothelper.coord_overlay import (  # noqa: E402
    CoordinateOverlay,
    _extrapolate_polyline_endpoints,
    _frame_to_curves,
    _intersect_polylines,
    _suppress_default_ticks,
    add_overlay_ticks,
)


def _axes(projection, **kw):
    fig = plt.figure(figsize=(7, 5))
    # tick_style='native' keeps astropy's default tick labels visible,
    # which is the baseline state these unit tests assert against (they
    # check what add_overlay_ticks' suppress_default kwarg toggles).
    # Without the override, make_wcs_frame's auto-trigger for globe-
    # class projections (SIN, SFL, PAR, ...) would suppress the
    # defaults before the test even runs.
    kw.setdefault('tick_style', 'native')
    ax = sph.make_wcs_frame(111, projection=projection, fig=fig, **kw)
    fig.canvas.draw()
    return fig, ax


# ---- _frame_to_curves: boundary detection ----

@pytest.mark.parametrize("projection,kwargs", [
    ("SIN", dict(center=(180, 30), fov_deg=70.0)),
    ("ARC", dict(center=(180, 0), fov_deg=70.0)),
    ("ZEA", dict(center=(180, 30), fov_deg=70.0)),
    ("AIT", dict(center=180)),
    ("MOL", dict(center=180)),
    ("SFL", dict(center=180)),
    ("PAR", dict(center=180)),
])
def test_frame_to_curves_custom_frame_returns_single_closed_boundary(
        projection, kwargs):
    """Custom astropy frames expose their boundary as a single 'c'
    spine — _frame_to_curves should return exactly one _FrameCurve
    named 'boundary' marked as closed."""
    fig, ax = _axes(projection, **kwargs)
    curves = _frame_to_curves(ax)
    assert len(curves) == 1
    fc = curves[0]
    assert fc.name == "boundary"
    assert fc.closed is True
    # Boundary polyline should be a non-trivial closed loop
    assert fc.xy_pix.shape[1] == 2
    assert fc.xy_pix.shape[0] > 4
    np.testing.assert_allclose(fc.xy_pix[0], fc.xy_pix[-1])
    plt.close(fig)


def test_frame_to_curves_rectangular_returns_four_spines():
    """Default rectangular frames expose four spines (b/t/l/r) and
    _frame_to_curves should return four _FrameCurves with axis-aligned
    bbox names."""
    fig, ax = _axes("CAR", center=180)
    curves = _frame_to_curves(ax)
    assert len(curves) == 4
    names = sorted(fc.name for fc in curves)
    assert names == ["bottom", "left", "right", "top"]
    for fc in curves:
        assert fc.xy_pix.shape == (2, 2)
        assert fc.closed is False
    plt.close(fig)


# ---- _suppress_default_ticks ----

@pytest.mark.parametrize("mode,lon_visible,lat_visible", [
    ("both", False, False),
    ("lon",  False, True),
    ("lat",  True, False),
    ("none", True, True),
])
def test_suppress_default_ticks_modes(mode, lon_visible, lat_visible):
    fig, ax = _axes("CAR", center=180)
    _suppress_default_ticks(ax, mode)
    # We can't read back set_ticks_visible state directly across
    # astropy versions; instead check the ticklabel visibility flag
    # via a documented public property.
    assert coord_ticklabels(ax.coords[0]).get_visible() is lon_visible
    assert coord_ticklabels(ax.coords[1]).get_visible() is lat_visible
    plt.close(fig)


def test_suppress_default_ticks_rejects_unknown_mode():
    fig, ax = _axes("CAR", center=180)
    with pytest.raises(ValueError, match="suppress_default must be"):
        _suppress_default_ticks(ax, "all")
    plt.close(fig)


# ---- add_overlay_ticks: end-to-end ----

def test_add_overlay_ticks_returns_overlay_with_ticks_on_boundary():
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    ov = add_overlay_ticks(ax)
    assert isinstance(ov, CoordinateOverlay)
    assert len(ov.gridticks) > 0
    # The boundary curve should be a single closed polyline.
    assert len(ov.frame_curves) == 1
    assert ov.frame_curves[0].name == "boundary"
    assert ov.frame_curves[0].closed is True
    plt.close(fig)


def test_add_overlay_ticks_produces_both_tick_and_label_artists():
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    ov = add_overlay_ticks(ax)
    assert len(ov.tick_artists) == len(ov.gridticks)
    assert len(ov.label_artists) == len(ov.gridticks)
    plt.close(fig)


def test_add_overlay_ticks_suppresses_default_lon_and_lat_by_default():
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    add_overlay_ticks(ax)
    assert coord_ticklabels(ax.coords[0]).get_visible() is False
    assert coord_ticklabels(ax.coords[1]).get_visible() is False
    plt.close(fig)


def test_add_overlay_ticks_suppress_default_lon_only():
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    add_overlay_ticks(ax, suppress_default="lon")
    assert coord_ticklabels(ax.coords[0]).get_visible() is False
    assert coord_ticklabels(ax.coords[1]).get_visible() is True
    plt.close(fig)


def test_add_overlay_ticks_suppress_default_none_keeps_both():
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    add_overlay_ticks(ax, suppress_default="none")
    assert coord_ticklabels(ax.coords[0]).get_visible() is True
    assert coord_ticklabels(ax.coords[1]).get_visible() is True
    plt.close(fig)


def test_add_overlay_ticks_rejects_unknown_suppress_default():
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    with pytest.raises(ValueError, match="suppress_default must be"):
        add_overlay_ticks(ax, suppress_default="default")
    plt.close(fig)


def test_add_overlay_ticks_custom_lon_lat_vals():
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    ov = add_overlay_ticks(ax, lon_vals=[150, 180, 210],
                           lat_vals=[0, 30])
    assert len(ov.lon_gridlines) == 3
    assert len(ov.lat_gridlines) == 2
    plt.close(fig)


def test_add_overlay_ticks_forwards_tick_and_label_kwargs():
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    ov = add_overlay_ticks(ax,
                            tick_kwargs={"length": 9, "color": "C2",
                                         "lw": 2.0},
                            label_kwargs={"fontsize": 13,
                                          "color": "purple"})
    for line in ov.tick_artists:
        assert line.get_color() == "C2"
        assert line.get_linewidth() == 2.0
    for text in ov.label_artists:
        assert text.get_color() == "purple"
        assert text.get_fontsize() == 13
    plt.close(fig)


def test_add_overlay_ticks_explicit_overlay_frame():
    """An ICRS axes overlaid with the galactic frame should pass
    that frame through to the underlying CoordinateOverlay."""
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0, frame="ICRS")
    ov = add_overlay_ticks(ax, frame="galactic",
                            lon_vals=[0, 90, 180, 270],
                            lat_vals=[0, 30])
    assert ov.frame == "galactic"
    plt.close(fig)


def test_add_overlay_ticks_works_on_rectangular_frame():
    """Rectangular frames should fall back to the 4 bbox spines."""
    fig, ax = _axes("CAR", center=180)
    ov = add_overlay_ticks(ax)
    assert len(ov.frame_curves) == 4
    plt.close(fig)


def test_add_overlay_ticks_top_level_export():
    """The helper should be reachable from the package top level."""
    assert sph.add_overlay_ticks is add_overlay_ticks


# ---- discovery numerical sanity ----

def test_add_overlay_ticks_discovered_points_lie_on_boundary():
    """Every discovered tick must sit on the boundary curve to
    within the polyline's sub-pixel sampling resolution."""
    fig, ax = _axes("SIN", center=(180, 30), fov_deg=70.0)
    ov = add_overlay_ticks(ax)
    boundary = ov.frame_curves[0].xy_pix
    centroid = boundary[:-1].mean(axis=0)
    # All ticks should be near the boundary at radius ~constant.
    radii = [np.linalg.norm(t.xy_pix - centroid) for t in ov.gridticks]
    median_r = np.median(radii)
    for r in radii:
        # SIN boundary is exactly circular — ticks within 1 pixel of
        # the median radius confirm they sit on the spine.
        assert abs(r - median_r) < 1.0
    plt.close(fig)


def test_add_overlay_ticks_ticks_on_robinson_custom_curve():
    """Smoke check that the helper works on a non-FITS custom-curve
    frame (Robinson) — at least one tick on each axis kind."""
    fig, ax = _axes("robinson", center=180)
    ov = add_overlay_ticks(ax, lon_vals=[60, 90, 180, 270],
                            lat_vals=[-60, -30, 0, 30, 60])
    lon_ticks = sum(t.kind == "lon" for t in ov.gridticks)
    lat_ticks = sum(t.kind == "lat" for t in ov.gridticks)
    assert lon_ticks >= 1
    assert lat_ticks >= 1
    plt.close(fig)


def test_add_overlay_ticks_filters_antimeridian_on_envelope_default():
    """On an envelope frame (Robinson here), the default lon_vals
    should exclude the antimeridian of the axes center — otherwise
    that meridian lies on the boundary curve and produces a cascade
    of spurious near-collinear intersection ticks."""
    fig, ax = _axes("robinson", center=180)
    ov = add_overlay_ticks(ax)
    antimeridian = 0.0  # (180 + 180) % 360
    assert antimeridian not in [float(v) for v in ov.lon_vals]
    # Sanity: the remaining default gridlines should produce a
    # tractable number of ticks (≤ 2 per meridian + ≤ 2 per parallel).
    n_lon = len(ov.lon_vals)
    n_lat = len(ov.lat_vals)
    assert len(ov.gridticks) <= 2 * n_lon + 2 * n_lat
    plt.close(fig)


def test_add_overlay_ticks_does_not_filter_antimeridian_on_rectangular():
    """Rectangular frames don't suffer the collinear-meridian
    degeneracy, so the antimeridian filter must NOT kick in for them
    — every default lon_vals entry stays."""
    fig, ax = _axes("CAR", center=180)
    ov = add_overlay_ticks(ax)
    expected = np.arange(0., 360., 30.)
    np.testing.assert_array_equal(np.asarray(ov.lon_vals), expected)
    plt.close(fig)


def test_add_overlay_ticks_explicit_lon_vals_not_filtered():
    """An explicit lon_vals list is passed through unchanged — the
    antimeridian filter only adjusts the *default*."""
    fig, ax = _axes("robinson", center=180)
    ov = add_overlay_ticks(ax, lon_vals=[0, 90, 180])
    assert [float(v) for v in ov.lon_vals] == [0.0, 90.0, 180.0]
    plt.close(fig)


# ---- polyline endpoint extension ----

def test_extrapolate_polyline_endpoints_min_extension():
    """A dense polyline (short segments) should still extend by at
    least the floor amount at each end."""
    xy = np.array([[0., 0.], [1., 0.], [2., 0.]])
    out = _extrapolate_polyline_endpoints(xy, min_extension=10.0)
    # Floor (10) > 1.5 × median segment length (1.5) → extension = 10
    np.testing.assert_allclose(out[0], [-10., 0.])
    np.testing.assert_allclose(out[-1], [12., 0.])


def test_extrapolate_polyline_endpoints_adaptive_density():
    """A sparse polyline (long segments) should extend by 1.5 ×
    median segment length, not the floor."""
    xy = np.array([[0., 0.], [10., 0.], [20., 0.]])
    out = _extrapolate_polyline_endpoints(xy, min_extension=2.0)
    # 1.5 × median (10) = 15 > floor (2) → extension = 15
    np.testing.assert_allclose(out[0], [-15., 0.])
    np.testing.assert_allclose(out[-1], [35., 0.])


def test_extrapolate_polyline_endpoints_short_input_unchanged():
    """A single-point polyline can't be extrapolated — return as-is."""
    xy = np.array([[5., 5.]])
    out = _extrapolate_polyline_endpoints(xy, min_extension=2.0)
    np.testing.assert_array_equal(out, xy)


def test_extrapolate_polyline_endpoints_degenerate_segments_unchanged():
    """A polyline whose only segment is degenerate (zero length) can't
    produce a tangent direction — return as-is."""
    xy = np.array([[1., 1.], [1., 1.], [1., 1.]])
    out = _extrapolate_polyline_endpoints(xy, min_extension=2.0)
    np.testing.assert_array_equal(out, xy)


# ---- intersection dedup ----

def test_intersect_polylines_dedup_merges_coincident_hits():
    """When the polyline extension causes the prepended segment AND
    the original first segment to both cross a boundary at the same
    point, dedup_tol > 0 must keep only one of them."""
    # Vertical "boundary" through x=0
    boundary = np.array([[0., -10.], [0., 10.]])
    # Polyline through x=0 with an extension that crosses again
    poly = np.array([[-1., 0.], [0., 0.], [1., 0.]])
    pts, _, _ = _intersect_polylines(poly, boundary, dedup_tol=0.5)
    # Without dedup we'd expect 2 (segment 0 and segment 1 both
    # touch x=0 at y=0); with dedup_tol=0.5 only 1 survives.
    assert len(pts) == 1


def test_intersect_polylines_dedup_disabled_keeps_both():
    """Setting dedup_tol=0 disables the merge — both coincident
    intersections come back."""
    boundary = np.array([[0., -10.], [0., 10.]])
    poly = np.array([[-1., 0.], [0., 0.], [1., 0.]])
    pts, _, _ = _intersect_polylines(poly, boundary, dedup_tol=0.0)
    assert len(pts) == 2
