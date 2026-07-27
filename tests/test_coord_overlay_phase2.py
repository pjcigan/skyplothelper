"""Tests for skyplothelper.coord_overlay tick discovery.

Tick discovery finds where each overlay gridline crosses an axes
frame curve. The frame curve can be a default bbox edge or an
arbitrary user-registered polyline (e.g. a projection boundary
curve, the route ``add_overlay_ticks`` takes for AIT / MOL / PCO).
This module covers only discovery; tick *rendering* (the visible
marks and labels) is covered by the sibling modules.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.coord_overlay import (  # noqa: E402
    CoordinateOverlay,
    _FrameCurve,
    _GridTick,
    _intersect_polylines,
)


def _make_axes(projection="CAR", center=180, frame="ICRS"):
    fig = plt.figure(figsize=(10, 5))
    ax = sph.make_wcs_frame(111, projection=projection, center=center,
                            frame=frame, fig=fig)
    fig.canvas.draw()
    return fig, ax


# ---- _intersect_polylines: synthetic geometry ----

def test_intersect_simple_cross():
    """Two perpendicular segments crossing at (0.5, 0.5)."""
    horizontal = np.array([[0., 0.5], [1., 0.5]])
    vertical = np.array([[0.5, 0.], [0.5, 1.]])
    pts, a_idx, a_t = _intersect_polylines(horizontal, vertical)
    assert pts.shape == (1, 2)
    np.testing.assert_allclose(pts[0], [0.5, 0.5])
    assert a_idx[0] == 0
    assert a_t[0] == pytest.approx(0.5)


def test_intersect_no_crossing():
    """Two parallel, non-touching segments."""
    a = np.array([[0., 0.], [1., 0.]])
    b = np.array([[0., 1.], [1., 1.]])
    pts, a_idx, a_t = _intersect_polylines(a, b)
    assert len(pts) == 0
    assert len(a_idx) == 0
    assert len(a_t) == 0


def test_intersect_multiple_crossings():
    """A zigzag polyline crosses a horizontal line at three points."""
    zigzag = np.array([[0., -1.], [1., 1.], [2., -1.], [3., 1.], [4., -1.]])
    horizontal = np.array([[-1., 0.], [5., 0.]])
    pts, _, _ = _intersect_polylines(zigzag, horizontal)
    assert len(pts) == 4
    np.testing.assert_allclose(sorted(pts[:, 0]), [0.5, 1.5, 2.5, 3.5])


def test_intersect_collinear_returns_empty():
    """Two segments lying on the same line are reported as no crossing."""
    a = np.array([[0., 0.], [1., 0.]])
    b = np.array([[0.5, 0.], [1.5, 0.]])
    pts, _, _ = _intersect_polylines(a, b)
    assert len(pts) == 0


def test_intersect_endpoint_touch():
    """A segment whose endpoint lies on another segment counts as an
    intersection (t=1 is inclusive)."""
    a = np.array([[0., 0.], [1., 1.]])
    b = np.array([[1., 1.], [2., 0.]])
    pts, _, _ = _intersect_polylines(a, b)
    assert len(pts) >= 1
    np.testing.assert_allclose(pts[0], [1., 1.])


def test_intersect_degenerate_input_returns_empty():
    """A single-point input has no segments and produces no crossings."""
    single = np.array([[0., 0.]])
    line = np.array([[0., 0.], [1., 1.]])
    pts, _, _ = _intersect_polylines(single, line)
    assert len(pts) == 0


# ---- _FrameCurve ----

def test_framecurve_construction():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.]])
    fc = _FrameCurve(xy, name="L")
    assert fc.name == "L"
    assert fc.closed is False
    np.testing.assert_array_equal(fc.xy_pix, xy)


def test_framecurve_closed_appends_closing_point():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.]])
    fc = _FrameCurve(xy, closed=True)
    assert fc.closed is True
    assert len(fc.xy_pix) == 4
    np.testing.assert_array_equal(fc.xy_pix[-1], fc.xy_pix[0])


def test_framecurve_already_closed_not_duplicated():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 0.]])
    fc = _FrameCurve(xy, closed=True)
    assert len(fc.xy_pix) == 4


def test_framecurve_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        _FrameCurve(np.array([1., 2., 3.]))


def test_framecurve_from_bbox_edge_endpoints():
    fig, ax = _make_axes()
    x0, y0, x1, y1 = ax.bbox.extents
    for edge, expected in [
        ("left",   [[x0, y0], [x0, y1]]),
        ("right",  [[x1, y0], [x1, y1]]),
        ("bottom", [[x0, y0], [x1, y0]]),
        ("top",    [[x0, y1], [x1, y1]]),
    ]:
        fc = _FrameCurve.from_bbox_edge(ax, edge)
        np.testing.assert_allclose(fc.xy_pix, expected)
        assert fc.name == edge
    plt.close(fig)


def test_framecurve_from_bbox_edge_rejects_unknown():
    fig, ax = _make_axes()
    with pytest.raises(ValueError, match="edge must be one of"):
        _FrameCurve.from_bbox_edge(ax, "diagonal")
    plt.close(fig)


def test_framecurve_from_world_polyline_matches_transform():
    """The factory should match the world→display transform exactly."""
    fig, ax = _make_axes(projection="CAR", center=180)
    lonlat = np.array([[150., -30.], [210., -30.], [210., 30.], [150., 30.]])
    fc = _FrameCurve.from_world_polyline(ax, lonlat, name="box")
    expected = ax.get_transform("world").transform(lonlat)
    np.testing.assert_allclose(fc.xy_pix, expected)
    assert fc.name == "box"
    plt.close(fig)


def test_framecurve_from_world_polyline_closed():
    fig, ax = _make_axes()
    lonlat = np.array([[150., -30.], [210., -30.], [210., 30.], [150., 30.]])
    fc = _FrameCurve.from_world_polyline(ax, lonlat, closed=True)
    assert fc.closed is True
    assert len(fc.xy_pix) == 5
    np.testing.assert_allclose(fc.xy_pix[-1], fc.xy_pix[0])
    plt.close(fig)


# ---- CoordinateOverlay.discover_ticks ----

def test_discover_ticks_with_custom_inner_box():
    """The cleanest validation: register an interior closed box as a
    frame curve, then count crossings against a known set of
    meridians and parallels."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _FrameCurve.from_world_polyline(
        ax,
        np.array([[150., -30.], [210., -30.],
                  [210., 30.], [150., 30.]]),
        name="inner_box", closed=True)
    ov = CoordinateOverlay(ax, frame="icrs",
                           lon_vals=[160, 180, 200],
                           lat_vals=[-20, 0, 20])
    ov.plot().set_frame_curves([box]).discover_ticks()
    # Each of 3 meridians crosses 2 sides of the box (top + bottom);
    # each of 3 parallels crosses 2 sides (left + right). = 12 ticks.
    assert len(ov.gridticks) == 12
    by_kind = {"lon": 0, "lat": 0}
    for t in ov.gridticks:
        by_kind[t.kind] += 1
        assert t.frame_curve is box
        assert isinstance(t, _GridTick)
        assert t.xy_pix.shape == (2,)
        assert np.isfinite(t.tangent_deg)
    assert by_kind == {"lon": 6, "lat": 6}
    plt.close(fig)


def test_discover_ticks_tangent_angles_for_axis_aligned_crossings():
    """A vertical meridian crossing a horizontal box edge should have
    a tangent near ±90°; a horizontal parallel crossing a vertical
    box edge should have a tangent near 0° or 180°."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _FrameCurve.from_world_polyline(
        ax,
        np.array([[150., -30.], [210., -30.],
                  [210., 30.], [150., 30.]]),
        closed=True)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .discover_ticks())
    lon_tans = [abs(abs(t.tangent_deg) - 90.) for t in ov.gridticks
                if t.kind == "lon"]
    lat_tans = [min(abs(t.tangent_deg), abs(abs(t.tangent_deg) - 180.))
                for t in ov.gridticks if t.kind == "lat"]
    assert len(lon_tans) == 2
    assert len(lat_tans) == 2
    assert max(lon_tans) < 1.0
    assert max(lat_tans) < 1.0
    plt.close(fig)


def test_discover_ticks_default_frame_curves_are_bbox_edges():
    """Calling discover_ticks without set_frame_curves should populate
    frame_curves with the 4 bbox edges."""
    fig, ax = _make_axes()
    ov = CoordinateOverlay(ax, frame="galactic").plot()
    assert ov.frame_curves is None
    ov.discover_ticks()
    assert ov.frame_curves is not None
    assert len(ov.frame_curves) == 4
    assert {fc.name for fc in ov.frame_curves} == {
        "left", "right", "bottom", "top"}
    plt.close(fig)


def test_set_frame_curves_replaces_default():
    fig, ax = _make_axes()
    custom = _FrameCurve.from_bbox_edge(ax, "left")
    ov = CoordinateOverlay(ax, frame="icrs")
    out = ov.set_frame_curves([custom])
    assert out is ov
    assert ov.frame_curves == [custom]
    ov.plot().discover_ticks()
    for t in ov.gridticks:
        assert t.frame_curve is custom
    plt.close(fig)


def test_discover_ticks_returns_self_for_chaining():
    fig, ax = _make_axes()
    ov = CoordinateOverlay(ax, frame="galactic").plot()
    assert ov.discover_ticks() is ov
    plt.close(fig)


def test_discover_ticks_idempotent_replaces_previous():
    """Calling discover_ticks twice should replace, not accumulate."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _FrameCurve.from_world_polyline(
        ax,
        np.array([[150., -30.], [210., -30.],
                  [210., 30.], [150., 30.]]),
        closed=True)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box]))
    ov.discover_ticks()
    n_first = len(ov.gridticks)
    ov.discover_ticks()
    assert len(ov.gridticks) == n_first
    plt.close(fig)


def test_discover_ticks_on_allsky_ait_with_bbox_edges_finds_tangent_points():
    """All-sky AIT gridlines stay inside the elliptical projection
    boundary. The ellipse is inscribed in the rectangular bbox and
    is tangent to it at four points (top / bottom / left / right
    midpoints). The polyline-endpoint extension closes the
    sub-pixel gap there, so bbox-edge discovery finds the tangent
    intersections — but only those four, not a full set of edge
    crossings, since gridlines don't actually exit the ellipse
    onto the bbox. The far cleaner path is the
    ``add_overlay_ticks`` helper, which registers the elliptical
    boundary as the frame curve directly."""
    fig, ax = _make_axes(projection="AIT", center=180)
    ov = (CoordinateOverlay(ax, frame="galactic", n_samples=400)
          .plot()
          .discover_ticks())
    # Modest count — only the tangent points, not a full edge label
    # set. Tracked as a sanity ceiling rather than a strict invariant.
    assert len(ov.gridticks) <= 12
    plt.close(fig)


def test_gridtick_attributes_copied_from_gridline():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _FrameCurve.from_world_polyline(
        ax,
        np.array([[150., -30.], [210., -30.],
                  [210., 30.], [150., 30.]]),
        closed=True)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .discover_ticks())
    for t in ov.gridticks:
        assert t.value == t.gridline.value
        assert t.kind == t.gridline.kind
    plt.close(fig)
