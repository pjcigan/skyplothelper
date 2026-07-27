"""Tests for skyplothelper.coord_overlay tick-mark rendering.

Tick rendering draws short line marks at every discovered tick
position, oriented along the gridline tangent and extending outward
(or inward / both) from the frame curve. The formatted text labels
on top of these are covered by the label-rendering sibling module.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.coord_overlay import (  # noqa: E402
    CoordinateOverlay,
    _FrameCurve,
)


def _make_axes(projection="CAR", center=180, frame="ICRS"):
    fig = plt.figure(figsize=(10, 5))
    ax = sph.make_wcs_frame(111, projection=projection, center=center,
                            frame=frame, fig=fig)
    fig.canvas.draw()
    return fig, ax


def _inner_box_curve(ax, lons=(150., 210.), lats=(-30., 30.)):
    lo0, lo1 = lons
    la0, la1 = lats
    return _FrameCurve.from_world_polyline(
        ax,
        np.array([[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1]]),
        closed=True, name="box")


# ---- _FrameCurve.outward_at ----

@pytest.mark.parametrize("edge,expected", [
    ("left",   [-1., 0.]),
    ("right",  [1., 0.]),
    ("bottom", [0., -1.]),
    ("top",    [0., 1.]),
])
def test_outward_at_bbox_edges(edge, expected):
    fig, ax = _make_axes()
    fc = _FrameCurve.from_bbox_edge(ax, edge)
    out = fc.outward_at([100., 100.])
    np.testing.assert_allclose(out, expected)
    plt.close(fig)


def test_outward_at_closed_curve_points_away_from_centroid():
    """For a closed inner box, outward_at should point radially away
    from the box centroid for any vertex."""
    xy = np.array([[10., 10.], [30., 10.], [30., 20.], [10., 20.]])
    fc = _FrameCurve(xy, closed=True, name="custom")
    centroid = xy.mean(axis=0)
    for vertex in xy:
        out = fc.outward_at(vertex)
        expected = vertex - centroid
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(out, expected)


def test_outward_at_degenerate_returns_finite_unit_vector():
    """A query point at the centroid should not produce NaN."""
    xy = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
    fc = _FrameCurve(xy, closed=True, name="custom")
    out = fc.outward_at(xy.mean(axis=0))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(np.linalg.norm(out), 1.0)


def test_outward_at_unit_norm():
    fig, ax = _make_axes()
    for edge in ("left", "right", "bottom", "top"):
        fc = _FrameCurve.from_bbox_edge(ax, edge)
        assert np.linalg.norm(fc.outward_at([0., 0.])) == pytest.approx(1.0)
    plt.close(fig)


# ---- render_ticks: basic ----

def test_render_ticks_one_artist_per_gridtick():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[160, 180, 200],
                            lat_vals=[-20, 0, 20])
          .plot()
          .set_frame_curves([box])
          .discover_ticks()
          .render_ticks())
    assert len(ov.tick_artists) == len(ov.gridticks) == 12
    for artist in ov.tick_artists:
        assert isinstance(artist, Line2D)
    plt.close(fig)


def test_render_ticks_auto_discovers_if_needed():
    """Calling render_ticks without an explicit discover_ticks should
    still produce ticks."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_ticks())
    assert len(ov.gridticks) > 0
    assert len(ov.tick_artists) == len(ov.gridticks)
    plt.close(fig)


def test_render_ticks_returns_self_for_chaining():
    fig, ax = _make_axes()
    ov = (CoordinateOverlay(ax, frame="galactic")
          .plot()
          .set_frame_curves([_inner_box_curve(ax)]))
    assert ov.render_ticks() is ov
    plt.close(fig)


def test_render_ticks_pixel_length_via_transdata():
    """Tick lines are stored in axes data coords (so rendering
    survives a savefig at a different dpi than construction), but
    their display-space length must match the requested pixel length."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_ticks(length=10))
    for line in ov.tick_artists:
        assert line.get_transform() is ax.transData
        data_xy = np.column_stack([line.get_xdata(), line.get_ydata()])
        disp_xy = ax.transData.transform(data_xy)
        L = np.linalg.norm(disp_xy[1] - disp_xy[0])
        assert L == pytest.approx(10., abs=1e-6)
    plt.close(fig)


def test_render_ticks_styling_kwargs_applied():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_ticks(length=8, lw=2.5, color="magenta", zorder=42))
    for line in ov.tick_artists:
        assert line.get_color() == "magenta"
        assert line.get_linewidth() == 2.5
        assert line.get_zorder() == 42
    plt.close(fig)


def test_render_ticks_default_clip_off():
    """clip_on must default to False so out-direction ticks stay
    visible past the axes bbox."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .render_ticks())
    for line in ov.tick_artists:
        assert line.get_clip_on() is False
    plt.close(fig)


# ---- render_ticks: directionality ----

def _line_endpoints_pix(ax, line):
    data_xy = np.column_stack([line.get_xdata(), line.get_ydata()])
    return ax.transData.transform(data_xy)


def test_render_ticks_direction_out_extends_outward():
    """For a tick on the right side of a centered inner box, an
    outward tick endpoint must have larger x than the starting
    intersection point (compared in display pixels)."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    # Single parallel at lat=0, crosses left + right of box
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .discover_ticks()
          .render_ticks(length=10, direction="out"))
    centroid = box.xy_pix.mean(axis=0)
    for line, tick in zip(ov.tick_artists, ov.gridticks):
        disp = _line_endpoints_pix(ax, line)
        # Start should be on the box edge (the tick's xy_pix)
        np.testing.assert_allclose(disp[0], tick.xy_pix, atol=1e-6)
        # End should be further from the centroid than start
        d_start = np.linalg.norm(disp[0] - centroid)
        d_end = np.linalg.norm(disp[1] - centroid)
        assert d_end > d_start
    plt.close(fig)


def test_render_ticks_direction_in_extends_inward():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[], lat_vals=[0])
          .plot()
          .set_frame_curves([box])
          .discover_ticks()
          .render_ticks(length=10, direction="in"))
    centroid = box.xy_pix.mean(axis=0)
    for line, tick in zip(ov.tick_artists, ov.gridticks):
        disp = _line_endpoints_pix(ax, line)
        d_start = np.linalg.norm(disp[0] - centroid)
        d_end = np.linalg.norm(disp[1] - centroid)
        assert d_end < d_start
    plt.close(fig)


def test_render_ticks_direction_both_centered_on_intersection():
    """direction='both' should produce a tick of total length
    2*length (in display pixels) centered on the intersection point."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[])
          .plot()
          .set_frame_curves([box])
          .discover_ticks()
          .render_ticks(length=10, direction="both"))
    for line, tick in zip(ov.tick_artists, ov.gridticks):
        disp = _line_endpoints_pix(ax, line)
        mid = (disp[0] + disp[1]) / 2
        np.testing.assert_allclose(mid, tick.xy_pix, atol=1e-6)
        L = np.linalg.norm(disp[1] - disp[0])
        assert L == pytest.approx(20., abs=1e-6)
    plt.close(fig)


def test_render_ticks_rejects_unknown_direction():
    fig, ax = _make_axes()
    ov = (CoordinateOverlay(ax, frame="galactic")
          .plot()
          .set_frame_curves([_inner_box_curve(ax)]))
    with pytest.raises(ValueError, match="direction must be"):
        ov.render_ticks(direction="sideways")
    plt.close(fig)


# ---- render_ticks: gridline-tangent orientation ----

def test_render_ticks_vertical_meridian_tangent_perpendicular_to_box_top():
    """A vertical meridian crossing the (horizontal) box top edge
    should produce a tick that is itself vertical (in display space)."""
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs",
                            lon_vals=[180], lat_vals=[])
          .plot()
          .set_frame_curves([box])
          .discover_ticks()
          .render_ticks(length=10))
    for line in ov.tick_artists:
        disp = _line_endpoints_pix(ax, line)
        dx = disp[1, 0] - disp[0, 0]
        dy = disp[1, 1] - disp[0, 1]
        assert abs(dx) < 1e-6
        assert abs(dy) == pytest.approx(10., abs=1e-6)
    plt.close(fig)


# ---- draw_frame_curves ----

def test_draw_frame_curves_one_artist_per_curve():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs")
          .set_frame_curves([box])
          .draw_frame_curves())
    assert len(ov.frame_curve_artists) == 1
    assert isinstance(ov.frame_curve_artists[0], Line2D)
    plt.close(fig)


def test_draw_frame_curves_default_fills_bbox_edges():
    """Without an explicit set_frame_curves, draw_frame_curves should
    auto-build the 4 bbox edges (matching discover_ticks behavior)."""
    fig, ax = _make_axes()
    ov = CoordinateOverlay(ax, frame="galactic")
    assert ov.frame_curves is None
    ov.draw_frame_curves()
    assert len(ov.frame_curves) == 4
    assert len(ov.frame_curve_artists) == 4
    plt.close(fig)


def test_draw_frame_curves_styling_applied():
    fig, ax = _make_axes(projection="CAR", center=180)
    box = _inner_box_curve(ax)
    ov = (CoordinateOverlay(ax, frame="icrs")
          .set_frame_curves([box])
          .draw_frame_curves(color="orange", lw=2.5, ls="--", zorder=11))
    line = ov.frame_curve_artists[0]
    assert line.get_color() == "orange"
    assert line.get_linewidth() == 2.5
    assert line.get_linestyle() == "--"
    assert line.get_zorder() == 11
    plt.close(fig)
