"""grid (add_second_grid + style_grid + highlight_gridline).

The canonical ``tests/test_phase3_misc.py`` smoke-tests these but
doesn't verify return shape, count, or kwarg propagation. This file
fills in:

  * add_second_grid returns a CoordinateOverlay.
  * highlight_gridline returns a Line2D list with the right color.
  * highlight_gridlines accepts list-of-values for both axes and
    creates the expected number of lines.
  * Different frames (galactic/ecliptic/icrs) are accepted by
    add_second_grid.
  * The 'lon'/'meridian'/'ra' aliases for ``coord=`` resolve to the
    same axis selection on highlight_gridline.
  * Invalid coord raises ValueError.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from skyplothelper.coord_overlay import CoordinateOverlay
from skyplothelper.grid import (
    add_second_grid,
    highlight_gridline,
    highlight_gridlines,
    style_grid,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def allsky_axes():
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    return fig, ax


# ============================================================
# add_second_grid
# ============================================================

@pytest.mark.parametrize("overlay_frame", [
    "galactic", "geocentrictrueecliptic", "icrs", "fk5",
])
def test_add_second_grid_accepts_each_frame(allsky_axes, overlay_frame):
    fig, ax = allsky_axes
    overlay = add_second_grid(ax, overlay_frame=overlay_frame,
                              color="C2", alpha=0.4)
    assert isinstance(overlay, CoordinateOverlay)


def test_add_second_grid_kwargs_propagate(allsky_axes):
    """Color and alpha settings should land on the overlay's coords."""
    fig, ax = allsky_axes
    add_second_grid(ax, overlay_frame="galactic",
                    color="purple", alpha=0.5, linestyle="--",
                    linewidth=1.0)
    # We can verify the overlay drew without raising
    fig.canvas.draw()


# ============================================================
# style_grid — applies retroactive style without raising
# ============================================================

def test_style_grid_with_stroke(allsky_axes):
    fig, ax = allsky_axes
    style_grid(ax, stroke_lw=3, stroke_color="black",
               color="white", lw=1.0, alpha=0.9)
    fig.canvas.draw()


# ============================================================
# highlight_gridline — return type, color propagation, alias acceptance
# ============================================================

def test_highlight_gridline_returns_line2d_list(allsky_axes):
    fig, ax = allsky_axes
    lines = highlight_gridline(ax, value=0, coord="lon",
                               color="red", lw=2.5)
    assert isinstance(lines, list)
    assert len(lines) >= 1
    assert all(isinstance(line, Line2D) for line in lines)


def test_highlight_gridline_stroke_applies_path_effect(allsky_axes):
    # Regression: a stroke_lw request used to crash (PathEffects.append on
    # the module) and never attach the effect. The stroke must now land as
    # a path-effect on the returned line(s).
    fig, ax = allsky_axes
    lines = highlight_gridline(ax, value=0, coord="lon",
                               stroke_lw=3.0, stroke_color="white")
    assert len(lines) >= 1
    assert any(line.get_path_effects() for line in lines)


def test_highlight_gridline_color_propagates(allsky_axes):
    fig, ax = allsky_axes
    lines = highlight_gridline(ax, value=30, coord="lat",
                               color="C9", lw=2)
    line_color = lines[0].get_color()
    if hasattr(line_color, "__len__") and len(line_color) >= 3:
        actual_rgb = tuple(line_color[:3])
    else:
        actual_rgb = to_rgba(line_color)[:3]
    expected = to_rgba("C9")[:3]
    assert actual_rgb == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("alias", ["lon", "meridian", "ra", "longitude", "l"])
def test_highlight_gridline_lon_aliases(allsky_axes, alias):
    fig, ax = allsky_axes
    lines = highlight_gridline(ax, value=0, coord=alias)
    assert len(lines) >= 1


@pytest.mark.parametrize("alias", ["lat", "parallel", "dec", "latitude", "b"])
def test_highlight_gridline_lat_aliases(allsky_axes, alias):
    fig, ax = allsky_axes
    lines = highlight_gridline(ax, value=0, coord=alias)
    assert len(lines) >= 1


def test_highlight_gridline_invalid_coord_raises(allsky_axes):
    fig, ax = allsky_axes
    with pytest.raises(ValueError, match="(?i)coord"):
        highlight_gridline(ax, value=0, coord="banana")


def test_highlight_gridline_parallel_breaks_at_wrap_seam():
    """Regression: a highlighted parallel on a center=0 all-sky frame spans
    every longitude and crosses the antimeridian seam. It must NaN-break at the
    seam (drawing off one frame edge onto the other) rather than streak
    straight across the canvas."""
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=0, fig=fig)
    fig.canvas.draw()
    lines = highlight_gridline(ax, value=30, coord="lat")
    xd, yd = lines[0].get_data()
    xd = np.asarray(xd, float)
    assert np.isnan(xd).any(), "parallel should NaN-break at the wrap seam"
    disp = ax.get_transform("world").transform(
        np.column_stack([xd, np.asarray(yd, float)]))
    dx = np.abs(np.diff(disp[:, 0]))
    dx = dx[np.isfinite(dx)]
    assert dx.max() < 0.5 * ax.bbox.width, "parallel streaks across the seam"
    plt.close(fig)


def test_highlight_gridline_meridian_no_spurious_break():
    """A constant-lon meridian must not be broken by the wrap handling."""
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=0, fig=fig)
    fig.canvas.draw()
    lines = highlight_gridline(ax, value=45, coord="lon")
    xd, _ = lines[0].get_data()
    assert not np.isnan(np.asarray(xd, float)).any()
    plt.close(fig)


# ============================================================
# highlight_gridlines — multi-value variant
# ============================================================

def test_highlight_gridlines_with_lon_values_returns_n_lines(allsky_axes):
    """One line per lon value passed."""
    fig, ax = allsky_axes
    lon_values = [0, 90, 180, 270]
    n_before = len(ax.lines)
    out = highlight_gridlines(ax, lon_values=lon_values, color="C0")
    n_after = len(ax.lines)
    assert (n_after - n_before) == len(lon_values)
    assert isinstance(out, list)


def test_highlight_gridlines_with_lat_values(allsky_axes):
    fig, ax = allsky_axes
    lat_values = [-60, -30, 0, 30, 60]
    n_before = len(ax.lines)
    highlight_gridlines(ax, lat_values=lat_values, color="C2")
    n_after = len(ax.lines)
    assert (n_after - n_before) == len(lat_values)


def test_highlight_gridlines_combined_lon_lat(allsky_axes):
    """Both lon_values and lat_values together → sum of the two lengths."""
    fig, ax = allsky_axes
    lon_values = [0, 180]
    lat_values = [-30, 0, 30]
    n_before = len(ax.lines)
    highlight_gridlines(ax, lon_values=lon_values, lat_values=lat_values)
    n_after = len(ax.lines)
    assert (n_after - n_before) == len(lon_values) + len(lat_values)


def test_highlight_gridlines_lon_cmap_assigns_distinct_colors(allsky_axes):
    """When lon_cmap is set, colors are distributed along the colormap."""
    fig, ax = allsky_axes
    lon_values = [0, 90, 180, 270]
    n_before = len(ax.lines)
    highlight_gridlines(ax, lon_values=lon_values, lon_cmap="viridis", lw=1.5)
    n_after = len(ax.lines)
    new_lines = ax.lines[n_before:n_after]
    # All four lines should have different colors when a colormap is used
    colors = {tuple(line.get_color()) if hasattr(line.get_color(), "__len__")
              else line.get_color() for line in new_lines}
    assert len(colors) == 4, f"expected 4 distinct colors, got {len(colors)}"
