"""overlays.planes (add_plane_overlay + add_great_circle).

Canonical ``tests/test_overlays/test_planes.py`` smoke-tests basic
construction. This file fills in:

  * Returned object types (Line2D list) and counts.
  * Per-plane default-color propagation.
  * The ``parallels`` parameter contributes additional lines.
  * A ``add_great_circle`` correctness check: every sample lies 90°
    away from the supplied pole (within FP tolerance).
  * Galactic-frame axes display the galactic plane horizontally
    (lat≈0 throughout in world coords).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D

from skyplothelper.overlays.planes import add_great_circle, add_plane_overlay
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
# add_plane_overlay — return shape, default color per plane
# ============================================================

@pytest.mark.parametrize("plane, expected_color", [
    ("galactic", "dimgray"),
    ("ecliptic", "goldenrod"),
    ("supergalactic", "steelblue"),
])
def test_add_plane_overlay_default_colors(allsky_axes, plane, expected_color):
    """The default color depends on which plane is requested."""
    fig, ax = allsky_axes
    lines = add_plane_overlay(ax, plane=plane)
    assert isinstance(lines, list)
    assert len(lines) >= 1
    assert all(isinstance(line, Line2D) for line in lines)
    # The expected color should match the first line's RGB tuple
    from matplotlib.colors import to_rgba
    assert lines[0].get_color() == expected_color or \
           tuple(lines[0].get_color()) == to_rgba(expected_color), (
               f"expected color {expected_color!r}, got {lines[0].get_color()!r}"
           )


def test_add_plane_overlay_custom_color_propagates(allsky_axes):
    fig, ax = allsky_axes
    lines = add_plane_overlay(ax, plane="galactic", color="C3")
    from matplotlib.colors import to_rgba
    assert tuple(lines[0].get_color()) == to_rgba("C3") or \
           lines[0].get_color() == "C3"


# ============================================================
# parallels parameter creates additional lines
# ============================================================

def test_add_plane_overlay_parallels_extends_line_list(allsky_axes):
    fig, ax = allsky_axes
    lines_no_par = add_plane_overlay(ax, plane="ecliptic")
    n_solo = len(lines_no_par)

    fig2 = plt.figure(figsize=(10, 5))
    ax2 = make_wcs_frame(111, projection="AIT", center=180, fig=fig2)
    fig2.canvas.draw()
    lines_with_par = add_plane_overlay(
        ax2, plane="ecliptic", parallels=[-10, 10],
    )
    # The two parallels should add at least 2 additional lines.
    assert len(lines_with_par) >= n_solo + 2


# ============================================================
# add_great_circle — every sample is 90° from the pole
# ============================================================

def test_add_great_circle_returns_lines(allsky_axes):
    fig, ax = allsky_axes
    lines = add_great_circle(
        ax, pole_lon=0.0, pole_lat=90.0, frame="pole",
        color="C3", lw=1.5,
    )
    assert isinstance(lines, list)
    assert len(lines) >= 1
    assert all(isinstance(line, Line2D) for line in lines)


def test_add_great_circle_samples_lie_90_deg_from_pole():
    """Geometric correctness: every point on a great circle is exactly
    90° away from its pole. This catches subtle frame / sampling
    regressions in add_great_circle."""
    pole_lon, pole_lat = 192.86, 27.13  # galactic-pole (ICRS)
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    lines = add_great_circle(
        ax, pole_lon=pole_lon, pole_lat=pole_lat, frame="pole",
        n_points=200,
    )
    # Pull the world-coord samples directly off the line(s).
    # add_great_circle plots in world-coords-via-transform, so the line
    # data are the raw lon/lat values.
    lons = np.concatenate([line.get_xdata() for line in lines])
    lats = np.concatenate([line.get_ydata() for line in lines])

    # Drop NaNs (segment breaks at antimeridian)
    valid = np.isfinite(lons) & np.isfinite(lats)
    lons = lons[valid]
    lats = lats[valid]
    assert len(lons) > 50, "expected at least 50 valid sample points"

    # Great-circle distance from pole to each sample
    cos_d = (np.sin(np.radians(pole_lat)) * np.sin(np.radians(lats)) +
             np.cos(np.radians(pole_lat)) * np.cos(np.radians(lats)) *
             np.cos(np.radians(lons - pole_lon)))
    d = np.degrees(np.arccos(np.clip(cos_d, -1, 1)))
    assert np.allclose(d, 90.0, atol=0.5), (
        f"every sample must be 90° from pole; "
        f"got d range [{d.min():.4f}, {d.max():.4f}]"
    )


# ============================================================
# Pixel-space jump detection on multi-face projections
# ============================================================

@pytest.mark.parametrize("projection", ["CSC", "TSC", "QSC", "XPH", "HPX"])
def test_great_circle_splits_at_projection_face_boundaries(projection):
    """Cube + HEALPix projections have internal face boundaries where
    adjacent (lon, lat) samples project to widely-separated pixels.
    The GP overlay must split into multiple segments at those boundaries
    rather than drawing straight lines across the canvas (the
    criss-cross artifact)."""
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection=projection, center=180, fig=fig)
    fig.canvas.draw()
    lines = add_plane_overlay(ax, plane="galactic")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_thresh = 0.25 * abs(xlim[1] - xlim[0])
    y_thresh = 0.25 * abs(ylim[1] - ylim[0])

    for line in lines:
        lons = np.asarray(line.get_xdata(), dtype=float)
        lats = np.asarray(line.get_ydata(), dtype=float)
        if lons.size < 2:
            continue
        x_pix, y_pix = ax.wcs.world_to_pixel_values(lons, lats)
        with np.errstate(invalid="ignore"):
            dx = np.abs(np.diff(x_pix))
            dy = np.abs(np.diff(y_pix))
        # Within a single segment, no consecutive sample pair should
        # exceed the pixel-jump threshold — that would be the very
        # criss-cross artifact this cluster fixed.
        assert not np.any(dx > x_thresh), (
            f"{projection}: segment has dx={dx.max():.1f} > {x_thresh:.1f} "
            f"(pixel-space jump within segment — line-splitting failed)"
        )
        assert not np.any(dy > y_thresh), (
            f"{projection}: segment has dy={dy.max():.1f} > {y_thresh:.1f} "
            f"(pixel-space jump within segment — line-splitting failed)"
        )


def test_great_circle_smooth_projections_unchanged_segment_count():
    """On smooth projections (AIT, MOL, CAR), the GP should remain a
    single continuous segment after splitting — pixel-jump detection
    must NOT introduce spurious splits where there are no
    discontinuities."""
    for projection in ("AIT", "MOL", "CAR"):
        fig = plt.figure(figsize=(10, 5))
        ax = make_wcs_frame(111, projection=projection, center=180, fig=fig)
        fig.canvas.draw()
        lines = add_plane_overlay(ax, plane="galactic")
        # Default add_plane_overlay returns one Line2D for the great
        # circle (no parallels) — pixel-jump split must not fragment it.
        assert len(lines) == 1, (
            f"{projection}: smooth projection should yield 1 GP segment, "
            f"got {len(lines)}"
        )
        plt.close(fig)


# ============================================================
# Galactic-frame axes display the galactic plane horizontally
# ============================================================

def test_galactic_plane_is_horizontal_in_galactic_frame_axes():
    """When the axes are themselves Galactic, the galactic plane
    should be a near-horizontal line (lat≈0 in world coords)."""
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection="AIT", center=0,
                        frame="Galactic", fig=fig)
    fig.canvas.draw()
    lines = add_plane_overlay(ax, plane="galactic")
    lats = np.concatenate([line.get_ydata() for line in lines])
    lats = lats[np.isfinite(lats)]
    assert np.max(np.abs(lats)) < 1.0, (
        f"galactic plane in galactic-frame axes should have |lat|<1°, "
        f"got max |lat|={np.max(np.abs(lats))}"
    )
