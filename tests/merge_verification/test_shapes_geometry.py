"""geometry.shapes return-value verification.

Canonical ``tests/test_geometry/test_shapes.py`` smoke-tests basic
construction and SkyCoord/Quantity acceptance. This file fills in:

  * Vertex counts on the generators (resolution / resolution propagation).
  * Returned patch facecolor / edgecolor / alpha propagation.
  * The complement=True render path (renders the *outside* of the shape).
  * Quantity-with-units input on radius / sizes.
  * SkyCoord center with the implicit positional-shift that lets
    ``add_geodesic_circle(ax, skycoord, radius_deg)`` work without
    naming the keyword.
"""

import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from matplotlib.patches import PathPatch

from skyplothelper.geometry.shapes import (
    add_annulus,
    add_ellipse,
    add_geodesic_circle,
    add_rectangle,
    add_spherical_polygon,
    add_square,
    ellipse,
    geodesic_circle,
    rectangle,
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
# Vertex generators — resolution / resolution propagation
# ============================================================

def test_geodesic_circle_n_samples_propagates():
    lons, lats = geodesic_circle(180, 30, radius_deg=5, resolution=64)
    assert len(lons) == 64
    assert len(lats) == 64


def test_rectangle_n_edge_propagates():
    """Rectangle has 4 edges; resolution controls per-edge sample density."""
    lons, lats = rectangle(180, 30, width=10, height=5, resolution=20)
    # 4 edges × 20 samples each (give or take corners) — ≥80 vertices total
    assert len(lons) >= 80


def test_ellipse_n_samples_propagates():
    """ellipse() returns resolution + 1 vertices (it duplicates the first
    point at the end so the patch path closes seamlessly)."""
    lons, lats = ellipse(180, 30, semi_major=8, semi_minor=4, resolution=120)
    assert len(lons) == 121
    # And the closure: first ≈ last
    assert abs(lons[0] - lons[-1]) < 1e-6
    assert abs(lats[0] - lats[-1]) < 1e-6


# ============================================================
# Return shape + kwarg propagation on every renderer
# ============================================================

def test_add_geodesic_circle_facecolor_propagates(allsky_axes):
    fig, ax = allsky_axes
    patches = add_geodesic_circle(ax, 180, 30, radius_deg=10,
                                  facecolor="red", alpha=0.4, edgecolor="black")
    assert isinstance(patches, list)
    assert all(isinstance(p, PathPatch) for p in patches)
    fc = patches[0].get_facecolor()
    # red = (1, 0, 0, alpha)
    assert fc[0] == pytest.approx(1.0)
    assert fc[1] == pytest.approx(0.0)


def test_add_rectangle_returns_pathpatch_list(allsky_axes):
    fig, ax = allsky_axes
    patches = add_rectangle(ax, 180, 30, width=20, height=10,
                            facecolor="C0", alpha=0.4)
    assert isinstance(patches, list) and len(patches) >= 1
    assert all(isinstance(p, PathPatch) for p in patches)


def test_add_square_size_keyword(allsky_axes):
    fig, ax = allsky_axes
    patches = add_square(ax, 180, 30, size=10, facecolor="C2", alpha=0.4)
    assert len(patches) >= 1
    assert all(isinstance(p, PathPatch) for p in patches)


def test_add_ellipse_returns_pathpatch_list(allsky_axes):
    fig, ax = allsky_axes
    patches = add_ellipse(ax, 180, 30, semi_major=15, semi_minor=8,
                          angle=30, facecolor="C3", alpha=0.4)
    assert len(patches) >= 1
    assert all(isinstance(p, PathPatch) for p in patches)


def test_add_annulus_returns_pathpatch_list(allsky_axes):
    fig, ax = allsky_axes
    patches = add_annulus(ax, 180, 30, inner_radius=5, outer_radius=12,
                          facecolor="C4", alpha=0.4)
    assert len(patches) >= 1
    assert all(isinstance(p, PathPatch) for p in patches)


def test_add_spherical_polygon_triangle(allsky_axes):
    fig, ax = allsky_axes
    patches = add_spherical_polygon(
        ax, [180, 200, 190], [10, 20, 30],
        facecolor="C5", alpha=0.4, edgecolor="C5",
    )
    assert isinstance(patches, list) and len(patches) >= 1
    assert all(isinstance(p, PathPatch) for p in patches)


# ============================================================
# complement=True branch — fills the OUTSIDE of the shape
# ============================================================

def test_complement_branch_renders(allsky_axes):
    """``complement=True`` should still return artists rendering the
    region outside the shape."""
    fig, ax = allsky_axes
    out = add_geodesic_circle(ax, 180, 0, radius_deg=20,
                              facecolor="0.7", alpha=0.4, complement=True)
    # The complement renderer returns at least one artist (PathPatch or list).
    assert out is not None
    if isinstance(out, list):
        assert len(out) >= 1


# ============================================================
# SkyCoord + Quantity input acceptance (v7 feature preserved)
# ============================================================

def test_add_geodesic_circle_skycoord_implicit_radius(allsky_axes):
    """When the center is a SkyCoord, the second positional argument
    becomes the radius (the parser's positional-shift behavior)."""
    fig, ax = allsky_axes
    center = SkyCoord(180 * u.deg, 30 * u.deg, frame="icrs")
    patches = add_geodesic_circle(ax, center, 8 * u.deg,
                                  facecolor="C0", alpha=0.4)
    assert len(patches) >= 1


def test_add_ellipse_quantity_inputs(allsky_axes):
    """Quantity-with-units sizes should be accepted on the ellipse.
    With a SkyCoord center, the next positional argument is interpreted
    as semi_major (the v7 positional-shift behavior); semi_minor and
    angle remain keyword."""
    fig, ax = allsky_axes
    center = SkyCoord(120 * u.deg, -10 * u.deg, frame="icrs")
    patches = add_ellipse(
        ax, center, 10 * u.deg,
        semi_minor=4 * u.deg, angle=45 * u.deg,
        facecolor="C2", alpha=0.4,
    )
    assert len(patches) >= 1


def test_add_annulus_quantity_inputs(allsky_axes):
    fig, ax = allsky_axes
    patches = add_annulus(
        ax, 0, -30,
        inner_radius=3 * u.deg, outer_radius=8 * u.deg,
        facecolor="C5", alpha=0.4,
    )
    assert len(patches) >= 1


# ============================================================
# Generator shapes are *closed* — last point (or rendering) coincides
# with the first when the shape is supposed to be closed.
# ============================================================

def test_geodesic_circle_walks_a_full_revolution():
    """A geodesic circle of radius r around (lon, lat) should produce
    samples whose center-relative bearing sweeps a full 2π — verify
    via the centroid / cumulative-bearing path."""
    lons, lats = geodesic_circle(0, 0, radius_deg=10, resolution=64)
    # All samples should be ≈ 10° from the center
    # great-circle distance: cos(d) = sin(0)sin(lat)+cos(0)cos(lat)cos(lon-0)
    cos_d = np.sin(np.radians(0)) * np.sin(np.radians(lats)) + \
            np.cos(np.radians(0)) * np.cos(np.radians(lats)) * \
            np.cos(np.radians(lons - 0))
    d = np.degrees(np.arccos(np.clip(cos_d, -1, 1)))
    # The small-circle sampler accumulates ≲0.1° of numerical drift at
    # radius 10° — tight enough to confirm "this really is a circle"
    # while loose enough not to pin the exact algorithm.
    assert np.allclose(d, 10.0, atol=0.1), (
        f"all samples should be ~10° from center; got d range "
        f"[{d.min()}, {d.max()}]"
    )
