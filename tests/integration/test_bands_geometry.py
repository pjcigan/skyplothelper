"""geometry bands + Tissot return-value verification.

Verifies that ``add_latitude_band``, ``add_longitude_band``,
``add_great_circle_band``, ``add_frame_band`` (patch + contour modes),
and ``tissot`` return / mutate as expected after the merge.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import PathPatch

from skyplothelper.geometry.bands import (
    add_frame_band,
    add_great_circle_band,
    add_latitude_band,
    add_longitude_band,
)
from skyplothelper.geometry.tissot import tissot
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def allsky_axes():
    """An AIT all-sky axes — the natural canvas for these bands."""
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    return fig, ax


# ============================================================
# add_latitude_band
# ============================================================

def test_add_latitude_band_returns_patch_list(allsky_axes):
    fig, ax = allsky_axes
    patches = add_latitude_band(ax, lat_min=-10, lat_max=10,
                                facecolor="orange", alpha=0.4)
    assert isinstance(patches, list)
    assert len(patches) >= 1
    assert all(isinstance(p, PathPatch) for p in patches)


def test_add_latitude_band_facecolor_propagates(allsky_axes):
    fig, ax = allsky_axes
    patches = add_latitude_band(ax, lat_min=20, lat_max=40,
                                facecolor="red", alpha=0.5)
    fc = patches[0].get_facecolor()
    # Red is RGBA (1, 0, 0, alpha) — alpha ~0.5
    assert fc[0] == pytest.approx(1.0)
    assert fc[1] == pytest.approx(0.0)


# ============================================================
# add_longitude_band
# ============================================================

def test_add_longitude_band_returns_patch_list(allsky_axes):
    fig, ax = allsky_axes
    patches = add_longitude_band(ax, lon_min=120, lon_max=240,
                                 facecolor="blue", alpha=0.3)
    assert isinstance(patches, list)
    assert len(patches) >= 1
    assert all(isinstance(p, PathPatch) for p in patches)


# ============================================================
# add_great_circle_band
# ============================================================

def test_add_great_circle_band_returns_patch_list(allsky_axes):
    fig, ax = allsky_axes
    patches = add_great_circle_band(
        ax, ra_pole=0.0, dec_pole=90.0, half_width=15,
        facecolor="green", alpha=0.4,
    )
    assert isinstance(patches, list)
    assert len(patches) >= 1
    assert all(isinstance(p, PathPatch) for p in patches)


# ============================================================
# add_frame_band — both rendering modes
# ============================================================

def test_add_frame_band_patch_mode_returns_patch_list(allsky_axes):
    fig, ax = allsky_axes
    out = add_frame_band(
        ax, lat_min=-5, lat_max=5, frame="galactic",
        backend="patch", facecolor="orange", alpha=0.4,
    )
    assert isinstance(out, list)
    assert len(out) >= 1


def test_add_frame_band_contour_mode_returns_artists(allsky_axes):
    """Contour mode returns either a single QuadContourSet or a
    (fill_set, edge_set) tuple. Verify a non-empty result either way."""
    fig, ax = allsky_axes
    out = add_frame_band(
        ax, lat_min=-5, lat_max=5, frame="galactic",
        backend="contour", facecolor="purple", alpha=0.4,
    )
    # Must return something usable (not None)
    assert out is not None


# ============================================================
# tissot
# ============================================================

def test_tissot_returns_patch_list(allsky_axes):
    fig, ax = allsky_axes
    patches = tissot(ax, rad_deg=5, resolution=50,
                     facecolor="C2", edgecolor="darkgreen", alpha=0.5)
    assert isinstance(patches, list)
    # Default lon/lat grids are 6x6 → 36 indicatrices each contributing >=1 patch
    assert len(patches) >= 36


def test_tissot_custom_grid_yields_correct_count(allsky_axes):
    """Explicit lon/lat grids → meshgrid product → that-many indicatrices."""
    fig, ax = allsky_axes
    patches = tissot(
        ax, rad_deg=3, lons=[0, 90, 180, 270], lats=[-30, 30],
        resolution=40, facecolor="C5", alpha=0.4,
    )
    # 4 lons × 2 lats = 8 indicatrices, each contributing ≥1 patch
    assert len(patches) >= 8


def test_add_frame_band_stroke_reaches_visible_outline():
    """stroke_color must land on the DRAWN band outline (the ax.plot edge
    lines), not the transparent+lw0 fill patch where it's invisible."""
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure(figsize=(5, 5))
    ax = make_wcs_frame(projection="AIT", center=0, fig=fig, grid=False)
    add_frame_band(ax, -15, 15, frame="galactic", edgecolor="yellow",
                   stroke_color="k", stroke_lw=4)
    stroked = [ln for ln in ax.lines if ln.get_path_effects()]
    assert stroked, "no band edge line carries the stroke"
    plt.close(fig)


def test_add_frame_band_no_stroke_leaves_outline_plain():
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure(figsize=(5, 5))
    ax = make_wcs_frame(projection="AIT", center=0, fig=fig, grid=False)
    add_frame_band(ax, -15, 15, frame="galactic", edgecolor="yellow")
    assert not any(ln.get_path_effects() for ln in ax.lines)
    plt.close(fig)


def test_add_frame_band_contour_stroke_on_edge():
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure(figsize=(5, 5))
    ax = make_wcs_frame(projection="AIT", center=0, fig=fig, grid=False)
    res = add_frame_band(ax, -15, 15, frame="galactic", backend="contour",
                         edgecolor="yellow", stroke_color="k", stroke_lw=4)
    edge_set = res[1] if isinstance(res, tuple) else None
    assert edge_set is not None
    try:
        has_pe = bool(edge_set.get_path_effects())
    except AttributeError:
        has_pe = any(c.get_path_effects() for c in edge_set.collections)
    assert has_pe, "contour edge not stroked"
    plt.close(fig)
