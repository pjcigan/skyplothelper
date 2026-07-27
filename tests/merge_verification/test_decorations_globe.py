"""globe decorations verification.

Verifies that the decoration helpers (``plot_ortho_grid``,
``add_checkered_border``, ``add_compass_rose``,
``add_scale_bar_cylindrical``, ``add_scale_bar_curved_parallel``) add the
expected artists to the axes after the merge. These helpers all return
None (they mutate the axes in place), so verification counts
``ax.lines`` / ``ax.patches`` / ``ax.texts`` deltas instead.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.offsetbox import AnchoredOffsetbox

from skyplothelper.globe.decorations import (
    add_checkered_border,
    add_compass_rose,
    add_scale_bar_curved_parallel,
    add_scale_bar_cylindrical,
    plot_ortho_grid,
)
from skyplothelper.globe.frame import make_globe_frame
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def globe_axes():
    fig = plt.figure(figsize=(6, 6))
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    return fig, ax


@pytest.fixture
def car_axes():
    """Plate Carrée axes — the natural target for add_scale_bar_cylindrical."""
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="CAR", center=0, fig=fig)
    fig.canvas.draw()
    return fig, ax


# ============================================================
# plot_ortho_grid
# ============================================================

def test_plot_ortho_grid_adds_lines():
    """Build a plain mpl axes (the function works on any axes), call
    plot_ortho_grid, verify lines were added."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    n_lines_before = len(ax.lines)

    plot_ortho_grid(ax, lon_0=0, lat_0=0, R=1.0,
                    lon_spacing=30, lat_spacing=30)
    n_lines_after = len(ax.lines)

    # Should add many grid lines (12 meridians + ~5 parallels + limb)
    assert n_lines_after - n_lines_before >= 10


def test_plot_ortho_grid_grid_step_more_lines_when_finer():
    """Finer grid spacing should produce more lines than a coarse grid."""
    fig, ax_coarse = plt.subplots(figsize=(6, 6))
    ax_coarse.set_xlim(-1.1, 1.1)
    ax_coarse.set_ylim(-1.1, 1.1)
    plot_ortho_grid(ax_coarse, lon_spacing=60, lat_spacing=60)
    n_coarse = len(ax_coarse.lines)

    fig2, ax_fine = plt.subplots(figsize=(6, 6))
    ax_fine.set_xlim(-1.1, 1.1)
    ax_fine.set_ylim(-1.1, 1.1)
    plot_ortho_grid(ax_fine, lon_spacing=15, lat_spacing=15)
    n_fine = len(ax_fine.lines)

    assert n_fine > n_coarse


# ============================================================
# add_checkered_border
# ============================================================

def test_add_checkered_border_adds_patches(globe_axes):
    fig, ax = globe_axes
    n_before = len(ax.patches)
    add_checkered_border(ax)
    n_after = len(ax.patches)
    # Many alternating-color patches around the limb
    assert n_after - n_before >= 8


def test_add_checkered_border_n_segments_propagates(globe_axes):
    """Explicitly requesting n_segments controls the patch count."""
    fig, ax = globe_axes
    n_before = len(ax.patches)
    add_checkered_border(ax, n_segments=12)
    n_after = len(ax.patches)
    # Border draws inner+outer rectangle pairs per segment, so the
    # exact count depends on internals — just verify it scales up.
    assert n_after - n_before >= 12


def test_add_checkered_border_globe_uses_circular_path(globe_axes):
    """make_globe_frame produces a circular limb → circular wedges."""
    import matplotlib.patches as mpatches
    fig, ax = globe_axes
    add_checkered_border(ax, n_segments=12)
    wedges = [p for p in ax.patches if isinstance(p, mpatches.Wedge)]
    rects = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
    # Auto-detect should pick the circular layout, producing Wedges,
    # not Rectangles.
    assert len(wedges) >= 12
    assert len(rects) == 0


def test_add_checkered_border_rectangular_path_unchanged():
    """Plain rectangular WCSAxes (CAR) still gets Rectangle patches."""
    import matplotlib.patches as mpatches
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="CAR", center=0, fig=fig)
    fig.canvas.draw()
    add_checkered_border(ax, segment_spacing_deg=30)
    rects = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
    wedges = [p for p in ax.patches if isinstance(p, mpatches.Wedge)]
    assert len(rects) > 0
    assert len(wedges) == 0


def test_add_checkered_border_frame_circular_explicit_plain_axes():
    """frame='circular' + radius= works on plain mpl axes (no WCS)."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    add_checkered_border(ax, frame='circular', radius=1.0,
                          segment_spacing_deg=15)
    wedges = [p for p in ax.patches if isinstance(p, mpatches.Wedge)]
    # 360 / 15 = 24 segments expected (rounded to even).
    assert len(wedges) == 24


def test_add_checkered_border_frame_rectangular_overrides_auto(globe_axes):
    """Explicit frame='rectangular' forces the rectangular layout even
    when the axes would auto-detect as circular."""
    import matplotlib.patches as mpatches
    fig, ax = globe_axes
    add_checkered_border(ax, frame='rectangular', n_segments=6)
    wedges = [p for p in ax.patches if isinstance(p, mpatches.Wedge)]
    rects = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
    assert len(rects) > 0
    assert len(wedges) == 0


def test_add_checkered_border_invalid_frame_raises():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="frame="):
        add_checkered_border(ax, frame='oval')


# ============================================================
# add_compass_rose
# ============================================================

def test_add_compass_rose_returns_anchoredoffsetbox(globe_axes):
    """add_compass_rose returns an AnchoredOffsetbox that wraps a
    DrawingArea containing the rose patches and labels."""
    fig, ax = globe_axes
    anchor = add_compass_rose(ax)
    assert isinstance(anchor, AnchoredOffsetbox)


def test_add_compass_rose_no_labels_drawingarea_smaller(globe_axes):
    """show_labels=False should produce a DrawingArea with fewer Text
    children than the labeled version."""
    fig, ax = globe_axes
    anchor_with = add_compass_rose(ax, show_labels=True)

    fig2 = plt.figure(figsize=(6, 6))
    ax2 = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig2.canvas.draw()
    anchor_without = add_compass_rose(ax2, show_labels=False)

    # The DrawingArea contains all the rose's children; count Text artists
    from matplotlib.text import Text
    da_with = anchor_with.get_child()
    da_without = anchor_without.get_child()

    n_text_with = sum(1 for a in da_with.get_children() if isinstance(a, Text))
    n_text_without = sum(1 for a in da_without.get_children()
                         if isinstance(a, Text))

    assert n_text_with > n_text_without
    assert n_text_without == 0


# ============================================================
# add_scale_bar_cylindrical (needs a CAR axes)
# ============================================================

def test_add_scale_bar_cylindrical_adds_artists(car_axes):
    fig, ax = car_axes
    n_patches_before = len(ax.patches)
    n_texts_before = len(ax.texts)
    add_scale_bar_cylindrical(ax, lat=0.0, length_km=1000.0)
    # Scale bar is typically a Rectangle/Line with a Text label
    assert (len(ax.patches) > n_patches_before
            or len(ax.lines) > 0)
    assert len(ax.texts) > n_texts_before


# ============================================================
# add_scale_bar_curved_parallel (needs a globe / ortho axes)
# ============================================================

def test_add_scale_bar_curved_parallel_adds_artists(globe_axes):
    fig, ax = globe_axes
    n_lines_before = len(ax.lines)
    n_texts_before = len(ax.texts)
    add_scale_bar_curved_parallel(ax, length_km=1000.0)
    # Scale bar adds line(s) + a text label
    assert (len(ax.lines) > n_lines_before
            or len(ax.patches) > 0)
    assert len(ax.texts) > n_texts_before
