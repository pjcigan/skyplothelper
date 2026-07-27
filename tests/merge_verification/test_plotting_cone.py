"""cone plotting return-value verification.

Verifies that the cone plotting helpers (``cone_scatter``, ``cone_plot``,
``cone_scatter_z``, ``cone_hexbin``, ``cone_pcolormesh``) return the
right kind of artist with the right contents after the merge.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection, PolyCollection, QuadMesh
from matplotlib.lines import Line2D

from skyplothelper.cone.frame import make_cone_frame
from skyplothelper.cone.plotting import (
    cone_hexbin,
    cone_pcolormesh,
    cone_plot,
    cone_scatter,
    cone_scatter_z,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def cone_axes():
    fig = plt.figure(figsize=(6, 6))
    ax = make_cone_frame(
        111, angle_center=180, angle_half_width=30,
        r_min=0, r_max=0.15, fig=fig,
    )
    fig.canvas.draw()
    return fig, ax


# ============================================================
# cone_scatter
# ============================================================

def test_cone_scatter_returns_pathcollection(cone_axes):
    fig, ax = cone_axes
    sc = cone_scatter(ax, [165, 180, 195], [0.05, 0.10, 0.05])
    assert isinstance(sc, PathCollection)


def test_cone_scatter_offsets_count_matches_input(cone_axes):
    fig, ax = cone_axes
    angles = np.linspace(155, 205, 25)
    rs = np.linspace(0.02, 0.13, 25)
    sc = cone_scatter(ax, angles, rs)
    assert sc.get_offsets().shape[0] == 25


def test_cone_scatter_color_propagates(cone_axes):
    fig, ax = cone_axes
    sc = cone_scatter(ax, [180], [0.05], color="red", s=40)
    fc = sc.get_facecolor()
    assert fc[0][0] == pytest.approx(1.0)
    assert fc[0][1] == pytest.approx(0.0)


# ============================================================
# cone_plot
# ============================================================

def test_cone_plot_returns_line2d_list(cone_axes):
    fig, ax = cone_axes
    lines = cone_plot(ax, [160, 180, 200], [0.05, 0.10, 0.05])
    assert isinstance(lines, list)
    assert len(lines) >= 1
    assert all(isinstance(line, Line2D) for line in lines)


def test_cone_plot_color_propagates(cone_axes):
    fig, ax = cone_axes
    lines = cone_plot(ax, [160, 180, 200], [0.05, 0.10, 0.05],
                      color="green", lw=2.0)
    color = lines[0].get_color()
    if isinstance(color, str):
        assert color == "green"
    else:
        # Green is (0, ~0.5, 0, 1) — exact value depends on mpl conversion
        assert color[1] > 0.3
    assert lines[0].get_linewidth() == pytest.approx(2.0)


# ============================================================
# cone_scatter_z (delegates to cone_scatter via redshift_to_r)
# ============================================================

def test_cone_scatter_z_returns_pathcollection_redshift(cone_axes):
    """When the frame's r_variable is 'redshift' (default), no cosmology
    is needed and the conversion is identity."""
    fig, ax = cone_axes
    sc = cone_scatter_z(ax, [170, 180, 190], [0.05, 0.10, 0.05])
    assert isinstance(sc, PathCollection)
    assert sc.get_offsets().shape[0] == 3


# ============================================================
# cone_hexbin
# ============================================================

def test_cone_hexbin_returns_polycollection(cone_axes):
    fig, ax = cone_axes
    rng = np.random.default_rng(42)
    angles = rng.uniform(155, 205, 500)
    rs = rng.uniform(0, 0.15, 500)
    hexes = cone_hexbin(ax, angles, rs, gridsize=12, cmap="viridis")
    # matplotlib.hexbin returns a PolyCollection
    assert isinstance(hexes, PolyCollection)


# ============================================================
# cone_pcolormesh
# ============================================================

def test_cone_pcolormesh_returns_quadmesh(cone_axes):
    fig, ax = cone_axes
    angle_edges = np.linspace(155, 205, 11)   # 10 angular bins
    r_edges = np.linspace(0, 0.15, 7)         # 6 radial bins
    H = np.arange(6 * 10, dtype=float).reshape(6, 10)
    qm = cone_pcolormesh(ax, angle_edges, r_edges, H, cmap="plasma")
    assert isinstance(qm, QuadMesh)
