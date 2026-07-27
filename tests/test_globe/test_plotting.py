"""Smoke tests for skyplothelper.globe.plotting."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyplothelper.globe.frame import make_globe_frame
from skyplothelper.globe.plotting import (
    plot_line_globe,
    plot_pcolormesh_globe,
    plot_scatter_globe,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_scatter_globe_smoke():
    fig = plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    plot_scatter_globe(ax, lons=[10, 20, 30], lats=[5, 0, -5])
    fig.canvas.draw()


def test_plot_scatter_globe_subsets_array_size_and_color():
    """Per-point array s/c of the full catalog length must be subset by the
    same hemisphere-visibility mask as lons/lats (else ax.scatter raises)."""
    fig = plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    rng = np.random.default_rng(0)
    lons = rng.uniform(-180, 180, 150)   # full-sky: some on the far side
    lats = rng.uniform(-90, 90, 150)
    sizes = rng.uniform(10, 80, 150)
    colors = rng.random(150)
    sc = plot_scatter_globe(ax, lons, lats, s=sizes, c=colors)  # must not raise
    fig.canvas.draw()
    n = sc.get_offsets().shape[0]
    assert 0 < n < 150                    # only the visible hemisphere drawn
    assert sc.get_sizes().shape[0] == n   # size array subset to match


def test_plot_scatter_globe_rgba_array_color():
    """An (N, 4) RGBA color array is subset row-wise too."""
    fig = plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    rng = np.random.default_rng(1)
    lons = rng.uniform(-180, 180, 120)
    lats = rng.uniform(-90, 90, 120)
    plot_scatter_globe(ax, lons, lats, c=rng.random((120, 4)))  # must not raise
    fig.canvas.draw()


def test_plot_line_globe_smoke():
    fig = plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    plot_line_globe(ax, lons=[10, 20, 30], lats=[5, 0, -5])
    fig.canvas.draw()


def test_plot_pcolormesh_globe_smoke():
    fig = plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    lon_grid, lat_grid = np.meshgrid(np.linspace(-30, 30, 20), np.linspace(-30, 30, 20))
    data = np.sin(np.radians(lon_grid))
    plot_pcolormesh_globe(ax, lon_grid, lat_grid, data)
    fig.canvas.draw()
