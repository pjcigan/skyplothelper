"""Smoke tests for skyplothelper.cone.plotting."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyplothelper.cone.frame import make_cone_frame
from skyplothelper.cone.plotting import (
    cone_hexbin,
    cone_plot,
    cone_scatter,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _basic_cone():
    fig = plt.figure()
    ax = make_cone_frame(
        111, angle_center=180, angle_half_width=30,
        r_min=0, r_max=0.15, fig=fig,
    )
    return fig, ax


def test_cone_scatter_smoke():
    fig, ax = _basic_cone()
    rng = np.random.default_rng(42)
    angles = rng.uniform(155, 205, 100)
    rs = rng.uniform(0, 0.15, 100)
    cone_scatter(ax, angles, rs, s=3)
    fig.canvas.draw()


def test_cone_plot_smoke():
    fig, ax = _basic_cone()
    cone_plot(ax, [160, 180, 200], [0.05, 0.10, 0.05])
    fig.canvas.draw()


def test_cone_hexbin_smoke():
    fig, ax = _basic_cone()
    rng = np.random.default_rng(7)
    angles = rng.uniform(155, 205, 500)
    rs = rng.uniform(0, 0.15, 500)
    cone_hexbin(ax, angles, rs, gridsize=20)
    fig.canvas.draw()
