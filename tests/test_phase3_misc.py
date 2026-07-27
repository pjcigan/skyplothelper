"""Smoke tests for the smaller support modules.

style, grid, figures, data_plots, cartopy_backend, queries, diagnostics.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from skyplothelper.diagnostics import describe_wcs
from skyplothelper.figures import allsky_figure
from skyplothelper.grid import (
    add_second_grid,
    highlight_gridline,
    style_grid,
)
from skyplothelper.style import _THEMES, set_base_style, set_theme, style_context
from skyplothelper.wcs_frame import dummy_allsky_hdr, make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- style ----

def test_themes_keys():
    assert "publication" in _THEMES


def test_set_theme_runs():
    set_theme("publication")


def test_set_base_style_runs():
    set_base_style("pretty1")


def test_style_context_runs():
    with style_context("pretty1"):
        plt.figure()


# ---- grid ----

def test_add_second_grid_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    add_second_grid(ax, overlay_frame="galactic")


def test_style_grid_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    style_grid(ax, color="red")


def test_highlight_gridline_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    highlight_gridline(ax, value=0, coord="lon", color="red")


# ---- figures ----

def test_allsky_figure_smoke():
    fig, ax = allsky_figure(projection="AIT", center=180)
    assert fig is not None


# ---- diagnostics ----

def test_describe_wcs_runs(capsys):
    hdr = dummy_allsky_hdr(center_LONdeg=180, projection="AIT")
    describe_wcs(hdr)
    out = capsys.readouterr().out
    assert "AIT" in out or "Aitoff" in out
