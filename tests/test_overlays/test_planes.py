"""Smoke tests for skyplothelper.overlays.planes."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from skyplothelper.overlays.planes import (
    add_great_circle,
    add_plane_overlay,
    make_gplane_in_RArange,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_make_gplane_in_RArange_returns_skycoord():
    coords = make_gplane_in_RArange(180, 100)
    # Returns a SkyCoord with 100 elements
    assert len(coords) == 100


@pytest.mark.parametrize("plane", ["galactic", "ecliptic", "supergalactic"])
def test_add_plane_overlay_runs(plane):
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    add_plane_overlay(ax, plane=plane)
    fig.canvas.draw()


def test_add_great_circle_runs():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    add_great_circle(ax, pole_lon=0., pole_lat=90., frame="galactic")
    fig.canvas.draw()


def test_add_great_circle_small_circle_not_fragmented_into_fan():
    """A small circle (lat_offset != 0) on an ICRS all-sky frame is
    double-valued in RA. Kept in path order it draws as a couple of clean
    segments; sorting it by longitude interleaved the two branches into a fan
    that fragmented into dozens of pieces. Regression for that bug."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=0, frame="ICRS", fig=fig)
    for off in (30, 45, -45):
        lines = add_great_circle(ax, frame="galactic", lat_offset=off)
        # A clean parallel is 1-3 seam-split segments; the fan was ~100.
        assert len(lines) <= 6, f"fan fragmentation at lat_offset={off}"
    fig.canvas.draw()


def test_add_plane_overlay_parallels_not_fragmented():
    """add_plane_overlay(parallels=...) draws each parallel via add_great_circle
    with a lat_offset, so they must not fan on an ICRS frame either."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=0, frame="ICRS", fig=fig)
    lines = add_plane_overlay(ax, plane="galactic", parallels=[-40, 40])
    # main plane (few segs) + two parallels (few each) — nowhere near a fan.
    assert len(lines) <= 15
    fig.canvas.draw()


# ===== Stroke kwarg uniformity =====

def test_add_great_circle_stroke_off_by_default():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    lines = add_great_circle(ax, frame="galactic")
    assert len(lines) > 0
    assert all(not ln.get_path_effects() for ln in lines)


def test_add_great_circle_stroke_on():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    lines = add_great_circle(ax, frame="galactic",
                              stroke_color="white", stroke_lw=3.0)
    assert len(lines) > 0
    assert all(len(ln.get_path_effects()) == 1 for ln in lines)
