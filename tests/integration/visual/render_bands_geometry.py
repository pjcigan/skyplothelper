"""Render the geometry band + Tissot helpers for visual eyeballing.

Covers: ``add_latitude_band``, ``add_longitude_band``,
``add_great_circle_band``, ``add_frame_band`` (patch + contour modes),
``tissot``.

Usage
-----
    python render_bands_geometry.py            # save PNGs to output/
    python render_bands_geometry.py --show     # display interactively
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper.geometry.bands import (
    add_frame_band,
    add_great_circle_band,
    add_latitude_band,
    add_longitude_band,
)
from skyplothelper.geometry.tissot import tissot
from skyplothelper.overlays.constellations import add_constellation_boundaries
from skyplothelper.overlays.planes import add_plane_overlay
from skyplothelper.ticks import format_ticklabels
from skyplothelper.wcs_frame import make_wcs_frame

# Builder registry — name → no-arg function that returns a Figure.
# Filled by the @_panel decorator below.
PANELS = {}


def _panel(name):
    """Register the decorated function as the builder for *name*."""
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _allsky(projection="AIT", center=180):
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection=projection, center=center, fig=fig)
    fig.canvas.draw()
    return fig, ax


@_panel("bands_01_latitude_bands")
def render_latitude_band():
    """add_latitude_band — equatorial and northern bands."""
    fig, ax = _allsky()
    add_latitude_band(ax, lat_min=-10, lat_max=10,
                      facecolor="orange", alpha=0.35,
                      edgecolor="darkorange", lw=0.8)
    add_latitude_band(ax, lat_min=30, lat_max=50,
                      facecolor="C0", alpha=0.30,
                      edgecolor="navy", lw=0.8)
    ax.set_title("add_latitude_band — equatorial (±10°) + northern (30°–50°)")
    return fig


@_panel("bands_02_longitude_band")
def render_longitude_band():
    """add_longitude_band — wedge across one hemisphere."""
    fig, ax = _allsky()
    add_longitude_band(ax, lon_min=120, lon_max=240,
                       facecolor="C2", alpha=0.30,
                       edgecolor="darkgreen", lw=0.8)
    ax.set_title("add_longitude_band — 120°–240° lon wedge")
    return fig


@_panel("bands_03_great_circle_band")
def render_great_circle_band():
    """add_great_circle_band — band centered on a custom great-circle pole."""
    fig, ax = _allsky()
    # Galactic-plane-like band: pole at galactic north (192.86°, 27.13°) ICRS
    add_great_circle_band(
        ax, ra_pole=192.86, dec_pole=27.13, half_width=15,
        facecolor="C3", alpha=0.4, edgecolor="darkred", lw=0.8,
    )
    ax.set_title("add_great_circle_band — ±15° around galactic-pole "
                 "great circle")
    return fig


@_panel("bands_04_frame_band_patch")
def render_frame_band_patch_mode():
    """add_frame_band — patch mode (D3-style antimeridian clipping)."""
    fig, ax = _allsky()
    add_frame_band(
        ax, lat_min=-10, lat_max=10, frame="galactic",
        backend="patch",
        facecolor="C1", alpha=0.4, edgecolor="orange", lw=0.8,
    )
    add_frame_band(
        ax, lat_min=-5, lat_max=5, frame="geocentrictrueecliptic",
        backend="patch",
        facecolor="C9", alpha=0.4, edgecolor="C0", lw=0.8,
    )
    ax.set_title("add_frame_band (backend='patch') — galactic ±10° + ecliptic ±5°")
    return fig


@_panel("bands_05_frame_band_contour")
def render_frame_band_contour_mode():
    """add_frame_band — contour mode (rasterize-then-render fallback)."""
    fig, ax = _allsky()
    add_frame_band(
        ax, lat_min=-10, lat_max=10, frame="galactic",
        backend="contour",
        facecolor="C1", alpha=0.4, edgecolor="orange", lw=0.8,
    )
    ax.set_title("add_frame_band (backend='contour') — galactic ±10°")
    return fig


@_panel("bands_06_tissot_default")
def render_tissot_default():
    """tissot — default 6×6 grid of indicatrices, 5° radius."""
    fig, ax = _allsky()
    tissot(ax, rad_deg=5, resolution=80,
           facecolor="C2", edgecolor="darkgreen",
           alpha=0.45, lw=0.6)
    ax.set_title("tissot — 6×6 default grid, 5° radius")
    return fig


@_panel("bands_07_tissot_mollweide")
def render_tissot_dense_with_grid():
    """tissot — denser grid + custom positions on multiple projections."""
    fig, ax = _allsky(projection="MOL")
    lons = np.linspace(-180, 180, 9, endpoint=False) + 22.5
    lats = np.linspace(-75, 75, 7)
    tissot(ax, rad_deg=4, lons=lons, lats=lats, resolution=80,
           facecolor="C5", edgecolor="indigo",
           alpha=0.4, lw=0.6)
    ax.set_title("tissot on Mollweide — 9×7 custom grid, 4° radius")
    return fig


@_panel("bands_08_combo")
def render_combo():
    """Putting several together: galactic frame band + ecliptic plane +
    constellation outlines + Tissot indicatrices."""
    fig, ax = _allsky()
    # Galactic plane band (frame_band)
    add_frame_band(
        ax, lat_min=-5, lat_max=5, frame="galactic",
        backend="patch",
        facecolor="C1", alpha=0.3, edgecolor="orange", lw=0.7,
    )
    # Ecliptic plane line
    add_plane_overlay(ax, plane="ecliptic", color="C9", lw=1.5)
    # Constellation boundaries (subset)
    add_constellation_boundaries(ax, color="0.6", lw=0.4)
    # Tissot indicatrices
    tissot(ax, rad_deg=4, lons=np.linspace(-180, 180, 6, endpoint=False) + 30,
           lats=np.linspace(-60, 60, 5), resolution=60,
           facecolor="C2", edgecolor="darkgreen", alpha=0.45, lw=0.5)
    format_ticklabels(ax, style="publication")
    ax.set_title("Combined: galactic band + ecliptic + constellations + tissot")
    return fig


def main():
    banner("geometry.bands + tissot — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
