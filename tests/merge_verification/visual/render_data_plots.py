"""Render data_plots helpers (plot_sky_vectors / plot_displacement /
plot_catalog) for visual eyeballing.

Produces:
  - data_01_sky_vectors.png         — random PM vectors on AIT (uniform color)
  - data_02_sky_vectors_color.png   — vectors colored by magnitude (auto colorbar)
  - data_03_displacement.png   — displacement arrows between two epochs
  - data_04_catalog_simple.png — plot_catalog with a tuple input
  - data_05_catalog_colorby.png — color-by-z and size-by-mag
  - data_06_catalog_with_labels.png — labeled catalog overlay
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show
from astropy.table import Table

from skyplothelper.data_plots import (
    plot_catalog,
    plot_displacement,
    plot_sky_vectors,
)
from skyplothelper.overlays.planes import add_plane_overlay
from skyplothelper.wcs_frame import make_wcs_frame

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _allsky():
    fig = plt.figure(figsize=(11, 5.5))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    return fig, ax


@_panel("data_01_sky_vectors")
def render_proper_motion_basic():
    fig, ax = _allsky()
    rng = np.random.default_rng(0)
    n = 30
    lon = rng.uniform(20, 340, n)
    lat = rng.uniform(-60, 60, n)
    # PM toward galactic center for "stellar streaming" effect
    pm_lon = -2.0 + rng.normal(0, 0.5, n)
    pm_lat = rng.normal(0, 0.5, n)
    plot_sky_vectors(ax, lon, lat, pm_lon, pm_lat,
                       units="mas", scale=3e7, color="C0",
                       label="proper motion (mas/yr)")
    add_plane_overlay(ax, plane="galactic", color="C3", lw=1.0)
    ax.set_title("plot_sky_vectors — 30 stars, mas/yr arrows on AIT")
    return fig


@_panel("data_02_sky_vectors_color")
def render_proper_motion_color_array():
    """The canonical "magnitude-coded arrows + colorbar" recipe via
    the ``color_by_magnitude=True`` + ``add_colorbar=True`` shortcut.
    Default ``pivot='middle'`` centers each arrow on its data point
    (the visually cleaner choice ported from the VSH plot_RDEM
    reference)."""
    fig, ax = _allsky()
    rng = np.random.default_rng(1)
    n = 50
    lon = rng.uniform(20, 340, n)
    lat = rng.uniform(-60, 60, n)
    pm_lon = rng.normal(0, 8, n)
    pm_lat = rng.normal(0, 8, n)
    plot_sky_vectors(ax, lon, lat, pm_lon, pm_lat,
                       units="mas", scale="auto", auto_target_deg=5.0,
                       color_by_magnitude=True, cmap="viridis",
                       add_colorbar=True, alpha=0.95)
    ax.set_title("plot_sky_vectors — color_by_magnitude + auto colorbar")
    return fig


@_panel("data_03_displacement")
def render_displacement():
    fig, ax = _allsky()
    rng = np.random.default_rng(2)
    n = 12
    lon1 = rng.uniform(40, 320, n)
    lat1 = rng.uniform(-50, 50, n)
    # Second epoch shifted by 5° in lon, ±2° in lat
    lon2 = lon1 + rng.normal(5, 2, n)
    lat2 = lat1 + rng.normal(0, 2, n)
    plot_displacement(ax, lon1, lat1, lon2, lat2,
                      color="C3", lw=1.5,
                      arrowstyle="->", connectionstyle="arc3,rad=0.0")
    # Mark start positions
    ax.scatter(lon1, lat1, transform=ax.get_transform("world"),
               s=20, color="C0", label="epoch 1", zorder=4)
    ax.scatter(lon2, lat2, transform=ax.get_transform("world"),
               s=20, color="C3", marker="x", label="epoch 2", zorder=4)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.set_title("plot_displacement — arrows from epoch 1 → epoch 2")
    return fig


@_panel("data_04_catalog_simple")
def render_catalog_simple():
    fig, ax = _allsky()
    rng = np.random.default_rng(3)
    n = 200
    ra = rng.uniform(0, 360, n)
    dec = rng.uniform(-80, 80, n)
    plot_catalog(ax, (ra, dec), color="C2", s=15, alpha=0.7)
    ax.set_title("plot_catalog((ra, dec)) — tuple input, 200 sources")
    return fig


@_panel("data_05_catalog_colorby")
def render_catalog_colorby_sizeby():
    fig, ax = _allsky()
    rng = np.random.default_rng(4)
    n = 150
    tbl = Table({
        "ra": rng.uniform(0, 360, n),
        "dec": rng.uniform(-80, 80, n),
        "z": rng.exponential(0.4, n),
        "mag": rng.uniform(12, 20, n),
    })
    plot_catalog(ax, tbl, ra_col="ra", dec_col="dec",
                 colorby="z", sizeby="mag",
                 cmap="plasma", smin=10, smax=120, alpha=0.85,
                 vmin=0, vmax=2.0, cbar=True, cbar_label="redshift z")
    ax.set_title("plot_catalog with colorby='z' + sizeby='mag' "
                 "(astropy Table input)")
    return fig


@_panel("data_06_catalog_with_labels")
def render_catalog_with_labels():
    fig, ax = _allsky()
    famous = {
        "ra":   [83.6, 187.7, 250.4, 299.5, 16.5,  37.9, 318.3, 5.2],
        "dec":  [22.0, 12.4,  36.5,  40.7,  -72.8, -19.7, -64.9, -77.8],
        "name": ["Crab", "M87", "M13", "Vega", "47Tuc", "Fornax",
                 "NGC 6744", "SMC"],
    }
    plot_catalog(ax, famous, ra_col="ra", dec_col="dec",
                 label_col="name", marker="*", color="C1",
                 s=80, label_fontsize=9, label_offset=(8, 8))
    ax.set_title("plot_catalog(label_col='name') — labeled famous "
                 "sky objects", fontsize=11)
    return fig


def main():
    banner("data_plots — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
