"""Render the globe plotting helpers for visual eyeballing.

Covers: ``plot_scatter_globe``, ``plot_line_globe``, ``plot_pcolormesh_globe``,
``plot_contour_globe``, ``imscatter`` / ``imscatter_rotated``.

Usage
-----
    python render_globe_plotting.py            # save PNGs to output/
    python render_globe_plotting.py --show     # display interactively
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper.globe.frame import make_globe_frame
from skyplothelper.globe.plotting import (
    imscatter,
    imscatter_rotated,
    plot_contour_globe,
    plot_line_globe,
    plot_pcolormesh_globe,
    plot_scatter_globe,
)

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _make_globe(center_lon=0.0, center_lat=23.44, fig=None):
    """Build a representative globe — Earth-tilt orientation by default."""
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    ax = make_globe_frame(
        111, center_LONdeg=center_lon, center_LATdeg=center_lat,
    )
    fig.canvas.draw()
    return fig, ax


def _scatter_data():
    """Shared scatter positions for the two scatter panels."""
    rng = np.random.default_rng(42)
    return rng.uniform(-180, 180, 200), rng.uniform(-90, 90, 200)


def _contour_grid():
    """Shared (lon, lat, data) grid for the two contour panels."""
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(-180, 180, 73),
        np.linspace(-90, 90, 37),
    )
    data = (np.sin(np.radians(lon_grid))
            * np.cos(np.radians(lat_grid * 1.5)))
    return lon_grid, lat_grid, data


def _icon():
    """Shared synthetic radio-dish icon for the two imscatter panels."""
    icon = np.zeros((20, 20, 4), dtype=float)
    for i in range(20):
        for j in range(20):
            if i + j < 20:
                icon[i, j] = (0.2, 0.5, 0.8, 1.0 - (i + j) / 25)
    return icon


@_panel("globe_01_scatter_uniform")
def render_scatter_uniform():
    """plot_scatter_globe — uniform scatter.

    (Note: ``hemisphere_only=True`` and ``False`` produce visually identical
    output for SIN projections because the WCS transform already culls
    back-hemisphere points; the flag is a Python-level optimization, not a
    visual switch. Both code paths are exercised in the assertion suite.)
    """
    lons, lats = _scatter_data()
    fig, ax = _make_globe()
    plot_scatter_globe(ax, lons, lats, hemisphere_only=True,
                       s=12, c="C0", edgecolor="k", linewidth=0.3)
    ax.set_title("plot_scatter_globe — uniform points")
    return fig


@_panel("globe_02_scatter_colormapped")
def render_scatter_colormapped():
    """plot_scatter_globe with c=array colormap mapping.

    Pre-filters to visible-hemisphere positions because c= alongside
    hemisphere_only=True can desynchronize values from positions.
    """
    from skyplothelper.globe.spherical import orthographic_visibility
    lons, lats = _scatter_data()
    vis = orthographic_visibility(lons, lats, lon_0=0., lat_0=23.44)
    lons_v, lats_v = lons[vis], lats[vis]

    fig, ax = _make_globe()
    plot_scatter_globe(ax, lons_v, lats_v, hemisphere_only=False,
                       s=20, c=np.abs(lats_v), cmap="plasma",
                       edgecolor="k", linewidth=0.3)
    ax.set_title("plot_scatter_globe — c=|lat| colormapped")
    return fig


@_panel("globe_03_line_polyline_segments")
def render_line():
    """plot_line_globe — a polyline of great-circle segments.

    Each pair of consecutive waypoints is connected by a great-circle arc
    (densified internally via ``great_circle_arc``); the overall path is
    a sequence of joined geodesic segments, not a single great circle.
    """
    fig, ax = _make_globe()
    lons = np.array([-90, -45, 0, 45, 90, 135])
    lats = np.array([-40, 0, 30, 50, 30, 0])
    plot_line_globe(ax, lons, lats, hemisphere_only=True,
                    color="orange", lw=2.5, marker="o", ms=6)
    ax.set_title("plot_line_globe — polyline of great-circle segments")
    return fig


@_panel("globe_04_pcolormesh_field")
def render_pcolormesh():
    """plot_pcolormesh_globe — a sin(lon)cos(lat) field."""
    fig, ax = _make_globe()
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(-180, 180, 91),
        np.linspace(-90, 90, 46),
    )
    data = np.sin(np.radians(lon_grid)) * np.cos(np.radians(lat_grid))
    plot_pcolormesh_globe(ax, lon_grid, lat_grid, data,
                          cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.85)
    ax.set_title("plot_pcolormesh_globe — sin(lon)·cos(lat)")
    return fig


@_panel("globe_05_contour_lines")
def render_contour_lines():
    """plot_contour_globe — line contours."""
    lon_grid, lat_grid, data = _contour_grid()
    fig, ax = _make_globe()
    plot_contour_globe(ax, lon_grid, lat_grid, data,
                       levels=np.linspace(-1, 1, 9),
                       colors="k", linewidths=0.8)
    ax.set_title("plot_contour_globe (lines)")
    return fig


@_panel("globe_06_contour_filled")
def render_contour_filled():
    """plot_contour_globe — filled contours."""
    lon_grid, lat_grid, data = _contour_grid()
    fig, ax = _make_globe()
    plot_contour_globe(ax, lon_grid, lat_grid, data,
                       levels=np.linspace(-1, 1, 11),
                       cmap="viridis", filled=True, alpha=0.8)
    ax.set_title("plot_contour_globe (filled)")
    return fig


@_panel("globe_07_imscatter")
def render_imscatter():
    """imscatter on a plain matplotlib axes."""
    icon = _icon()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    imscatter(np.array([1.5, 4.0, 6.5, 8.5]),
              np.array([2.5, 3.5, 1.5, 3.0]),
              icon, ax=ax, zoom=0.7)
    ax.set_title("imscatter — icons at (x, y) coords")
    ax.set_aspect("equal")
    return fig


@_panel("globe_08_imscatter_rotated")
def render_imscatter_rotated():
    """imscatter_rotated — per-point rotation."""
    icon = _icon()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    imscatter_rotated(
        np.array([1.5, 4.0, 6.5, 8.5]),
        np.array([2.5, 3.5, 1.5, 3.0]),
        icon, rotations=[0, 30, 60, 90], ax=ax, zoom=0.7,
    )
    ax.set_title("imscatter_rotated — per-point rotation")
    ax.set_aspect("equal")
    return fig


def main():
    banner("globe.plotting — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
