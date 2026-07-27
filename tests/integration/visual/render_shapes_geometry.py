"""Render the geometry.shapes renderers for visual eyeballing.

Covers: ``geodesic_circle``, ``rectangle``, ``ellipse`` (vertex
generators), and the renderers ``add_geodesic_circle``,
``add_spherical_polygon``, ``add_rectangle``, ``add_square``,
``add_ellipse``, ``add_annulus`` (and the complement=True branch).

Usage
-----
    python render_shapes_geometry.py            # save PNGs to output/
    python render_shapes_geometry.py --show     # display interactively
"""

import sys

import astropy.units as u
import matplotlib.pyplot as plt
from _common import banner, save_or_show
from astropy.coordinates import SkyCoord

from skyplothelper.constants import REGION_PALETTE_NAMED as _PAL
from skyplothelper.geometry.shapes import (
    add_annulus,
    add_ellipse,
    add_geodesic_circle,
    add_rectangle,
    add_spherical_polygon,
    add_square,
)
from skyplothelper.overlays.planes import add_plane_overlay
from skyplothelper.ticks import format_ticklabels
from skyplothelper.wcs_frame import make_wcs_frame


def _shade(name, alpha=0.4):
    """Curated REGION_PALETTE color + matching darker edge."""
    edges = {
        'teal': 'darkslategray', 'cyan': 'teal', 'mustard': 'darkgoldenrod',
        'coral': 'firebrick', 'orange': 'darkorange', 'rust': '#7a2200',
        'tan': 'sienna', 'peach': 'sienna', 'deep_teal': '#0e2228',
    }
    return dict(facecolor=_PAL[name], edgecolor=edges.get(name, 'black'),
                alpha=alpha)

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _allsky(projection="AIT", center=180):
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection=projection, center=center, fig=fig)
    fig.canvas.draw()
    return fig, ax


@_panel("shapes_01_geodesic_circles")
def render_geodesic_circles():
    """add_geodesic_circle — circles in different projections / centers / radii."""
    fig, ax = _allsky()
    add_geodesic_circle(ax, 60, 30, radius_deg=15, lw=0.8, **_shade("teal"))
    add_geodesic_circle(ax, 200, -10, radius_deg=8, lw=0.8, **_shade("cyan"))
    add_geodesic_circle(ax, 270, 65, radius_deg=20, lw=0.8, **_shade("coral"))
    # SkyCoord center + Quantity radius
    sc = SkyCoord(330 * u.deg, -50 * u.deg, frame="icrs")
    add_geodesic_circle(ax, sc, 12 * u.deg, lw=0.8, **_shade("mustard"))
    ax.set_title("add_geodesic_circle — four circles, mixed inputs "
                 "(float/SkyCoord/Quantity)")
    return fig


@_panel("shapes_02_rectangles_squares")
def render_rectangles_and_squares():
    """add_rectangle and add_square with rotation."""
    fig, ax = _allsky()
    add_rectangle(ax, 60, 0, width=40, height=20, angle=0, **_shade("teal"))
    add_rectangle(ax, 180, 30, width=30, height=15, angle=30, **_shade("orange"))
    add_square(ax, 280, -30, size=20, angle=0, **_shade("cyan"))
    add_square(ax, 320, 50, size=25, angle=45, **_shade("coral"))
    ax.set_title("add_rectangle / add_square — varied centers, sizes, "
                 "and rotations")
    return fig


@_panel("shapes_03_ellipses")
def render_ellipses():
    """add_ellipse with various axis ratios and orientations."""
    fig, ax = _allsky()
    add_ellipse(ax, 60, 0, semi_major=20, semi_minor=8, angle=0, **_shade("teal"))
    add_ellipse(ax, 180, 30, semi_major=15, semi_minor=10, angle=30, **_shade("orange"))
    add_ellipse(ax, 280, -40, semi_major=18, semi_minor=5, angle=70, **_shade("cyan"))
    # SkyCoord + Quantity (positional-shift behavior: semi_major is positional)
    sc = SkyCoord(330 * u.deg, 60 * u.deg, frame="icrs")
    add_ellipse(ax, sc, 25 * u.deg, semi_minor=8 * u.deg, angle=110 * u.deg,
                **_shade("coral"))
    ax.set_title("add_ellipse — varied semi-major/minor and orientation")
    return fig


@_panel("shapes_04_annulus")
def render_annulus():
    """add_annulus — concentric inner and outer radii."""
    fig, ax = _allsky()
    add_annulus(ax, 90, 0, inner_radius=8, outer_radius=18, lw=0.8,
                **_shade("teal"))
    add_annulus(ax, 270, 30, inner_radius=5, outer_radius=15, lw=0.8,
                **_shade("cyan"))
    add_annulus(ax, 180, -50, inner_radius=10, outer_radius=20, lw=0.8,
                **_shade("coral"))
    ax.set_title("add_annulus — three rings of varied inner/outer radii")
    return fig


@_panel("shapes_05_polygons")
def render_spherical_polygon():
    """add_spherical_polygon — irregular sky region."""
    fig, ax = _allsky()
    # An irregular pentagon spanning ~50° lon × ~40° lat
    lons = [60, 120, 130, 90, 50]
    lats = [-10, -20, 20, 30, 5]
    add_spherical_polygon(ax, lons, lats, lw=0.8, **_shade("rust"))
    # A small triangle near the north pole — exercises pole containment
    lons_tri = [180, 240, 60]
    lats_tri = [70, 80, 70]
    add_spherical_polygon(ax, lons_tri, lats_tri, lw=0.8,
                          **_shade("orange", alpha=0.5))
    ax.set_title("add_spherical_polygon — irregular pentagon + "
                 "polar-cap triangle")
    return fig


@_panel("shapes_06_complement")
def render_complement():
    """complement=True — fill the sky OUTSIDE the shape."""
    fig, ax = _allsky()
    add_geodesic_circle(ax, 180, 0, radius_deg=30,
                        facecolor="0.7", alpha=0.5, edgecolor="black",
                        lw=1.0, complement=True)
    ax.set_title("add_geodesic_circle(complement=True) — fills outside the circle")
    return fig


@_panel("shapes_07_combined")
def render_combined():
    """Showcase: many shapes overlapping in one plot."""
    fig, ax = _allsky()
    # Galactic plane reference
    add_plane_overlay(ax, plane="galactic", color="0.3", lw=1.0)
    # Mixed shapes (curated palette for visual coherence)
    add_geodesic_circle(ax, 60, 30, radius_deg=12, **_shade("teal", alpha=0.35))
    add_rectangle(ax, 200, -20, width=30, height=15, angle=10,
                  **_shade("orange", alpha=0.35))
    add_ellipse(ax, 290, 50, semi_major=20, semi_minor=10, angle=45,
                **_shade("cyan", alpha=0.35))
    add_annulus(ax, 130, -40, inner_radius=6, outer_radius=14,
                **_shade("coral", alpha=0.35))
    add_spherical_polygon(ax, [340, 360, 20, 0], [-10, 0, 10, 5],
                          **_shade("rust", alpha=0.35))
    format_ticklabels(ax, style="publication")
    ax.set_title("Combined: circle + rectangle + ellipse + annulus + polygon")
    return fig


def main():
    banner("geometry.shapes — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
