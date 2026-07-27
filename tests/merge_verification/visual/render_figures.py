"""Render the high-level figure constructors for visual eyeballing.

Produces:
  - figures_01_allsky_default.png   — allsky_figure() default
  - figures_02_allsky_galactic.png  — allsky_figure(MOL, frame='Galactic')
  - figures_03_offset_default.png   — offset_figure() with 1° FOV
  - figures_04_offset_subdeg.png    — offset_figure() with 5′ FOV
  - figures_05_projection_gallery_default.png — default 6-projection set
  - figures_06_projection_gallery_custom.png — custom list (AIT/MOL/SFL/CAR)

Usage
-----
    python render_figures.py            # save PNGs to output/
    python render_figures.py --show     # display interactively
"""

import sys

import numpy as np
from _common import banner, save_or_show

from skyplothelper.figures import (
    allsky_figure,
    offset_figure,
    projection_gallery,
)
from skyplothelper.overlays.planes import add_plane_overlay

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


@_panel("figures_01_allsky_default")
def render_allsky_default():
    fig, ax = allsky_figure(projection="AIT", center=180)
    add_plane_overlay(ax, plane="galactic", color="C3", lw=1.0)
    ax.set_title("allsky_figure(projection='AIT', center=180) — defaults",
                 fontsize=11)
    return fig


@_panel("figures_02_allsky_galactic")
def render_allsky_galactic():
    fig, ax = allsky_figure(projection="MOL", center=0, frame="Galactic")
    add_plane_overlay(ax, plane="ecliptic", color="C2", lw=1.0,
                      label="ecliptic")
    add_plane_overlay(ax, plane="supergalactic", color="C5", lw=1.0,
                      label="supergalactic")
    ax.set_title("allsky_figure(projection='MOL', frame='Galactic')",
                 fontsize=11)
    return fig


@_panel("figures_03_offset_default")
def render_offset_default():
    """1-degree FOV around galactic center (0, 0) Galactic."""
    fig, ax = offset_figure(
        center=(266.4051, -28.9362),  # SgrA* in ICRS
        fov_deg=1.0, projection="TAN", frame="ICRS",
    )
    # Add a small circle to show the FOV scale
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 100)
    r = rng.uniform(0, 0.4, 100)
    lons = 266.4051 + r * np.cos(theta) / np.cos(np.radians(-28.9362))
    lats = -28.9362 + r * np.sin(theta)
    ax.scatter(lons, lats, transform=ax.get_transform("world"),
               s=10, c="C0", alpha=0.6)
    ax.set_title("offset_figure(center=SgrA*, fov_deg=1.0)", fontsize=11)
    return fig


@_panel("figures_04_offset_subdeg")
def render_offset_subdeg():
    """5-arcmin FOV — small enough that arcmin/arcsec offsets make sense."""
    fig, ax = offset_figure(
        center=(83.6324, 22.0145),  # M1 / Crab
        fov_deg=5 / 60.0, projection="TAN", style="offset_arcsec",
    )
    rng = np.random.default_rng(1)
    n = 50
    lons = 83.6324 + rng.normal(0, 0.5 / 60.0, n) / np.cos(np.radians(22.0145))
    lats = 22.0145 + rng.normal(0, 0.5 / 60.0, n)
    ax.scatter(lons, lats, transform=ax.get_transform("world"),
               s=8, c="C3", alpha=0.7)
    ax.set_title("offset_figure(center=Crab, fov_deg=5′) — offset_arcsec ticks",
                 fontsize=11)
    return fig


@_panel("figures_05_projection_gallery_default")
def render_projection_gallery_default():
    """Default 6-projection comparison of a smoothed random HEALPix map.
    The default list is
    ['AIT', 'MOL', 'SFL', 'CAR', 'PAR', 'PCO']."""
    fig, _axes = projection_gallery(title="projection_gallery() — default 6-projection set", ncols=3)
    return fig


@_panel("figures_06_projection_gallery_custom")
def render_projection_gallery_custom():
    """Custom projection list — all-sky pseudocylindrical comparison."""
    fig, _axes = projection_gallery(
        projections=["AIT", "MOL", "SFL", "CAR"],
        ncols=2,
        title="projection_gallery(projections=['AIT', 'MOL', 'SFL', 'CAR'], ncols=2)",
    )
    return fig


def main():
    banner("figures — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
