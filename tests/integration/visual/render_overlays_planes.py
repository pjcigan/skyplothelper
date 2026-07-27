"""Render the overlays.planes helpers (add_plane_overlay + add_great_circle)
for visual eyeballing.

Produces:
  - planes_01_galactic_default.png — galactic plane on AIT (ICRS)
  - planes_02_ecliptic_with_parallels.png — ecliptic + ±23.4° parallels
  - planes_03_supergalactic.png — supergalactic plane
  - planes_04_all_three_planes.png — galactic + ecliptic + supergalactic
  - planes_05_great_circle_custom_pole.png — add_great_circle around an
    arbitrary user-supplied pole
  - planes_06_galactic_frame_axes.png — galactic plane on Galactic-frame
    axes (renders horizontally)

Usage
-----
    python render_overlays_planes.py            # save PNGs to output/
    python render_overlays_planes.py --show     # display interactively
"""

import sys

import matplotlib.pyplot as plt
from _common import banner, save_or_show

from skyplothelper.overlays.planes import add_great_circle, add_plane_overlay
from skyplothelper.wcs_frame import make_wcs_frame

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _allsky(projection="AIT", center=180, frame="ICRS"):
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection=projection, center=center,
                        frame=frame, fig=fig)
    fig.canvas.draw()
    return fig, ax


@_panel("planes_01_galactic_default")
def render_galactic_default():
    fig, ax = _allsky()
    add_plane_overlay(ax, plane="galactic")
    ax.set_title("add_plane_overlay(plane='galactic') — AIT (ICRS), default style")
    return fig


@_panel("planes_02_ecliptic_with_parallels")
def render_ecliptic_with_parallels():
    fig, ax = _allsky()
    add_plane_overlay(ax, plane="ecliptic", lw=1.5,
                      parallels=[-23.4, 23.4], parallel_alpha=0.6)
    ax.set_title("add_plane_overlay(plane='ecliptic', parallels=[±23.4°])")
    return fig


@_panel("planes_03_supergalactic")
def render_supergalactic():
    fig, ax = _allsky()
    add_plane_overlay(ax, plane="supergalactic", lw=1.5,
                      parallels=[-10, 10], parallel_alpha=0.5)
    ax.set_title("add_plane_overlay(plane='supergalactic') with ±10° parallels")
    return fig


@_panel("planes_04_all_three_planes")
def render_all_three():
    fig, ax = _allsky()
    add_plane_overlay(ax, plane="galactic", lw=1.5)
    add_plane_overlay(ax, plane="ecliptic", lw=1.5)
    add_plane_overlay(ax, plane="supergalactic", lw=1.5)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.set_title("Galactic + Ecliptic + Supergalactic planes overlaid (ICRS)")
    return fig


@_panel("planes_05_great_circle_custom_pole")
def render_great_circle_custom_pole():
    fig, ax = _allsky()
    # The Galactic-pole great circle (matches add_plane_overlay('galactic'))
    add_great_circle(ax, pole_lon=192.86, pole_lat=27.13, frame="pole",
                     color="C3", lw=1.5, label="galactic pole")
    # An offset great circle (10° "north" of it)
    add_great_circle(ax, pole_lon=192.86, pole_lat=27.13, frame="pole",
                     lat_offset=10, color="C1", ls="--", lw=1.0,
                     label="lat_offset=+10°")
    add_great_circle(ax, pole_lon=192.86, pole_lat=27.13, frame="pole",
                     lat_offset=-10, color="C2", ls="--", lw=1.0,
                     label="lat_offset=-10°")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.set_title("add_great_circle(frame='pole') — galactic pole + ±10° offsets")
    return fig


@_panel("planes_06_galactic_frame_axes")
def render_galactic_frame_axes():
    fig, ax = _allsky(frame="Galactic", center=0)
    add_plane_overlay(ax, plane="galactic", lw=2.0)
    add_plane_overlay(ax, plane="ecliptic", lw=1.5)
    add_plane_overlay(ax, plane="supergalactic", lw=1.5)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.set_title("Galactic-frame AIT axes — galactic plane is horizontal "
                 "by definition")
    return fig


def main():
    banner("overlays.planes — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
