"""Render the VSH forward model (vsh_field / vsh_shift_frame) as arrows
and frame-displacement overlays for visual eyeballing.

Produces:
  - vsh_01_rotation_field.png   — pure rotation (R) vector field (quiver)
  - vsh_02_glide_field.png      — pure glide (D) dipole field (quiver)
  - vsh_03_quadrupole_field.png — an l=2 quadrupole term field (quiver)
  - vsh_04_frame_displacement.png — whole-frame before→after arrows for a
                                    rotation+glide combo (plot_displacement)
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper.data_plots import plot_displacement, plot_sky_vectors
from skyplothelper.overlays.planes import add_plane_overlay
from skyplothelper.vsh import vsh_field, vsh_shift_frame
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


def _grid(n_lon=18, n_lat=9, lat_max=80.0):
    lons = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    lats = np.linspace(-lat_max, lat_max, n_lat)
    glon, glat = np.meshgrid(lons, lats)
    return glon.ravel(), glat.ravel()


def _field_panel(ax, params, color):
    """Draw the VSH field on a uniform grid as quiver arrows. Params are
    in degrees, so the field is too; ``scale='auto'`` sizes the arrows."""
    lon, lat = _grid()
    dlon, dlat = vsh_field(lon, lat, params)
    plot_sky_vectors(ax, lon, lat, dlon, dlat,
                     units="deg", scale="auto", auto_target_deg=12.0,
                     color=color, alpha=0.9, pivot="tail")


@_panel("vsh_01_rotation_field")
def render_rotation_field():
    fig, ax = _allsky()
    # Tilt (R1, R2) + spin about the pole (R3), in degrees.
    _field_panel(ax, [4.0, 0.0, 8.0, 0, 0, 0], color="C0")
    ax.set_title("vsh_field — rotation R=(4, 0, 8)° (tilt + polar spin)")
    return fig


@_panel("vsh_02_glide_field")
def render_glide_field():
    fig, ax = _allsky()
    # Glide / dipole flow toward a direction, in degrees.
    _field_panel(ax, [0, 0, 0, 0.0, 8.0, 4.0], color="C3")
    ax.set_title("vsh_field — glide D=(0, 8, 4)° (dipole flow)")
    return fig


@_panel("vsh_03_quadrupole_field")
def render_quadrupole_field():
    fig, ax = _allsky()
    # An l=2 quadrupole: E_20 (poloidal) + E_22_Re terms.
    params = np.zeros(16)
    params[6] = 8.0    # E_20
    params[12] = 6.0   # E_22_Re
    _field_panel(ax, params, color="C4")
    ax.set_title("vsh_field — l=2 quadrupole (E20=8°, E22_Re=6°)")
    return fig


@_panel("vsh_04_frame_displacement")
def render_frame_displacement():
    fig, ax = _allsky()
    # Rotation + glide combo; exaggerate so the whole-frame shear is
    # clearly visible as before→after arrows.
    params = [3.0, -2.0, 6.0, 0.0, 5.0, 2.0]
    lon, lat, lon_s, lat_s = vsh_shift_frame(params, n_lon=18, n_lat=9,
                                             lat_max=80.0, scale=1.0)
    # plot_displacement follows the geodesic and is seam-aware, so sources
    # sitting on the wrap edge render correctly without special handling.
    plot_displacement(ax, lon, lat, lon_s, lat_s,
                      color="C2", lw=1.2, arrowstyle="->")
    ax.scatter(lon, lat, transform=ax.get_transform("world"),
               s=8, color="0.4", zorder=4)
    add_plane_overlay(ax, plane="galactic", color="C3", lw=0.8, alpha=0.5)
    ax.set_title("vsh_shift_frame — whole-frame displacement "
                 "(rotation + glide)")
    return fig


def main():
    banner("vsh forward model — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
