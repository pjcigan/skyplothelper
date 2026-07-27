"""Render the cartopy_backend helpers (make_cartopy_frame /
cartopy_figure) for visual eyeballing.

Produces:
  - cartopy_01_projection_grid.png — six common projections in a 2x3 grid
  - cartopy_02_features.png        — coastlines + land + ocean toggles
  - cartopy_03_orthographic_centered.png — orthographic centered on a city
"""

import sys

import matplotlib.pyplot as plt
from _common import banner, save_or_show

try:
    from skyplothelper.cartopy_backend import (
        cartopy_figure,
        make_cartopy_frame,
    )
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


# Conditional registration: when cartopy isn't installed, leave
# PANELS empty so pytest-mpl's collection produces no tests for
# this module.
if _HAS_CARTOPY:

    @_panel("cartopy_01_projection_grid")
    def render_projection_grid():
        projections = ["mollweide", "robinson", "sinusoidal",
                       "plate_carree", "orthographic", "lambert_azimuthal"]
        fig = plt.figure(figsize=(15, 9))
        for idx, proj in enumerate(projections, start=1):
            ax = make_cartopy_frame(subplotnumber=230 + idx,
                                    projection=proj, center=0,
                                    coastlines=True, land=True, ocean=True,
                                    fig=fig)
            ax.set_title(proj, fontsize=10)
        fig.suptitle("make_cartopy_frame — six common projections",
                     fontsize=13)
        fig.subplots_adjust(top=0.93, hspace=0.3, wspace=0.15)
        return fig

    @_panel("cartopy_02_features")
    def render_features():
        fig = plt.figure(figsize=(15, 5.5))
        feature_combos = [
            ("coastlines only", dict(coastlines=True)),
            ("land + ocean", dict(land=True, ocean=True)),
            ("all (coastlines + land + ocean)",
             dict(coastlines=True, land=True, ocean=True)),
        ]
        for idx, (title, kw) in enumerate(feature_combos, start=1):
            ax = make_cartopy_frame(subplotnumber=130 + idx,
                                    projection="robinson", center=0,
                                    fig=fig, **kw)
            ax.set_title(title, fontsize=10)
        fig.suptitle("make_cartopy_frame — coastline / land / ocean "
                     "feature toggles", fontsize=12)
        fig.subplots_adjust(top=0.85, wspace=0.15)
        return fig

    @_panel("cartopy_03_orthographic_centered")
    def render_orthographic_centered():
        """An orthographic globe centered on Earth's surface."""
        fig, ax = cartopy_figure(projection="orthographic", center=(0, 50),
                                 coastlines=True, land=True, ocean=True,
                                 figsize=(7, 7))
        ax.set_title("cartopy_figure(projection='orthographic', "
                     "center=(0, 50)) — globe centered on (0°E, 50°N)",
                     fontsize=10)
        return fig


def main():
    banner("cartopy_backend — merge-verification visual gallery")
    if not _HAS_CARTOPY:
        print("cartopy not installed — skipping all renders.")
        return
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
