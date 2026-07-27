"""Render day/night terminator blends for visual eyeballing.

Compares the legacy image-space ``blend='gaussian'`` smoothing against the
physical ``blend='elevation'`` mode (solar-elevation field + transfer
curves), using real NASA Blue Marble (day) and Black Marble (night-lights)
Plate Carrée world maps.

The example images live in ``examples/data/`` and are untracked, so the
panels here only register when both images are present locally — i.e. these
baselines are opt-in and won't break clones that lack the imagery.

Produces (when the images are available):
  - nightshade_01_curve_comparison.png — gaussian vs the three elevation
      curves (linear / smoothstep / twilight), full globe.
  - nightshade_02_twilight_bands.png  — smoothstep terminator at civil
      (-6°), nautical (-12°), and astronomical (-18°) twilight depths.
"""

import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper.globe.nightshade import (
    _subsolar_lonlat,
    make_nightshade_blend,
)

try:
    import cartopy.crs as ccrs
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


# --- example imagery (untracked; panels only register if both exist) ---
_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "examples", "data"))
_DAY_PATH = os.path.join(_DATA_DIR, "world.topo.bathy.200412.3x5400x2700.jpg")
_NIGHT_PATH = os.path.join(_DATA_DIR, "BlackMarble_2016_01deg.jpg")
_HAVE_IMAGES = os.path.exists(_DAY_PATH) and os.path.exists(_NIGHT_PATH)

# A fixed instant so the terminator geometry (and therefore the baseline) is
# deterministic. Northern-summer afternoon UTC gives a nicely tilted line.
_DATE = datetime.datetime(2024, 6, 21, 18, 0, 0)
_EXTENT = (-180.0, 180.0, -90.0, 90.0)


def _load_map(path, width, height):
    """Load a Plate Carrée JPG and resample to (height, width) RGB in [0, 1]."""
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((width, height))
    return np.asarray(im, dtype=float) / 255.0


def _composite(ax, day, night, title, **blend_kw):
    """Daytime base + night-lights overlay cross-faded at the terminator."""
    ax.imshow(day, extent=_EXTENT, origin="upper", aspect="auto")
    rgba = make_nightshade_blend(night, _DATE, **blend_kw)
    ax.imshow(rgba, extent=_EXTENT, origin="upper", aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


if _HAVE_IMAGES:

    @_panel("nightshade_01_curve_comparison")
    def render_curve_comparison():
        """Gaussian vs the three elevation transfer curves, full globe."""
        w, h = 720, 360
        day = _load_map(_DAY_PATH, w, h)
        night = _load_map(_NIGHT_PATH, w, h)

        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        # Image-space Gaussian (the original method) for reference.
        try:
            _composite(axes[0, 0], day, night,
                       "blend='gaussian' (image-space, fixed pixel width)",
                       blend="gaussian")
        except ImportError:
            axes[0, 0].imshow(day, extent=_EXTENT, origin="upper",
                              aspect="auto")
            axes[0, 0].set_title("blend='gaussian' (cartopy/scipy unavailable)",
                                 fontsize=9)
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])
        # The three physical curves, all over the full 0 → -18° twilight band.
        _composite(axes[0, 1], day, night,
                   "blend='elevation', curve='linear'  (≈ perceived brightness)",
                   blend="elevation", curve="linear")
        _composite(axes[1, 0], day, night,
                   "blend='elevation', curve='smoothstep'  (raised cosine)",
                   blend="elevation", curve="smoothstep")
        _composite(axes[1, 1], day, night,
                   "blend='elevation', curve='twilight'  (radiometric falloff)",
                   blend="elevation", curve="twilight", twilight_decay=6.0)
        fig.suptitle(
            "Nightshade blend — terminator softening methods "
            f"({_DATE:%Y-%m-%d %H:%M} UTC)", fontsize=13)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.03,
                            hspace=0.18, wspace=0.05)
        return fig

    @_panel("nightshade_02_twilight_bands")
    def render_twilight_bands():
        """How the night-edge threshold sets the terminator (band) width."""
        w, h = 720, 360
        day = _load_map(_DAY_PATH, w, h)
        night = _load_map(_NIGHT_PATH, w, h)

        fig, axes = plt.subplots(3, 1, figsize=(11, 11))
        for ax, h_night, label in zip(
                axes, (-6.0, -12.0, -18.0),
                ("civil (h_night = -6°)", "nautical (h_night = -12°)",
                 "astronomical (h_night = -18°)")):
            _composite(ax, day, night,
                       f"smoothstep — {label}",
                       blend="elevation", curve="smoothstep", h_night=h_night)
        fig.suptitle("Nightshade blend — twilight-band depth "
                     "(curve='smoothstep')", fontsize=13)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                            hspace=0.15)
        return fig

    if _HAS_CARTOPY:

        @_panel("nightshade_03_orthographic_terminator")
        def render_orthographic_terminator():
            """Same four blends on an orthographic globe centered on the
            terminator, so the day→night transition runs through the disk
            center (least-distorted spot) for close inspection."""
            w, h = 1024, 512   # a little more resolution for the warped globe
            day = _load_map(_DAY_PATH, w, h)
            night = _load_map(_NIGHT_PATH, w, h)

            # The terminator crosses the equator 90° of longitude from the
            # sub-solar point; centring there puts the day/night edge straight
            # down the middle of the visible disk.
            lon0, _dec = _subsolar_lonlat(_DATE)
            proj = ccrs.Orthographic(central_longitude=lon0 + 90.0,
                                     central_latitude=0.0)

            specs = [
                ("blend='gaussian'", dict(blend="gaussian")),
                ("elevation / linear", dict(blend="elevation", curve="linear")),
                ("elevation / smoothstep",
                 dict(blend="elevation", curve="smoothstep")),
                ("elevation / twilight",
                 dict(blend="elevation", curve="twilight", twilight_decay=6.0)),
            ]
            fig = plt.figure(figsize=(12, 12))
            for i, (title, blend_kw) in enumerate(specs, start=1):
                ax = fig.add_subplot(2, 2, i, projection=proj)
                ax.set_global()
                # cartopy reprojects both equirectangular layers onto the
                # globe; the night RGBA's alpha does the cross-fade.
                ax.imshow(day, extent=_EXTENT, transform=ccrs.PlateCarree(),
                          origin="upper")
                try:
                    rgba = make_nightshade_blend(night, _DATE, **blend_kw)
                    ax.imshow(rgba, extent=_EXTENT,
                              transform=ccrs.PlateCarree(), origin="upper")
                except ImportError:
                    title += " (unavailable)"
                ax.set_title(title, fontsize=10)
            fig.suptitle(
                "Nightshade blend — orthographic, centered on the terminator "
                f"({_DATE:%Y-%m-%d %H:%M} UTC)", fontsize=13)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                                hspace=0.08, wspace=0.05)
            return fig


def main():
    banner("nightshade day/night blend — gallery")
    if not _HAVE_IMAGES:
        print(f"  example imagery not found under {_DATA_DIR} — skipping.")
        print(f"    expected: {os.path.basename(_DAY_PATH)}")
        print(f"              {os.path.basename(_NIGHT_PATH)}")
        return 0
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
