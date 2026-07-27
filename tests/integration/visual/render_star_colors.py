"""Render perceived-star-color sequences for visual eyeballing.

Produces:
  - starcolor_01_index_sequences.png — one horizontal color-sequence strip per
    method (teff_to_rgb, bv_to_rgb, bp_rp_to_rgb, color_index_to_rgb g-r / J-K),
    each annotated with spectral type + the input value at that point.
  - starcolor_02_sun_agreement.png — the Sun rendered five ways (Teff and all
    four color indices) as adjacent swatches; they should read the same white,
    the visual proof that every index resolves to one Teff scale.

These back the perceived-color functionality's visual regression: a real change
to the tristimulus pipeline, the Ballesteros B-V relation, or the Pecaut &
Mamajek (2013) interpolation tables will visibly shift a strip.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper import star_colors as _sc
from skyplothelper.star_colors import (
    bp_rp_to_rgb,
    bv_to_rgb,
    color_index_to_rgb,
    teff_to_rgb,
)

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


# Representative Pecaut & Mamajek (2013) dwarf-sequence anchors. Every strip
# shares ONE temperature axis, so a given spectral type sits at the same x
# across all five methods; the per-method columns give the input VALUE that
# type carries in each system (`nan` = index not tabulated for that type, so
# that method's colored box stops short there).
#   SpT     Teff     B-V    BP-RP    g-r     J-K
_ANCHORS = [
    ("B0", 31400,  -0.30,  np.nan,  -0.59,  -0.226),
    ("B5", 15700,  -0.156, np.nan,  -0.413, -0.094),
    ("A0",  9700,   0.00,  -0.037,  -0.25,  -0.004),
    ("F0",  7220,   0.295,  0.377,   0.10,   0.141),
    ("G2",  5770,   0.65,   0.823,   0.476,  0.366),
    ("K0",  5270,   0.816,  0.983,   0.62,   0.478),
    ("K5",  4440,   1.15,   1.43,    1.04,   0.70),
    ("M0",  3850,   1.42,   1.84,    np.nan, 0.817),
    ("M4",  3210,   1.65,   2.94,    np.nan, 0.837),
]

# Shared temperature axis: hot (blue) on the left, cool (red) on the right —
# the same direction as the color indices, so every strip reads left→right the
# same way. Log-spaced so the cool dwarfs (where color changes fastest) breathe.
_T_HOT, _T_COOL = 32000.0, 3000.0


def _u_of_teff(teff):
    """Position in [0, 1] along the shared axis (0 = hottest/left)."""
    lo, hi = np.log10(_T_COOL), np.log10(_T_HOT)
    return (hi - np.log10(teff)) / (hi - lo)


def _bv_of_teff(teff):
    """Invert the Ballesteros B-V→Teff relation numerically (B-V is defined at
    every type, so this strip spans the whole axis)."""
    bv = np.linspace(-0.35, 2.0, 4000)
    t = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
    return np.interp(teff, t[::-1], bv[::-1])          # t descends in bv


def _index_of_teff(teff, key):
    """Tabulated color index as a function of Teff (NaN outside the index's
    tabulated range, so the strip's box shortens there)."""
    tt = _sc._TEFF_COL
    cc = _sc._COLOR_COLS[key]
    m = np.isfinite(cc)
    order = np.argsort(tt[m])
    ta, ca = tt[m][order], cc[m][order]
    out = np.interp(teff, ta, ca, left=np.nan, right=np.nan)
    return np.where((teff < ta.min()) | (teff > ta.max()), np.nan, out)


def _strip(ax, title, teff_axis, input_arr, method, anchor_col, fmt):
    """Draw one perceived-color strip on the shared temperature axis.

    ``input_arr`` is this method's input at each ``teff_axis`` sample; cells
    whose input is NaN (index not defined there) render transparent so the
    colored box stops short. Anchor ticks show the method's own input value.
    """
    rgb = method(input_arr)
    finite = np.isfinite(rgb).all(axis=-1) & np.isfinite(input_arr)
    rgba = np.zeros((len(input_arr), 4))
    rgba[:, :3] = np.clip(np.nan_to_num(rgb), 0.0, 1.0)
    rgba[:, 3] = finite.astype(float)                  # transparent where undef
    ax.imshow(rgba[np.newaxis, :, :], extent=(0.0, 1.0, 0.0, 1.0),
              aspect="auto", origin="lower", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title, fontsize=9.5, loc="left", pad=3)

    # Shared vertical guides at every anchor type (aligned across all strips).
    us = [_u_of_teff(row[1]) for row in _ANCHORS]
    for u in us:
        ax.axvline(u, color="#777", lw=0.5, alpha=0.35)

    # Per-method input-value ticks at the aligned type positions (de-crowded).
    ticks, labels, last = [], [], -1.0
    for row, u in zip(_ANCHORS, us):
        v = row[anchor_col]
        if not np.isfinite(v) or u - last < 0.05:
            continue
        ticks.append(u)
        labels.append(fmt(v))
        last = u
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)


@_panel("starcolor_01_index_sequences")
def _sequences():
    fig, axes = plt.subplots(5, 1, figsize=(9, 7.4), constrained_layout=True)
    teff = np.geomspace(_T_HOT, _T_COOL, 600)          # hot→cool == left→right

    _strip(axes[0], "teff_to_rgb(teff)  —  effective temperature [K]",
           teff, teff, teff_to_rgb, 1, lambda v: f"{v:.0f}")
    _strip(axes[1], "bv_to_rgb(bv)  —  Johnson B-V  (Ballesteros 2012)",
           teff, _bv_of_teff(teff), bv_to_rgb, 2, lambda v: f"{v:.2f}")
    _strip(axes[2], "bp_rp_to_rgb(bp_rp)  —  Gaia BP-RP  (Pecaut & Mamajek 2013)",
           teff, _index_of_teff(teff, "bp-rp"), bp_rp_to_rgb, 3,
           lambda v: f"{v:.2f}")
    _strip(axes[3], "color_index_to_rgb(v, 'g-r')  —  SDSS g-r  (Pecaut & Mamajek 2013)",
           teff, _index_of_teff(teff, "g-r"),
           lambda v: color_index_to_rgb(v, "g-r"), 4, lambda v: f"{v:.2f}")
    _strip(axes[4], "color_index_to_rgb(v, 'J-K')  —  2MASS J-K  (Pecaut & Mamajek 2013)",
           teff, _index_of_teff(teff, "j-k"),
           lambda v: color_index_to_rgb(v, "J-K"), 5, lambda v: f"{v:.2f}")

    # Shared spectral-type row along the top of the first strip.
    sax = axes[0].secondary_xaxis("top")
    sax.set_xticks([_u_of_teff(row[1]) for row in _ANCHORS])
    sax.set_xticklabels([row[0] for row in _ANCHORS], fontsize=8)
    sax.tick_params(length=0)

    fig.suptitle("Perceived star-color sequences  —  shared temperature axis "
                 "(hot/blue left → cool/red right)", fontsize=12)
    return fig


@_panel("starcolor_02_sun_agreement")
def _sun_agreement():
    """The Sun by every route should be the same near-white swatch."""
    swatches = [
        ("teff_to_rgb\n5772 K", teff_to_rgb(5772)),
        ("bv_to_rgb\nB-V 0.65", bv_to_rgb(0.65)),
        ("bp_rp_to_rgb\nBP-RP 0.82", bp_rp_to_rgb(0.823)),
        ("g-r 0.476", color_index_to_rgb(0.476, "g-r")),
        ("J-K 0.366", color_index_to_rgb(0.366, "J-K")),
    ]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for i, (label, rgb) in enumerate(swatches):
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1.0,
                                   color=np.clip(rgb, 0, 1)))
        ax.text(i + 0.46, -0.12, label, ha="center", va="top", fontsize=8)
    ax.set_xlim(-0.1, len(swatches))
    ax.set_ylim(-0.5, 1.05)
    ax.axis("off")
    ax.set_title("The Sun, five ways — one Teff scale, one color", fontsize=11)
    return fig


def main():
    banner("star_colors — perceived-color visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
