# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FITS Images & Quicklook
#
# Displaying an astronomical image well is mostly about **scaling** — choosing which
# pixel values map to the ends of the colormap, and how the values in between are
# distributed — so that faint structure shows without burning out the bright peaks.
# skyplothelper wraps that scaling stack, the one-call "quicklook" figures built on
# top of it, colorbars that read true physical units, and contour overlays, all on
# real WCS axes — through to multi-band RGB composites and reprojecting images onto
# a shared grid so they can be overlaid.
#
# This tutorial works through each in turn, anchored on a recurring real image — a
# VLBA radio map of the galaxy 3C 84 — with a supporting cast of other real datasets
# (M51, SN 1987A, NGC 602, the Crab Nebula), and closes by combining the tools into
# a publication-ready figure. It answers the two questions every image raises:
# **"how do I show my data this way?"** and **"how do I adjust it?"**
#
# ## Contents
#
# 1. [Interval and stretch](#1.-Interval-and-stretch)
# 2. [Colorbars and how to style them](#2.-Colorbars-and-how-to-style-them)
# 3. [Contour overlays](#3.-Contour-overlays)
# 4. [Quicklook in one call](#4.-Quicklook-in-one-call)
# 5. [Signed data and symmetric log](#5.-Signed-data-and-symmetric-log)
# 6. [Data cubes and channel maps](#6.-Data-cubes-and-channel-maps)
# 7. [RGB composites](#7.-RGB-composites)
# 8. [Reprojecting images to overlay them](#8.-Reprojecting-images-to-overlay-them)
# 9. [Putting it together](#9.-Putting-it-together)
# 10. [Where to go next](#10.-Where-to-go-next)
#
# > **Where the data comes from:** every image in this tutorial is a real, bundled
# > dataset under `examples/data/`. That directory's `README.md` lists the source,
# > instrument, and provenance of each file (3C 84, M51, SN 1987A, NGC 602, the Crab
# > Nebula, the DDO 70 H I cube, and the Gaia field), so you can trace or re-download
# > any of them.

# %%
import os
import tempfile
import warnings

import astropy.units as u
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import multicolorfits as mcf
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.colors import to_rgb
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

import skyplothelper as sph

# Give every figure a clean, consistent skyplothelper look via the 'structural' base
# style (tidy spines, ticks, and fonts). It composes with whatever theme or palette
# is active — including the dark mode of these docs.
sph.set_style(base="structural")

# %% [markdown]
# Throughout, we lean on one recurring **data anchor**: a VLBA 15 GHz image of the
# radio galaxy **3C 84** (0316+413 / NGC 1275, the brightest galaxy in the Perseus
# cluster), from the MOJAVE monitoring program. It is an ideal teaching image —
# a brilliant, compact core a few thousand times brighter than the faint jet that
# trails away from it, all sitting on a sea of low-level noise. Showing the core
# *and* the jet in one frame is exactly the problem image scaling exists to solve.
#
# Substitute your own FITS file for `FITS_PATH` and everything below applies
# unchanged.

# %%
FITS_PATH = "../../examples/data/0316+413.u.stacked.icd.fits"

# Real FITS headers carry small WCS quirks (e.g. a DATE-OBS that astropy wants to
# normalize); silence only that specific, harmless fix so output cells stay clean.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FITSFixedWarning)
    with fits.open(FITS_PATH) as hdul:
        # The image is a 4-D radio cube (1, 1, NY, NX) with two degenerate axes;
        # np.squeeze drops them to the 2-D sky image we actually plot. See §1.
        data = np.squeeze(hdul[0].data).astype(float)
        hdr = hdul[0].header
    wcs = WCS(hdr).celestial                # the 2-D sky WCS for WCSAxes plots

# Off-source noise the radio way: the standard deviation of a source-free corner.
# (The bright core lives at frame center; a 150-px corner box is pure background.)
rms = float(np.std(data[:150, :150]))
peak = float(np.nanmax(data))
print(f"image {data.shape},  peak = {peak:.3f} Jy/beam,  noise = {rms*1e3:.3f} mJy/beam"
      f"  (dynamic range ~ {peak/rms:.0f}:1)")

# All the real structure sits within a few tens of mas of the core; the rest of the
# 1024-px frame is noise and dirty-beam rings. This helper crops a WCSAxes to a
# symmetric window (in mas) about the core, so the showcase figures zoom to the jet.
CORE = (hdr["CRVAL1"], hdr["CRVAL2"])       # 3C 84 core position (deg)
_mas_per_px = abs(hdr["CDELT2"]) * 3.6e6


def zoom_to_core(ax, half_mas=20.0):
    """Crop a WCSAxes built on `wcs` to ±half_mas about the 3C 84 core."""
    cx, cy = wcs.world_to_pixel_values(*CORE)
    half = half_mas / _mas_per_px
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def tidy_radec(ax, nlon=4, nlat=5, size=8):
    """Thin out and shrink RA/Dec tick labels — long sexagesimal labels crowd at
    these mas zooms. (A WCSAxes convenience, not specific to this image.)"""
    ax.coords[0].set_ticks(number=nlon)
    ax.coords[1].set_ticks(number=nlat)
    ax.coords[0].set_ticklabel(size=size)
    ax.coords[1].set_ticklabel(size=size)


def mojave_norm():
    """A symmetric-log normalization matching the look the MOJAVE survey publishes for
    3C 84 — linear within ~5 mJy of zero, logarithmic out to the 3 Jy peak — which
    reveals the jet's full dynamic range. (A `symlog` stretch suits any high-contrast
    source like this one, not just signed data; §5 covers the symmetric stretches.)"""
    return sph.make_norm(stretch="symlog", clip="manual", vmin=0, vmax=peak, a=5e-3)


def _load2d(name):
    """Load a bundled example FITS as (data, 2-D celestial WCS)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        h = fits.open(f"../../examples/data/{name}")[0]
        return np.squeeze(h.data).astype(float), WCS(h.header).celestial


def _load2d_northup(name):
    """Load a bundled FITS and reproject it onto a clean north-up, east-left grid.
    For images whose native WCS is rotated or flipped — like the bundled M51
    exposure, which is stored north-*down* — this puts the sky in the familiar
    orientation. (A one-line preview of the reprojection machinery in §8.)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        h = fits.open(f"../../examples/data/{name}")[0]
        data, src = np.squeeze(h.data).astype(float), WCS(h.header).celestial
    ny, nx = data.shape
    scale = float(np.mean(proj_plane_pixel_scales(src)))    # deg / pixel
    cra, cdec = src.pixel_to_world_values(nx / 2, ny / 2)   # sky at the image center
    up = WCS(naxis=2)                                       # a standard N-up, E-left WCS
    up.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    up.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    up.wcs.crval = [float(cra), float(cdec)]
    up.wcs.cd = [[-scale, 0.0], [0.0, scale]]
    out, _ = reproject_interp((data, src), up, shape_out=(ny, nx))
    return out, up


# %% [markdown]
# A small helper picks an annotation color palette that suits whichever page the
# figure lands on — these docs come in a light and a dark version, and the same
# trick works for your own light/dark themes. It reads the active figure background
# and returns the matching in-theme palette, so a single code path looks right in both.

# %%
def annotation_palette():
    """Pick the in-theme annotation palette for the active (light or dark) style."""
    r, g, b = to_rgb(mpl.rcParams["figure.facecolor"])
    dark = (0.299 * r + 0.587 * g + 0.114 * b) < 0.5
    return sph.ANNOTATION_PALETTES["dark" if dark else "publication"]


def theme_ink():
    """Foreground ('ink') color that reads on the active page — black on a light
    background, white on a dark one. The same luminance test as `annotation_palette`,
    for the few artists that must be colored *explicitly* rather than inheriting the
    theme (a `quicklook` axis/label color, a title stroke)."""
    r, g, b = to_rgb(mpl.rcParams["figure.facecolor"])
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"


# %% [markdown]
# ## The problem in one figure
#
# Why does scaling deserve a whole tutorial? Because of *dynamic range*. The 3C 84
# core is several thousand times brighter than the jet beside it, so a naive
# **linear** display (the same interval, 0 to the peak) spends the entire colormap on
# the core and renders everything else black. The *same data, same interval*, under a
# **symmetric-log** stretch — linear near zero, logarithmic toward the bright peak —
# reveals the jet and counter-jet that were there all along:

# %%
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), subplot_kw={"projection": wcs})
opener_norms = [("linear stretch — only the core",
                 sph.make_norm(stretch="linear", clip="manual", vmin=0, vmax=peak)),
                ("symmetric-log stretch — the jet appears", mojave_norm())]
for ax, (title, nrm) in zip(axes, opener_norms):
    ax.imshow(data, origin="lower", cmap="sph.deepsky", norm=nrm)
    zoom_to_core(ax, half_mas=17)
    tidy_radec(ax, nlon=2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Right Ascension (J2000)")
axes[0].set_ylabel("Declination (J2000)")
# Right panel shares the left's declination axis — hide its redundant labels.
axes[1].coords[1].set_ticklabel_visible(False)
axes[1].coords[1].set_axislabel("")
# Point out the structure the stretch recovered: text off to the side, arrow onto the
# middle of the faint southern jet that was invisible in the linear panel.
axes[1].annotate("faint jet", xy=(0.5, 0.20), xytext=(0.93, 0.34),
                 xycoords="axes fraction", ha="right", va="center", color="white",
                 fontsize=11, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.6),
                 path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
# Square WCSAxes leave big vertical margins under tight_layout; pack them by hand.
fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98, top=0.90, bottom=0.13)

# %% [markdown]
# Same pixels, same colormap, same interval — only the *stretch* changed. That is the
# single most visible lever in image display, and only the first: the rest of this
# tutorial adds the others — the interval, colorbars in true units, contour overlays,
# one-call quicklooks, RGB composites, and reprojection — and shows how to combine them.

# %% [markdown]
# ## 1. Interval and stretch
#
# Displaying an image well comes down to two **orthogonal** choices:
#
# - the **interval** — *which* pixel values land at the two ends of the colormap
#   (everything below the low end is floored, everything above the high end is
#   saturated);
# - the **stretch** — *how* values are distributed between those ends (linearly,
#   or compressed so faint detail near the floor gets more of the colormap).
#
# skyplothelper exposes both at three levels of convenience: `rescale_image()`
# (arrays in, scaled array out), `make_norm()` (a matplotlib normalization, so the
# original data is untouched), and `auto_stretch()` (it inspects the data and
# *recommends* a choice). We'll meet all three in this section.
#
# ### What is this image?
#
# Before scaling anything, it is worth knowing what the FITS file actually
# contains. `describe_wcs()` prints a friendly summary of any header or WCS — the
# projection, reference coordinate, pixel scale, and (for radio data) the
# synthesized beam:

# %%
sph.describe_wcs(hdr)

# %% [markdown]
# > **Note:** this file is a 4-D radio cube with shape `(1, 1, 1024, 1024)` — two
# > degenerate Stokes/frequency axes wrapping the 2-D sky image. We dropped them
# > with `np.squeeze` when loading (above); `describe_wcs` is unbothered and reports
# > the celestial part. For headers that *also* need their WCS trimmed to 2-D,
# > `squeeze_image()` and `force_hdr_to_2D()` do that job (see the guide).
#
# Now the pixel values themselves. `describe_image()` prints the distribution
# statistics that drive every scaling decision, and `auto_stretch()` turns them
# into a recommendation:

# %%
sph.describe_image(data)
stretch_rec, reason = sph.auto_stretch(data)
print(f"\nauto_stretch recommends: {stretch_rec!r}  ({reason})")

# %% [markdown]
# The "dynamic range" line is the crux: the core is several thousand times brighter
# than the noise. `auto_stretch` plays it safe with `sqrt`, but for a contrast this
# extreme we will reach for the more aggressive `asinh`/`log` stretches below —
# `auto_stretch` is a sensible *starting point*, not the last word.
#
# ### The interval × stretch gallery
#
# The two choices are easiest to understand side by side. The top row fixes a
# sensible interval and varies only the **stretch**; the bottom row fixes `asinh`
# and varies only the **interval method**. Watch the faint jet emerge as the
# stretch compresses the bright core, and watch the interval method decide how much
# noise comes along with it.

# %%
CMAP = "sph.deepsky"    # the through-black skyplothelper map we use for 3C 84 throughout
# A floor a little below zero (the noise is symmetric about 0) and a ceiling well
# below the 3 Jy peak: clipping the core is the price of revealing the faint jet.
VMIN, VMAX = -3 * rms, 0.05

fig, axes = plt.subplots(2, 4, figsize=(11, 6))
# Top row: one interval, four stretches. The interval is the FULL range (0..peak), so
# the row varies only the stretch — linear then spends the colormap on the bright core
# and shows little else, while the jet emerges as each stretch compresses the core more.
for ax, stretch in zip(axes[0], ["linear", "sqrt", "log", "asinh"]):
    scaled = sph.rescale_image(data, stretch=stretch, clip="manual", vmin=0, vmax=peak)
    ax.imshow(scaled, origin="lower", cmap=CMAP)
    ax.set_title(f"stretch = {stretch}", fontsize=9)
# Bottom row: one stretch, four interval methods.
intervals = [("percentile (phi=99.9)", dict(clip="percentile", phi=99.9)),
             ("sigma (3σ)", dict(clip="sigma")),
             ("zscale", dict(clip="zscale")),
             ("manual", dict(clip="manual", vmin=VMIN, vmax=VMAX))]
for ax, (label, kw) in zip(axes[1], intervals):
    scaled = sph.rescale_image(data, stretch="asinh", **kw)
    ax.imshow(scaled, origin="lower", cmap=CMAP)
    ax.set_title(f"asinh · {label}", fontsize=9)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].set_ylabel("vary stretch\n(fixed interval)", fontsize=9)
axes[1, 0].set_ylabel("vary interval\n(fixed asinh)", fontsize=9)
fig.suptitle("3C 84 (VLBA 15 GHz): the same data, scaled eight ways", fontsize=12)
fig.tight_layout()

# %% [markdown]
# A `linear` stretch shows essentially nothing but the core — almost all of the
# colormap is spent on the handful of brightest pixels. `sqrt` and `log` recover
# progressively more, and `asinh` (inverse-hyperbolic-sine, the radio/optical
# workhorse) gives the smoothest faint-to-bright ramp. Along the bottom, a tight
# percentile or `sigma` interval shows the faintest emission but drags in noise,
# while `zscale` (the DS9 algorithm) and a hand-picked `manual` range trade a little
# of that faint structure for a cleaner background.
#
# ### The same problem on an extended source
#
# Dynamic range is not just a radio-interferometry quirk — it bites for any source
# with a bright center and faint outskirts. Here is a raw optical image of the
# Whirlpool galaxy (M51, a B+V ground-based exposure): a **linear** display (0 to the
# peak) shows only the two bright nuclei — the galaxy core and the companion NGC 5195 —
# and nothing else. An **asinh** stretch brings out the full spiral, the tidal bridge
# between the two, and the faint outer arms. This is a clear case where choosing a non-linear stretch gives much better results.

# %%
# The bundled M51 exposure is stored north-*down*; _load2d_northup reprojects it
# onto a standard north-up, east-left grid first (see the helper near the top).
m51, m51_wcs = _load2d_northup("m51_optical.fits")
m51_peak = float(np.nanmax(m51))

fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.6), subplot_kw={"projection": m51_wcs})
axes[0].imshow(m51, origin="lower", cmap="sph.nebula",
               norm=sph.make_norm(stretch="linear", clip="manual", vmin=0, vmax=m51_peak))
axes[0].set_title("linear — bright cores only", fontsize=11)
axes[1].imshow(m51, origin="lower", cmap="sph.nebula",
               norm=sph.make_norm(stretch="asinh", clip="percentile", plo=30, phi=99.8, data=m51))
axes[1].set_title("asinh — arms and companion appear", fontsize=11)
for ax in axes:
    ax.coords[0].set_ticklabel(size=7)
    ax.coords[1].set_ticklabel(size=7)
    ax.set_xlabel("Right Ascension (J2000)")
axes[0].set_ylabel("Declination (J2000)")
axes[1].coords[1].set_ticklabel_visible(False)
axes[1].coords[1].set_axislabel("")
fig.subplots_adjust(wspace=0.05, left=0.09, right=0.98, top=0.92, bottom=0.10)

# %% [markdown]
# > **Tip:** the `clip=` methods are convenience wrappers over lower-level helpers
# > that return the raw `(vmin, vmax)` for a custom pipeline: `clip_percentile`,
# > `clip_sigma`, `clip_zscale`, and the unified `auto_interval`. `adjust_gamma`
# > applies a separate gamma correction if you need one — the
# > [multicolorfits documentation](https://github.com/pjcigan/multicolorfits)
# > discusses gamma's role in image display in more depth.
#
# ### A norm for your own imshow
#
# `rescale_image` returns a *new array* of scaled values — convenient, but a
# colorbar drawn over it would read 0–1, not Jy/beam. `make_norm()` instead packages
# the identical interval+stretch choice as a matplotlib `Normalize`, so you hand
# `imshow` the **original** data and the colorbar reads true physical units:

# %%
fig, ax = plt.subplots(figsize=(5.2, 4.6))
norm = sph.make_norm(stretch="asinh", clip="manual", vmin=VMIN, vmax=VMAX, data=data)
im = ax.imshow(data, origin="lower", cmap=CMAP, norm=norm)
ax.set_xticks([])
ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("Jy / beam")
ax.set_title("make_norm: data untouched, colorbar in true units", fontsize=10)
fig.tight_layout()

# %% [markdown]
# The colorbar tick spacing follows the `asinh` stretch — bunched at the bright end,
# spread at the faint end — which is exactly the point: it shows the *real* Jy/beam
# value everywhere. We build on this in [§2](#2.-Colorbars-and-how-to-style-them).
#
# ### The stretch menu
#
# `list_stretches()` prints every available stretch:

# %%
sph.list_stretches()

# %% [markdown]
# Grouped by what they are for:
#
# | Stretch | What it does | Reach for it when… | Caveat |
# |---|---|---|---|
# | `linear` | no remapping | data already low-contrast | hides faint structure beside a bright peak |
# | `sqrt` / `squared` | mild boost / mild suppress | gentle faint-end lift / de-emphasis | `squared` buries the faint end further |
# | `log` | logarithmic | positive data spanning decades | undefined at ≤0 — needs a positive floor |
# | `asinh` / `sinh` | soft-log / its inverse | the everyday high-dynamic-range default | `asinh` is the one you want 90% of the time |
# | `power` | configurable γ via `a=` | a custom curve between linear and log | you must tune `a` |
# | `histeq` | histogram equalization | maximizing visible contrast | distorts relative brightness; needs astropy |
# | `symlog` / `symmetric_log` | linear core + log wings | signed data (residuals, Stokes Q/U) *and* high-contrast positive sources | full theory in [§5](#5.-Signed-data-and-symmetric-log) |
#
# > **Tip:** `asinh` is a sensible default for almost any astronomical image with
# > a bright source and faint surroundings — start there, then adjust the interval.
# > `log` and `sqrt` can also work well, depending on the distribution of the data, try a few options to see what works best for your image.
# > For *extreme* dynamic range (like the several-thousand-to-one of 3C 84), or log-scaled values with positive and negative values, step up
# > to `symlog`, which is what the rest of this tutorial uses for the radio maps.

# %% [markdown]
# ## 2. Colorbars and how to style them
#
# A colorbar is where all that scaling pays off — it lets a reader recover physical
# values from colors. But that only works if the colorbar knows the *real* data
# range, and there is a subtle trap. From here on we settle on the **symmetric-log
# normalization** from the opener (`mojave_norm()` above — the MOJAVE-style look),
# **zoom to the central jet** (the outer frame is just noise), and plot on the image's
# real WCS axes.
#
# - `rescale_image()` returns a **new array** of values squashed into 0–1. Hand
#   *that* to `imshow` and a colorbar describes the 0–1 scaled numbers — physically
#   meaningless.
# - `make_norm()` hands `imshow` the **original** data plus a normalization, so the
#   colorbar reads true Jy/beam. **This is the one you want for use with colorbars.**
#
# Same image, same look, two colorbars — only the right-hand one gives the true data values:

# %%
# constrained_layout + inset bars pack the two square frames large and close together;
# a divider bar between them (or plain tight_layout) leaves them small and far apart.
fig = plt.figure(figsize=(9.0, 4.6), constrained_layout=True)

# Left: the trap — a colorbar over rescaled (0–1) values.
ax0 = fig.add_subplot(1, 2, 1, projection=wcs)
scaled = sph.rescale_image(data, stretch="symlog", clip="manual", vmin=0, vmax=peak, a=5e-3)
im0 = ax0.imshow(scaled, origin="lower", cmap=CMAP)
zoom_to_core(ax0)
tidy_radec(ax0)
sph.add_colorbar(im0, ax=ax0, mode="inset", label="scaled 0–1 (meaningless)")
ax0.set_xlabel("Right Ascension (J2000)")
ax0.set_ylabel("Declination (J2000)")
ax0.set_title("rescale_image → colorbar lies", fontsize=11)

# Right: make_norm keeps the data real, so the bar reads Jy/beam. Its declination
# axis repeats the left panel's, so hide it and let the two images sit closer.
ax1 = fig.add_subplot(1, 2, 2, projection=wcs)
norm = mojave_norm()
im1 = ax1.imshow(data, origin="lower", cmap=CMAP, norm=norm)
zoom_to_core(ax1)
tidy_radec(ax1)
sph.add_colorbar(im1, ax=ax1, mode="inset", label="Jy/beam (true values)")
ax1.set_xlabel("Right Ascension (J2000)")
ax1.coords[1].set_ticklabel_visible(False)
ax1.coords[1].set_axislabel("")
ax1.set_title("make_norm → colorbar reads true values", fontsize=11)

# %% [markdown]
# The two images are pixel-for-pixel identical; only the colorbar differs. Reach for
# `make_norm` whenever the colorbar matters (which is almost always), and keep
# `rescale_image` for when you genuinely want a scaled array — compositing channels
# into an RGB image (§7), say, where there are no physical units to preserve.
#
# > **Note:** `sph.add_colorbar()` fixes a real annoyance — a plain `plt.colorbar` sizes
# > itself to the axes *bounding box*, not the rendered image, so on a fixed-aspect
# > frame (any `imshow`, every WCS axes) it overshoots or falls short. `sph.add_colorbar`
# > matches the image height; its `mode=` and styling controls are covered just below.
#
# ### Choosing a colormap
#
# The colormap is the other half of how an image reads. matplotlib ships dozens; a
# small selection that suits astronomical data, all perceptually reasonable — starting
# from `viridis`, matplotlib's own default:

# %%
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4),
                         subplot_kw={"projection": wcs})
for ax, cmap in zip(axes, ["viridis", "inferno", "cubehelix", "bone_r"]):
    ax.imshow(data, origin="lower", cmap=cmap, norm=norm)
    zoom_to_core(ax)
    ax.set_title(cmap, fontsize=10)
    for c in ax.coords:
        c.set_ticklabel_visible(False)
        c.set_ticks_visible(False)
fig.suptitle("The same 3C 84 image under four colormaps", fontsize=12)
fig.tight_layout()

# %% [markdown]
# `viridis` is the safe, perceptually-uniform default; `inferno` (and its near-twin
# `magma`) give punchier, high-contrast renderings; `cubehelix` increases
# monotonically in *printed* brightness (safe in grayscale); `bone_r` is a clean
# black-on-white, grayscale-adjacent look that reads well as a print or journal
# figure. The choice is aesthetic, but favor a perceptually uniform map — avoid
# `jet`, whose false bright bands may give the false impression of importance to features in the data that a perceptually-uniform map would not (ironic that we downplay it here, as `jet` was originally created to colorize astronomical jet simulation images).

# %% [markdown]
# ### skyplothelper's bundled colormaps
#
# Beyond the matplotlib maps, skyplothelper ships a small **curated set of image
# colormaps** — the raster twin of the cycle palettes used for scatter/line colors (see
# [Themes, Palettes & Fonts](styling.ipynb)). `show_colormaps()` previews the whole set
# at a glance: 12 **sequential** maps (luminance-smoothed, for ordinary intensity
# images) and 6 **diverging** `diff_*` maps (for signed / difference data):

# %%
_ = sph.show_colormaps()   # assign the returned Figure so it isn't echoed twice

# %% [markdown]
# `list_colormaps()` returns the names. Use any of them anywhere a matplotlib colormap
# goes, three equivalent ways — a `cmap='sph.<name>'` string, a cmocean-style attribute
# `sph.colormaps.<name>`, or `get_colormap('<name>')` for the object; append `_r` to the
# string to reverse. Four of the sequential maps on the M51 image from §1: `deepsky`
# (the through-black map we reach for in place of `magma`), `nebula` (a warmer
# purple-brown cast), `thicket` (green/forest), and `mesa_r` — the **reversed** form of
# `mesa`, previewing both the `_r` trick and the terracotta `mesa` map we use for the
# radio images later:

# %%
m51_cnorm = sph.make_norm(stretch="asinh", clip="percentile", plo=30, phi=99.8, data=m51)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), subplot_kw={"projection": m51_wcs})
for ax, cmap in zip(axes, ["sph.deepsky", "sph.nebula", "sph.thicket", "sph.mesa_r"]):
    ax.imshow(m51, origin="lower", cmap=cmap, norm=m51_cnorm)
    ax.set_title(cmap, fontsize=10)
    for c in ax.coords:
        c.set_ticklabel_visible(False)
        c.set_ticks_visible(False)
fig.suptitle("Four bundled sequential colormaps on M51", fontsize=12)
fig.tight_layout()

# %% [markdown]
# For **signed** data — residuals, difference images, velocity fields — reach for a
# **diverging** `diff_*` map instead, anchored so zero sits at the neutral midpoint
# (pair it with a symmetric `vmin=-lim, vmax=+lim`, as in [§5](#5.-Signed-data-and-symmetric-log)).
# Their off-beat color pairings are deliberate — distinct from the standard red-blue, so
# a difference map never gets mistaken for an intensity image. Here on the residual of a
# model fit (data − model), where the fit mis-centered one source and under-estimated
# another's flux — exactly the signed structure a diverging map is built for:

# %%
# A realistic "fit residual" = data − model. The fitted model mis-centered the bright
# source (leaving a +/− dipole) and under-estimated the fainter one's flux (a positive
# blob); zero sits at the neutral midpoint color, positive and negative to either side.
rng = np.random.default_rng(7)
gx, gy = np.meshgrid(np.linspace(-10, 10, 240), np.linspace(-10, 10, 240))
src_a = np.exp(-((gx + 3)**2 + (gy - 1)**2) / (2 * 1.6**2))          # bright source
src_b = 0.7 * np.exp(-((gx - 4)**2 + (gy + 2)**2) / (2 * 2.0**2))    # fainter source
model = (np.exp(-((gx + 2.5)**2 + (gy - 1.4)**2) / (2 * 1.6**2))     # mis-centered
         + 0.60 * np.exp(-((gx - 4)**2 + (gy + 2)**2) / (2 * 2.0**2)))  # flux too low
residual = (src_a + src_b) - model + 0.01 * rng.standard_normal((240, 240))
rlim = float(np.abs(residual).max())
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, cmap in zip(axes, ["sph.diff_blueorange", "sph.diff_tealorange", "sph.diff_purplegreen"]):
    ax.imshow(residual, origin="lower", cmap=cmap, vmin=-rlim, vmax=rlim)
    ax.set_title(cmap, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Three diverging diff_* colormaps on a fit residual (data − model)", fontsize=12)
fig.tight_layout()

# %% [markdown]
# > **Note:** this is a *curated* set — baked lookup tables for the luminance-smoothed
# > sequential maps, and exact neutral anchors for the `diff_*` maps. Treat them as a
# > ready palette to *use*; a full colormap-authoring toolkit is a separate future
# > package. For how they relate to the categorical cycle palettes, see
# > [Themes, Palettes & Fonts](styling.ipynb) and the
# > [styling guide](../guide/styling.md#image-colormaps).

# %% [markdown]
# ### Placement: divider, inset, and simple
#
# On a fixed-aspect axes a plain colorbar mis-sizes itself; `add_colorbar`'s `mode=`
# fixes it three ways. Each mode *matches the image* — the core fix — so they differ
# mainly in layout (below). For contrast, the middle panel instead drops a bar *fully
# inside* the frame via a hand-placed `cax`:
#
# | `mode` | how it places the bar | best for |
# |---|---|---|
# | `'divider'` *(default)* | reserves a slot beside the image; the image shrinks to fit | multi-panel rows — bars never overlap a neighbor |
# | `'inset'` | floats just outside; the image keeps its full size | a single panel where the image shouldn't shrink (e.g. the [§9](#9.-Putting-it-together) capstone) |
# | `'simple'` | classic `plt.colorbar` | quick one-offs; it accounts for bars already placed, so call it twice to stack several |
#
# All three also take `orientation='horizontal'` for a bar beneath the image, or
# `location=` (`'right'` *(default)* / `'left'` / `'top'` / `'bottom'`) to pick a side
# outright — it moves the ticks and label to the outer edge for you, so a left- or
# top-side bar is a one-liner rather than a hand-rolled `inset_axes`. (We put it to work
# on the moment-0 map in [§6](#6.-Data-cubes-and-channel-maps).)

# %%
# A generic synthetic blob field for the layout demos (bundled sph.deepsky colormap).
bx, by = np.meshgrid(np.linspace(-3, 3, 160), np.linspace(-3, 3, 160))
blob = np.exp(-((bx - 0.4)**2 + (by + 0.3)**2)) + 0.5 * np.exp(-((bx + 1.2)**2 + (by - 1)**2) / 0.6)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
# 'divider' — reserves a slot beside the image (the image shrinks to fit).
im = axes[0].imshow(blob, origin="lower", cmap="sph.deepsky")
sph.add_colorbar(im, ax=axes[0], label="value", mode="divider")
axes[0].set_title("mode='divider'", fontsize=10)
# A bar dropped *fully inside* the frame: hand add_colorbar a cax you place in
# axes-fraction coords (a light stroke keeps its ticks legible over the image).
im = axes[1].imshow(blob, origin="lower", cmap="sph.deepsky")
sph.add_colorbar(im, cax=axes[1].inset_axes([0.08, 0.10, 0.5, 0.05]),
                 orientation="horizontal", stroke_color="white", stroke_lw=2.5)
axes[1].set_title("inset via cax — inside the frame", fontsize=10)
# 'simple' — the classic plt.colorbar.
im = axes[2].imshow(blob, origin="lower", cmap="sph.deepsky")
sph.add_colorbar(im, ax=axes[2], label="value", mode="simple")
axes[2].set_title("mode='simple'", fontsize=10)
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Colorbar placement", fontsize=12)
fig.tight_layout()

# %% [markdown]
# The `mode=` values (`'divider'` / `'inset'` / `'simple'`) all *match the image* and
# differ only in layout — `'divider'` reserves space so the image shrinks, `'inset'`
# floats a same-size bar just outside without shrinking it (handy inside an ImageGrid),
# `'simple'` is the classic bar. For a bar genuinely *inside* the frame — the compact
# middle panel above — hand `add_colorbar` your own `cax=ax.inset_axes([...])`.

# %% [markdown]
# ### Styling the bar
#
# A colorbar's tick marks, labels, and spine are black by default — which **vanishes
# against the dark end of a colormap** (where the bar's own inward ticks sit) or on a
# dark page. `add_colorbar` can lay a contrasting **stroke** (outline) on them via
# `stroke_color` (with `stroke_targets` = `'both'` / `'ticks'` / `'spine'`); for full
# control, recolor the pieces directly with `cb.ax.tick_params(...)`, `cb.set_label(...,
# color=...)`, and `cb.ax.spines[...]`. The difference is clearest with inward ticks:

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for ax, stroke in zip(axes, [False, True]):
    im = ax.imshow(blob, origin="lower", cmap="sph.deepsky")
    cb = sph.add_colorbar(im, ax=ax, label="value", aspect=9,
                          **({"stroke_color": "white", "stroke_lw": 2.6} if stroke else {}))
    # Force the default *black* ticks + spine so the point lands under any page theme
    # (a dark page would otherwise auto-lighten them, hiding the very problem shown
    # here). The wider bar (aspect=9) keeps the inward marks from spanning its width;
    # the tick *labels* stay theme-colored since they sit on the light page margin.
    cb.ax.tick_params(which="both", direction="in", length=6, color="k",
                      labelcolor=theme_ink())
    cb.outline.set_edgecolor("k")
    cb.set_label("value", color=theme_ink())
    ax.set_title("stroked — visible all the way down" if stroke else
                 "default — black ticks lost on the dark end", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()

# %% [markdown]
# > **Note:** this matters doubly on a dark background, where a black spine disappears
# > entirely — the dark-theme version of these docs is exactly that case, so a light
# > stroke keeps the bar legible in both.
#
# ### Two colorbars in one figure
#
# When a figure carries two color-mapped datasets, each needs its own bar. The
# `divider`/`inset` modes each claim the same spot, so place the pair yourself with
# `cax=` (an `inset_axes` you size and position). First, two crossing cmapped scatter
# trends, with **half-height bars stacked** for a compact key:

# %%
rng = np.random.default_rng(5)
xx = np.linspace(0, 10, 140)
rising = 0.8 * xx + rng.normal(0, 0.6, xx.size)
falling = 8 - 0.7 * xx + rng.normal(0, 0.6, xx.size)

fig, ax = plt.subplots(figsize=(6.6, 4.6))
s1 = ax.scatter(xx, rising, c=xx, cmap="sph.mulberry", s=16)
s2 = ax.scatter(xx, falling, c=xx, cmap="sph.lagoon", s=16)
ax.set_xlabel("x")
ax.set_ylabel("y")
sph.add_colorbar(s1, cax=ax.inset_axes([1.03, 0.54, 0.045, 0.44]), label="rising trend")
_ = sph.add_colorbar(s2, cax=ax.inset_axes([1.03, 0.02, 0.045, 0.44]), label="falling trend")

# %% [markdown]
# And a cmapped background image with a **differently-cmapped contour overlay** — two
# full-height bars, side by side with a small gap. The overlay runs symmetrically about
# zero, so it takes a diverging **`diff_bluebrown`** map; its blue/brown also reads more
# clearly against the green `thicket` background than a warm sequential would. Placing the
# pair by hand, nudge the second `cax` far enough right that it clears the first bar's label:

# %%
overlay = np.sin(1.5 * bx) * np.cos(1.5 * by)
olim = float(np.abs(overlay).max())
fig, ax = plt.subplots(figsize=(6.8, 4.8))
im = ax.imshow(blob, origin="lower", cmap="sph.thicket", extent=[-3, 3, -3, 3])
# Signed overlay → a diverging map, centered by symmetric levels so zero sits at neutral.
cs = ax.contour(bx, by, overlay, levels=np.linspace(-olim, olim, 9),
                cmap="sph.diff_bluebrown", linewidths=1.4)
ax.set_xticks([])
ax.set_yticks([])
sph.add_colorbar(im, cax=ax.inset_axes([1.03, 0, 0.045, 1]), label="image")
_ = sph.add_colorbar(cs, cax=ax.inset_axes([1.24, 0, 0.045, 1]), label="overlay")  # nudged right, clear of the first label

# %% [markdown]
# ## 3. Contour overlays
#
# A stretch makes faint structure *visible*; contours make it *quantitative* —
# isophotes at known brightness levels let a reader read off how bright each feature
# is and compare figures fairly. The radio convention is levels spaced by powers of
# two above a multiple of the off-source noise: a ladder like 5σ, 10σ, 20σ, 40σ, …,
# with the **−5σ** level drawn dashed so any negative (deconvolution) artifacts are
# honestly shown. Starting a few σ up keeps the lowest contour clear of noise.
#
# `add_contour_overlay()` draws contours on a WCSAxes from world-coordinate grids,
# which `header_coord_grids()` builds for us:

# %%
lon, lat = sph.header_coord_grids(wcs)      # per-pixel RA, Dec grids (degrees)

# Powers-of-two ladder from 5σ up to the peak, plus a dashed -5σ level.
base = 5 * rms
ladder = base * 2.0 ** np.arange(0, np.log2(peak / base))
levels = np.concatenate([[-base], ladder])
print(f"contour levels (mJy/beam): {', '.join(f'{1e3*lev:.2f}' for lev in levels)}")

PAL = annotation_palette()
# The mesa_r image is light-backed and mode-invariant, so the contours use a fixed
# deep blue (readable on it in both the light and dark builds of these docs) rather
# than a theme-swapped accent that would wash out on one of them.
C3_CONTOUR = "#1f4e79"

fig = plt.figure(figsize=(5.6, 5.4))
ax = fig.add_subplot(111, projection=wcs)
# A muted image backdrop so the contours carry the information.
ax.imshow(data, origin="lower", cmap="sph.mesa_r", norm=norm, alpha=0.9)
# Matplotlib draws negative levels dashed automatically when given one color.
sph.add_contour_overlay(ax, lon, lat, data, levels=levels,
                        colors=C3_CONTOUR, linewidths=0.8)
zoom_to_core(ax, half_mas=16)
tidy_radec(ax)
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination (J2000)")
ax.set_title("3C 84 — contours at 5σ × 2ⁿ (−5σ dashed)", fontsize=11)

# %% [markdown]
# The contours trace the jet down the frame in even brightness steps, and a few
# dashed rings mark where the map dips 5σ *below* zero — a sanity check on the noise
# floor. Because the levels are tied to the measured `rms`, the same recipe gives a
# comparable figure for any radio map.
#
# Here it is on a second MOJAVE source — the blazar **1502+106**, a one-sided
# core-plus-jet — with its own noise (rms ≈ 0.11 mJy/beam) and a ladder from ~3.5σ:

# %%
jet, jet_wcs = _load2d("1502+106.u.stacked.icd.fits")
jet_lon, jet_lat = sph.header_coord_grids(jet_wcs)
jet_peak = float(np.nanmax(jet))
# MOJAVE-matching contours for this source: 0.39 mJy/beam in powers of two.
jet_levels = 0.39e-3 * 2.0 ** np.arange(0, np.log2(jet_peak / 0.39e-3))

fig = plt.figure(figsize=(5.6, 5.4))
ax = fig.add_subplot(111, projection=jet_wcs)
ax.imshow(jet, origin="lower", cmap="sph.mesa_r", alpha=0.9,
          norm=sph.make_norm(stretch="symlog", clip="manual", vmin=0, vmax=jet_peak, a=5e-3))
sph.add_contour_overlay(ax, jet_lon, jet_lat, jet, levels=jet_levels,
                        colors=C3_CONTOUR, linewidths=0.8)
ax.set_xlim(180, 320)        # pixel window on the central core-jet region
ax.set_ylim(180, 320)
tidy_radec(ax)
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination (J2000)")
ax.set_title("1502+106 — contours at 0.39 mJy × 2ⁿ", fontsize=11)

# %% [markdown]
# > **Tip:** `add_contour_overlay` takes either an integer (let matplotlib choose
# > that many evenly-spaced levels) or an explicit array (as here). Pass `filled=True`
# > for filled bands, `cmap=` to color levels by value, or any `contour`/`contourf`
# > keyword. For contours that share the image's own normalization (so faint
# > log-spaced levels spread out on the colorbar), pass the same `norm=`.

# %% [markdown]
# ## 4. Quicklook in one call
#
# Everything so far — load, measure the noise, scale, draw a beam, lay down σ-spaced
# contours, label it — is the *standard* radio-map recipe. `quicklook_fits()` does
# all of it from a path in one call, returning a `QuicklookResult` you can keep
# tweaking. Here we also switch on **offset coordinates** (more easily readable for tiny fields of view, such as in a VLBI image - relative arcsec/mas from a reference, rather than absolute RA/Dec) and crop to a
# 34 mas window about the core:

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FITSFixedWarning)   # same harmless DATE-OBS fix as at load
    # obs_date='' drops the date stamp: this MOJAVE *stack* carries the placeholder
    # DATE-OBS='MULTIEPOCH', which quicklook would faithfully print next to the title.
    res = sph.quicklook_fits(FITS_PATH, image=True, norm=mojave_norm(),
                             colormap="sph.deepsky", obs_date="",
                             offset_coords=True, field_size=34,
                             facecolor="none", axcolor=theme_ink(), info_color=theme_ink())
res.ax.set_title("3C 84 — VLBA 15 GHz (quicklook_fits)", fontsize=11)
res.fig.set_size_inches(6.6, 7.2)

# %% [markdown]
# From a single call we get the deepsky symmetric-log image (we hand it our
# `mojave_norm()` via `norm=` and name the `colormap`) with a colorbar in the header's own
# units, an auto-measured noise and σ-contour ladder, the synthesized beam (lower-left),
# relative-mas axes zeroed on the core, and an info block (peak, RMS, contour spec, beam).
# Note we never say what color the contours should be: marks drawn *on* the image sample
# their ink from the colormap, so they come out white over a dark map like `deepsky` and
# black over a reversed one like `gray_r` — pass `contour_color=` to override.
# Passing `facecolor='none'` lets the figure have a transparent outer canvas background, so it
# follows the light/dark theme of these docs like the others. The returned
# `QuicklookResult` exposes `.fig`, `.ax`, `.image`, `.contour_set`, `.colorbar`, and
# `.info_text`, so any piece is yours to restyle afterward — as we just did with the
# title.
#
# `quicklook_fits` takes a *path*; its siblings take data you already have in
# memory: `quicklook_plot(array, header=…)` draws onto an existing (or new) axes,
# and `quicklook_figure(array, …)` builds the figure for you; `quicklook_plot` also
# accepts a `MomentMap` (§6) directly, applying the moment defaults (order colormap,
# a centered velocity scale, a *Moment N* tag). They share every keyword, so the same
# call adjusts the result. Below, a small **synthetic** radio
# source (a bright core plus two jet knots — handy when you want a clean, known
# target) shows three display modes:

# %%
# A synthetic radio source: a TAN image with a Gaussian core + two offset knots +
# noise, wrapped in a minimal radio header (beam, units, name). Substitute your own.
rng = np.random.default_rng(42)
nx = ny = 256
gx, gy = np.meshgrid(np.linspace(-50, 50, nx), np.linspace(-50, 50, ny))
synth = (0.022 * np.exp(-(gx**2 + gy**2) / (2 * 5**2))
         + 0.005 * np.exp(-((gx - 15)**2 + (gy - 8)**2) / (2 * 3**2))
         + 0.002 * np.exp(-((gx - 25)**2 + (gy - 14)**2) / (2 * 4**2))
         + 1e-4 * rng.standard_normal((ny, nx)))
synth_hdr = fits.Header({
    "NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny, "CRPIX1": nx / 2 + 0.5,
    "CRPIX2": ny / 2 + 0.5, "CRVAL1": 330.075, "CRVAL2": 22.015,
    "CDELT1": -0.0001 / 3600, "CDELT2": 0.0001 / 3600, "CUNIT1": "deg",
    "CUNIT2": "deg", "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "BMAJ": 1.2 / 3600e3, "BMIN": 0.6 / 3600e3, "BPA": 15.0,
    "BUNIT": "JY/BEAM", "OBJECT": "J2200+2200", "DATE-OBS": "2023-06-15"})

fig = plt.figure(figsize=(12, 4.2))
# `image` and `colorbar` are stated per panel rather than left to the defaults: the left
# panel is *deliberately* contour-only, and holding the colorbar off keeps all three
# panels the same width so the comparison is about the display mode, nothing else.
modes = [("single-color contours",
          dict(image=False, color=PAL["accent"], frame_color=theme_ink())),
         ("grayscale image + cmap contours",
          dict(image=True, colormap="gist_yarg", contour_cmap="viridis")),
         ("image + white contours (asinh)",
          dict(image=True, colormap="sph.deepsky", contour_color="w", stretch="asinh"))]
for i, (title, kw) in enumerate(modes):
    ax = fig.add_subplot(1, 3, i + 1, projection=WCS(synth_hdr, naxis=2))
    # offset_coords gives short relative-mas ticks (this ~25 mas synthetic field would
    # otherwise get long, near-identical sexagesimal labels); facecolor='none' + axcolor
    # let the panels follow the page theme.
    sph.quicklook_plot(synth, ax=ax, header=synth_hdr, show_info=False, colorbar=False,
                       obs_date="", label="", offset_coords=True, offset_units="mas",
                       facecolor="none", axcolor=theme_ink(), **kw)
    ax.coords[0].axislabels.set_fontsize(9)
    ax.coords[1].axislabels.set_fontsize(9)
    ax.set_title(title, fontsize=9)
fig.suptitle("quicklook_plot display modes (synthetic source)", fontsize=12)
fig.tight_layout()

# %% [markdown]
# Contours alone (left) keep a figure print-light; a grayscale image with colored
# contours (center) shows extent *and* structure; a colormapped image with white
# contours (right) is the punchy presentation default.
#
# A quicklook is an **image with a colorbar by default** — that is what the name implies —
# so the two *subtractive* knobs are the ones to know: `image=False` gives the contour-only
# look above (the colorbar bows out with it), and `colorbar=False` keeps the image but drops
# the value bar, as the panels above do to stay the same width. Other knobs worth knowing:
# `contour_start`/`contour_factor` set the σ-ladder, `negative_contours` toggles the dashed
# sub-zero levels (`n_negative` caps how many), and `contour_lw='scaled'` thickens brighter
# contours.

# %% [markdown]
# ### Frame color and legibility
#
# You may have noticed the main quicklook figure earlier came out with a **gray frame
# and ticks**, not black. That is automatic: `quicklook` is mode-aware, and because a
# filled image with a dark colormap (`magma`, `inferno`, …) runs to black at the low
# end, a black frame would disappear into the dark edges — so `image=True` defaults to a
# gray (0.5) frame. Two knobs take over when you want more:
#
# - `frame_color=...` sets the frame/tick color outright (e.g. to match a theme);
# - `frame_stroke='white'` (or `{'color': …, 'lw': …}`) draws a contrasting outline
#   around the frame and ticks — the robust choice when you can't predict whether the
#   image edges are light or dark.
#
# On a full `sph.deepsky` fill, a plain black frame loses its inward ticks against the
# dark image (and the whole frame vanishes on a dark page — flip these docs to dark mode
# to see it). A white stroke makes them more visible by giving them an outline, and `frame_color` takes full control of the frame color itself:

# %%
# colorbar=False: this figure is about the *frame* — a value bar would only shrink the
# panels and pull the eye off the ticks it is meant to show.
frame_demo = dict(image=True, colormap="sph.deepsky", contours=False, show_info=False,
                  colorbar=False, obs_date="", label="", offset_coords=True,
                  offset_units="mas", facecolor="none", axcolor=theme_ink())
fig = plt.figure(figsize=(12, 4.0))
for i, (title, kw) in enumerate([
        ("frame_color='black' — lost at dark edges", dict(frame_color="black")),
        ("frame_stroke='white' — robust anywhere", dict(frame_stroke="white")),
        ("frame_color='#33bbee' — full control", dict(frame_color="#33bbee"))]):
    ax = fig.add_subplot(1, 3, i + 1, projection=WCS(synth_hdr, naxis=2))
    sph.quicklook_plot(synth, ax=ax, header=synth_hdr, **frame_demo, **kw)
    ax.coords[0].axislabels.set_fontsize(9)   # quicklook's WCS axis labels (12) dwarf the
    ax.coords[1].axislabels.set_fontsize(9)   # ticks/title; match the title size for tidiness
    ax.set_title(title, fontsize=9)
fig.suptitle("quicklook frame legibility on a dark colormap", fontsize=12)
fig.tight_layout()

# %% [markdown]
# The takeaway mirrors the colorbar stroking from
# [§2](#2.-Colorbars-and-how-to-style-them): the **gray default is the simple, safe
# choice**, a **stroke is robust on any background**, and **`frame_color` gives full
# control**.

# %% [markdown]
# When you want *just the pixels* — no beam, no contours, no info furniture —
# `simpleimageplot()` (array → axes) and `simpleimage_figure()` (array + header →
# figure) give a clean minimal display. Like the quicklook family, each returns a result
# whose artists you can restyle afterward — below, we stroke the title so it stays
# legible on any background:

# %%
fig, ax = plt.subplots(figsize=(4.8, 4.4))
res_simple = sph.simpleimageplot(synth, ax=ax, cmap="sph.deepsky", colorbar=True,
                                 tickcolor="white", labelcolor=theme_ink(),
                                 cbar_label="Jy / beam", axtitle="simpleimageplot — bare pixels")
# res_simple is a (fig, ax, image, colorbar) result — reach into any piece to restyle it.
res_simple.ax.title.set_path_effects(
    [pe.withStroke(linewidth=2.5, foreground=("black" if theme_ink() == "white" else "white"))])
fig.tight_layout()

# %% [markdown]
# ### Not just radio
#
# `quicklook` works on any 2-D FITS — here on an **optical** HST image, the Hα ring of
# SN 1987A, a different part of the spectrum entirely. The σ-contours now trace the
# ionized ring rather than a jet.
#
# Note how little we have to say: the image, its colorbar, and the σ-contours are all
# defaults, and the contours pick white out of `nebula` on their own. Two small touches
# make the bar read better here. The header's `BUNIT` is `Jy/pix` (not the radio
# `Jy/beam`), but the values run to only a few ×10⁻⁴ Jy, so `display_factor=1000` with
# `unit="mJy/pix"` relabels the bar in **milli**-Jy and drops the leading zeros. And
# because an `asinh` stretch expands the near-zero range, matplotlib's own major ticks
# bunch up at the bright end and leave the lower bar unlabeled — an explicit
# 1/2/5-per-decade `set_ticks` labels it the whole way down, and `cbar_minor_ticks=False`
# drops the now-redundant minors. (The info block is switched **off**: its peak/RMS
# readout is written for radio maps.)

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FITSFixedWarning)
    opt = sph.quicklook_fits("../../examples/data/sn1987a_hst_F656N.fits",
                             show_info=False, colormap="sph.nebula", stretch="asinh",
                             source_name="SN 1987A — HST Hα",
                             display_factor=1000, unit="mJy/pix", cbar_minor_ticks=False,
                             facecolor="none", axcolor=theme_ink(), info_color=theme_ink())
# asinh bunches its auto major ticks at the bright end; an explicit 1/2/5-per-decade set
# (values in the display's mJy/pix units) labels the bar the whole way down its expanded low
# end. With that full set the auto minor ticks only clutter — cbar_minor_ticks=False above.
opt.colorbar.set_ticks([0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
opt.fig.set_size_inches(6, 6)

# %% [markdown]
# ## 5. Signed data and symmetric log
#
# Everything so far assumed *positive* data. But plenty of images are **signed** —
# imaging residuals, Stokes Q/U polarization, line-of-sight velocity fields — with
# meaningful structure on *both* sides of zero, often spanning decades each way.
# A positive-only stretch like `log` or `asinh` cannot cope: it either fails on the
# negative values or maps them asymmetrically, hiding half the signal.
#
# The fix is a **symmetric** stretch: linear through a small window around zero, then
# logarithmic into *both* the positive and negative wings. skyplothelper offers two:
#
# - `stretch='symlog'` — matplotlib's `SymLogNorm`.
# - `stretch='symmetric_log'` — a C¹-continuous variant from the optional `pysymlog`
#   package (smoother across the linear-to-log transition).
#
# Here is a synthetic signed field — a bipolar pair of strong lobes plus fainter
# secondary structure and noise, the shape an imaging-residual or Stokes-Q map takes —
# rendered with one of the diverging `diff_*` colormaps from
# [§2](#2.-Colorbars-and-how-to-style-them), the natural pairing for signed data:

# %%
rng = np.random.default_rng(3)
m = 256
mx, my = np.meshgrid(np.linspace(-50, 50, m), np.linspace(-50, 50, m))
signed = (0.8 * np.exp(-((mx - 12)**2 + (my - 6)**2) / (2 * 7**2))     # + lobe
          - 0.8 * np.exp(-((mx + 12)**2 + (my + 6)**2) / (2 * 7**2))   # − lobe
          + 0.03 * np.exp(-(mx**2 + (my - 22)**2) / (2 * 9**2))        # faint +
          - 0.03 * np.exp(-(mx**2 + (my + 22)**2) / (2 * 9**2))        # faint −
          + 0.002 * rng.standard_normal((m, m)))
lim = float(np.abs(signed).max())

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
panels = [("log  ✗ loses negatives", "log"),
          ("asinh  ✗ not symmetric", "asinh"),
          ("symlog  ✓", "symlog"),
          ("symmetric_log  ✓ smoother", "symmetric_log")]
for ax, (title, stretch) in zip(axes, panels):
    norm_s = sph.make_norm(stretch=stretch, vmin=-lim, vmax=lim, data=signed, clip="manual")
    ax.imshow(signed, origin="lower", cmap="sph.diff_blueorange", norm=norm_s)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("A signed (bipolar) field under four stretches — diverging diff_* colormap",
             fontsize=12)
fig.tight_layout()

# %% [markdown]
# `log` collapses the entire negative lobe to one flat color; `asinh` keeps both
# lobes but, mapping the data as if it ran 0→1, gives the negatives short shrift and
# washes out the faint secondary blobs. The diverging colormap makes both failures
# visible at a glance: zero should sit at the **pale neutral midpoint**, yet `log` and
# `asinh` drag the zero-level background deep into the orange half. Both symmetric
# stretches put it back where it belongs and recover the full bipolar structure,
# including the faint outer features.
#
# So why prefer `symmetric_log` over plain `symlog`? `SymLogNorm` is linear within
# `±linthresh` and logarithmic outside, but its **slope jumps** at that boundary —
# a kink that can show up as a faint contour/banding artifact right where data
# crosses it. `pysymlog`'s `symmetric_log` smooths the transition so the slope is
# continuous. The transfer function (value → colormap position) and its slope make
# the difference plain:

# %%
nrm_symlog = sph.make_norm(stretch="symlog", vmin=-lim, vmax=lim, clip="manual")
nrm_symmetric = sph.make_norm(stretch="symmetric_log", vmin=-lim, vmax=lim, clip="manual")
lt = nrm_symlog.linthresh                       # auto-picked ~1% of the range
v = np.linspace(-0.08, 0.08, 4001)              # zoom into the transition region

# Line colors from the uranometria cycle palette — legible on light and dark pages.
urano = sph.CYCLE_PALETTES["uranometria"]["colors"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
for nrm, lbl, c in [(nrm_symlog, "symlog (mpl)", urano[0]),
                    (nrm_symmetric, "symmetric_log (pysymlog)", urano[2])]:
    y = nrm(v)
    axL.plot(v, y, lw=1.6, label=lbl, color=c)
    axR.plot(v[1:], np.diff(y) / np.diff(v), lw=1.5, label=lbl, color=c)
for ax in (axL, axR):
    for s in (-lt, lt):
        ax.axvline(s, ls=":", c="0.6", lw=1)
    ax.legend(fontsize=9)
axL.set(title="Transfer function (zoom near zero)", xlabel="data value",
        ylabel="colormap position")
axR.set(title="Slope: symlog jumps at ±linthresh; symmetric_log is smooth",
        xlabel="data value", ylabel="slope")
fig.tight_layout()

# %% [markdown]
# The dotted lines mark ±`linthresh` (chosen automatically at ~1% of the data range;
# tune it via the `a=` keyword). At those points `symlog`'s slope steps
# discontinuously — the right panel shows the flat-topped box — whereas
# `symmetric_log` rises and falls smoothly. For most figures the visual difference is
# subtle, but for quantitative or polarization work the smooth version avoids an
# artifact exactly where the faint signed structure lives.

# %% [markdown]
# ### The linear zone at image scale
#
# The transfer curves are the *why*; here is the *what*. `SymLogNorm`'s **linear zone** —
# everything within ±`linthresh` — collapses to a single flat band of color, so structure
# that lives *below* the threshold renders as a featureless sea even though it is
# mathematically still there. Data with real **sub-threshold texture** — a faint diffuse
# component beneath bright compact sources — is exactly where that bites. Here is a
# synthetic **magnetogram**: a fine bipolar "quiet-network" background (\|B\| well under the
# threshold) beneath scattered small bipoles and a few strong active regions, spanning
# ~3 decades of amplitude on each side of zero:

# %%
def grf(n, slope, seed):
    """A power-law (fractal) Gaussian random field, normalized to unit variance."""
    r = np.random.default_rng(seed)
    kx, ky = np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None]
    k = np.hypot(kx, ky)
    k[0, 0] = 1.0                                        # avoid 0/0 at the DC term
    spec = k ** (slope / 2) * (r.normal(size=(n, n)) + 1j * r.normal(size=(n, n)))
    f = np.fft.ifft2(spec).real
    return (f - f.mean()) / f.std()


def make_magnetogram(n=500, seed=3):
    """A synthetic *signed* field: a faint sub-threshold GRF 'network' beneath
    log-amplitude bipoles (± Gaussian pairs) and a few strong 'active regions'."""
    r = np.random.default_rng(seed)
    img = 0.15 * grf(n, -1.5, seed + 1)                  # quiet network, |B| < ~0.5
    yy, xx = np.mgrid[0:n, 0:n]

    def bipole(x0, y0, peak, size, ang):                 # a +/- Gaussian pair
        dx, dy = 1.1 * size * np.cos(ang), 1.1 * size * np.sin(ang)
        for sign, ox, oy in ((+1, dx, dy), (-1, -dx, -dy)):
            img[:] += sign * peak * np.exp(
                -(((xx - x0 - ox) ** 2 + (yy - y0 - oy) ** 2) / (2 * size ** 2)))

    for _ in range(120):                                 # small/medium bipoles, peaks 0.3-8
        bipole(r.uniform(0, n), r.uniform(0, n),
               10 ** r.uniform(np.log10(0.3), np.log10(8)),
               r.uniform(1.5, 3.5), r.uniform(0, np.pi))
    for _ in range(8):                                   # strong active regions, peaks 30-100
        bipole(r.uniform(0.1 * n, 0.9 * n), r.uniform(0.1 * n, 0.9 * n),
               10 ** r.uniform(np.log10(30), np.log10(100)),
               r.uniform(5, 9), r.uniform(0, np.pi))
    return img


mag = make_magnetogram()

# %%
# The one field under three settings, in a signed diverging map — zero at the pale
# midpoint — the natural pairing for signed data (the diff_* colormap from §2).
specs = [("mpl symlog  (linthresh=1)", "symlog", 1.0),
         ("mpl symlog  (linthresh=0.02)", "symlog", 0.02),
         ("pysymlog  (shift=0.02)", "symmetric_log", 0.02)]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
for ax, (title, stretch, a) in zip(axes, specs):
    norm_m = sph.make_norm(stretch=stretch, vmin=-100, vmax=100, a=a)
    im = ax.imshow(mag, origin="lower", cmap="sph.diff_blueorange", norm=norm_m)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    # sph.add_colorbar, not fig.colorbar: it sizes to the image (not the axes bbox) and
    # keeps the minor tick marks — which is exactly what shows the ticks bunch at the seam.
    sph.add_colorbar(im, ax=ax, mode="divider")
fig.suptitle("The same signed field under three symmetric-log settings", fontsize=12)
fig.tight_layout()

# %% [markdown]
# **Left** — the default `symlog` (`linthresh=1`): the quiet network vanishes into a flat
# pale sea. Everything under the threshold is one neutral color; only the strong bipoles
# read. **Middle** — lower `linthresh` to `0.02` and `symlog` *does* recover the network,
# but it drags the linear-to-log kink down to ±0.02 and re-flattens everything below that
# new floor (watch the colorbar's ticks bunch at the seam). **Right** — `symmetric_log` at
# the same `shift=0.02` recovers the identical network with a *smooth* knee: no slope
# discontinuity, and a clean log-spaced tick ladder straight through zero.
#
# The honest reading: at *matched* settings the two families nearly coincide away from the
# transition. The practical win is that `pysymlog`'s soft knee is yours to place anywhere —
# push it to the data floor and the sub-threshold structure appears, with none of the kink
# that lowering `symlog`'s `linthresh` forces into the data.

# %% [markdown]
# The same kink bites in **one dimension**, too. Here is a synthetic radio field whose noise
# is genuinely *symmetric-log distributed* (linear near zero, exponential wings, real
# negatives from sidelobes), with a source well above it — binned once evenly in `symlog`
# space and once with `pysymlog`'s `symlogbin_histogram`:

# %%
import pysymlog  # noqa: E402  (optional dep; this figure is the pysymlog showcase)

# A synthetic radio field: symmetric-log-distributed noise (linear near zero, exponential
# wings, real negatives from sidelobes) plus a smooth source well above it.
rng = np.random.default_rng(7)
ny = nx = 320
sy, sx = np.mgrid[-1:1:ny*1j, -1:1:nx*1j]
sky_rms = 0.9e-3
sky_noise = np.clip(sky_rms * np.sinh(rng.normal(0.0, 1.25, (ny, nx))), -30 * sky_rms, 30 * sky_rms)
sky = (sky_noise
       + 0.05 * np.exp(-((sx - 0.05)**2 + (sy + 0.03)**2) / (2 * 0.24**2))
       + 0.014 * np.exp(-((sx + 0.42)**2 + (sy - 0.33)**2) / (2 * 0.09**2)))
sky_vlim = float(sky.max())
sky_lt = 2.5e-3                       # linear threshold ~ a few × the noise rms
sn_mpl = sph.make_norm(stretch="symlog", vmin=-sky_vlim, vmax=sky_vlim, clip="manual", a=sky_lt / sky_vlim)
lt_eff = sn_mpl.linthresh

NB = 24
mpl_edges = np.asarray(sn_mpl.inverse(np.linspace(0, 1, NB + 1)))   # bins even in symlog space
counts_mpl, _ = np.histogram(sky.ravel(), bins=mpl_edges)
counts_psl, psl_edges = pysymlog.symlogbin_histogram(sky.ravel(), NB, limits=[-sky_vlim, sky_vlim])

ink = theme_ink()   # bin outlines: dark on light pages, light on dark — legible in both
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(mpl_edges[:-1], counts_mpl, width=np.diff(mpl_edges), align="edge",
          facecolor=urano[0], edgecolor=ink, linewidth=0.6)
ax[0].set_title("flux distribution — symlog bins", fontsize=10)
ax[1].bar(psl_edges[:-1], counts_psl, width=np.diff(psl_edges), align="edge",
          facecolor=urano[2], edgecolor=ink, linewidth=0.6)
ax[1].set_title("flux distribution — pysymlog bins", fontsize=10)
for a in ax:
    a.set_xscale("symlog", linthresh=lt_eff)
    a.set_xlim(-sky_vlim, sky_vlim)
    a.set_xlabel("flux (Jy/beam)")
    for s in (-lt_eff, lt_eff):
        a.axvline(s, ls=":", c="0.55", lw=0.8)
fig.suptitle("The same kink in 1-D: a symmetric-log flux distribution, binned two ways", fontsize=12)
fig.tight_layout()

# %% [markdown]
# The dotted lines mark ±`linthresh`. The **symlog-binned** histogram (left) throws a
# spurious spike where the bin widths jump from linear to logarithmic; `pysymlog`'s
# continuous binning (right) removes it, so the true symmetric-log shape reads cleanly. For
# a *picture* the stretch choice is often cosmetic — but for any *quantitative* read of
# signed data near zero (fitting a noise model, measuring a flux distribution), the
# continuous version is the honest one.

# %% [markdown]
# > **Note:** `symmetric_log` needs the optional `pysymlog` package
# > (`pip install skyplothelper[pysymlog]`); without it, `symlog` is the built-in
# > fallback. The dedicated `pysymlog` documentation covers the math in depth.

# %% [markdown]
# ## 6. Data cubes and channel maps
#
# So far every image was a single 2-D frame. Spectral-line observations add a third
# axis: a **data cube** of sky images, one per velocity (or frequency) channel. The
# standard static view is a **channel map** — a grid of slices sampled across the
# line, every panel on the *same* brightness scale so the eye can follow the emission
# as it moves across the field with velocity.
#
# `channel_map()` builds that grid from a cube (a path or an array) in one call: it
# squeezes the data, picks a shared normalization, lays out the panels, reads the
# spectral WCS for per-channel velocity labels, thins the ticks, and adds a single
# shared colorbar. The bundled example is a VLA H I (21 cm) cube of the dwarf galaxy
# **DDO 70 (Sextans B)** — 43 velocity channels spanning the line:

# %%
CUBE = "../../examples/data/ddo70_hi_subcube.fits"
res = sph.channel_map(CUBE, channels=9, ncols=3, cmap="sph.dusk",
                      suptitle="DDO 70 (Sextans B) — VLA H I channel maps")

# %% [markdown]
# Read the grid like a flip-book: the emission appears at the high-velocity end,
# sweeps across the galaxy, and fades away — the classic signature of a rotating gas
# disk, each channel picking out the gas at one line-of-sight velocity. Every panel
# shares one normalization (so brightness is directly comparable and the single
# colorbar is honest for all nine), and by default only one panel carries tick labels
# to keep the grid clean. The call returns a `ChannelMapResult` — `.fig`, `.axes`,
# `.images`, `.colorbar`, `.velocities`, `.panel(ch)` — so every piece stays editable.
#
# ### Adjusting it
#
# The knobs are all one keyword away. A **publication-style** grid — panels flush
# (`pad=0`), relative **offset coordinates** (ΔRA/ΔDec from the center), panel
# backgrounds bled to the colormap's dark end, and a **beam** + **scale bar**:

# %%
# The bundled sub-cube carries no beam header; inject the real ~15" VLA H I beam so
# the beam marker has something to draw. (Everything else reads from the cube's WCS.)
ddo_hdr = fits.getheader(CUBE)
ddo_hdr["BMAJ"], ddo_hdr["BMIN"], ddo_hdr["BPA"] = 15 / 3600, 12 / 3600, 30.0

_ = sph.channel_map(CUBE, header=ddo_hdr, channels=9, ncols=3, cmap="sph.dusk",
                    pad=0.0, coords="offset", panel_facecolor="cmap_min",
                    beam=True, scalebar=120, scalebar_label="2′",
                    scalebar_kwargs=dict(color="white"),   # reads better than black here
                    suptitle="paper style — flush panels, offset coords, beam + scale bar")

# %% [markdown]
# Other knobs, each a keyword: **tick density** (`ticks='minimal'` default / `'complete'`
# / `'plain'`); **channel selection & preprocessing** (`channels=` a count or explicit
# list, `every_N=`, `average=`, `smooth='hanning'`, `order=`, `start_panel=`); and
# **labels** (`label='channel'` for numbers, `'auto'` for velocity/frequency, or a
# callable). Two patterns go further, using the editable result.
#
# **Reference contours on every panel** — the classic "line vs. total" figure: overlay
# the velocity-integrated (moment-0) outline on each channel to see how each slice sits
# within the whole. `res.axes` is just the panel grid, so loop over it:

# %%
res = sph.channel_map(CUBE, channels=9, ncols=3, cmap="sph.dusk", pad=0.02,
                      suptitle="each channel against the integrated H I outline")
moment0 = np.nansum(np.squeeze(fits.getdata(CUBE)).astype(float), axis=0)
levels = np.nanpercentile(moment0[moment0 > 0], [80, 95])
for ax in res.axes.flat:
    if ax.get_visible():
        ax.contour(moment0, levels=levels, colors="cyan", linewidths=0.6)

# %% [markdown]
# **A moment-0 summary panel** — collapse the velocity axis into one integrated map and
# drop it in with `moment0=True`, in its own colormap (`moment0_cmap=`) to set it apart.
# Give it its own colorbar with `add_colorbar` — a good place to meet its new
# **`location=`** argument (a one-liner for a left/top/bottom bar; it moves the ticks and
# label to the outer side for you). On a `channel_map` panel pass `mode='inset'` — the
# default `'divider'` fights the shared grid layout:

# %%
res = sph.channel_map(CUBE, channels=8, ncols=3, cmap="sph.dusk",
                      moment0=True, moment0_cmap="sph.deepsky",
                      suptitle="channels + a moment-0 summary panel")
# The moment-0 panel sits top-left; give it a dedicated bar on its outer (left) side.
# aspect=12 (vs the default 25) makes it about twice as thick so it reads at panel size.
cb0 = sph.add_colorbar(res.moment0_image, ax=res.moment0_image.axes, mode="inset",
                       location="left", aspect=12, pad=0.12, label=res.moment0_units)
cb0.ax.tick_params(labelsize=6)

# %% [markdown]
# **Editing the result.** Everything the call draws comes back on the `ChannelMapResult`
# as a live artist, so you can keep tuning after the fact. Here, on a compact 3×2 grid, we
# recolor one velocity label, title a single panel via `.panel()`, drop a marker onto
# another, and relabel the shared colorbar — all *after* the one-line construction:

# %%
res = sph.channel_map(CUBE, channels=6, ncols=3, cmap="sph.dusk",
                      suptitle="editing the result after the fact")
res.labels[0].set_color("gold")                              # recolor one velocity label
res.panel(int(res.channels[3])).set_title(                   # title a single panel
    "near the line peak", fontsize=8, color="cyan")
res.axes.flat[4].plot(112, 112, marker="+", ms=15, mew=1.6, color="cyan")   # mark a spot
res.colorbar.set_label("H I brightness (Jy/beam)")           # relabel the shared bar

# %% [markdown]
# For reference, the full `ChannelMapResult` — every piece is yours to restyle or query:
#
# | Attribute | What it holds |
# |---|---|
# | `.fig` | the matplotlib `Figure` |
# | `.axes` | 2-D array of the panel axes (WCSAxes) |
# | `.images` | the `AxesImage` for each drawn channel |
# | `.colorbar` | the shared `Colorbar` (`None` if `colorbar=False`) |
# | `.norm` | the one `Normalize` applied to every panel |
# | `.channels` | the (processed) channel indices drawn |
# | `.velocities` | per-channel `(value, unit)` read from the spectral WCS |
# | `.labels` | the per-channel corner `Text` artists |
# | `.beam` | the beam artist (`None` unless `beam=True`) |
# | `.scalebar` | the scale-bar artist (`None` unless set) |
# | `.moment0_image` | the moment-0 `AxesImage` (`None` unless `moment0=True`) |
# | `.moment0_units` | the moment-0 units string |
# | `.panel(ch)` | method → the axes for channel `ch` |

# %% [markdown]
# A single channel is just a 2-D image, so everything from §1–§4 (stretches, colorbars,
# contours, quicklook) applies to `res.panel(ch)` or a raw `cube[ch]` unchanged. And a
# **4-D** cube (a degenerate Stokes axis wrapping the spectral one) is squeezed for you —
# the same `np.squeeze` / `force_hdr_to_2D()` trick from §1.
#
# > **Tip:** the animated counterpart of a channel map — stepping through the cube as
# > a movie — is a one-liner in the [Animations](animations.ipynb) tutorial, which
# > plays this very cube.

# %% [markdown]
# ### The cube object: `DataCube`
#
# `channel_map` above took a *path*, but under the hood it loads the file into a
# `sph.DataCube` — a light *(data + WCS)* holder that squeezes the cube, splits the
# celestial and spectral WCS, and reads the per-channel velocities. It's public, so you can
# hold that object, inspect it, transform it, and hand it back to `channel_map` (your call
# above already works either way — `sph.channel_map(cube, …)` is identical). The transforms
# each return a *new* cube, so they chain — handy for a quick, coarse pass over a big cube
# before the full render:

# %%
cube = sph.DataCube.from_fits(CUBE)      # also accepts an ndarray, HDU/HDUList, or SpectralCube
print("channel 21 sits at", cube.spectral_label(21))   # velocity label, unit read from the WCS
print("one plane is", cube.channel(21).shape, "pixels")

small = cube.spatial_downsample(2).spectral_bin(2)      # half the pixels, channels binned in pairs
print(cube.shape, "->", small.shape)                    # cube.smooth(5) hanning-smooths spectrally
cube                                                    # repr: channels, size, spectral-axis kind

# %% [markdown]
# ### Moment maps: collapsing the cube
#
# The other thing you do with a cube is **collapse the spectral axis into a moment map**.
# `cube.moment(0/1/2)` (or the named `moment0`/`moment1`/`moment2`) returns a `MomentMap`:
# the integrated intensity (∫ I d*v*), the intensity-weighted **velocity field**, and the
# velocity **dispersion**.
#
# **The one thing you almost certainly want to do:** moments 1 and 2 divide by ∫ I, so over noise that
# denominator collapses and the map turns to garbage — you can drop the noise with
# `threshold=`, a few × the cube RMS is a great choice. (It's a plain scalar cut; for real per-channel
# masking, reach for [`spectral_cube`](https://spectral-cube.readthedocs.io).) A quick RMS
# off a couple of line-free edge channels is enough to set it:

# %%
cube_rms = float(np.nanstd(np.concatenate([cube.data[:3], cube.data[-3:]])))   # line-free edge channels

m0 = cube.moment0(unit="km/s")                           # ∫ I dv — integrated H I
m1 = cube.moment1(unit="km/s", threshold=3 * cube_rms)   # velocity field  (the cut matters)
m2 = cube.moment2(unit="km/s", threshold=3 * cube_rms)   # velocity dispersion

# .plot() is order-aware: a diverging blue↔orange scale centered on the systemic velocity
# for m1, sequential maps for m0/m2, each with its own unit-labeled colorbar. We give m0
# the same percentile-linear norm the channel-map summary panel uses (cleaner than the
# auto-stretch here), pin the dispersion at vmin=0 (a spread is ≥ 0) in a distinct cmap, and
# widen wspace so the colorbars don't crowd the next panel's tick labels.
m0_norm = sph.make_norm(stretch="linear", clip="percentile", plo=0.5, phi=99.8,
                        data=m0.data[np.isfinite(m0.data)])
fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
fig.get_layout_engine().set(wspace=0.12)
fig.suptitle("DDO 70 H I — the three moment maps", fontsize=12)
ink = theme_ink()   # black on light pages, white on dark: transparent canvas + legible ticks in both
panels = [(m0, "moment 0 — integrated H I", dict(norm=m0_norm)),
          (m1, "moment 1 — velocity field", {}),
          (m2, "moment 2 — velocity dispersion", dict(vmin=0, cmap="sph.penumbra"))]
for i, (mm, panel_title, extra) in enumerate(panels):
    ax = fig.add_subplot(1, 3, i + 1, projection=mm.wcs)
    mm.plot(ax=ax, title="", show_info=False, label="", obs_date="",
            facecolor="none", axcolor=ink, **extra)
    ax.set_title(panel_title, fontsize=11)
    ax.coords[0].axislabels.set_fontsize(9)
    ax.coords[1].axislabels.set_fontsize(9)

# %% [markdown]
# The velocity field is the memorable panel: the disk splits clean into blue (approaching)
# and orange (receding) halves — DDO 70's rotation, straight out of the cube.
#
# **Adjusting the plot.** `.plot()` forwards to the same quicklook machinery, so every knob
# from §1–§4 is available: `center=` pins the diverging scale on a systemic velocity (for a
# map already in *absolute* velocity), `cmap=` overrides the order default, `stretch=` /
# `vmin=` / `vmax=` reset the scaling, `beam=True` / `scalebar=<arcsec>` add the header beam
# and a scale bar, and `title=` sets the heading (default: the header `OBJECT`).

# %% [markdown]
# ### Bringing your own moment map
#
# Often the moment map is *already made* — a pipeline (CASA, `spectral_cube`, …) handed you
# a moment-1 FITS file, for example. `sph.MomentMap.from_fits` wraps it for the same order-aware `.plot()`;
# `order=` tells it how to color the map (a file written by `.to_fits` carries a `MOMORDER`
# keyword, so the order round-trips and you can omit it). Here we write one out — beam and
# `OBJECT` kept in the header — then read it straight back:

# %%
# Stand in for a pipeline product: write m1 out (with the real ~15" beam header from earlier),
# then load it fresh. In practice you'd just point from_fits at your existing file.
cube_beam = sph.DataCube.from_fits(CUBE, header=ddo_hdr)
mom1_path = os.path.join(tempfile.gettempdir(), "ddo70_mom1.fits")
cube_beam.moment1(unit="km/s", threshold=3 * cube_rms).to_fits(mom1_path, overwrite=True)

mm = sph.MomentMap.from_fits(mom1_path)   # order recovered from the MOMORDER keyword
res = mm.plot(facecolor="none", axcolor=theme_ink(),   # order-aware colors + title + beam, from the header;
              info_color=theme_ink())                  # transparent canvas + theme-following text (light/dark)

# %% [markdown]
# The name (`SEXTANSB`), the beam, and the moment order all survived the round-trip — they
# travel in the FITS header, so a map loaded from disk plots exactly like one computed here.
# `quicklook_plot` (§4) recognizes a `MomentMap` too, if you want the moment defaults
# without the extra `.plot()` conveniences.

# %% [markdown]
# ## 7. RGB composites
#
# Everything so far displayed a *single* image. But the most informative figures
# often combine several bands — different filters, or different telescopes entirely —
# into one color composite, so structures that appear in different wavelengths can be
# compared at a glance. skyplothelper pairs naturally with
# [`multicolorfits`](https://github.com/pjcigan/multicolorfits) (`mcf`) for this: it
# builds the composite array, and we display it on a sky frame just like any image.
#
# The recipe is three steps per band, then a combine:
#
# 1. **`to_grey_rgb`** — scale one band (an interval + stretch, exactly like §1)
#    into a grayscale RGB frame;
# 2. **`colorize_image`** — tint that frame a chosen color;
# 3. **`combine_multicolor`** — add the colored frames into the final RGB image.
#
# ### A worked example: SN 1987A in four HST filters
#
# Our centerpiece is a press-release-style optical composite of the supernova remnant
# **SN 1987A** from 2014 HST imaging (WFC3/UVIS; NASA, ESA, R. Kirshner & P. Challis):
# a broadband filter (F625W) plus three emission lines — Hα (F656N), [N II] (F658N),
# and [O III] (F502N). Four filters, one frame.
# The field is wide enough to show the famous **triple-ring system**: the bright inner
# equatorial ring plus the two fainter outer rings above and below it.

# %%
sn_625W, sn_wcs = _load2d("sn1987a_hst_F625W.fits")   # broadband continuum
sn_656N, _ = _load2d("sn1987a_hst_F656N.fits")        # Hα
sn_658N, _ = _load2d("sn1987a_hst_F658N.fits")        # [N II]
sn_502N, _ = _load2d("sn1987a_hst_F502N.fits")        # [O III]
sn_bands = [sn_625W, sn_656N, sn_658N, sn_502N]

# %% [markdown]
# **Step 1 — scale each band.** `to_grey_rgb` takes the same interval/stretch
# ideas as §1: here a `log` stretch with a percentile floor to suppress noise. The
# result is a grayscale image, shown for each band below:

# %%
sn_gray = [mcf.to_grey_rgb(d, rescalefn="log", scaletype="perc", min_max=[40, 99.9])
           for d in sn_bands]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, g, name in zip(axes, sn_gray,
                       ["F625W", "Hα (F656N)", "[N II] (F658N)", "[O III] (F502N)"]):
    ax.imshow(g, origin="lower")
    ax.set_title(name, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Step 1: each band scaled to grayscale", fontsize=12)
fig.tight_layout()

# %% [markdown]
# **Steps 2 & 3 — colorize and combine.** Each grayscale band is tinted and the
# colored frames are added together. Before the composite, a quick note on **naming
# the channels**: skyplothelper ships reference tables of standard band wavelengths and
# facility resolutions, handy for captioning — `FILTER_BANDS` (optical/IR photometric
# bands), `RADIO_BANDS` (radio frequency ranges), and `FACILITY_RESOLUTION`:

# %%
print("FILTER_BANDS['R'] :", sph.FILTER_BANDS["R"])   # F625W is a broad red filter ≈ R
print("RADIO_BANDS['Ku']  :", sph.RADIO_BANDS["Ku"], "GHz")  # e.g. the §1 3C 84 VLBA image
print("FACILITY_RESOLUTION['vlba_X'] :", sph.FACILITY_RESOLUTION["vlba_X"])

# %% [markdown]
# `add_bandlabels` then stacks those channel names on the figure in the matching
# colors — a legend keyed to the composite. `mcf.combo_swatch` supplies the complementary
# piece: the channel colors drawn as **overlapping circles**, pushed through the *same*
# colorize-and-combine pipeline as the image itself, so every overlap shows the honest
# mixed color you get where those bands coincide. Here is the finished composite, in a
# yellow/blue/purple/red ("YBPR") scheme, with the swatch tucked into the quiet corner:

# %%
SN_COLORS = ["#FBFCCF", "#729FCF", "#75507B", "#EF2929"]
SN_LABELS = ["F625W", "Hα (F656N)", "[N II] (F658N)", "[O III] (F502N)"]
sn_rgb = mcf.combine_multicolor(
    [mcf.colorize_image(g, c, colorintype="hex", gammacorr_color=2.2)
     for g, c in zip(sn_gray, SN_COLORS)], gamma=2.2)

fig = plt.figure(figsize=(6.4, 6.2))
ax = fig.add_subplot(111, projection=sn_wcs)
ax.imshow(sn_rgb, origin="lower")
sph.format_ticklabels(ax, style="publication")
sph.add_bandlabels(ax, SN_LABELS, SN_COLORS, fontsize=10)
# The color-blend key: same colors, same gamma, same additive ('rgb') combine as the image,
# so the overlaps are the real mixed colors. Its RGBA is transparent outside the circles.
swatch = mcf.combo_swatch(SN_COLORS, mode="rgb", gamma=2.2, background="black")
# On-figure size is the inset rect (combo_swatch's size= is only the raster resolution).
ax_sw = ax.inset_axes([0.83, 0.02, 0.14, 0.14])
ax_sw.imshow(swatch["rgba"], origin="lower")
ax_sw.axis("off")
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination (J2000)")
ax.set_title("SN 1987A — HST 2014 (four-band optical)", fontsize=12)
fig.tight_layout()

# %% [markdown]
# The broadband continuum, Hα, [N II], and [O III] each pick out a different part of
# the shocked rings and inner debris — structure no single filter shows on its own.
# The same three-step recipe builds any composite; below, two more on bundled data —
# the star-forming region **NGC 602** (Spitzer IR + HST optical R/B) and the **Crab
# Nebula** (four HST filters):

# %%
def _composite(bands, colors, recipes):
    """Scale each (data, recipe) to grayscale -> colorize -> combine."""
    gray = [mcf.to_grey_rgb(d, **r) for d, r in zip(bands, recipes)]
    return mcf.combine_multicolor(
        [mcf.colorize_image(g, c, colorintype="hex", gammacorr_color=2.2)
         for g, c in zip(gray, colors)], gamma=2.2)


# NGC 602 — the frames are rotated ~90° from north, so reproject to a north-up
# header first (via mcf) for clean WCS axes.
_ngc_hdu = fits.open("../../examples/data/ngc602_IR.fits")[0]
ngc_ir, ngc_h = np.squeeze(_ngc_hdu.data).astype(float), _ngc_hdu.header
ngc_R = np.squeeze(fits.open("../../examples/data/ngc602_R.fits")[0].data).astype(float)
ngc_B = np.squeeze(fits.open("../../examples/data/ngc602_B.fits")[0].data).astype(float)
ngc_uphdr = mcf.make_simple_header(ngc_h)
ngc_uphdr["CROTA2"] = 0
ngc_bands = [mcf.reproject_image(d, ngc_h, ngc_uphdr) for d in (ngc_ir, ngc_R, ngc_B)]
ngc_rgb = _composite(ngc_bands, ["#BE599E", "#DEA215", "#77C0F9"],
                     [dict(rescalefn="linear")] * 3)

crab_bands = [np.squeeze(fits.open(f"../../examples/data/crab_hst_{b}.fits")[0].data).astype(float)
              for b in ["F502N", "F547M", "F631N", "F673N"]]
crab_rgb = _composite(crab_bands, ["#a40000", "#581ac7", "#7decff", "#e9b96e"],
                      [dict(rescalefn="asinh", scaletype="perc", min_max=mm)
                       for mm in ([10, 99.8], [5, 99.94], [50, 99.8], [30, 98])])

fig = plt.figure(figsize=(11, 5.6))
axn = fig.add_subplot(1, 2, 1, projection=WCS(ngc_uphdr).celestial)
axn.imshow(ngc_rgb, origin="lower")
sph.format_ticklabels(axn, style="publication")
sph.add_bandlabels(axn, ["IR", "R", "B"], ["#BE599E", "#DEA215", "#77C0F9"], fontsize=10)
axn.set_title("NGC 602 — Spitzer IR + HST R/B", fontsize=11)
axc = fig.add_subplot(1, 2, 2, projection=WCS(fits.open(
    "../../examples/data/crab_hst_F673N.fits")[0].header).celestial)
axc.imshow(crab_rgb, origin="lower")
sph.format_ticklabels(axc, style="publication")
# The F547M label uses a lighter tint of its channel purple so it stays readable
# against the dark background; the other labels match their channels exactly.
sph.add_bandlabels(axc, ["F502N", "F547M", "F631N", "F673N"],
                   ["#a40000", "#8e6fe0", "#7decff", "#e9b96e"], fontsize=9)
axc.set_title("Crab Nebula — four HST filters", fontsize=11)
fig.tight_layout()

# %% [markdown]
# ### Placing an image on a sky field
#
# Each composite above sits on its own pixel grid. To show *where* a small deep image
# actually falls on the sky — and how your own data can share one frame with a catalog —
# drape it onto a wider `make_wcs_frame` field. Here NGC 602's composite is reprojected
# (`sph.reproject_rgb_map`, the RGB analog of the §8 reprojection) onto a 13′ ICRS frame
# over a **Gaia** star field, each star in its *true perceived color* (`sph.bp_rp_to_rgb`,
# straight from Gaia's `BP−RP`). It carries a compass and an angular scale bar, plus a
# `Ruler` reading the **physical** size in parsecs at the SMC's distance — a different
# measurement from the angular bar, so it takes a distinct color. The canvas is set black
# to read like a real field image; the frame *outside* still follows the page's theme.

# %%
# The Gaia field is cached under examples/data/query_cache/ so this renders offline and
# identically every run. The one-off cone search that built it:
#   gaia = sph.search_vizier("I/355/gaiadr3", ngc_center, radius=12,
#                            columns=["RA_ICRS", "DE_ICRS", "Gmag", "BP-RP"])
gaia = Table.read("../../examples/data/query_cache/gaia_ngc602.ecsv")

# Re-composite NGC 602 with a percentile floor so its faint edges fall to black at the
# drape border rather than ending in a hard box; reuse the reprojected bands from above.
field_rgb = _composite(ngc_bands, ["#BE599E", "#DEA215", "#77C0F9"],
                       [dict(rescalefn="linear", scaletype="perc", min_max=[30, 99.8])] * 3)
ngc_center = WCS(ngc_uphdr).celestial.pixel_to_world(field_rgb.shape[1] / 2,
                                                     field_rgb.shape[0] / 2)

# The 'structural' base gives inward ticks and is read at frame-build time, so scope it to
# make_wcs_frame with rc_context — the tick direction bakes in and later figures are safe.
with mpl.rc_context():
    sph.set_base_style("structural")
    axf = sph.make_wcs_frame(projection="TAN", center=ngc_center, fov_deg=0.22,
                             frame="ICRS", gridcolor="0.5", gridalpha=0.6)
axf.get_figure().set_size_inches(6.6, 6.6)          # a prominent showcase figure
axf.set_facecolor("black")                          # black canvas; the margin stays themed
axf.coords[0].set_ticks(number=8, color="0.5")      # frame furniture readable on black
axf.coords[1].set_ticks(number=10, color="0.5")
axf.coords.frame.set_color("0.5")
sph.format_ticklabels(axf, style="decimal", decimal_places=2, color=theme_ink())

# Drape the composite onto the frame, then lay a faint grid *over* it.
drape = sph.reproject_rgb_map(fits.PrimaryHDU(field_rgb, header=ngc_uphdr),
                              axf.wcs, shape_out=axf.wcs.pixel_shape[::-1])
axf.imshow(drape, origin="lower", zorder=4)
axf.coords.grid(color="0.5", alpha=0.6, zorder=6)   # match make_wcs_frame's gridalpha

# Gaia stars: sized by brightness, colored by perceived color from BP−RP. A non-finite
# color (missing photometry) is masked out rather than drawn a misleading gray.
bp_rp = np.array(gaia["BP-RP"], float)
gmag = np.array(gaia["Gmag"], float)
finite = np.isfinite(bp_rp) & np.isfinite(gmag)
star_rgb = sph.bp_rp_to_rgb(bp_rp[finite])
shown = np.isfinite(star_rgb).all(axis=1)
star_size = np.clip(32 * 10 ** (-0.4 * 0.7 * (gmag[finite] - 12)), 0.25, 32)
axf.scatter(np.asarray(gaia["RA_ICRS"])[finite][shown],
            np.asarray(gaia["DE_ICRS"])[finite][shown],
            transform=axf.get_transform("world"), s=star_size[shown],
            c=star_rgb[shown], edgecolors="none", zorder=3, alpha=0.95)

# Cream compass + angular scale bar; the NGC 602 label with a short leader.
sph.add_compass(axf, loc="lower left", color="#D9D5C5", stroke_color="k", stroke_lw=1.6)
sph.add_sizebar_asec(axf, axf.wcs.to_header(), 60, "1′", color="#D9D5C5", loc=4)
label_tip = ngc_center.directional_offset_by(0 * u.deg, 1.7 * u.arcmin)
label_txt = ngc_center.directional_offset_by(0 * u.deg, 3.3 * u.arcmin)
axf.annotate("NGC 602", xy=(label_tip.ra.deg, label_tip.dec.deg),
             xytext=(label_txt.ra.deg, label_txt.dec.deg),
             xycoords=axf.get_transform("world"), textcoords=axf.get_transform("world"),
             color="#D9D5C5", fontsize=11, ha="center", va="bottom", zorder=11,
             path_effects=[pe.withStroke(linewidth=2.2, foreground="k")],
             arrowprops=dict(arrowstyle="-", color="#D9D5C5", lw=0.8, shrinkA=2, shrinkB=2))

# A round 50 pc, projected to an angle at the SMC's 62 kpc distance and set along the
# field's lower-left↔upper-right diagonal. Ruler(convert=dict(distance=..., unit=...))
# would auto-label it; a single title reads cleaner here than per-tick physical labels.
half = ((50 / 2) / (62 * 1000) * u.rad).to(u.arcmin)
mid = ngc_center.directional_offset_by(225 * u.deg, 2.1 * u.arcmin)
ruler = sph.Ruler(mid.directional_offset_by(315 * u.deg, half),
                  mid.directional_offset_by(135 * u.deg, half), ax=axf, coord_type="auto",
                  color="#8FBFAF", stroke_color="k", n_ticks=4, label_fmt=lambda val, unit: "",
                  title="50 pc", title_color="#8FBFAF", title_side="left", zorder=10)
axf.set_title("NGC 602 placed in its Gaia star field (SMC)", fontsize=12, color=theme_ink())
_ = ruler.add_to(axf)

# %% [markdown]
# > **Note:** RGB compositing needs the optional `multicolorfits` package
# > (`pip install multicolorfits`). `mcf` covers the *color* science — interactive
# > color picking, gamma, inverse schemes — in depth; here we just need its three-step
# > combine. The annotation toolkit that captions these figures (beams, scale bars, and
# > the full band-label treatment) is the subject of the
# > [Annotations & Overlays](annotations.ipynb) tutorial.

# %% [markdown]
# ## 8. Reprojecting images to overlay them
#
# Images from different instruments almost never share a pixel grid — different
# resolution, orientation, and field of view. To **overlay** them — radio over optical,
# a multi-wavelength composite — you first **reproject** one onto the other's WCS so the
# pixels line up. The `reproject` package does the resampling. We use the bundled
# SN 1987A data, where the HST optical (a ~9″ WFC3 field) and the ALMA radio dust (a
# 600² grid) sit on genuinely different grids.
#
# > **Note:** this needs the optional `reproject` package (`pip install reproject`). To
# > display *one* image in a chosen projection or coordinate system (an all-sky frame, a
# > different sky system), see [A Tour of Projections](projections.ipynb); here we line
# > one image up with *another image's* grid.
#
# ### Onto another image's grid
#
# `reproject_interp` takes the source `(data, wcs)` plus the **target** WCS and shape and
# returns the source resampled onto that grid. We put the ALMA dust onto the HST grid,
# then overlay it two ways — as **contours** (the classic "radio contours over optical"
# convention) and as a **transparent colormap** (the reprojected image itself). Both
# panels zoom to the ALMA field, which covers only the central third of the wider HST one:

# %%
hst_opt, hst_wcs = _load2d("sn1987a_hst_F656N.fits")       # HST Hα — optical ring
alma_dust, alma_wcs = _load2d("sn1987a_alma_315GHz.fits")  # ALMA 315 GHz — radio dust

# Resample the ALMA image onto the HST pixel grid.
alma_on_hst, _ = reproject_interp((alma_dust, alma_wcs), hst_wcs, shape_out=hst_opt.shape)


def zoom_to_alma(ax):
    """Crop an HST-grid axes to the (smaller) ALMA footprint."""
    ny_a, nx_a = alma_dust.shape
    ra, dec = alma_wcs.pixel_to_world_values([-0.5, nx_a - 0.5, nx_a - 0.5, -0.5],
                                             [-0.5, -0.5, ny_a - 0.5, ny_a - 0.5])
    px, py = hst_wcs.world_to_pixel_values(ra, dec)
    ax.set_xlim(min(px), max(px))
    ax.set_ylim(min(py), max(py))


hst_norm = sph.make_norm(stretch="asinh", clip="percentile", plo=40, phi=99.7, data=hst_opt)
hlon, hlat = sph.header_coord_grids(hst_wcs)
alma_levels = np.nanpercentile(alma_on_hst, [88, 94, 97, 99, 99.7])

fig = plt.figure(figsize=(9.4, 5.0), constrained_layout=True)
# Left — the classic: cmap background image + contours of the reprojected other image.
axL = fig.add_subplot(1, 2, 1, projection=hst_wcs)
axL.imshow(hst_opt, origin="lower", cmap="sph.mesa_r", norm=hst_norm)
sph.add_contour_overlay(axL, hlon, hlat, alma_on_hst, levels=alma_levels,
                        colors="#1f4e79", linewidths=1.0)
zoom_to_alma(axL)
for c in axL.coords:
    c.set_ticks(number=3)
    c.set_ticklabel(size=7)
axL.set_xlabel("Right Ascension (J2000)")
axL.set_ylabel("Declination (J2000)")
axL.set_title("contours — ALMA dust over HST Hα", fontsize=11)
# Right — the alternate: the reprojected ALMA image itself, as a transparent colormap.
axR = fig.add_subplot(1, 2, 2, projection=hst_wcs)
axR.imshow(hst_opt, origin="lower", cmap="sph.mesa_r", norm=hst_norm)
# Mask the radio below its noise, and keep it well below opaque (alpha ~0.35) so the
# underlying Hα ring reads *through* the dust — the point is to see *both* distributions,
# which is what makes a transparent image overlay a viable alternative to contours.
radio = np.ma.masked_less(alma_on_hst, np.nanpercentile(alma_on_hst, 88))
axR.imshow(radio, origin="lower", cmap="hot", alpha=0.35,
           norm=sph.make_norm(stretch="asinh", clip="percentile", plo=70, phi=99.9, data=alma_on_hst))
zoom_to_alma(axR)
for c in axR.coords:
    c.set_ticks(number=3)
    c.set_ticklabel(size=7)
axR.coords[1].set_ticklabel_visible(False)
axR.coords[1].set_axislabel("")
axR.set_xlabel("Right Ascension (J2000)")
axR.set_title("image — reprojected ALMA (hot) over HST Hα", fontsize=11)

# %% [markdown]
# The cold dust sits inside the optical ring — the same multi-wavelength story as §7's
# composite, but here the two images started on different grids and one was resampled
# onto the other. **Contours** (left) are the print-friendly convention and keep the
# backdrop readable; an **image overlay** (right) shows the radio morphology in full.
#
# ### A common grid for several images
#
# Rather than favor one image's grid, you can let `reproject` compute the *optimal*
# common WCS that covers a whole set of images. `find_optimal_celestial_wcs` returns a
# WCS and shape spanning them all; reproject each onto it and combine. First, though, it
# helps to *see* what reprojection does — the two images cover very different amounts of
# sky, and only once they share a grid can they be combined:

# %%
common_wcs, common_shape = find_optimal_celestial_wcs(
    [(hst_opt, hst_wcs), (alma_dust, alma_wcs)], resolution=0.04 * u.arcsec)
opt_c, _ = reproject_interp((hst_opt, hst_wcs), common_wcs, shape_out=common_shape)
rad_c, _ = reproject_interp((alma_dust, alma_wcs), common_wcs, shape_out=common_shape)

# A three-panel explainer: HST fills the field; ALMA *looks* like a full image on its own
# native grid but actually spans only the central portion of the sky; reprojected onto the
# common grid, its true (smaller) footprint and pixel-alignment with HST become clear.
hnorm = sph.make_norm(stretch="asinh", clip="percentile", plo=40, phi=99.7, data=hst_opt)
anorm_n = sph.make_norm(stretch="asinh", clip="percentile", plo=60, phi=99.8, data=alma_dust)
anorm_c = sph.make_norm(stretch="asinh", clip="percentile", plo=60, phi=99.8, data=rad_c)

fig = plt.figure(figsize=(12, 4.4))
ax1 = fig.add_subplot(1, 3, 1, projection=common_wcs)
ax1.imshow(opt_c, origin="lower", cmap="sph.lagoon", norm=hnorm)
ax1.set_title("HST Hα — wide optical field", fontsize=10)
ax2 = fig.add_subplot(1, 3, 2, projection=alma_wcs)
ax2.imshow(alma_dust, origin="lower", cmap="sph.sunset", norm=anorm_n)
ax2.set_title("ALMA — native grid (fills its own smaller field)", fontsize=10)
ax3 = fig.add_subplot(1, 3, 3, projection=common_wcs)
# nan_to_num so the off-footprint region reads as the colormap's dark low end
# (matching the other panels) rather than a stark transparent/white box.
ax3.imshow(np.nan_to_num(rad_c), origin="lower", cmap="sph.sunset", norm=anorm_c)
ax3.set_title("ALMA — reprojected onto the common grid", fontsize=10)
# The same 3" scale bar on each panel — ~most of the ALMA field's width. On the native
# ALMA panel it spans most of the frame; on the wide common-grid panels the identical
# angular length looks short, making the sky-extent difference obvious at a glance.
sph.add_sizebar_asec(ax1, common_wcs.to_header(), 3.0, "3″", color="white", loc=4)
sph.add_sizebar_asec(ax2, alma_wcs.to_header(), 3.0, "3″", color="white", loc=4)
sph.add_sizebar_asec(ax3, common_wcs.to_header(), 3.0, "3″", color="white", loc=4)
for ax in (ax1, ax2, ax3):
    for c in ax.coords:
        c.set_ticklabel_visible(False)
        c.set_ticks_visible(False)
fig.suptitle("What reprojection does: two images onto one grid", fontsize=12)
fig.tight_layout()

# %% [markdown]
# Note the scale bar in the middle panel: on its own grid the ALMA dust *looks* like a full image,
# yet panels 1 and 3 — both on the common grid, so directly comparable — show it occupies
# only the central portion of the HST field. The identical 3″ scale bar makes it concrete:
# it spans most of the native ALMA frame but only a third of the wide common-grid frames.
# Now that the two share a grid pixel-for-pixel, a combined **two-color overlay** is
# straightforward:

# %%
def _gray(d, floor):
    return mcf.to_grey_rgb(np.nan_to_num(d), rescalefn="asinh", scaletype="perc",
                                min_max=[floor, 99.7])


two_color = mcf.combine_multicolor(
    [mcf.colorize_image(_gray(opt_c, 40), "#77C0F9", colorintype="hex", gammacorr_color=2.2),
     mcf.colorize_image(_gray(rad_c, 96), "#FF6A1A", colorintype="hex", gammacorr_color=2.2)],
    gamma=2.2)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection=common_wcs)
ax.imshow(two_color, origin="lower")
sph.format_ticklabels(ax, style="publication")
sph.add_bandlabels(ax, ["HST Hα (optical)", "ALMA 315 GHz (radio)"],
                   ["#77C0F9", "#FF6A1A"], fontsize=10)
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination (J2000)")
ax.set_title("Optical + radio on one common grid", fontsize=11)

# %% [markdown]
# `find_optimal_celestial_wcs` chose a grid that frames both fields (we pinned its
# resolution to the HST pixel scale; omit `resolution=` to default to the finest input).
# This is the building block for true mosaics — `reproject.mosaicking.reproject_and_coadd`
# stitches many tiles onto such a grid.
#
# > **Tip:** to keep a reprojected image for later use, write it out with its new header —
# > `fits.writeto("out.fits", opt_c, common_wcs.to_header())`. And for an image that is
# > *already* a 3-channel RGB FITS, `sph.reproject_rgb_map()` reprojects all three
# > channels in one call.

# %% [markdown]
# ## 9. Putting it together
#
# Each tool in this tutorial does one job; a real science figure uses them together.
# Here is the full stack on 3C 84 — the symmetric-log `make_norm` from §1–§2 (true
# Jy/beam values), a colorbar, and the σ-spaced contour ladder from §3 — composed into
# one publication-ready panel:

# %%
fig = plt.figure(figsize=(6.2, 5.6))
ax = fig.add_subplot(111, projection=wcs)

# §1/§2 — the symmetric-log norm, so the data stays in real units.
norm = mojave_norm()
im = ax.imshow(data, origin="lower", cmap="sph.deepsky", norm=norm)

# §3 — the 5σ × 2ⁿ contour ladder, in white to read over the deepsky image.
base = 5 * rms
levels = np.concatenate([[-base], base * 2.0 ** np.arange(0, np.log2(peak / base))])
sph.add_contour_overlay(ax, lon, lat, data, levels=levels, colors="white", linewidths=0.5)

# §2 — a colorbar that reads true Jy/beam. mode='inset' floats it beside the image
# without shrinking the (fixed-aspect) frame — the right choice for a single panel.
sph.add_colorbar(im, ax=ax, label="Jy / beam", mode="inset")

zoom_to_core(ax, half_mas=17)
tidy_radec(ax, nlon=3)
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination (J2000)")
ax.set_title("3C 84 — VLBA 15 GHz", fontsize=13, fontweight="bold")

# %% [markdown]
# That is the standard image-display workflow end to end — and exactly the recipe
# `quicklook_fits()` (§4) automates when you want it in a single call. Swap in your
# own FITS file at the top and the whole notebook re-renders on your data.
#
# ## 10. Where to go next
#
# | If you want to… | Go to |
# |---|---|
# | Add beams, scale bars, compasses, band labels, rulers | [Annotations & Overlays](annotations.ipynb) |
# | Style the frame's ticks, grids, labels, and themes | [Decorating Frames](decorating_frames.ipynb) |
# | Themes, palettes, and the bundled colormaps in depth | [Themes, Palettes & Fonts](styling.ipynb) |
# | Reproject an image into a different projection or all-sky frame | [A Tour of Projections](projections.ipynb) |
# | Animate a data cube, a globe, or a time series | [Animations](animations.ipynb) |
# | Bin an image or catalog onto a HEALPix map | [HEALPix Workflows](healpix_workflows.ipynb) |
# | The full API and header utilities | [Images & FITS guide](../guide/images.md) |
