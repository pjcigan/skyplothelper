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
# # HEALPix Workflows
#
# HEALPix is a powerful way to put **all-sky data on a grid** (or any data into well-defined sky area bins): it tiles the sphere
# into pixels of equal solid angle, so a density map reads fairly from pole to equator.
# skyplothelper covers the plotting side of that workflow end to end — binning a catalog
# into a map, rendering the map onto any frame the package can build, changing
# resolution and smoothing, running spatial queries, and staying tractable at very high
# resolution with a sparse representation.
#
# This tutorial walks the whole chain on a recurring **synthetic all-sky catalog** —
# built with a Galactic-plane concentration, a few compact clusters, and a uniform
# background — plus a real FITS image of the galaxy M51. Throughout it answers two
# questions these maps raise: **"how do I show my data this way?"** and **"how do I
# adjust it?"**
#
# ## Contents
#
# 1. [From catalog to map](#1.-From-catalog-to-map)
# 2. [Rendering maps](#2.-Rendering-maps)
# 3. [Resolution and smoothing](#3.-Resolution-and-smoothing)
# 4. [Spatial queries and pixel geometry](#4.-Spatial-queries-and-pixel-geometry)
# 5. [High-resolution and zoomed maps](#5.-High-resolution-and-zoomed-maps)
# 6. [Drawing on top of a map](#6.-Drawing-on-top-of-a-map)
# 7. [Putting it together](#7.-Putting-it-together)
# 8. [Where to go next](#8.-Where-to-go-next)

# %%
import time
import warnings

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D

import skyplothelper as sph

# Give every figure a clean, consistent skyplothelper look via the 'structural' base
# style (tidy spines, ticks, and fonts). set_style applies each layer (base / theme /
# palette) independently, so this base composes with whatever theme + palette is
# active — including the dark theme the documentation's dark-mode pass sets.
sph.set_style(base="structural")

# Data series (the catalog scatters below) are colored from an sph cycle palette;
# 'uranometria' reads well on both the light and dark documentation pages.
sph.set_palette("uranometria")
URAN = sph.CYCLE_PALETTES["uranometria"]["colors"]

# Slightly thicker coordinate graticules than the structural default — the thin
# 0.5-pt lines are hard to see over the density colormaps.
plt.rcParams["grid.linewidth"] = 1.0

# %% [markdown]
# Throughout we work with one recurring dataset: a synthetic all-sky catalog of
# ~11,000 sources, built to look like a real survey — a dense concentration along the
# **Galactic plane**, a few **compact clusters**, and a **uniform background**, each
# source carrying a (log-distributed) flux. It is deliberately synthetic so we can dial
# the structure to make each tool's behavior obvious; swap in your own
# `ra` / `dec` / `flux` arrays and everything below applies unchanged.

# %%
def make_synthetic_sky(seed=42):
    """A reproducible all-sky catalog: Galactic-plane disk + clusters + field."""
    rng = np.random.default_rng(seed)

    # (1) Galactic-plane concentration: uniform in Galactic longitude, peaked at
    #     b=0 (a Laplace distribution makes a realistic exponential disk), then
    #     converted to equatorial (ICRS) coordinates.
    n_disk = 6000
    gal = SkyCoord(l=rng.uniform(0, 360, n_disk) * u.deg,
                   b=np.clip(rng.laplace(0, 7, n_disk), -89, 89) * u.deg,
                   frame="galactic").icrs

    # (2) Three compact clusters (Gaussian blobs) at fixed equatorial positions.
    clusters = [(45.0, 10.0, 2.0, 400), (210.0, -25.0, 1.5, 350),
                (310.0, 55.0, 2.5, 500)]
    ra_c, dec_c = [], []
    for cra, cdec, sig, nc in clusters:
        ra_c.append(rng.normal(cra, sig / np.cos(np.radians(cdec)), nc))
        dec_c.append(rng.normal(cdec, sig, nc))
    ra_c = np.concatenate(ra_c) % 360
    dec_c = np.clip(np.concatenate(dec_c), -89, 89)

    # (3) Uniform background, sampled equal-area (arcsin in dec, not uniform-in-dec).
    n_field = 4000
    ra_f = rng.uniform(0, 360, n_field)
    dec_f = np.degrees(np.arcsin(rng.uniform(-1, 1, n_field)))

    ra = np.concatenate([gal.ra.deg, ra_c, ra_f])
    dec = np.concatenate([gal.dec.deg, dec_c, dec_f])
    kind = np.array(["disk"] * n_disk + ["cluster"] * len(ra_c)
                    + ["field"] * n_field)
    # Flux: log-distributed (lognormal), with the clusters a few times brighter.
    flux = rng.lognormal(0.0, 1.0, len(ra))
    flux[kind == "cluster"] *= 3.0
    return ra, dec, flux, kind


ra, dec, flux, kind = make_synthetic_sky()
print(f"{len(ra)} sources  "
      f"(disk: {(kind=='disk').sum()}, cluster: {(kind=='cluster').sum()}, "
      f"field: {(kind=='field').sum()})")
print(f"flux range: {flux.min():.3f} – {flux.max():.1f} "
      f"(median {np.median(flux):.2f})")

# %% [markdown]
# ## Why HEALPix?
#
# HEALPix (**H**ierarchical **E**qual **A**rea iso**L**atitude **Pix**elization) cuts
# the sphere into pixels that all subtend the **same solid angle**. That single
# property is what makes it a useful standard for all-sky data: "sources per pixel" is a
# fair density everywhere, with no area distortion to correct for. The figure below
# shows the tiling wrapping a globe (left) and our catalog binned into it (right).
#
# One subtle point to be aware of: the tiles are equal-**area**, not
# equal-**shape** on the sphere's surface. Toward the poles each diamond is sheared
# into a different outline even though its area never changes — a distinction we will
# see per-tile in §4.

# %%
# Build a smoothed Galactic density map for the right-hand globe.
_gal = SkyCoord(ra * u.deg, dec * u.deg).galactic
_gden = sph.healpix_smooth(
    sph.bin_data_as_healpix(_gal.l.deg, _gal.b.deg, np.ones_like(flux),
                            nside=32, statistic="count", blank_value=0)[0],
    sigma_deg=3.0)

fig = plt.figure(figsize=(13, 5.6))
# A faint coordinate graticule on both globes, for spatial reference (drawn under
# the tiles via a low zorder).
grid_kw = dict(projection="SIN", center=(60, 30), frame="galactic",
               lon_spacing=30, lat_spacing=30, gridcolor="0.6", gridalpha=0.6)
# Left: the bare tessellation on a globe, tiles tinted by latitude band.
ax1 = sph.make_wcs_frame((1, 2, 1), fig=fig, **grid_kw)
fig.canvas.draw()
allpix = np.arange(hp.nside2npix(8))
_, tile_lat = hp.pix2ang(8, allpix, lonlat=True)
sph.plot_healpix_sparse(allpix, tile_lat, nside=8, ax=ax1, cmap="sph.thicket",
                        show_boundaries=True, boundary_color="w", boundary_lw=0.5,
                        set_extent=False, zorder=0)
ax1.set_title("Equal-area tiles wrap the sphere (nside=8)", fontsize=10)
# Right: the catalog density on the same globe. 'sph.deepsky' is one of the
# package's bundled colormaps (sph.show_colormaps() previews the set; the FITS
# Images tutorial tours them) — we use it for every density map in this tutorial.
ax2 = sph.make_wcs_frame((1, 2, 2), fig=fig, **grid_kw)
fig.canvas.draw()
sph.plot_healpix_map(_gden, ax=ax2, cmap="sph.deepsky", vmin=0,
                     vmax=np.percentile(_gden[_gden > 0], 99), zorder=0)
ax2.set_title("A source catalog binned into HEALPix tiles at nside=32", fontsize=10)
# The tick labels sit over busy colormaps, so give them a fine contrasting stroke.
# The stroke tracks the page: a light stroke lifts the dark light-mode labels, a dark
# one lifts the light dark-mode labels.
r, g, b = to_rgb(plt.rcParams["figure.facecolor"])
label_stroke = "k" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "w"
for a in (ax1, ax2):
    sph.format_ticklabels(a, stroke_lw=1.6, stroke_color=label_stroke,
                          axis_labels=False)
fig.suptitle("HEALPix — an equal-area pixelization of the sphere", fontsize=13)
fig.subplots_adjust(top=0.88, wspace=0.18)
plt.show()

# %% [markdown]
# ## 1. From catalog to map
#
# A catalog is a list of positions; a **HEALPix map** is a fixed array of equal-area
# pixels covering the whole sphere, indexed `0 … 12·nside² − 1`. *Binning* a catalog
# is the act of counting (or averaging) sources into those pixels. Because every pixel
# subtends the **same solid angle**, each count stands for the same patch of sky at the
# pole as on the equator, so "counts per pixel" is a fair density everywhere — which is
# exactly why HEALPix is the natural tool for all-sky density work.
#
# Here is the raw catalog as points (colored by population), the starting material we
# will bin into a map:

# %%
ax = sph.make_wcs_frame(111, projection="AIT", center=180)
ax.figure.set_size_inches(11, 5.5)
for kind_name, color in zip(["field", "disk", "cluster"], URAN):
    sel = kind == kind_name
    ax.scatter(ra[sel], dec[sel], transform=ax.get_transform("world"),
               s=4, color=color, alpha=0.5, edgecolors="none", label=kind_name)
ax.legend(loc="lower right", markerscale=2, framealpha=0.9)
ax.set_title(f"The synthetic catalog as points ({len(ra):,} sources)", fontsize=11)
plt.show()

# %% [markdown]
# Plotted as points, the dense Galactic plane is already obvious, but overlapping
# markers hide the true density and there is no quantitative reading. *Binning* fixes
# both. The headline function is `bin_data_as_healpix(lons, lats, data, nside,
# statistic=)`. It returns a **4-tuple** — the dense map array first, then arrays used
# internally for rendering — so unpack the map as element `[0]`:

# %%
nside = 32

count_map = sph.bin_data_as_healpix(
    ra, dec, np.ones_like(flux), nside=nside,
    statistic="count", blank_value=0)[0]
mean_map = sph.bin_data_as_healpix(
    ra, dec, flux, nside=nside, statistic="mean")[0]

print(f"nside={nside}  ->  {hp.nside2npix(nside):,} pixels total")
print(f"occupied pixels: {(count_map > 0).sum():,}   "
      f"peak count: {int(np.nanmax(count_map))}")

# %% [markdown]
# `statistic=` selects the aggregation. `'count'` gives a **source-density** map (how
# many sources landed in each pixel); `'mean'` (the default) gives the **mean of a
# data column** per pixel — here the mean flux. `'sum'`, `'median'`, and `'std'` are
# also available. Empty pixels take `blank_value` (`NaN` by default, or `0` for a
# count map so the background reads as truly empty).
#
# We render these two maps with `plot_healpix_allsky`, which draws an all-sky map
# onto a frame you've already built (the full tour of the rendering helpers comes in
# §2). Binning into the **Galactic** frame puts the disk along the horizontal, so
# the plane is unmistakable:

# %%
# Bin in Galactic coordinates so the disk lands on b = 0 (horizontal).
gal = SkyCoord(ra * u.deg, dec * u.deg).galactic
gcount = sph.bin_data_as_healpix(gal.l.deg, gal.b.deg, np.ones_like(flux),
                                 nside=nside, statistic="count", blank_value=0)[0]
gmean = sph.bin_data_as_healpix(gal.l.deg, gal.b.deg, flux,
                                nside=nside, statistic="mean")[0]

fig = plt.figure(figsize=(14, 4.6))
for col, (m, label, cmap) in enumerate([
    (gcount, "statistic='count'  (source density)", "sph.deepsky"),
    (gmean, "statistic='mean'  (mean flux per pixel)", "sph.dusk"),
], start=1):
    ax = sph.make_wcs_frame((1, 2, col), projection="MOL", center=0,
                            frame="galactic", fig=fig)
    sph.plot_healpix_allsky(m, ax=ax, cmap=cmap, colorbar=True)
    ax.set_title(label, fontsize=11)
fig.suptitle("The synthetic catalog binned into a HEALPix map (Galactic MOL, "
             f"nside={nside})", fontsize=12)
fig.subplots_adjust(top=0.86, wspace=0.25)
plt.show()

# %% [markdown]
# The count map shows the Galactic plane lit up plus the three compact clusters; the
# mean-flux map is roughly uniform *except* in the clusters, whose sources we made
# intrinsically brighter — the two statistics answer genuinely different questions
# ("how many?" vs "how bright, on average?").
#
# Beyond the preset strings (`'mean'`, `'sum'`, `'median'`, `'min'`, `'max'`, `'std'`,
# `'count'`), `statistic=` also accepts a **callable** — it receives the 1-D array of
# values that fell in each pixel and returns a scalar, so you can map any reduction you
# like (a high percentile to chase bright outliers, a robust estimator, a trimmed
# mean):

# %%
p90_map = sph.bin_data_as_healpix(
    ra, dec, flux, nside=nside,
    statistic=lambda v: np.percentile(v, 90))[0]
occupied = np.isfinite(p90_map)
print(f"90th-percentile flux per pixel — range over occupied pixels: "
      f"{p90_map[occupied].min():.2f} to {p90_map[occupied].max():.2f}")

# %% [markdown]
# ### Choosing nside, and the sparse alternative
#
# `nside` sets the resolution: it must be a power of two, and each step doubles it
# (so the pixel count quadruples). Rather than guess, ask for a **target pixel scale**
# and let `auto_nside` pick the nearest power-of-two nside that meets it:

# %%
for res in (5.0, 1.0, 0.25):
    ns, actual = sph.auto_nside(resolution_deg=res)
    print(f"target {res:>4}°  ->  nside={ns:<6}  "
          f"(actual pixel scale {actual/3600:.3f}°, "
          f"{hp.nside2npix(ns):,} pixels)")

# %% [markdown]
# Notice how fast the pixel count grows: a 0.25° map already needs millions of pixels,
# almost all of which a modest catalog leaves empty. For that case, bin **sparsely** —
# `bin_data_sparse` returns only the *occupied* pixel indices and their values, never
# allocating the full `12·nside²` array, so it stays usable at arbitrarily high nside
# (we put this to work in §5):

# %%
ipix, vals = sph.bin_data_sparse(ra, dec, np.ones_like(flux),
                                 nside=256, statistic="count")
print(f"nside=256 dense map would be {hp.nside2npix(256):,} pixels;")
print(f"bin_data_sparse keeps only the {len(ipix):,} occupied ones "
      f"({100*len(ipix)/hp.nside2npix(256):.2f}% of the sphere).")

# %% [markdown]
# > **Note:** for the common "just count my sources" case there is an even shorter
# > path — `sources_to_healpix_bins(lons, lats, nside)` (a position list straight to a
# > binned map) and `sources_to_healpix_plot(...)` (straight to a plotted figure). They
# > are `bin_data_as_healpix(..., statistic='count')` with the data column filled in
# > for you.

# %% [markdown]
# ## 2. Rendering maps
#
# Once you have a map array, skyplothelper draws it onto **any** frame the package can
# build — the same array on an all-sky oval, a globe, or a zoomed field. Three entry
# points cover the range from "one line" to "full control":
#
# | Function | What it does | Use when |
# |---|---|---|
# | `healpix_allsky_figure(m, ...)` | Builds the figure **and** an all-sky frame, renders, adds a colorbar; returns a `HealpixResult` (`fig`, `ax`, `mappable`, `colorbar`) | You just want an all-sky map, fast (like `healpy.mollview`, but on a real WCSAxes) |
# | `plot_healpix_allsky(m, ax=...)` | Renders an all-sky map onto an **existing** WCSAxes | You built the all-sky frame yourself (e.g. a multi-panel figure) |
# | `plot_healpix_map(m, ax=...)` | Renders onto an existing frame in **any** projection / coordinate system, full-sky or a `lonlatlims=` sub-window | The map goes on a globe, a tangent field, or beside other panels |
#
# The one-call form is the quickest start. `HealpixResult` is a named tuple, so you
# can keep drawing on `result.ax` and relabel `result.colorbar` afterward:

# %%
result = sph.healpix_allsky_figure(count_map, projection="AIT", center=180,
                                   cmap="sph.deepsky")
result.colorbar.set_label("sources per pixel")
result.ax.set_title("healpix_allsky_figure — one call, full all-sky figure",
                    fontsize=11)
plt.show()

# %% [markdown]
# > **Tip — a `mollview`-style bottom colorbar.** Prefer the colorbar underneath the
# > map, the way `healpy.mollview` lays it out? Pass `colorbar=False` to skip the
# > default side bar, then place your own with `sph.add_colorbar` and
# > `location="bottom"` (or `orientation="horizontal"`). It sizes the bar to the map,
# > reserves its own space so nothing overlaps, and carries the same minor ticks and
# > optional stroke as every other sph colorbar — none of which a bare `plt.colorbar`
# > adds for you. Plain `plt.colorbar(res.mappable, ax=res.ax, orientation=...)` still
# > works if you would rather stay in stock matplotlib.

# %%
# A coarse map renders instantly — fine for a layout demo.
demo = sph.bin_data_as_healpix(ra, dec, np.ones_like(flux), nside=8,
                               statistic="count", blank_value=0)[0]
res = sph.healpix_allsky_figure(demo, projection="MOL", center=180, cmap="sph.deepsky",
                                colorbar=False)
sph.add_colorbar(res.mappable, ax=res.ax, location="bottom",
                 label="sources per pixel")
res.ax.set_title("Bottom colorbar, healpy.mollview style", fontsize=11)
plt.show()

# %% [markdown]
# ### One map, many frames
#
# To place the *same* array on a frame you built yourself, use `plot_healpix_map`. It
# is the engine behind the all-sky helpers, but it accepts any projection — so a single
# density map lands on an equal-area oval, a sinusoidal map, an orthographic **globe**,
# and a zoomed **tangent** field with the same one-line call (the projection trade-offs
# themselves are the subject of [A Tour of Projections](projections.ipynb)):

# %%
# A coarser map (nside=16) puts more sources in each pixel, and clipping the color
# scale to the 96th percentile lets the Galactic band stand out from the field.
gallery_map = sph.bin_data_as_healpix(ra, dec, np.ones_like(flux), nside=16,
                                      statistic="count", blank_value=0)[0]
vmax = np.percentile(gallery_map[gallery_map > 0], 96)

panels = [
    ("AIT", 180,        "AIT — equal-area oval"),
    ("MOL", 180,        "MOL — equal-area oval"),
    ("SFL", 180,        "SFL — sinusoidal"),
    ("robinson", 180,   "Robinson — compromise"),
    ("BON", 180,        "BON — Bonne pseudoconic"),
    ("CAR", 180,        "CAR — plate carrée"),
    ("SIN", (180, 0),   "SIN — orthographic globe"),
    ("SIN", (300, 20),  "SIN — tilted globe (plane on face)"),
    ("HPX", 180,        "HPX — HEALPix on HEALPix"),
]
fig = plt.figure(figsize=(14, 11))
for i, (proj, center, label) in enumerate(panels, start=1):
    ax = sph.make_wcs_frame((3, 3, i), projection=proj, center=center, fig=fig)
    fig.canvas.draw()
    sph.plot_healpix_map(gallery_map, ax=ax, cmap="sph.deepsky", vmin=0, vmax=vmax)
    ax.set_title(label, fontsize=10)
fig.suptitle("plot_healpix_map — the same density map (nside=16) on nine frames",
             fontsize=12)
fig.subplots_adjust(top=0.93, hspace=0.32, wspace=0.28)
plt.show()

# %% [markdown]
# > **Note — the pixelization is not the projection.** Two different things share the
# > HEALPix name: the equal-area *pixelization* this tutorial is about, and the *HPX
# > projection* in the last panel (whose diamond layout mirrors the pixel scheme). As
# > the other eight panels show, a HEALPix **map** can be drawn in **any** projection —
# > you never need HPX to plot one. On interrupted frames like HPX and the quad-cubes,
# > cells that would bridge a facet seam are blanked automatically (`mask_seams=True`)
# > so the map doesn't smear across the gaps. HPX and its butterfly cousin XPH are also
# > pole-locked; [A Tour of Projections](projections.ipynb) has that story.
#
# The Galactic plane reads as a horizontal band only in a Galactic frame — on these
# **equatorial** frames it cuts diagonally, exactly as it should. Which raises the one
# HEALPix-rendering gotcha worth internalizing.
#
# ### A map carries its own coordinate frame
#
# A HEALPix array is just numbers indexed by pixel; it does not record *which* sky
# frame those pixels were defined in. If you bin in Galactic coordinates and then draw
# on an equatorial frame without saying so, the map lands rotated. Tell the renderer
# the map's **own** frame and it rotates the data onto the displayed frame for you:

# %%
vmax_g = np.percentile(gcount[gcount > 0], 97)
fig = plt.figure(figsize=(14, 4.6))
# Left: the Galactic-binned map drawn on a Galactic frame — disk horizontal.
ax1 = sph.make_wcs_frame((1, 2, 1), projection="AIT", center=0,
                         frame="galactic", fig=fig)
sph.plot_healpix_allsky(gcount, ax=ax1, cmap="sph.deepsky", colorbar=False,
                        vmin=0, vmax=vmax_g)
ax1.set_title("Galactic map on a Galactic frame — plane horizontal", fontsize=10)
# Right: the SAME array on an equatorial frame. healpix_to_canvas resamples the map
# onto this axes' pixel grid, and frame='galactic' tells it the data is Galactic — so
# it rotates into equatorial coordinates (plane now diagonal — the true sky geometry).
ax2 = sph.make_wcs_frame((1, 2, 2), projection="AIT", center=180,
                         frame="icrs", fig=fig)
arr, extent = sph.healpix_to_canvas(gcount, ax2, frame="galactic")
ax2.imshow(arr, extent=extent, origin="lower", cmap="sph.deepsky", aspect="auto",
           interpolation="nearest", vmin=0, vmax=vmax_g)
ax2.set_title("Same array rotated onto an equatorial frame", fontsize=10)
fig.suptitle("healpix_to_canvas(..., frame='galactic') rotates a map between "
             "coordinate systems", fontsize=12)
fig.subplots_adjust(top=0.86, wspace=0.2)
plt.show()

# %% [markdown]
# `healpix_allsky_figure(gcount, frame='galactic')` does exactly this rotation for you
# in one call — pass the map's own `frame=` and it lands correctly on whatever frame
# the figure is built in.

# %% [markdown]
# ### Rendering backends
#
# `plot_healpix_allsky` can render a map four ways, via `backend=` and `sampling=`
# (`plot_healpix_sparse` shares the `patch` machinery). At high resolution they are
# visually indistinguishable — the choice is about **speed** and **return type**. The
# one place they differ is up close, as the figure below shows at a low nside: the
# three raster paths sample the map onto the canvas, while `patch` draws each tile as a
# true diamond.
#
# | `backend` / `sampling` | How it draws | Returns | Notes |
# |---|---|---|---|
# | `imshow` / `canvas` (default) | samples the map onto the frame's canvas pixels | `AxesImage` | fastest, lowest memory |
# | `pcolormesh` / `canvas` | same sampling, as a quad mesh | `QuadMesh` | per-cell event picking |
# | `pcolormesh` / `lonlat` | an older lon/lat-meshgrid path | `QuadMesh` | kept for compatibility and debugging |
# | `patch` | each tile as a true polygon | `PatchCollection` | exact tile edges; much slower as nside increases |
#
# Reach for `patch` (or `plot_healpix_sparse`) only when you actually want to *see* or
# annotate individual tile boundaries; otherwise the default `imshow` is the right
# call. All the raster paths sample the map at `xyres_pix=(2000, 1000)` by default —
# raise that for a large print figure so the sampling stays finer than the output.

# %%
# A smooth analytic field at a low nside makes both the agreement and the one real
# difference legible: the raster backends sample the map onto the canvas, while `patch`
# draws each tile as a true diamond. The field is signed (cosine lobes in −1..1), so a
# diverging colormap — here one of the bundled sph.diff_* maps — is the honest choice.
ns_b = 16
plon, plat = hp.pix2ang(ns_b, np.arange(hp.nside2npix(ns_b)), lonlat=True)
smooth_field = np.cos(np.radians(2 * plon)) * np.cos(np.radians(2 * plat))

backends = [("imshow", "canvas", "imshow + canvas (default)"),
            ("pcolormesh", "canvas", "pcolormesh + canvas"),
            ("pcolormesh", "lonlat", "pcolormesh + lonlat (older path)"),
            ("patch", "canvas", "patch (true tile polygons)")]

# Warm each backend once on a scratch figure so the timed pass below isn't charged
# for one-time import/compile costs (which would otherwise inflate the first panel).
_scratch = plt.figure()
for backend, sampling, _ in backends:
    _ax = sph.make_wcs_frame(111, projection="AIT", center=180, fig=_scratch)
    sph.plot_healpix_allsky(smooth_field, ax=_ax, backend=backend, sampling=sampling,
                            colorbar=False)
    _scratch.clf()
plt.close(_scratch)

fig = plt.figure(figsize=(13, 6.8))
for i, (backend, sampling, label) in enumerate(backends, start=1):
    ax = sph.make_wcs_frame((2, 2, i), projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    t0 = time.perf_counter()      # time just the render call
    sph.plot_healpix_allsky(smooth_field, ax=ax, cmap="sph.diff_bluebrown",
                            vmin=-1, vmax=1, backend=backend, sampling=sampling,
                            colorbar=False)
    ax.set_title(f"{label}\n{1e3 * (time.perf_counter() - t0):.0f} ms", fontsize=10)
fig.suptitle(f"plot_healpix_allsky backends — same field, cos(2λ)cos(2β), "
             f"nside={ns_b}", fontsize=12)
fig.subplots_adjust(top=0.90, hspace=0.38, wspace=0.18)
plt.show()

# %% [markdown]
# The render times above are modest at this resolution, but they scale very
# differently: the raster backends are near-constant (they sample a fixed canvas),
# while `patch` grows with the tile count, which quadruples with every step in nside.
# By nside=32 it is already several seconds, and much above nside≈64 it stops being
# practical — which is why `imshow` is the default. Treat the numbers in the titles as
# rough: they are a single timing on one machine, and they move with whatever else it
# happens to be doing.

# %% [markdown]
# ### RING vs NESTED ordering
#
# The coordinate frame is not the only metadata a HEALPix array cannot carry. There
# are two conventions for *numbering* the pixels: **RING** (rows of constant latitude
# — the default everywhere in skyplothelper and in healpy) and **NESTED** (a
# hierarchical scheme many released survey maps use). The array itself is just values
# in pixel order, so *you* have to say which convention it follows. The renderers and
# queries accept `nest=True` for NESTED input (`plot_healpix_allsky`,
# `plot_healpix_sparse`, the queries, `bin_data_sparse`, `image_to_healpix`); anything
# without a `nest` option expects RING — convert a NESTED map once with
# `healpy.reorder(m, n2r=True)` and move on.
#
# Get it wrong and the symptom is unmistakable — the map shatters into fine-grained,
# scrambled tiles:

# %%
# A finer field (nside=32) makes the scrambling dramatic. Start from the RING-ordered
# analytic field, then reorder it to NESTED.
rlon, rlat = hp.pix2ang(32, np.arange(hp.nside2npix(32)), lonlat=True)
ring_field = np.cos(np.radians(2 * rlon)) * np.cos(np.radians(2 * rlat))
nested_field = hp.reorder(ring_field, r2n=True)        # the same map, NESTED-ordered

fig = plt.figure(figsize=(14, 4.6))
ax1 = sph.make_wcs_frame((1, 2, 1), projection="MOL", center=180, fig=fig)
sph.plot_healpix_allsky(nested_field, ax=ax1, nest=True, colorbar=False,
                        cmap="sph.diff_bluebrown", vmin=-1, vmax=1)
ax1.set_title("NESTED map rendered with nest=True — correct", fontsize=10)
ax2 = sph.make_wcs_frame((1, 2, 2), projection="MOL", center=180, fig=fig)
sph.plot_healpix_allsky(nested_field, ax=ax2, colorbar=False,     # default nest=False
                        cmap="sph.diff_bluebrown", vmin=-1, vmax=1)
ax2.set_title("Same array assumed RING — scrambled", fontsize=10)
fig.suptitle("A map that renders as scrambled tiles is almost always an "
             "ordering mismatch — pass nest=True", fontsize=12)
fig.subplots_adjust(top=0.86, wspace=0.2)
plt.show()

# %% [markdown]
# ## 3. Resolution and smoothing
#
# A raw count map is **noisy**: with a finite catalog, neighboring pixels scatter just
# from counting statistics. The standard treatment is to **smooth** with a beam, then
# optionally move to a coarser resolution. skyplothelper gives you each step as a
# one-liner that operates directly on the map array.
#
# `healpix_smooth` convolves the map with a Gaussian on the sphere (specify the beam as
# `sigma_deg`, `sigma_arcmin`, `sigma_arcsec`, or `beam_fwhm_arcmin`):

# %%
smooth_map = sph.healpix_smooth(gcount, sigma_deg=3.0)

fig = plt.figure(figsize=(14, 4.6))
for col, (m, label) in enumerate([
    (gcount, f"raw counts (nside={nside}) — speckled by counting noise"),
    (smooth_map, "smoothed with σ=3°  (healpix_smooth)"),
], start=1):
    ax = sph.make_wcs_frame((1, 2, col), projection="MOL", center=0,
                            frame="galactic", fig=fig)
    sph.plot_healpix_allsky(m, ax=ax, cmap="sph.deepsky", colorbar=True)
    ax.set_title(label, fontsize=10)
fig.suptitle("healpix_smooth — a Gaussian beam turns a speckled count map into a "
             "continuous density field", fontsize=12)
fig.subplots_adjust(top=0.86, wspace=0.25)
plt.show()

# %% [markdown]
# ### Changing resolution
#
# `healpix_upgrade` and `healpix_downgrade` move a map between nside levels —
# downgrading averages blocks of pixels together (good for shrinking a smoothed map to
# its true information content), upgrading subdivides (handy for combining maps that
# live at different resolutions). Rendered with the **patches** backend, the tile
# structure is explicit, so the resolution change is obvious from the tile size itself:

# %%
m8 = sph.healpix_smooth(
    sph.bin_data_as_healpix(gal.l.deg, gal.b.deg, np.ones_like(flux),
                            nside=8, statistic="count", blank_value=0)[0],
    sigma_deg=10.0)
m_down = sph.healpix_downgrade(m8, nside_out=4)
m_up = sph.healpix_upgrade(m8, nside_out=16)

vlo, vhi = float(np.nanmin(m8)), float(np.nanmax(m8))
fig = plt.figure(figsize=(15, 5.2))
for col, (m, label) in enumerate([
    (m_down, f"downgraded → nside=4  ({hp.nside2npix(4)} pixels)"),
    (m8,     f"original nside=8  ({hp.nside2npix(8)} pixels)"),
    (m_up,   f"upgraded → nside=16  ({hp.nside2npix(16):,} pixels)"),
], start=1):
    ax = sph.make_wcs_frame((1, 3, col), projection="MOL", center=0,
                            frame="galactic", fig=fig)
    fig.canvas.draw()
    n = hp.npix2nside(len(m))
    sph.plot_healpix_sparse(np.arange(len(m)), m, nside=n, ax=ax, backend="patch",
                            cmap="sph.deepsky", vmin=vlo, vmax=vhi,
                            show_boundaries=True, boundary_color="0.3",
                            boundary_lw=0.3, set_extent=False)
    ax.set_title(label, fontsize=10)
fig.suptitle("healpix_downgrade / healpix_upgrade — the same field at three "
             "resolutions (patches backend; tile structure visible)", fontsize=12)
fig.subplots_adjust(top=0.88, wspace=0.12)
plt.show()

# %% [markdown]
# Upgrading does not *add* information — the nside=16 panel is the nside=8 map with
# each tile subdivided — but it lets you bring maps to a common resolution before
# arithmetic. `healpix_combine` does that arithmetic, merging two same-nside maps with
# `operation='add'`, `'subtract'`, `'multiply'`, or `'divide'` (returning the combined
# map and its nside):

# %%
combined, ns_out = sph.healpix_combine(gcount, smooth_map, operation="add")
print(f"combined two nside={nside} maps with operation='add'  ->  "
      f"nside={ns_out}, {len(combined):,} pixels")

# %% [markdown]
# > **Note:** the usual recipe for a noisy survey-count map is **bin fine → smooth →
# > downgrade**: bin at high resolution so you don't lose real structure, smooth to the
# > beam you trust, then downgrade to the resolution that smoothing actually supports.

# %% [markdown]
# ## 4. Spatial queries and pixel geometry
#
# HEALPix is also an **indexing scheme**: "which pixels fall inside this region?" is a
# fast lookup, which makes it the natural engine for footprint and masking logic — *is
# this source inside the survey area? inside the masked zone?* Two queries cover most
# needs:
#
# - `healpix_circle_query(lon, lat, radius_deg, nside)` — pixels inside a disk.
# - `healpix_polygon_query(vertices_deg, nside)` — pixels inside a polygon (it handles
#   antimeridian-straddling shapes).
#
# Both return arrays of pixel indices. Below, three disks (orange) and two boxes
# (blue, one straddling the 0°/360° seam) on an all-sky frame. The **left** panel
# shows the selected pixels at nside=64; the **right** repeats the query at a lower
# nside with *every* tile outlined — filled and empty alike — so you can see the
# underlying grid the query is selecting from.

# %%
# In-theme query colors pulled from the uranometria palette: orange disks, blue boxes.
QCMAP = ListedColormap([URAN[1], URAN[0]])     # value 1 -> orange, 2 -> blue
circles = [(120.0, 60.0, 12.0), (210.0, 0.0, 10.0), (40.0, -50.0, 15.0)]
boxes = [[(255, -45), (285, -45), (285, -15), (255, -15)],
         [(345, 15), (15, 15), (15, 45), (345, 45)]]


def build_query_map(ns):
    m = np.full(hp.nside2npix(ns), np.nan)
    for lon, lat, r in circles:
        m[sph.healpix_circle_query(lon, lat, r, ns)] = 1.0
    for verts in boxes:
        m[sph.healpix_polygon_query(verts, nside=ns)] = 2.0
    return m


fig = plt.figure(figsize=(15, 4.6))
# Left: high-resolution selection, fill only.
ax1 = sph.make_wcs_frame((1, 2, 1), projection="AIT", center=180, fig=fig)
fig.canvas.draw()
sph.plot_healpix_map(build_query_map(64), ax=ax1, cmap=QCMAP, vmin=1, vmax=2)
ax1.set_title("Selected pixels at nside=64", fontsize=10)
# Right: lower nside with every tile outlined (empty tiles show only their border).
ns_grid = 16
qmap16 = build_query_map(ns_grid)
ax2 = sph.make_wcs_frame((1, 2, 2), projection="AIT", center=180, fig=fig)
fig.canvas.draw()
sph.plot_healpix_sparse(np.arange(len(qmap16)), qmap16, nside=ns_grid, ax=ax2,
                        backend="patch", cmap=QCMAP, vmin=1, vmax=2,
                        show_boundaries=True, boundary_color="0.6", boundary_lw=0.15,
                        set_extent=False)
ax2.set_title(f"Same query at nside={ns_grid}, all tiles outlined", fontsize=10)
fig.suptitle("healpix_circle_query (orange disks) + healpix_polygon_query "
             "(blue boxes, one across the seam)", fontsize=12)
fig.subplots_adjust(top=0.86, wspace=0.2)
plt.show()

# %% [markdown]
# One adjustment worth knowing on both queries: by default a pixel is selected only
# when its **center** lies inside the region (`inclusive=False`). Pass
# `inclusive=True` to also catch every pixel that merely *overlaps* the boundary —
# the safer choice for a mask that must not miss edge sources:

# %%
strict = sph.healpix_circle_query(210.0, 0.0, 10.0, 64)
generous = sph.healpix_circle_query(210.0, 0.0, 10.0, 64, inclusive=True)
print(f"10° disk at nside=64:  centers-inside {len(strict)} pixels;  "
      f"any-overlap {len(generous)} pixels")

# %% [markdown]
# ### From a region to "which of my sources are in it?"
#
# Because the query returns pixel indices, region membership is a fast set test: bin
# the catalog's positions to the *same* nside and ask which sources land in a query's
# pixels. No per-source angular-distance loop:

# %%
nside_q = 64
region_pix = sph.healpix_circle_query(210.0, 0.0, 10.0, nside_q)   # the equatorial disk
src_pix = hp.ang2pix(nside_q, ra, dec, lonlat=True)                # each source's pixel
inside = np.isin(src_pix, region_pix)
print(f"{inside.sum()} of {len(ra)} catalog sources fall inside the 10°-radius "
      f"disk at (210°, 0°)")

# %% [markdown]
# ### Drawing individual tiles
#
# To outline or annotate *specific* pixels, two tools help. `healpix_pixel_corners`
# returns the corner coordinates of given pixels (for drawing them yourself), and the
# **patches** backend tags every rendered polygon with its pixel id on
# `collection.patch_pixel_index` — so you can find and restyle one tile after the fact.
#
# This is also where the equal-area-but-not-equal-shape caveat from the opener
# becomes visible tile by tile. The pixels are equal-**area**, but *not*
# equal-**shape**: a tile near the pole covers the same
# solid angle as one on the equator, yet it is sheared into a different outline. Below
# we render a small cluster of tiles, highlight one in each frame, and label pixel ids
# on the zoom — the highlighted tile's outline changes shape from frame to frame and
# from the equator poleward, even though its area never does:

# %%
nside_h = 8
hpix = sph.healpix_circle_query(180.0, 20.0, 18.0, nside_h)
hvals = np.random.default_rng(7).uniform(0, 1, len(hpix))
# Target tile: the one whose center is closest to (180°, 20°).
plon, plat = hp.pix2ang(nside_h, hpix, lonlat=True)
target = int(hpix[np.argmin(np.hypot((plon - 180 + 180) % 360 - 180, plat - 20))])

panels = [("AIT", 180,        False, "AIT — one tile among many"),
          ("SIN", (180, 0),   False, "SIN globe — limb-clipped tiles"),
          ("TAN", (180, 20),  True,  "TAN zoom — pixel ids labeled")]
fig = plt.figure(figsize=(15, 5))
for i, (proj, center, zoom, label) in enumerate(panels, start=1):
    ax = sph.make_wcs_frame((1, 3, i), projection=proj, center=center, fig=fig)
    fig.canvas.draw()
    pc = sph.plot_healpix_sparse(hpix, hvals, nside=nside_h, ax=ax,
                                 cmap="sph.lagoon",
                                 show_boundaries=True, boundary_color="0.4",
                                 boundary_lw=0.4, set_extent=zoom)
    # Trace the target tile's outline in red via patch_pixel_index.
    mask = pc.patch_pixel_index == target
    for path in np.asarray(pc.get_paths(), dtype=object)[mask]:
        v = path.vertices
        ax.plot(v[:, 0], v[:, 1], color="#e8000b", lw=2.5, zorder=5,
                solid_joinstyle="round")
    if zoom:                                  # label every tile on the close-up
        for path, pid in zip(pc.get_paths(), pc.patch_pixel_index):
            cx, cy = path.vertices.mean(axis=0)
            ax.text(cx, cy, str(int(pid)), ha="center", va="center", fontsize=8,
                    color=("#e8000b" if pid == target else "0.2"),
                    fontweight=("bold" if pid == target else "normal"), zorder=6)
    ax.set_title(label, fontsize=10)
fig.suptitle(f"plot_healpix_sparse — highlight one tile (pixel {target}, "
             f"nside={nside_h}) via pc.patch_pixel_index", fontsize=12)
fig.subplots_adjust(top=0.88, wspace=0.28)
plt.show()

# %% [markdown]
# > **See also:** the [Regions & Spherical Polygons](regions.ipynb) tutorial builds
# > arbitrary set-algebra footprints (`CompoundRegion`); combine one with a polygon
# > query to ask "which HEALPix pixels does my footprint cover?" — the same membership
# > pattern as above, scaled up to a real survey mask.

# %% [markdown]
# ## 5. High-resolution and zoomed maps
#
# Everything so far binned into a **dense** array of `12·nside²` pixels. That is fine
# at all-sky resolutions, but it does not scale: a 1′ map (nside≈4096) is 200 million
# pixels, and a 13″ map (nside≈16384) is 3 *billion* — far too large to allocate when
# your data only touches a tiny patch of sky.
#
# `bin_data_sparse` is the way out: it returns just the **occupied** pixels (indices +
# values), never allocating the full array, so it works at *any* nside. Paired with
# `plot_healpix_sparse` — which draws only those pixels, as true polygons — it gives a
# crisp zoomed map where a dense one is impossible. Take a deep field of ~37,000
# sources concentrated in a few degrees of sky and bin it at nside=256:

# %%
def deep_field(seed=11):
    """A dense ~6° field around (45°, 10°): a broad blob plus two compact clumps."""
    rng = np.random.default_rng(seed)
    cra, cdec = 45.0, 10.0
    ra = np.concatenate([rng.normal(cra, 2.0 / np.cos(np.radians(cdec)), 30000),
                         rng.normal(cra - 1.5, 0.4, 4000),
                         rng.normal(cra + 1.2, 0.5, 3000)])
    dec = np.concatenate([rng.normal(cdec, 2.0, 30000),
                          rng.normal(cdec + 1.0, 0.4, 4000),
                          rng.normal(cdec - 1.3, 0.5, 3000)])
    return ra, dec, (cra, cdec)


dra, ddec, (cra, cdec) = deep_field()
nside_zoom = 256
zi, zv = sph.bin_data_sparse(dra, ddec, np.ones_like(dra), nside=nside_zoom,
                             statistic="count")
print(f"nside={nside_zoom}: a dense map would be {hp.nside2npix(nside_zoom):,} "
      f"pixels; only {len(zi):,} are occupied by this field.")

fig = plt.figure(figsize=(13, 5.6))
# Left: the raw sources as tiny dots — the input to the binning.
ax1 = sph.make_wcs_frame((1, 2, 1), projection="TAN", center=(cra, cdec),
                         fov_deg=10, grid=False, fig=fig)
fig.canvas.draw()
ax1.scatter(dra, ddec, transform=ax1.get_transform("world"), s=1,
            color=URAN[1], alpha=0.3, edgecolor="none")
ax1.set_title(f"raw sources ({len(dra):,} points)", fontsize=11)
# Right: the same field binned sparsely — the density map.
ax2 = sph.make_wcs_frame((1, 2, 2), projection="TAN", center=(cra, cdec),
                         fov_deg=10, grid=False, fig=fig)
fig.canvas.draw()
pc = sph.plot_healpix_sparse(zi, zv, nside=nside_zoom, ax=ax2, cmap="sph.sunset",
                             show_boundaries=False, set_extent=True)
sph.add_colorbar(pc, ax=ax2, label="sources per pixel", shrink=0.85)
ax2.set_title(f"binned sparsely at nside={nside_zoom}", fontsize=11)
fig.suptitle("From raw counts to a density map — bin_data_sparse on a deep field",
             fontsize=12)
fig.subplots_adjust(top=0.88, wspace=0.2)
plt.show()

# %% [markdown]
# Sparse mode makes no assumption that the occupied pixels are **contiguous**, and it
# handles pixels (and connected groups of them) that **straddle the frame seam** just
# like any other. The left panel below scatters isolated pixels all over the sky; the
# right shows a connected cap centered on longitude 0°, which splits across both edges
# of the 180°-centered frame and renders without a smear:

# %%
ns_s = 32
rng = np.random.default_rng(3)
# Left: 60 isolated, non-adjacent pixels scattered across the whole sphere.
loose_pix = rng.choice(hp.nside2npix(ns_s), size=60, replace=False)
loose_val = rng.uniform(0, 1, len(loose_pix))
# Right: a connected cap on the antimeridian (centered at lon 0°), which the
# 180°-centered frame must split into its left and right edges.
seam_pix = hp.query_disc(ns_s, hp.ang2vec(0, 20, lonlat=True), np.radians(18))
seam_val = rng.uniform(0, 1, len(seam_pix))

fig = plt.figure(figsize=(15, 4.6))
for i, (pix, val, label) in enumerate([
    (loose_pix, loose_val, f"Non-contiguous — {len(loose_pix)} isolated pixels"),
    (seam_pix, seam_val, "A connected cap split across the 0°/360° seam"),
], start=1):
    ax = sph.make_wcs_frame((1, 2, i), projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    sph.plot_healpix_sparse(pix, val, nside=ns_s, ax=ax, cmap="sph.evening",
                            show_boundaries=True, boundary_color="0.4",
                            boundary_lw=0.3, set_extent=False)
    ax.set_title(label, fontsize=10)
fig.suptitle("Sparse pixels need not be contiguous, and groups cross the seam cleanly",
             fontsize=12)
fig.subplots_adjust(top=0.86, wspace=0.2)
plt.show()

# %% [markdown]
# ### Putting a FITS image on the HEALPix grid
#
# The same idea turns a **FITS image** into a HEALPix map: bin each image pixel's value
# into the HEALPix cell that contains its sky position. `image_to_healpix` does it in
# one call — hand it an array + header, an HDU, or just a file path:
#
# ```python
# r = sph.image_to_healpix(data, header, nside="auto", statistic="mean")
# ```
#
# It reads each pixel's world coordinate from the WCS and bins through the same sparse
# machinery, so it stays tractable even when `nside='auto'` matches the image's native
# (here arcsecond-scale) resolution. We use a wide-field optical image of the spiral
# galaxy **M51**; first, the fully automatic call:

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FITSFixedWarning)
    with fits.open("../../examples/data/m51_optical.fits") as hdul:
        img = np.squeeze(hdul[0].data).astype(float)
        iwcs = WCS(hdul[0].header).celestial

# One call, everything automatic. nside='auto' matches the image's (arcsecond-scale)
# pixel size; because this is a small field, sparse='auto' returns a HealpixBins
# (pixels, values, nside, counts) instead of a giant mostly-empty full-sky array.
auto = sph.image_to_healpix(img, iwcs)
print(f"auto  ->  {type(auto).__name__} at nside={auto.nside:,} "
      f"({len(auto.pixels):,} occupied pixels)")

# %% [markdown]
# **Why the sparse result is a named tuple — and why you must keep its `nside`.** A
# sparse HEALPix map is just *(pixel indices, values)*, and those indices do **not**
# encode the resolution: pixel 50,000 could belong to an nside=128 map or an nside=4096
# one — there is no way to tell from the indices alone. So the nside has to travel
# *with* the data, which is exactly why the sparse return is a `HealpixBins` carrying
# `.nside`, and why `plot_healpix_sparse` takes nside as a **required** argument:
#
# ```python
# sph.plot_healpix_sparse(r.pixels, r.values, r.nside, ax=ax)   # r.nside is mandatory
# ```
#
# A *dense* full-sky map is the opposite case — its length is exactly `12·nside²`, so
# the nside is recoverable. `plot_healpix_map` does that for you, and
# `nside_from_array` exposes it directly:

# %%
print(f"dense count_map: length {len(count_map):,}  ->  "
      f"nside {sph.nside_from_array(count_map)}")

# %% [markdown]
# For a figure with *visible* tiles we ask for coarser ~13″ cells (`nside=16384`) and
# render the sparse result, passing its `nside` straight through to the plotter:

# %%
r = sph.image_to_healpix(img, iwcs, nside=16384, statistic="mean", sparse=True,
                         return_counts=True)
center = iwcs.pixel_to_world(img.shape[1] / 2, img.shape[0] / 2)
fig = plt.figure(figsize=(13, 5.6))
# Left: the original image on its own WCS.
ax1 = fig.add_subplot(1, 2, 1, projection=iwcs)
ax1.imshow(img, origin="lower", cmap="sph.deepsky",
           vmin=np.percentile(img, 5), vmax=np.percentile(img, 99.5))
ax1.set_title("original FITS image (M51)", fontsize=11)
ax1.coords[0].set_axislabel("RA")
ax1.coords[1].set_axislabel("Dec")
# A compass on each panel makes the sky orientation explicit — this image's WCS puts
# North *down*, which is correct, not a mistake, and the HEALPix panel inherits it.
# White with a black stroke reads cleanly over the dark image.
sph.add_compass(ax1, loc="lower left", color="white", stroke_color="k")
# Right: the same image as a HEALPix map. The HealpixBins carries nside, so it feeds
# straight into plot_healpix_sparse.
ax2 = sph.make_wcs_frame((1, 2, 2), projection="TAN",
                         center=(center.ra.deg, center.dec.deg), fov_deg=0.24,
                         grid=False, fig=fig)
fig.canvas.draw()
sph.plot_healpix_sparse(r.pixels, r.values, r.nside, ax=ax2, cmap="sph.deepsky",
                        vmin=np.percentile(r.values, 5),
                        vmax=np.percentile(r.values, 99.5),
                        show_boundaries=False, set_extent=True)
sph.add_compass(ax2, loc="lower left", color="white", stroke_color="k")
ax2.set_title(f"image_to_healpix (sparse, nside={r.nside})", fontsize=11)
fig.suptitle("A FITS image on the HEALPix grid, in one call with image_to_healpix",
             fontsize=12)
fig.subplots_adjust(top=0.88, wspace=0.2)
plt.show()

# %% [markdown]
# A few more `image_to_healpix` knobs worth knowing:
#
# - **`nside=`** accepts `'auto'` (match the image scale), an explicit integer, or a
#   target-resolution string like `'5arcmin'`. **`statistic=`** takes the same presets —
#   and the same **callable** — as the catalog binners.
# - **`frame=`** rebins into another coordinate system, e.g. drop an equatorial image
#   onto a Galactic HEALPix grid in one step: `image_to_healpix(img, iwcs,
#   frame='galactic')`.
# - **`return_counts=True`** adds a per-cell coverage count (how many image pixels
#   landed in each tile) — the key to masking thin, under-sampled cells, especially when
#   the HEALPix grid is finer than the image:

# %%
# r above was built with return_counts=True, so r.counts is already populated.
print(f"image pixels per occupied cell — min {int(r.counts.min())}, "
      f"median {int(np.median(r.counts))}, max {int(r.counts.max())}")
well_sampled = r.counts >= 3
print(f"{well_sampled.sum():,} of {len(r.pixels):,} cells have ≥3 image pixels "
      "(mask the rest for a cleaner map)")

# %% [markdown]
# > **Note — binning vs. resampling.** `image_to_healpix` *bins* pixel values (no flux
# > interpolation), which is what you want when the HEALPix cells are comparable to or
# > coarser than the image pixels. For a flux-conserving, interpolating reprojection,
# > the community [`reproject`](https://reproject.readthedocs.io) package offers
# > `reproject_to_healpix((data, wcs), 'icrs', nside=...)` — though it builds a
# > **full-sky dense** array, so a fine nside over a small field is memory-bound.

# %% [markdown]
# ## 6. Drawing on top of a map
#
# A rendered HEALPix map is just an image on an ordinary WCSAxes, so anything else —
# from skyplothelper or plain matplotlib — layers on top of it in world coordinates.
# Below, a density background carries **contours** of the same field, a **scatter** of
# the brightest sources, and a `CompoundRegion` **footprint** outline, all on one frame:

# %%
density = sph.healpix_smooth(
    sph.bin_data_as_healpix(ra, dec, np.ones_like(flux), nside=32,
                            statistic="count", blank_value=0)[0],
    sigma_deg=3.0)

res = sph.healpix_allsky_figure(density, projection="AIT", center=180,
                                cmap="sph.deepsky", figsize=(12, 6))
res.colorbar.set_label("smoothed source density")
ax = res.ax
world = ax.get_transform("world")

# (a) Contours of the same field — healpix_to_celestial gives the lon/lat/value mesh.
clon, clat, cval = sph.healpix_to_celestial(density, lonlatlims="allsky",
                                            center_deg=180)
ax.contour(clon, clat, cval, levels=5, colors="white", linewidths=0.6, alpha=0.7,
           transform=world)

# (b) Scatter the brightest 1% of sources on top.
bright = flux > np.percentile(flux, 99)
ax.scatter(ra[bright], dec[bright], transform=world, s=14, facecolor=URAN[2],
           edgecolor="k", linewidth=0.3, zorder=5, label="brightest 1%")

# (c) A CompoundRegion footprint, outlined (no fill).
sph.CompoundRegion(ax).add_circle(150, 35, 30).render(
    facecolor="none", edgecolor=URAN[4], lw=2.0, zorder=6)
ax.legend(loc="lower left", framealpha=0.9)
ax.set_title("A density map with three overlays: contours, a scatter, and a region",
             fontsize=12)
plt.show()

# %% [markdown]
# Contours come straight from the `healpix_to_celestial` mesh; the scatter and the
# region outline are ordinary artists drawn in world coordinates. Nothing about the map
# being HEALPix changes how you annotate it.

# %% [markdown]
# ### Layering a second HEALPix map
#
# The overlay can itself be a HEALPix map. Because every HEALPix layer lives on the
# same angular grid, you can drop a **sparse** array on top of a full-sky one — a
# region of interest, a second survey's footprint, a different quantity — each with its
# own colormap and its own colorbar. Here a full-sky **source-count** background carries
# a **mean-flux** overlay confined to one cap, with `add_colorbar` called twice — the
# second onto a hand-placed inset axes far enough right to clear the first bar's label:

# %%
# Base layer: the full-sky source-count density (as above).
base = sph.healpix_smooth(
    sph.bin_data_as_healpix(ra, dec, np.ones_like(flux), nside=32,
                            statistic="count", blank_value=0)[0],
    sigma_deg=3.0)
# Overlay layer: mean flux, smoothed to fill each cell, shown only inside a cap.
flux_map = sph.bin_data_as_healpix(ra, dec, flux, nside=32, statistic="mean")[0]
flux_map = sph.healpix_smooth(np.where(np.isnan(flux_map), 0.0, flux_map),
                              sigma_deg=4.0)
cap = sph.healpix_circle_query(300, 35, 42, nside=32)

res = sph.healpix_allsky_figure(base, projection="AIT", center=180,
                                cmap="sph.deepsky", figsize=(12, 6.4),
                                colorbar=False)
overlay = sph.plot_healpix_sparse(cap, flux_map[cap], nside=32, ax=res.ax,
                                  cmap="sph.moss", show_boundaries=False,
                                  set_extent=False, zorder=3)
sph.add_colorbar(res.mappable, ax=res.ax, label="all-sky source density")
# Second bar on its own inset, shifted right (x=1.25) so it clears the first label.
cax2 = res.ax.inset_axes([1.25, 0.0, 0.04, 1.0])
sph.add_colorbar(overlay, cax=cax2, label="mean flux (cap)")
res.ax.set_title("Two HEALPix layers on one grid — full-sky density + a cap overlay",
                 fontsize=12)
plt.show()

# %% [markdown]
# ## 7. Putting it together
#
# The pieces compose into the workflow you will actually run: **bin** a catalog into a
# density map, **smooth** and **render** it, define a **footprint**, and ask **which
# sources fall inside it** — using a `CompoundRegion`'s own membership test to color the
# catalog by in/out. The footprint here is set algebra: a cap with a smaller cap masked
# out of it. One extra layer keeps the pixelization from disappearing under all of that:
# a pass of tile outlines drawn straight over the map.

# %%
# A low nside keeps the cells big enough to read individually under the overplotted
# catalog.
density = sph.healpix_smooth(
    sph.bin_data_as_healpix(ra, dec, np.ones_like(flux), nside=16,
                            statistic="count", blank_value=0)[0],
    sigma_deg=2.0)

res = sph.healpix_allsky_figure(density, projection="AIT", center=180,
                                cmap="sph.deepsky", figsize=(12, 6))
res.colorbar.set_label("smoothed source density")
ax = res.ax
world = ax.get_transform("world")

# Draw the tile edges over the raster, so the pixelization stays visible under the
# catalog. Passing values=None turns the patch backend into a pure outline layer:
# every tile becomes an unfilled polygon (facecolor='none') with only its boundary
# stroked. Keep it hair-thin and mostly transparent — a full-sky mesh at nside=16 is
# a lot of ink, and at full strength it swamps everything drawn on top of it.
sph.plot_healpix_sparse(np.arange(len(density)), None, nside=16, ax=ax,
                        facecolor="none", show_boundaries=True,
                        boundary_color="white", boundary_lw=0.3, alpha=0.15,
                        set_extent=False, zorder=3)

# A compound footprint (a 28° cap with an 8° hole) and its exact membership test.
footprint = sph.CompoundRegion(ax).add_circle(60, 20, 28).subtract_circle(70, 28, 8)
inside = footprint.contains_points(ra, dec)

# Color the catalog by the region's own in/out logic, then outline the footprint.
# Both scatters stay semi-transparent so the map underneath keeps showing through.
ax.scatter(ra[~inside], dec[~inside], transform=world, s=5, color=URAN[0],
           alpha=0.28, edgecolor="none", zorder=4)
ax.scatter(ra[inside], dec[inside], transform=world, s=7, color=URAN[1],
           alpha=0.75, edgecolor="none", zorder=5)
footprint.render(facecolor="none", edgecolor="white", lw=1.8, zorder=6)
# Transparent points make an unreadable legend key, so give the legend opaque
# stand-in markers instead of letting it inherit the plotted artists' alpha.
ax.legend(handles=[Line2D([], [], ls="none", marker="o", color=URAN[1],
                          label="inside"),
                   Line2D([], [], ls="none", marker="o", color=URAN[0],
                          label="outside")],
          loc="lower left", framealpha=0.9)
ax.set_title(f"Catalog density + a masked survey footprint — "
             f"{int(inside.sum())} of {len(ra):,} sources inside", fontsize=12)
plt.show()

# %% [markdown]
# That figure exercises the whole chain: `bin_data_as_healpix` → `healpix_smooth` →
# `healpix_allsky_figure` for the backdrop, a tile-edge pass from
# `plot_healpix_sparse`, a `CompoundRegion` for the footprint, and its
# `contains_points` for membership.
#
# The outline pass is worth stealing on its own. `plot_healpix_sparse(pixels, None,
# nside, facecolor='none', show_boundaries=True)` draws the pixelization over *any*
# map — the quickest way to show a reader what the cells actually are, and how their
# size compares to the structure you are measuring. Keep it faint: a full-sky mesh is
# thousands of lines, and at full strength it competes with everything else on the
# figure. Swap in your own catalog and footprint and the same calls answer "how dense
# is my sky, and how much of my sample lands in this region?"

# %% [markdown]
# ## 8. Where to go next
#
# | To do this | See |
# |---|---|
# | Build set-algebra footprints and test region membership | [Regions & Spherical Polygons](regions.ipynb) |
# | Understand the HPX/XPH projections (and why they pole-lock) vs. drawing a map on a globe | [A Tour of Projections](projections.ipynb) |
# | Display FITS images with full scaling control (the M51 image came from here) | [FITS Images & Quicklook](fits_images.ipynb) |
# | Drape an all-sky map over a rotatable celestial or planetary globe | [Globe & Planet Plotting](globe_plots.ipynb) |
# | Pull real catalogs to bin (SIMBAD / VizieR / SkyView) | **Catalogs: Querying & Plotting** *(coming soon)* |
# | The full function listing and parameters | [HEALPix guide](../guide/healpix.md) |
