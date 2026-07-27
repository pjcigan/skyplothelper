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
# # A Tour of Projections
#
# The [Getting Started](getting_started.ipynb) tutorial built a couple of frames
# without dwelling on the choice of projection. This tutorial slows down on that
# choice: what projections skyplothelper offers, how the three frame *shapes*
# differ and when to use each, how to aim and size a frame, and — useful for
# making impressive plots — how to drop a real image onto a sky map and even
# move it between coordinate systems.
#
# By the end you'll know how to pick a projection deliberately, and how to get
# pixels (a panorama, a survey image) onto whatever frame you've built.
#
# ## Contents
#
# 1. [The projection zoo](#1.-The-projection-zoo)
# 2. [Visualizing distortions in projections](#2.-Visualizing-distortions-in-projections)
# 3. [Frame shapes and builders](#3.-Frame-shapes-and-builders)
# 4. [Steering a frame](#4.-Steering-a-frame)
# 5. [Putting an image on the sky](#5.-Putting-an-image-on-the-sky)
# 6. [Reprojecting between coordinate systems](#6.-Reprojecting-between-coordinate-systems)
# 7. [Sky versus Earth orientation](#7.-Sky-versus-Earth-orientation)
# 8. [The full projection gallery](#8.-The-full-projection-gallery)
# 9. [Putting it together](#9.-Putting-it-together)
# 10. [Where to go next](#10.-Where-to-go-next)

# %%
import numpy as np
import matplotlib.pyplot as plt

import skyplothelper as sph

# base='structural' tightens only the frame/tick *geometry* (inward ticks, tidy
# spines) — it leaves colors and fonts to the docs light/dark theme, so it
# composes with the dark-figure pass (which sets a theme on top).
sph.set_style(base="structural")

# %% [markdown]
# ## 1. The projection zoo
#
# <video controls autoplay loop muted playsinline poster="../_static/manim/projections__what-is-a-projection.poster.jpg" width="100%" style="max-width:760px;display:block;margin:0.5em auto;" aria-label="What a projection is: starlight arrives inward at the observer at the center of the celestial sphere and crosses a flat pane to form a gnomonic TAN chart of Orion, then the whole sky is flattened into an all-sky map"><source src="../_static/manim/projections__what-is-a-projection.mp4" type="video/mp4"></video>
#
# *What "projection" really means, before any code. You sit at the center of the
# celestial sphere, and starlight arrives from every direction. Catch one patch on
# a flat pane, and where each ray crosses it is that star's place on a map — a
# gnomonic (TAN) chart of Orion. Flatten the whole sphere instead, and you get
# an all-sky map. Every projection in this tutorial is a different way of doing
# that flattening.*
#
# A map projection is a compromise: the sphere cannot be flattened without
# distorting *something* — area, shape, distance, or direction. Different
# projections choose different things to preserve. skyplothelper exposes the
# whole astropy/wcslib projection set plus a few classic compromise projections,
# and `list_projections()` prints the registry:

# %%
sph.list_projections()

# %% [markdown]
# That printout is exhaustive; here is the same set distilled **by family**, with
# what each is typically optimal for and the main caveat to keep in mind:
#
# | Family | Codes | Best for — and caveats |
# |---|---|---|
# | **Cylindrical** | `CAR`, `CEA`, `MER`, `CYP` | simple straight lon/lat grids; reprojection input. Severe area blow-up toward the poles (`MER` sends them to infinity). |
# | **Pseudocylindrical / elliptical** | `AIT`, `MOL`, `SFL`, `PAR` | the everyday all-sky maps — all equal-area, so honest for densities and coverage. |
# | **Compromise** | `robinson`, `winkel_tripel`, `eckert_iv`, `kavrayskiy`, `mcbryde` | outreach and overview figures; neither equal-area nor conformal (they split the difference for looks). |
# | **Zenithal (disk)** | `SIN`, `ZEA`, `STG`, `ARC` | hemisphere / globe views — `SIN` the "3-D sphere" look, `ZEA` equal-area polar caps, `STG` low-distortion wide fields. |
# | **Tangent / perspective (field)** | `TAN`, `AIR`, `AZP`, `SZP` | individual/focused imaging fields — use them *zoomed* (`fov_deg=`), not draped with all-sky data. |
# | **Conic** | `COD`, `COE`, `COO`, `COP` | mid-latitude bands; tune the standard parallels with `pv2_1=`/`pv2_2=`. |
# | **HEALPix / quad-cube** | `HPX`, `XPH`, `CSC`, `TSC`, `QSC` | pixelized all-sky data. **Pole-locked** — their center latitude is forced to 0 (see Section 4). |
# | **Bonne / polyconic** | `BON`, `PCO` | distinctive equal-area / atlas-style all-sky outlines; rotate in longitude only (their outline is pole-anchored — see Section 4). |
#
# The fastest way to *feel* the differences is to render the same data through
# several projections. We'll use a synthetic test pattern — a band around the
# equator plus two great circles through the poles — chosen so that *two* kinds
# of distortion jump out: how the edges stretch, and how tightly things converge
# at the poles.

# %%
import healpy as hp

# nside=32 keeps the notebook quick to re-run; the smooth test pattern looks
# essentially identical at higher resolution. (Raise it for publication-clean
# pixelized projections — see the note in Section 8.)
nside = 32
lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
lon_r, lat_r = np.radians(lon), np.radians(lat)


def great_circle_band(sep_deg, width=7.0):
    """A soft band of angular half-width ~`width`° around a great circle."""
    return np.exp(-0.5 * (sep_deg / width) ** 2)


# Equator band (distance from b = 0 is just |b|), plus the two meridional great
# circles through the poles at lon = 0/180 and lon = 90/270:
equator = great_circle_band(np.abs(lat))
merid_a = great_circle_band(
    np.degrees(np.arcsin(np.clip(np.abs(np.cos(lat_r) * np.sin(lon_r)), 0, 1))))
merid_b = great_circle_band(
    np.degrees(np.arcsin(np.clip(np.abs(np.cos(lat_r) * np.cos(lon_r)), 0, 1))))
demo_map = np.maximum.reduce([equator, merid_a, merid_b])


def show_grid(axes):
    """Make the lon/lat graticule legible over the copper test pattern, so each
    projection's *distortion of the grid itself* is visible — the quickest read
    on what a projection does. The maps below are drawn at ``zorder=0`` so the
    frame's own graticule (correctly projected, and correctly rolled for oblique
    / ``lonpole`` views) shows through; this just brightens it to a solid white
    line. Works on a single axes or the list `projection_gallery()` returns."""
    for ax in np.atleast_1d(axes).ravel():
        sph.style_grid(ax, color="white", alpha=0.5, lw=0.6, ls="-")


# %% [markdown]
# First, to set the mental picture, here is that pattern on the *actual sphere* —
# an orthographic (`SIN`) globe seen from three viewing angles. The equator and
# the two great circles are simply three great circles on a ball; every flat map
# further down is an attempt to peel or stretch this sphere onto the page, and
# the distortions are what that particular way of peeling does to it:

# %%
fig = plt.figure(figsize=(12, 4.2))
for i, (clon, clat) in enumerate([(0, 20), (30, 20), (30, 50)], start=1):
    ax = sph.make_globe_frame(130 + i, center_LONdeg=clon, center_LATdeg=clat,
                              radesys="galactic", Naxispix=400)
    sph.plot_healpix_map(demo_map, ax, cmap="copper", zorder=0)
    show_grid(ax)
    ax.set_title(f"globe view ({clon}°, {clat}°)", fontsize=9)
fig.suptitle("The same test pattern on a sphere (orthographic, three views)")
plt.show()

# %% [markdown]
# Now flatten it. `projection_gallery()` renders the same map through any list of
# projections at once — here the four most common all-sky maps. We draw the map
# at `zorder=0` and brighten the frame's own graticule over it (the `show_grid`
# helper above), so you can read each projection by how it bends the grid:

# %%
fig, axes = sph.projection_gallery(
    demo_map, projections=["AIT", "MOL", "SFL", "robinson"],
    center=0, frame="galactic", cmap="copper", ncols=2, zorder=0)
show_grid(axes)
plt.show()

# %% [markdown]
# Notice how the *same* pattern renders in each: Hammer–Aitoff (`AIT`) and
# Mollweide (`MOL`) are equal-area ovals (a square degree covers the same canvas
# area everywhere — the honest choice for densities and coverage); Sanson–
# Flamsteed (`SFL`) keeps parallels straight and evenly spaced but bows the
# meridians; and Robinson is a *compromise* that trades a little of everything
# for a pleasing overall look. Watch the two pole-crossing great circles
# especially — how tightly they pinch together at the top and bottom is a direct
# read-out of each projection's polar distortion.
#
# A rough decision guide:
#
# - **Equal-area** (`AIT`, `MOL`, `SFL`, `CEA`, `ZEA`) — when *how much* matters:
#   source counts, survey footprints, HEALPix maps.
# - **Conformal** (`TAN`, `STG`, `MER`) — when local *shape* matters: imaging
#   fields, morphology.
# - **Compromise** (Robinson, Winkel Tripel, Eckert IV, ...) — when *looks*
#   matter most: outreach and overview figures.
#
# > **Tip:** `list_projections(allsky=True)` filters to the full-sky
# > projections, and `list_projections(shape="circular")` to the globe/disk
# > ones — handy when you know the shape you want but not the code.

# %% [markdown]
# ## 2. Visualizing distortions in projections
#
# The great-circle pattern above shows distortion *qualitatively* — you can see
# the meridians pinch at the poles. The classic way to *measure* it is Tissot's
# indicatrix: scatter a grid of identical small circles across the sphere and
# watch what each projection does to them. A projection that preserves area keeps
# every circle's *area* constant (though it may shear them); a conformal one keeps
# every circle *round* (though their sizes grow); most projections preserve
# neither. `sph.tissot()` drops a grid of these indicatrices onto any frame.
#
# Here are three frames we've already met, each telling a different story:
#
# - **`CAR`** (equirectangular): the simplest map — the circles balloon toward
#   the poles, distorting both area and shape.
# - **`SIN`** (orthographic, the globe view from above): near the center the
#   indicatrices are nearly true, but they foreshorten toward the limb as the
#   sphere curves away.
# - **`AIT`** (Hammer-Aitoff, equal-area): every indicatrix keeps the same
#   *area*, but shears into a tilted ellipse toward the edges.

# %%
fig = plt.figure(figsize=(13, 3))
for i, code in enumerate(["CAR", "SIN", "AIT"], start=1):
    ax = sph.make_wcs_frame(130 + i, code, center=0, frame="galactic", fig=fig)
    # lons stop at ±120 so no indicatrix lands exactly on the SIN limb (every
    # point at lon ±90 is 90° from this center); clip="none" then needs no
    # boundary intersection, so the near-limb circles on SIN simply foreshorten.
    sph.tissot(ax, rad_deg=7, lons=np.arange(-120, 121, 60),
               lats=np.arange(-60, 61, 30),
               facecolor="peru", edgecolor="saddlebrown", alpha=0.6, lw=0.6,
               clip="none")
    ax.set_title(code, fontsize=9)
fig.subplots_adjust(top=0.9, wspace=0.3)
plt.show()

# %% [markdown]
# Read across the three: `CAR`'s circles grow without bound toward the poles
# (neither area nor shape preserved); `SIN`'s stay round but shrink toward the
# limb (the foreshortening of a sphere seen edge-on); `AIT`'s all enclose the same
# area but tilt and shear away from center — the price an equal-area map pays to
# stay honest about *how much* sky each region covers.
#
# > **Tip:** the indicatrix size is just a visual scale — tune it with
# > `tissot(rad_deg=...)` (and the grid spacing with `lons=`/`lats=`) so the
# > circles are large enough to read without overlapping on the projection you're
# > checking.

# %% [markdown]
# ## 3. Frame shapes and builders
#
# Every frame is a real astropy `WCSAxes`, but the *outline* comes in three
# shapes, and the convenience builders are organized around them:
#
# | Shape | Looks like | Typical projections | Builder |
# |---|---|---|---|
# | **elliptical** | full-sky oval | AIT, MOL | `allsky_figure()` |
# | **circular** | globe / hemisphere disk | SIN, ARC, STG, ZEA | `make_globe_frame()`, `make_planet_frame()` |
# | **rectangular** | bounded field | TAN, CAR, MER, conics | `offset_figure()` |
#
# `make_wcs_frame()` is the master builder underneath all of them — it can make
# any shape and projection; the others are conveniences for the common cases.
# Here is one of each, side by side:

# %%
fig = plt.figure(figsize=(13, 3.0))

# Elliptical: a full-sky Aitoff oval
sph.make_wcs_frame(131, "AIT", center=0, fig=fig)
# Circular: an orthographic globe looking at the celestial sphere
sph.make_globe_frame(132, center_LONdeg=0, center_LATdeg=20)
# Rectangular: a bounded tangent-plane field on the Crab Nebula
sph.make_wcs_frame(133, "TAN", center=(83.63, 22.01), fov_deg=6, fig=fig)

fig.suptitle("The three frame shapes: elliptical · circular · rectangular")
plt.show()

# %% [markdown]
# All three are WCSAxes, so they accept the same `sph.add_*` / `sph.plot_*`
# helpers and the same `ax.plot(..., transform=ax.get_transform("world"))`
# pattern. Two specialized backends round out the set: the globe builders use a
# circular frame for hemisphere/globe views (see the
# [Globe and Planet Plotting](globe_plots.ipynb) tutorial), and
# `make_cartopy_frame()` returns a cartopy GeoAxes when you want cartopy's
# terrestrial feature stack.

# %% [markdown]
# ## 4. Steering a frame
#
# Three knobs aim and size a frame:
#
# - **`center=`** (or `center_lon=`/`center_lat=`) — what longitude/latitude
#   sits at the middle of the map.
# - **`fov_deg=`** (or FITS-style `cdelt=` + `npix=`) — the field of view, for
#   *bounded* projections. All-sky projections ignore it and always show the
#   whole sphere.
# - **`lonpole=`/`latpole=`** — the projection's rotation, for the cases that
#   need it.
#
# Centering is the knob you'll reach for most often. The same all-sky projection
# can look different depending on what you put at the center, and you can use
# this to bring data on one part of the sky into focus in the middle of the plot:

# %%
fig = plt.figure(figsize=(12, 4))
sph.make_wcs_frame(121, "MOL", center=180, fig=fig)   # 0h at the edges
sph.make_wcs_frame(122, "MOL", center=0, fig=fig)     # 0h in the middle
fig.suptitle("Same projection, different center (RA 180 vs 0 at mid-map)")
plt.show()

# %% [markdown]
# **Oblique aspect, and the projections that can't tilt.** Passing a *latitude*
# as well — `center=(lon, lat)` — tilts most frames into an oblique aspect (the
# Section 1 globe views did exactly this). A few projections are the exception:
# the pixelization-based ones — `HPX`, `XPH`, and the quad-cubes `CSC`, `TSC`,
# `QSC` — are **locked to the poles** (their center *latitude* is forced to 0),
# because the HEALPix / cube tilings are defined relative to the poles and an
# oblique frame would misalign the tiles from the data. A center *longitude*
# still shifts them normally.
#
# A softer case: `BON` (Bonne) and `PCO` (polyconic) *do* accept a latitude
# offset — it's mathematically valid — but their outline is anchored at a pole,
# so a latitude tilt reads oddly and the perimeter can overflow the frame.
# Rotate these in *longitude* only for clean results.
#
# > **Note — the projection vs. the map.** That lock is a property of the `HPX`
# > *projection*, not of HEALPix data. A binned HEALPix *map* can be viewed at
# > any orientation — you just render it on a frame that *does* rotate. Bin your
# > catalog into a HEALPix map, then drape it on a tilted `SIN`/orthographic
# > globe (exactly the Section 1 sphere views) and spin that to any angle; only
# > the `HPX` projection *itself* — the fixed diamond-grid all-sky layout — stays
# > pole-locked. Binning, reprojecting, and viewing HEALPix maps across
# > projections is the subject of the [HEALPix Workflows](healpix_workflows.ipynb)
# > tutorial.

# %% [markdown]
# **Rotating a frame: `lonpole`.** The third knob, `lonpole=`, rolls the
# projection about its center — use this when you want a specific orientation
# rather than the default "north up." It is easiest to *see* on a globe: here is
# the same orthographic hemisphere, looking at the same point, rolled by 0°, 45°,
# and 90°. (This is exactly the roll that `euler_to_fits_ortho()`'s `perspective`
# angle sets for the tilted Earth in Section 7.)

# %%
fig = plt.figure(figsize=(12, 4.2))
for i, lp in enumerate([0, 45, 90], start=1):
    ax = sph.make_globe_frame(130 + i, center_LONdeg=0, center_LATdeg=20,
                              radesys="galactic", Naxispix=350, lonpole=lp)
    sph.plot_healpix_map(demo_map, ax, cmap="copper", zorder=0)
    show_grid(ax)   # the graticule rolls with lonpole, making the roll obvious
    ax.set_title(f"lonpole={lp}°", fontsize=9)
fig.suptitle("The same globe, rolled by lonpole")
plt.show()

# %% [markdown]
# For bounded frames, `fov_deg` sets how much sky you see. Compare a wide
# context field with a tight zoom on the same target:

# %%
fig = plt.figure(figsize=(11, 4.5))
sph.make_wcs_frame(121, "TAN", center=(83.63, 22.01), fov_deg=10, fig=fig)
sph.make_wcs_frame(122, "TAN", center=(83.63, 22.01), fov_deg=1.5, fig=fig)
fig.suptitle("Same target, different field of view (10° vs 1.5°)")
plt.show()

# %% [markdown]
# > **Note:** `fov_deg` only applies to bounded (rectangular) projections.
# > Passing it to an all-sky projection has no effect — those always show the
# > full sphere. If you want a *zoomed* all-sky-style view, use a globe frame
# > (circular) or a bounded projection instead.

# %% [markdown]
# ## 5. Putting an image on the sky
#
# Frames aren't just for points and lines — you can lay a raster image into one.
# The classic case is a single telescope observation — say, a single galaxy or
# deep-sky field — but it can also be an all-sky panorama (a Milky Way photo, a
# survey mosaic) as a backdrop over the entire frame. Two helpers do the work:
#
# - `load_sky_image()` reads an equirectangular image and wraps it in a matching
#   all-sky WCS header.
# - `reproject_background()` resamples that image onto whatever frame you've
#   built (requires the optional `reproject` package).
#
# > **Two loaders, same idea.** `load_sky_image()` and `pseudofits_from_image()`
# > both wrap a flat equirectangular image in a `CAR` WCS so the reprojection
# > machinery can resample it; they differ in packaging and reach.
# > `load_sky_image()` returns a ready `(array, header)` pair and pairs with
# > `reproject_background()` — convenient for sky panoramas.
# > `pseudofits_from_image()` returns a single FITS HDU, adds a `geo=True`
# > option for geographic (Earth/planet) maps and a `gmst_deg` Earth-rotation
# > offset, and pairs with `reproject_rgb_map()`. Either works for a sky
# > backdrop; use `pseudofits_from_image()` when the source is geographic
# > (as we do for the Earth in Section 7).
#
# The one thing to get right is the **`frame=` of the source image** — it must
# match the coordinate layout the panorama was drawn in. This NOIRLab all-sky
# photo plotted below is laid out in *galactic* coordinates (the Milky Way runs
# straight across the middle of the raw image), so we load it as `frame="galactic"`.
#
# ### Start with the source: an equirectangular (plate carrée) image
#
# Panoramas like this are stored as *equirectangular* images — the
# **plate carrée** (`CAR`) projection — where longitude maps linearly to the
# x-axis and latitude to the y-axis, so the whole sky fills a simple 2:1
# rectangle. That regular grid is the key: because we know the source is
# plate carrée, `load_sky_image()` can attach a matching `CAR` WCS header (note
# the `-CAR` in its `CTYPE`s), and *that* is what makes reprojecting it onto any
# other projection possible. Here is the source in its own native frame:

# %%
# fig-slug: plate-carree-source
pano = "../../examples/data/Allsky_noirlab2430b_1280x640.jpg"

img, hdr = sph.load_sky_image(pano, frame="galactic", center=0)
print("source projection (CTYPE):", hdr["CTYPE1"], hdr["CTYPE2"])

fig = plt.figure(figsize=(11, 5.5))
ax = sph.make_wcs_frame(111, "CAR", center=0, frame="galactic",
                        npix=(1200, 600), fig=fig)
ax.imshow(sph.reproject_background(img, hdr, ax))
ax.set_title("The source panorama in its native plate carrée (CAR) projection")
plt.show()

# %% [markdown]
# Straight, evenly spaced gridlines — the signature of plate carrée. Now we can
# resample it onto a *curved* all-sky projection. `reproject_background()` does
# the work, sampling the image onto the target frame's own pixel grid. The
# all-sky builders default to a coarse grid (fine for scatter plots and small
# thumbnails, where it keeps things fast); for a crisp high-resolution backdrop
# that's too coarse, so we request a finer grid with `npix=(NAXIS1, NAXIS2)`,
# keeping the 2:1 ratio for an all-sky oval (here `(1200, 600)`):

# %%
# fig-slug: panorama-on-aitoff
fig, ax = sph.allsky_figure(projection="AIT", center=0, frame="galactic",
                            npix=(1200, 600), figsize=(11, 5.5))
background = sph.reproject_background(img, hdr, ax)
ax.imshow(background)
ax.set_title("NOIRLab all-sky panorama on a Galactic Aitoff frame")
plt.show()

# %% [markdown]
# > **Note:** `npix=` sets the frame's synthetic-WCS pixel dimensions
# > `(NAXIS1, NAXIS2)` — i.e. how finely the sky is sampled onto the canvas. It
# > only matters when you're *reprojecting imagery* onto the frame; for points,
# > lines, and overlays the default coarse grid is perfectly sharp (those are
# > drawn as vectors, not resampled). Match the ratio to the frame shape: ~2:1
# > for an all-sky oval, square for a globe or a square field.

# %% [markdown]
# The Milky Way sits right along the equator of the map, exactly where the
# galactic plane belongs. If you instead loaded this image as `frame="ICRS"`,
# the labels would say one thing while the pixels meant another, and the band
# would land in the wrong place — a common and confusing mistake. **Match the
# source frame to the image's actual layout.**

# %% [markdown]
# ### The same image on a globe
#
# This isn't limited to flat all-sky maps. A circular globe frame
# (`make_globe_frame()`, an orthographic `SIN` view by default) takes the same
# reprojection, draping the panorama over a hemisphere — here looking straight
# at the galactic center. The globe builder's resolution knob is `Naxispix` (if
# your raster image looks pixelated, try increasing `Naxispix`):

# %%
img, hdr = sph.load_sky_image(pano, frame="galactic", center=0)

fig = plt.figure(figsize=(6.5, 6.5))
ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=0,
                          radesys="galactic", Naxispix=1000)
ax.imshow(sph.reproject_background(img, hdr, ax))
ax.set_title("Full-sky image projected onto a celestial globe (galactic center)")
plt.show()

# %% [markdown]
# ## 6. Reprojecting between coordinate systems
#
# Because `reproject_background()` does a full WCS transform, the *target* frame
# doesn't have to be in the same coordinate system as the source. Build the
# frame in whatever system you want and the image is resampled into it —
# gridlines, labels, and overlays all follow the target frame.
#
# Here we take the same galactic-layout panorama and display it on an **ICRS
# (equatorial)** frame. The Milky Way now cuts diagonally across the sky, the
# way it actually does in equatorial coordinates — and overlaying the galactic
# plane confirms the transform landed it correctly:

# %%
img, hdr = sph.load_sky_image(pano, frame="galactic", center=0)

fig, ax = sph.allsky_figure(projection="AIT", center=180, frame="ICRS",
                            npix=(1200, 600), figsize=(11, 5.5))
background = sph.reproject_background(img, hdr, ax)
ax.imshow(background)

# The galactic plane, drawn on the equatorial frame, traces the band:
sph.add_plane_overlay(ax, plane="galactic", color="cyan", lw=1.2,
                      label="galactic plane")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("The same panorama reprojected into equatorial coordinates")
plt.show()

# %% [markdown]
# The cyan galactic-plane line lies right along the bright band — visual proof
# that the image was genuinely transformed from galactic into equatorial
# coordinates, not merely relabeled.
#
# > **Note:** The same machinery works for science FITS images: reproject a FITS
# > image onto a frame built in a different system and it lands correctly, with
# > the new frame's gridlines. Displaying and stretching FITS data (single-
# > channel images, colorbars, contours) is the subject of the
# > [FITS Images & Quicklook](fits_images.ipynb) tutorial; here we've focused
# > on full-sky backdrops.

# %% [markdown]
# ## 7. Sky versus Earth orientation
#
# One convention deserves a closer look here because it is *projection-shaped*:
# which way longitude runs. Looking **up** at the sky with north up, east is to
# the **left**; looking **down** at a map of the Earth, east is to the right.
# skyplothelper is an astronomy-focused package, so every frame defaults to the
# sky convention. The flip is a single keyword when making your frame:
#
# - `direction='sky'` (aliases `'astro'`, `'east-left'`) — the default.
# - `direction='geo'` (aliases `'earth'`, `'east-right'`) — terrestrial/planetary.
#
# The same plate carrée frame, both ways. To make the flip unmistakable we add
# an **eastward arrow** (pointing toward increasing longitude) and tint both the
# arrow's points and the **longitude tick labels** by their longitude value — so
# "color runs eastward." In `sky` mode east is to the left, in `geo` mode it's to
# the right, and the whole color gradient reverses with it:

# %%
import matplotlib as mpl

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(-180, 180)             # longitude -> color
wrap = lambda v: (np.asarray(v) + 180) % 360 - 180  # to [-180, 180)


def color_lon_labels(ax):
    """Recolor the bottom longitude tick labels by their value: read the
    rendered labels off the coordinate helper, hide them, redraw in color.
    Match the latitude labels' size and boldface them for readability, and
    nudge them a few points below the spine: ``.data`` gives the tick position
    *on* the spine (without astropy's label pad), so we re-add that pad via an
    offset, otherwise the redrawn labels ride up against the frame border."""
    ax.get_figure().canvas.draw()                   # materialize the labels
    tlab = ax.coords[0].ticklabels
    vals, strs, pos = tlab.world["b"], tlab.text["b"], tlab.data["b"]
    ax.coords[0].set_ticklabel_visible(False)
    for v, s, (dx, dy) in zip(vals, strs, pos):
        ax.annotate(s, xy=(dx, dy), xytext=(0, -5), textcoords="offset points",
                    color=cmap(norm(wrap(v))), ha="center", va="top",
                    fontsize=10, fontweight="bold", clip_on=False)


fig = plt.figure(figsize=(11, 4.4))
for col, direction, title in [(1, "sky", "direction='sky' (east left)"),
                              (2, "geo", "direction='geo' (east right)")]:
    ax = sph.make_wcs_frame(120 + col, "CAR", center=0, fig=fig,
                            direction=direction)
    tr = ax.get_transform("world")
    lons = np.linspace(-45, 45, 9)
    ax.scatter(lons, np.zeros_like(lons), transform=tr, s=22,
               c=cmap(norm(lons)), zorder=5)
    ax.annotate("", xy=(58, 0), xytext=(45, 0), xycoords=tr, textcoords=tr,
                arrowprops=dict(arrowstyle="-|>", color=cmap(norm(58)), lw=2.5,
                                mutation_scale=22))
    # The theme's text color (not a fixed gray) keeps this legible on a dark
    # canvas too:
    ax.text(0, 22, "increasing longitude (East)", transform=tr,
            color=plt.rcParams["text.color"], ha="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    color_lon_labels(ax)
plt.show()

# %% [markdown]
# There is a second, related setting: `radesys=`/`frame=`. Sky frames default to
# `'ICRS'`; Earth maps want a body-fixed frame (`'ITRS'`). Rather than remember
# to set both `direction='geo'` and the right frame, simply use
# `make_planet_frame()`, which bundles the geographic direction with the
# body-fixed coordinate frame for Earth and the other planets in one call.
#
# > **Important:** Don't "fix" a mirrored-looking Earth map by flipping the axis
# > by hand — that desynchronizes the data from the labels. Use `direction='geo'`
# > or `make_planet_frame()`. The full treatment of globes and planets is in the
# > [Globe and Planet Plotting](globe_plots.ipynb) tutorial.

# %% [markdown]
# ### A geo example: a tilted, spinning Earth
#
# The same projection machinery serves the Earth and the other planets — and it
# isn't limited to looking straight on. A globe's orientation is three physical
# Euler angles (spin/rotation, obliquity/axial tilt, and a perspective roll);
# `euler_to_fits_ortho()` converts those intuitive angles into the FITS
# `(center_lon, center_lat, lonpole)` the frame builders take. Here we give the
# Earth its real 23.4° obliquity plus a spin that brings the Americas into view,
# and drape NASA's "Black Marble" night-lights map over it. `make_planet_frame()`
# supplies the geographic direction and body-fixed frame; the geographic raster
# is loaded with `pseudofits_from_image(..., geo=True)` and resampled with
# `reproject_rgb_map()`. Two finishing touches: the image is drawn *below* the
# graticule (`zorder=-10`), and `add_overlay_ticks()` redraws the ticks and
# labels in light gray — with the latitude labels parked on the −30° meridian
# (out in the Atlantic) so they stay clear of the Americas.

# %%
# Three physical Euler angles → FITS SIN orientation params:
center_lon, center_lat, lonpole = sph.euler_to_fits_ortho(
    rotation=65, obliquity=23.4, perspective=15)

earth = sph.pseudofits_from_image(
    "../../examples/data/BlackMarble_2016_01deg.jpg", geo=True)

fig = plt.figure(figsize=(6.5, 6.5))
ax = sph.make_planet_frame(111, body="earth", center_LONdeg=center_lon,
                           center_LATdeg=center_lat, lonpole=lonpole,
                           Naxispix=1000, tick_style="native")

# Project the RGB raster onto the frame's pixel grid, below the graticule:
out_hdr = ax.wcs.to_header()
nx = round(ax.get_xlim()[1] - ax.get_xlim()[0])
ny = round(ax.get_ylim()[1] - ax.get_ylim()[0])
out_hdr["NAXIS1"], out_hdr["NAXIS2"] = nx, ny
ax.imshow(sph.reproject_rgb_map(earth, out_hdr, shape_out=(ny, nx)), zorder=-10)

# Light ticks/labels so they read on the night side; place the latitude labels
# on the -30° meridian (out in the Atlantic) to keep them off the Americas:
sph.add_overlay_ticks(ax, lon_at="axis", lat_at="lon=-30",
                      tick_kwargs={"color": "0.9"},
                      label_kwargs={"color": "0.9"})
ax.set_title("Black Marble: Earth at 23.4° obliquity, spun toward the Americas")
plt.show()

# %% [markdown]
# > **Note:** skyplothelper is first and foremost an *astronomy* package — but
# > the Earth is a planet too, and both a WCSAxes path (`make_planet_frame()`,
# > shown here) and a cartopy path (`make_cartopy_frame()`, with cartopy's full
# > terrestrial feature stack) are supported. Tilted globes, planet textures,
# > nightshade, and surface features get the full treatment in the
# > [Globe and Planet Plotting](globe_plots.ipynb) tutorial; this is just a
# > taste that the projection machinery carries straight over to geographic
# > frames.

# %% [markdown]
# ## 8. The full projection gallery
#
# Section 1 showed a handful of all-sky projections; skyplothelper supports about
# thirty. Rather than one giant grid, here they are **by family**, each rendered
# with the same test pattern from Section 1 so you can compare how each treats
# the sphere. (`projection_gallery()` accepts any list of codes, and
# `list_projections()` lists them all — including the filters
# `list_projections(shape=...)` and `list_projections(allsky=...)`. It samples
# the map onto each frame and handles the antimeridian seam and each projection's
# visible boundary for you. Its `cmap=` takes any registered colormap, including
# the bundled `sph.*` palettes — see
# [Themes, Palettes & Fonts](styling.ipynb).)
#
# **Cylindrical** — longitude and latitude map to straight, perpendicular axes.
# Simple and predictable; the spacing is what distinguishes them (equirectangular,
# equal-area, Mercator, perspective):

# %%
fig, axes = sph.projection_gallery(
    demo_map, projections=["CAR", "CEA", "MER", "CYP"],
    center=0, frame="galactic", cmap="copper", ncols=2, zorder=0)
show_grid(axes)
plt.show()

# %% [markdown]
# **Pseudocylindrical and elliptical** — curved meridians close the whole sky
# into an oval or lens. The equal-area ovals (`AIT`, `MOL`) are the usual choices
# for all-sky figures; `SFL` and `PAR` keep straight, evenly spaced parallels:

# %%
fig, axes = sph.projection_gallery(
    demo_map, projections=["AIT", "MOL", "SFL", "PAR"],
    center=0, frame="galactic", cmap="copper", ncols=2, zorder=0)
show_grid(axes)
plt.show()

# %% [markdown]
# **Compromise projections** — Robinson, Winkel Tripel, Eckert IV, Kavrayskiy
# VII, and McBryde–Thomas are neither equal-area nor conformal; they balance both
# for a pleasing overall look — the usual choice for outreach and overview
# figures:

# %%
fig, axes = sph.projection_gallery(
    demo_map,
    projections=["robinson", "winkel_tripel", "eckert_iv", "kavrayskiy", "mcbryde"],
    center=0, frame="galactic", cmap="copper", ncols=3, zorder=0)
show_grid(axes)
plt.show()

# %% [markdown]
# **Zenithal (azimuthal)** — the sphere projected onto a plane, giving circular
# disks, each showing one hemisphere (the far side falls behind the limb). `SIN`
# is the globe view, `ZEA` is equal-area, `STG` is conformal, and `ARC` is
# equidistant:

# %%
fig, axes = sph.projection_gallery(
    demo_map, projections=["SIN", "ZEA", "STG", "ARC"],
    center=0, frame="galactic", cmap="copper", ncols=2, zorder=0)
show_grid(axes)
plt.show()

# %% [markdown]
# (The *perspective* zenithals — the bounded tangent plane `TAN` (the
# imaging/interferometry workhorse from Sections 3–4), plus `AIR`, `AZP`, and
# `SZP` — show a *field* rather than the whole sky, so use them zoomed in with
# `fov_deg=` rather than draping all-sky data over them.)
#
# **Conic** — the sphere projected onto a cone opening from a standard parallel:
# one possible choice for a mid-latitude band. The all-sky wedge keeps the data
# within the cone's fan (standard parallels tunable via `pv2_1=`/`pv2_2=`):

# %%
fig, axes = sph.projection_gallery(
    demo_map, projections=["COD", "COE", "COO", "COP"],
    center=0, frame="galactic", cmap="copper", ncols=2, zorder=0)
show_grid(axes)
plt.show()

# %% [markdown]
# What "tuning the standard parallels" means in practice: `pv2_1=` sets the
# standard parallel — the latitude where the cone touches the sphere and
# distortion is lowest (the default is 45°) — and `pv2_2=` optionally splits it
# into *two* parallels at `pv2_1 ± pv2_2`, spreading the low-distortion zone
# across a band. Aim them at the latitudes your data lives at, and the whole fan
# reshapes around that choice:

# %%
fig = plt.figure(figsize=(13, 3.6))
for i, (kwargs, title) in enumerate([
        ({}, "default (pv2_1=45)"),
        ({"pv2_1": 20}, "pv2_1=20 (low-latitude cone)"),
        ({"pv2_1": 60, "pv2_2": 15}, "pv2_1=60, pv2_2=15")], start=1):
    ax = sph.make_wcs_frame(130 + i, "COE", center=0, frame="galactic",
                            fig=fig, **kwargs)
    sph.plot_healpix_map(demo_map, ax, cmap="copper", zorder=0)
    show_grid(ax)
    ax.set_title(title, fontsize=9)
plt.show()

# %% [markdown]
# **HEALPix and quad-cubes** — projections built for *pixelized* all-sky data.
# `HPX` is the HEALPix all-sky face and `XPH` its polar "butterfly"; the quad-cube
# codes (`CSC`, `TSC`, `QSC`) unfold the sphere onto cube faces. Note that `HPX`
# is the HEALPix *frame* (the fixed all-sky diamond layout), distinct from
# *binning* data into a HEALPix map. Binning, sparse plotting, resolution changes,
# and spatial queries get their own [HEALPix Workflows](healpix_workflows.ipynb)
# tutorial:

# %%
fig, axes = sph.projection_gallery(
    demo_map, projections=["HPX", "XPH", "CSC", "TSC", "QSC"],
    center=0, frame="galactic", cmap="copper", ncols=3, zorder=0)
show_grid(axes)
plt.show()

# %% [markdown]
# > **Note:** at low HEALPix resolution these pixelized projections can show
# > faint resolution-dependent edges; raise `nside` for publication-clean figures.
#
# The quad-cube codes are designed to be read **one face at a time** — each
# square face is a low-distortion view of about a sixth of the sky. The all-sky
# unfold above is handy for a full map, but for a single region pass `fov_deg=` to
# zoom to one face (here ~70°, centered on a face; `XPH`'s central diamond shown
# alongside):

# %%
fig = plt.figure(figsize=(9, 9))
for i, code in enumerate(["CSC", "TSC", "QSC", "XPH"], start=1):
    ax = sph.make_wcs_frame(220 + i, code, center=(0, 0), fov_deg=70, fig=fig)
    ax.set_title(code)
fig.suptitle("Quad-cube and HEALPix-butterfly faces (single-face views)")
plt.show()

# %% [markdown]
# **Bonne and polyconic** — two more bounded outlines that still render the whole
# sky: the equal-area heart-shaped Bonne (`BON`) and the polyconic egg (`PCO`):

# %%
fig, axes = sph.projection_gallery(
    demo_map, projections=["BON", "PCO"],
    center=0, frame="galactic", cmap="copper", ncols=2, zorder=0)
show_grid(axes)
plt.show()

# %% [markdown]
# > **Tip:** `projection_gallery()` (and `plot_healpix_map()`) clean up the
# > tricky all-sky cases for you — masking quads that cross the antimeridian seam
# > and clipping data to each projection's true boundary. If you ever hand-roll a
# > `pcolormesh` on one of these frames, `sph.mask_seam_crossing_quads(...)` and
# > `sph.clip_to_projection_boundary(...)` apply the same two steps.

# %% [markdown]
# ## 9. Putting it together
#
# Every figure so far has used a synthetic test pattern or a stock panorama. But
# the reason you pick a projection is almost always to show *your own data* — a
# source catalog, a survey. So let's close by putting a real catalog on the sky,
# switching projections under it, and composing a finished all-sky figure.
#
# `plot_catalog()` scatters a catalog in world coordinates straight onto any
# frame. It accepts an astropy `Table`, a pandas `DataFrame`, a dict, or plain
# `(ra, dec)` arrays, auto-detects common column names, and can color- or
# size-code the points by any column — with built-in scaling for the skewed
# quantities catalogs are full of. Our catalog is the **ICRF3 defining
# sources** — the 303 quasars that anchor the International Celestial Reference
# Frame — bundled with the examples as a small CSV:

# %%
from astropy.table import Table

icrf = Table.read("../../examples/data/icrf3_defining.csv")
print(len(icrf), "sources;", icrf.colnames)

fig = plt.figure(figsize=(13, 4))
for i, code in enumerate(["AIT", "PAR"], start=1):
    ax = sph.make_wcs_frame(120 + i, code, center=0, frame="ICRS", fig=fig)
    # Encode each source's observing history in its marker size. The session
    # count is very skewed (a few quasars dominate), so `size_scale="sqrt"`
    # shapes it before mapping into the smin..smax range:
    sph.plot_catalog(ax, icrf, ra_col="ra_deg", dec_col="dec_deg",
                     sizeby="n_sess", size_scale="sqrt", smin=8, smax=160,
                     color="darkorange", alpha=0.8,
                     edgecolors="0.2", linewidths=0.3)
    ax.set_title(code, fontsize=10)
# y= lifts the suptitle clear of the taller PAR panel's own title:
fig.suptitle("The same catalog, two projections "
             "(swapped the projection identifier, nothing else)", y=1.04)
plt.show()

# %% [markdown]
# Switching projection is a one-line change: the `plot_catalog()` call is
# identical for both panels — only the projection code passed to
# `make_wcs_frame()` differs. Marker size encodes how many VLBI sessions have
# observed each source, so the heavily monitored quasars stand out at a glance.
#
# Now the payoff — and a chance to pull the whole tutorial together. We build a
# single equal-area Aitoff frame (the honest choice when *coverage* is the point),
# **overlay the all-sky panorama onto it** exactly as in Section 5, then layer the
# catalog on top in one `plot_catalog()` call that does the encoding work for us:
# marker *size* carries the session count (`size_scale="sqrt"`), while *color*
# encodes each source's positional precision (`colorby="pos_err_mas"`,
# `color_scale="log"`). The `plasma` map reads naturally on the dark sky — the
# well-measured sub-mas sources sit at the cool end, the few less-certain ones glow
# hot — and `cmap_range` (trimming the colormap's darkest end) plus a light marker
# edge keep the points legible against the Milky Way. A thin galactic-plane line
# completes the reference geometry:

# %%
pano = "../../examples/data/Allsky_noirlab2430b_1280x640.jpg"
img, hdr = sph.load_sky_image(pano, frame="galactic", center=0)

fig, ax = sph.allsky_figure(projection="AIT", center=180, frame="ICRS",
                            npix=(1200, 600), figsize=(11, 5.5))
ax.imshow(sph.reproject_background(img, hdr, ax), zorder=-10)

# A single call carries all four encodings. Size = sessions (sqrt-scaled);
# color = positional precision on a log scale, with vmax clipped to the 95th
# percentile so the tightly clustered sub-mas values spread across the colormap;
# `cmap_range` trims plasma's darkest end so the points stay legible on the night
# sky; and the colorbar is returned with plain-decimal (not scientific) ticks:
sc, cb = sph.plot_catalog(
    ax, icrf, ra_col="ra_deg", dec_col="dec_deg",
    sizeby="n_sess", size_scale="sqrt", smin=8, smax=160,
    colorby="pos_err_mas", color_scale="log",
    vmax=np.percentile(icrf["pos_err_mas"], 95),
    cmap="plasma", cmap_range=(0.2, 1.0),
    alpha=0.95, edgecolors="0.7", linewidths=0.4,
    cbar=True, cbar_label="position uncertainty (mas)",
    cbar_ticks=[0.05, 0.1, 0.2], cbar_format="{x:g}")
sph.add_plane_overlay(ax, plane="galactic", color="cyan", lw=1.0, alpha=0.6,
                      label="galactic plane")

# Hours-only longitude labels (drop minutes/seconds) for a clean all-sky read,
# stroked so they stay legible over BOTH the dark image and the canvas margin.
# The treatment adapts to the active theme: a dark label with a light (white)
# stroke on a light canvas, a light label with a dark stroke on a dark one —
# keyed off the theme's text color so the same cell works either way.
from matplotlib.colors import to_rgb

_dark_theme = sum(w * x for w, x in zip((0.299, 0.587, 0.114),
                  to_rgb(plt.rcParams["text.color"]))) > 0.5
_lab_color, _lab_stroke = ("0.9", "0.2") if _dark_theme else ("0.2", "white")
sph.format_ticklabels(ax, style="allsky_hours", color=_lab_color,
                      stroke_color=_lab_stroke, stroke_lw=0.8)

ax.legend(loc="lower right", fontsize=8)
ax.set_title("The ICRF3 defining sources on the night sky")
plt.show()

# %% [markdown]
# That is the figure the whole tutorial has been building toward: a real catalog
# on a projection chosen on purpose, *over* a reprojected sky image, with marker
# size and color carrying two more dimensions of the data and a reference line on
# top. Swap in your own `Table` — point `ra_col`/`dec_col` at its columns and
# `colorby=`/`sizeby=` at whatever you want to encode (with `color_scale=`/
# `size_scale=` to tame skewed values) — and the same few lines give you your own
# all-sky map. (Note too that unlike
# the optical surveys hidden behind galactic dust, these radio-loud quasars spread
# right across the Milky Way, so the reference frame tiles the entire sky.)

# %% [markdown]
# ## 10. Where to go next
#
# You can now choose a projection on purpose, build it at the right shape,
# center, and size, and get imagery onto it — including across coordinate
# systems. From here:
#
# | If you want to... | Go to |
# |---|---|
# | Fine-tune grids, ticks, and labels on these frames | [Decorating Frames](decorating_frames.ipynb) |
# | Restyle everything at once — themes, palettes, fonts | [Themes, Palettes & Fonts](styling.ipynb) |
# | Overlay a second coordinate system | [Overlay Coordinate Grids](overlay_grids.ipynb) |
# | Display and stretch science FITS images | [FITS Images & Quicklook](fits_images.ipynb) |
# | Draw globes and tilted planets | [Globe and Planet Plotting](globe_plots.ipynb) |
# | Bin, view, and query HEALPix maps | [HEALPix Workflows](healpix_workflows.ipynb) |
# | Make z–RA wedge ("cone") and bowtie plots | [Cone & Bowtie Plots](cone_bowtie.ipynb) |
# | Drag-rotate and explore these frames interactively | [Interactive Plotting with Plotly](interactive_plotly.ipynb) |
#
# The companion reference for this material is the
# [Frames & projections](../guide/frames.md) guide page.
# %%
