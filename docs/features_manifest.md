# Feature Gallery manifest

This file is the editable source of truth for the Feature Gallery.
`make_features.py` parses it and generates the thumbnail images, the
index grid page, and one detail page per entry. To regenerate after
editing (from the repo root):

    cd docs && cenv && python make_features.py && make html

Format: `## headings` are categories (order = page order); `###`
headings are entries. After an entry title, optional metadata lines
(`- guide:`, `- api:`, `- data: examples`), then description prose,
then exactly one ```python fenced code block. The code must be
self-contained (its own imports) and leave the target figure as `fig`
or as the current matplotlib figure. Paths are relative to the repo
root. Keep snippets seeded (`default_rng(N)`) so regeneration is
deterministic.

Styling: the renderer sets the package's own annotation palettes
(`publication` light / `denim` dark) with the dual-mode `uranometria`
data cycle, so `C0`–`C9` resolve to in-house colors in both modes. Use
those cycle indices for accents and the bundled `sph.*` colormaps for
image/density maps — not `tab:*` or bare `magma`/`viridis`. The full
features held back from this curated pass (nightshade / tilted-textured
globes, co-visibility regions, image-stamp markers, RGB composites,
symmetric-log, data cubes, zoom insets, quicklook, more tick/label knobs)
may be added in a later expansion.

## All-sky maps

### All-sky frame with overlays
- guide: frames
- api: allsky_figure, add_plane_overlay, add_constellation_boundaries
Elliptical full-sky frames in any projection, with coordinate-plane and
constellation overlays.

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_plane_overlay(ax, plane="ecliptic", color="C1", label="Ecliptic")
sph.add_constellation_boundaries(ax)
sph.add_constellation_labels(ax)
```

### Second coordinate grid
- guide: ticks
- api: add_second_grid
Another coordinate system's graticule drawn over the primary frame —
here galactic gridlines on an ICRS Mollweide map.

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="MOL", center=180)
# A warm, dashed overlay so the galactic graticule can't be mistaken for
# the frame's own (neutral, solid) equatorial grid.
sph.add_second_grid(ax, overlay_frame="galactic", color="C1", alpha=0.95,
                    linewidth=1.1, linestyle="--")
```

### Projection gallery
- guide: frames
- api: projection_gallery, bin_data_as_healpix
The same all-sky map through several projections at once — the full FITS
set plus non-FITS cartographers' projections like Robinson.

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(4)
# A uniform sky plus a galactic-plane overdensity, so the demo map has
# real structure to carry through each projection.
# Center the overdensity on the gallery center (180) so the zenithal
# panels — which only ever show a hemisphere — have structure in frame.
lon = np.concatenate([rng.uniform(0, 360, 40000),
                      rng.normal(180, 22, 30000) % 360])
lat = np.concatenate([np.degrees(np.arcsin(rng.uniform(-1, 1, 40000))),
                      np.clip(rng.normal(-12, 12, 30000), -89, 89)])
demo, *_ = sph.bin_data_as_healpix(lon, lat, np.ones_like(lon),
                                   nside=16, statistic="count", blank_value=0)
# A spread across the families: elliptical and pseudocylindrical all-sky,
# a cylindrical, the HEALPix quad-cube, and the zenithal projections (which
# by nature show only part of the sky).
fig, axes = sph.projection_gallery(
    demo, projections=["AIT", "MOL", "robinson", "CAR", "HPX", "SFL",
                       "TAN", "SIN", "ZEA"],
    center=180, ncols=3, cmap="sph.deepsky")
# A thin graticule on each panel — the whole point of the comparison is how
# each projection deforms the coordinate grid, which a bare map can't show.
# The map has to be pushed below the gridlines (WCSAxes draws its grid at a
# fixed stage, so the image would otherwise cover it), and the tick labels
# are dropped: unreadable at this size and they clutter the curved frames.
for ax in np.ravel(axes):
    for art in list(ax.collections) + list(ax.images):
        art.set_zorder(-10)
    try:
        for c in (0, 1):
            ax.coords[c].grid(True, color="0.85", linewidth=0.6, alpha=0.95,
                              linestyle="-")
            ax.coords[c].set_ticklabel_visible(False)
            ax.coords[c].set_axislabel("")
    except Exception:          # non-WCS frames (robinson & co.)
        sph.style_grid(ax, color="0.85", lw=0.6, alpha=0.95, ls="-")

# Some frames draw their in-frame labels as separate tagged artists, which
# set_ticklabel_visible doesn't reach — delete them after a draw.
fig.canvas.draw()
for ax in np.ravel(axes):
    for art in list(ax.lines) + list(ax.texts) + list(ax.collections):
        if getattr(art, "_sph_overlay_ticklabel", False):
            art.remove()
    for txt in list(ax.texts):
        txt.remove()
```

### Image on the sky
- guide: images
- api: load_sky_image, reproject_background
- data: examples
Resampling a sky panorama onto a curved projection — the all-sky NOIRLab
image reprojected onto a galactic Aitoff frame.

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

pano = "examples/data/Allsky_noirlab2430b_1280x640.jpg"
img, hdr = sph.load_sky_image(pano, frame="galactic", center=0)
fig = plt.figure(figsize=(6.4, 3.4))
ax = sph.make_wcs_frame(111, "AIT", center=0, frame="galactic",
                        npix=(1200, 600), fig=fig)
ax.imshow(sph.reproject_background(img, hdr, ax))
sph.add_plane_overlay(ax, plane="ecliptic", color="C1", lw=1.0)
```

### Insets — a zoom and a circular cutout
- guide: globe
- api: reproject_inset_axes, mark_inset_axes, connect_inset_axes
- data: examples
Two detail views on one all-sky catalog map: a rectangular tangent-plane zoom
(boxed and connected) resolves a dense knot, and a circular orthographic cutout
drops the NOIRLab Milky Way panorama with the catalog scattered over it, marked
by a circle and its two tangent connectors.

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skyplothelper as sph
from astropy.visualization.wcsaxes.frame import EllipticalFrame

# Mode-aware decoration colors (the gallery renders one snippet light + dark).
from matplotlib.colors import to_rgb
_r, _g, _b = to_rgb(mpl.rcParams["figure.facecolor"])
DARK = 0.299 * _r + 0.587 * _g + 0.114 * _b < 0.5
PAL = sph.ANNOTATION_PALETTES["dark" if DARK else "publication"]
MARK = PAL["accent2"]        # rectangular-zoom marker + connectors, on-theme
WARM = "#F5B301"             # amber reads over the (always-dark) Milky Way image

# A synthetic all-sky source catalog in galactic coords, plus a dense knot for
# the rectangular zoom to resolve.
rng = np.random.default_rng(11)
n = 620
l = rng.uniform(-180, 180, n)
b = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
KNOT = (-58.0, 26.0)
nk = 120
lk = KNOT[0] + rng.normal(0, 1.1, nk) / np.cos(np.radians(KNOT[1]))
bk = KNOT[1] + rng.normal(0, 1.1, nk)
cat = {"l": np.r_[l, lk], "b": np.r_[b, bk],
       "z": 10 ** rng.uniform(-2.0, -0.8, n + nk),
       "bright": 10 ** rng.uniform(0, 2, n + nk)}

# A smooth density field peaking on the knot, contoured in yellow - showing an
# inset is a normal Axes that takes the same overlays as the parent.
gl, gb = np.meshgrid(np.linspace(-180, 180, 361), np.linspace(-90, 90, 181))
_dl = ((gl - KNOT[0] + 180) % 360 - 180) * np.cos(np.radians(gb))
dens = np.exp(-0.5 * ((_dl / 5.0) ** 2 + ((gb - KNOT[1]) / 5.0) ** 2))

def science(target, smin, smax):
    sc = sph.plot_catalog(target, cat, lon_col="l", lat_col="b", frame="galactic",
                          colorby="z", sizeby="bright", cmap="sph.dusk",
                          size_scale="sqrt", smin=smin, smax=smax,
                          vmin=0.01, vmax=0.16, alpha=0.85)
    sph.add_contour_overlay(target, gl, gb, dens, levels=[0.3, 0.6],
                            colors="gold", linewidths=1.4, alpha=0.9)
    return sc

# Parent: all-sky galactic map of the catalog.
panorama, pano_hdr = sph.load_sky_image(
    "examples/data/Allsky_noirlab2430b_1280x640.jpg", frame="galactic", center=0)
fig = plt.figure(figsize=(13, 6.5))
ax = sph.make_wcs_frame(111, projection="AIT", center=0, frame="galactic", fig=fig)
science(ax, smin=4, smax=55)
ax.set_title("Two insets on one all-sky field: a rectangular zoom and a circular cutout",
             fontsize=12)

# Right: rectangular TAN zoom on the knot (marked + connected).
zoom = sph.reproject_inset_axes(ax, rect=[0.70, 0.10, 0.28, 0.40], transform="figure",
                                projection="TAN", center=KNOT, size=(14, 12))
science(zoom, smin=16, smax=110)
for c in (0, 1):
    zoom.coords[c].axislabels.set_visible(False)
sph.mark_inset_axes(ax, zoom, edgecolor=MARK, linewidth=1.8)
sph.connect_inset_axes(ax, zoom, color=MARK, linewidth=1.3, corners='diagonal')

# Left: circular SIN inset on a second (galactic-plane) patch, showing the
# NOIRLab panorama with the catalog scattered over it.
PATCH = (48.0, 0.0)
circ = sph.reproject_inset_axes(ax, rect=[0.03, 0.10, 0.26, 0.52], transform="figure",
                                projection="SIN", center=PATCH, size=45, npix=1000,
                                frame_class=EllipticalFrame)
circ.imshow(sph.reproject_background(panorama, pano_hdr, circ), origin="lower")
science(circ, smin=8, smax=70)
for c in (0, 1):
    circ.coords[c].set_ticklabel(color="0.9")
    circ.coords[c].axislabels.set_visible(False)
sph.mark_inset_axes(ax, circ, style="circle", center=PATCH, radius=20,
                    edgecolor=WARM, linewidth=1.8)
sph.connect_inset_axes(ax, circ, color=WARM, linewidth=1.2)
fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.04)
```

## Ticks & grids

### Tick label styles & conventions
- guide: ticks
- api: format_ticklabels, make_wcs_frame
One call — `format_ticklabels(ax, style=...)` — swaps the whole labeling
convention (sexagesimal superscripts, a colon convention, decimal degrees, or
relative offsets); on a curved frame `make_wcs_frame(tick_rotation=...)` sets
whether the labels ride the gridline tangent or lie flat.

```python
import skyplothelper as sph
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 7.2))
# One call - sph.format_ticklabels(ax, style=...) - swaps the whole labeling
# convention: sexagesimal superscripts, a colon convention, decimal degrees, or
# relative offsets. Only the last needs a matched (mas-scale) field of view.
presets = [
    ("publication", None),   # sexagesimal h/m/s superscripts (the default)
    ("casa",        None),   # colon convention, 05:42:00
    ("decimal",     None),   # decimal degrees
    ("offset_mas",  4e-4),   # relative offsets in mas (a +/-40 mas field)
]
for i, (style, cdelt_asec) in enumerate(presets, start=1):
    kw = dict(projection="TAN", center=(180.0, 30.0), fig=fig)
    if cdelt_asec is not None:
        kw.update(cdelt=cdelt_asec / 3600.0, npix=200)
    ax = sph.make_wcs_frame((2, 3, i), **kw)
    sph.format_ticklabels(ax, style=style)
    ax.set_title("style=%r" % style, fontsize=9)
# On a curved frame the labels ride the gridline tangent by default, or lie
# flat - set once at build time with tick_rotation=.
for j, rot in enumerate(["tangent", "horizontal"], start=5):
    ax = sph.make_wcs_frame((2, 3, j), projection="SIN", center=(60.0, 35.0),
                            fig=fig, tick_rotation=rot)
    ax.set_title("tick_rotation=%r" % rot, fontsize=9)
fig.suptitle("Tick label styles & conventions", fontsize=13)
fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.05,
                    hspace=0.45, wspace=0.40)
```

### Offset coordinate ticks
- guide: ticks
- api: apply_offset_ticks, make_wcs_frame
Relative offset labels (Δα cos δ, Δδ) about a reference position, with
units that walk from degrees down to μas as the field shrinks.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(7)
fig = plt.figure(figsize=(4.8, 4.4))
ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01), fov_deg=0.2, fig=fig)
ra = 83.63 + (rng.random(30) - 0.5) * 0.16
dec = 22.01 + (rng.random(30) - 0.5) * 0.16
ax.scatter(ra, dec, transform=ax.get_transform("world"), s=14, color="C0")
sph.apply_offset_ticks(ax, ref_ra_deg=83.63, ref_dec_deg=22.01)
```

### Highlighted gridlines
- guide: ticks
- api: highlight_gridlines
Emphasizing individual meridians and parallels — a colormapped family of
parallels and a few accented meridians — without redrawing the frame.

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="MOL", center=180)
sph.highlight_gridlines(ax, lat_values=list(range(-60, 61, 30)),
                        lat_cmap="sph.dusk", lw=2.2)
sph.highlight_gridlines(ax, lon_values=[0, 90, 180, 270], color="C1", lw=1.6)
```

### Coordinate overlay grid
- guide: ticks
- api: add_coord_overlay
A second coordinate system's full graticule — styled gridlines, ticks, and
labels — laid over the primary frame (galactic over an ICRS map).

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_coord_overlay(ax, frame="galactic", color="C1", lw=0.8, alpha=0.8)
```

## Fields & images

### Tangent-plane field
- guide: frames
- api: make_wcs_frame, add_compass
A zoomed gnomonic (TAN) field on a target, with catalog points and a
compass.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(7)
fig = plt.figure(figsize=(4.8, 4.4))
ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01), fov_deg=0.2, fig=fig)
ra = 83.63 + (rng.random(40) - 0.5) * 0.16
dec = 22.01 + (rng.random(40) - 0.5) * 0.16
ax.scatter(ra, dec, transform=ax.get_transform("world"), s=14, color="C0")
sph.add_compass(ax)
```

### FITS image display
- guide: images
- api: make_norm, squeeze_image, add_beam
- data: examples
A real FITS image on WCS axes with an asinh/zscale stretch and the
synthesized beam (VLBA 15 GHz image of 3C 84 from the MOJAVE program).

```python
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import skyplothelper as sph

hdu = fits.open("examples/data/0316+413.u.stacked.icd.fits")[0]
data, hdr = sph.squeeze_image(hdu.data, hdu.header)   # drop degenerate axes
wcs = WCS(hdr).celestial
fig = plt.figure(figsize=(4.8, 4.6))
ax = fig.add_subplot(111, projection=wcs)
# symlog: linear within ~5 mJy of zero, logarithmic out to the ~3 Jy peak.
# This is the look the MOJAVE survey publishes, and it reveals the jet's
# full dynamic range where a linear or zscale stretch shows only the core.
ax.imshow(data, origin="lower", cmap="sph.deepsky",
          norm=sph.make_norm(stretch="symlog", clip="manual", vmin=0,
                             vmax=float(np.nanmax(data)), a=5e-3))
# All the structure sits within a few tens of mas of the core: zoom there.
cx, cy = wcs.world_to_pixel_values(hdr["CRVAL1"], hdr["CRVAL2"])
half = 20.0 / (abs(hdr["CDELT2"]) * 3.6e6)
ax.set_xlim(cx - half, cx + half)
ax.set_ylim(cy - half, cy + half)
ax.coords[0].set_ticks(number=4)
ax.coords[1].set_ticks(number=5)
ax.coords[0].set_ticklabel(size=8)
ax.coords[1].set_ticklabel(size=8)
ax.coords[0].set_axislabel("RA (J2000)", fontsize=9)
ax.coords[1].set_axislabel("Dec (J2000)", fontsize=9)
sph.add_beam(ax, hdr, facecolor="white", edgecolor="white")
```

### Basic RGB composite
- guide: images
- api: rescale_image
- data: examples
A three-color composite from separate filter images (NGC 602 in
HST B / R / IR), each stretched and stacked into an RGB frame.

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import skyplothelper as sph

def band(path):
    d = np.squeeze(fits.getdata(path)).astype(float)
    lo, hi = np.nanpercentile(d, [1.0, 99.6])
    return np.clip((d - lo) / (hi - lo), 0, 1)

rgb = np.dstack([band("examples/data/ngc602_IR.fits"),
                 band("examples/data/ngc602_R.fits"),
                 band("examples/data/ngc602_B.fits")]) ** 0.8
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.imshow(rgb, origin="lower")
ax.set_xticks([])
ax.set_yticks([])
```

### Advanced RGB composite
- guide: images
- api: rescale_image
- data: examples
Four narrow-band frames combined with
[`multicolorfits`](https://github.com/pjcigan/multicolorfits): each band is
scaled to grayscale, tinted its own hue, then added together — so emission
lines keep distinct colors instead of being forced into R/G/B channels.
SN 1987A in four HST filters.

```python
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import multicolorfits as mcf
import skyplothelper as sph

BANDS = ["sn1987a_hst_F625W.fits", "sn1987a_hst_F656N.fits",
         "sn1987a_hst_F658N.fits", "sn1987a_hst_F502N.fits"]
COLORS = ["#FBFCCF", "#729FCF", "#75507B", "#EF2929"]   # continuum, Ha, [N II], [O III]

hdus = [fits.open(f"examples/data/{b}")[0] for b in BANDS]
wcs = WCS(hdus[0].header).celestial
# 1. scale each band  2. tint it  3. add the tinted frames together
gray = [mcf.to_grey_rgb(np.squeeze(h.data).astype(float), rescalefn="log",
                        scaletype="perc", min_max=[40, 99.9]) for h in hdus]
rgb = mcf.combine_multicolor(
    [mcf.colorize_image(g, c, colorintype="hex", gammacorr_color=2.2)
     for g, c in zip(gray, COLORS)], gamma=2.2)

LABELS = ["F625W (continuum)", "Hα (F656N)", "[N II] (F658N)", "[O III] (F502N)"]

# The gallery renders this snippet in both modes, so pick the legend
# palette off the active figure background rather than hard-coding it.
from matplotlib.colors import to_rgb
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
fig = plt.figure(figsize=(4.8, 4.6))
ax = fig.add_subplot(111, projection=wcs)
ax.imshow(rgb, origin="lower")
ax.coords[0].set_axislabel("RA (J2000)", fontsize=9)
ax.coords[1].set_axislabel("Dec (J2000)", fontsize=9)
ax.coords[0].set_ticklabel(size=8)
ax.coords[1].set_ticklabel(size=8)
# Key which hue came from which filter — round swatches in the tint colors.
(sph.MultiLegend(ax, loc="lower left", palette=LP, fontsize=6.5,
                 framealpha=0.75)
    .add_color("Filter", dict(zip(LABELS, COLORS)), swatch="marker")
    .draw())
```

### Colorbar styles
- guide: images
- api: add_colorbar, rescale_image
A stretched image with a WCS-aware colorbar — `add_colorbar` sizes to a
fixed-aspect image axes correctly where a bare `plt.colorbar` would not.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(2)
ny = nx = 220
yy, xx = np.mgrid[0:ny, 0:nx]
# A diffuse extended source plus compact knots on a noise floor: something
# with real dynamic range for the stretch and the colorbar to describe.
img = 180 * np.exp(-((((xx - 108) / 46.) ** 2 + ((yy - 104) / 30.) ** 2)) ** 1.1)
for sx, sy, amp, w in [(58, 150, 90, 3.4), (166, 66, 130, 2.8),
                       (150, 158, 55, 2.2), (78, 62, 70, 2.6),
                       (196, 128, 40, 2.0)]:
    img += amp * np.exp(-(((xx - sx) / w) ** 2 + ((yy - sy) / w) ** 2))
img += rng.normal(0, 1.6, img.shape)
fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(img, origin="lower", cmap="sph.lagoon",
               norm=sph.make_norm(stretch="asinh", clip="percentile",
                                  phi=99.5, data=img))
ax.set_xticks([])
ax.set_yticks([])
sph.add_colorbar(im, ax=ax, mode="divider", label="surface brightness")
```

### Contour overlays
- guide: images
- api: add_contour_overlay
World-coordinate contours drawn on a WCS frame — for tracing structure or
overlaying one dataset on another.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(1)
fig = plt.figure(figsize=(4.8, 4.4))
ax = sph.make_wcs_frame(111, "TAN", center=(150.0, 2.2), fov_deg=0.4, fig=fig)
gx, gy = np.meshgrid(np.linspace(-0.18, 0.18, 120), np.linspace(-0.18, 0.18, 120))
blob = np.exp(-((gx - 0.03) ** 2 + gy ** 2) / 0.004) \
    + 0.7 * np.exp(-((gx + 0.05) ** 2 + (gy - 0.04) ** 2) / 0.002)
lon = 150.0 + gx / np.cos(np.radians(2.2))
lat = 2.2 + gy
sph.add_contour_overlay(ax, lon, lat, blob, levels=7, cmap="sph.deepsky",
                        linewidths=1.1)
```

### Signed data & symmetric log
- guide: images
- api: make_norm
A residual/Stokes-style image with positive and negative values, shown
through a symmetric-log norm (linear core, log wings) so the colorbar reads
true values.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(3)
gx, gy = np.meshgrid(np.linspace(-1, 1, 160), np.linspace(-1, 1, 160))
signed = (np.exp(-((gx - 0.3) ** 2 + gy ** 2) / 0.03)
          - np.exp(-((gx + 0.3) ** 2 + gy ** 2) / 0.03)) \
    + 0.05 * rng.standard_normal(gx.shape)
lim = float(np.nanmax(np.abs(signed)))
fig, ax = plt.subplots(figsize=(4.8, 4.0))
im = ax.imshow(signed, origin="lower", cmap="sph.diff_blueorange",
               norm=sph.make_norm(stretch="symlog", clip="manual",
                                  vmin=-lim, vmax=lim, a=0.05))
ax.set_xticks([])
ax.set_yticks([])
sph.add_colorbar(im, ax=ax, mode="divider", label="residual")
```

### Data cube channel maps
- guide: images
- api: channel_map, ChannelMapResult
- data: examples
`channel_map` turns a spectral-line cube (the DDO 70 HI sub-cube) into a
uniform panel grid on one shared normalization — velocity labels, the shared
colorbar, and sparse ticks all handled in one call.

```python
import skyplothelper as sph

res = sph.channel_map("examples/data/ddo70_hi_subcube.fits",
                      channels=9, ncols=3, cmap="sph.dusk")
fig = res.fig
```

### Quicklook in one call
- guide: images
- api: quicklook_fits
- data: examples
The standard radio-map recipe — open, stretch, frame, annotate — from a
single call on a FITS path (VLBA 15 GHz image of 3C 84).

```python
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import skyplothelper as sph

path = "examples/data/0316+413.u.stacked.icd.fits"
peak = float(np.nanmax(np.squeeze(fits.getdata(path))))
# Hand quicklook the same symmetric-log norm the survey uses, ask for
# relative-mas axes, and crop to the jet — the standard radio-map look.
# quicklook builds its own figure, so hand it the active theme's colors —
# otherwise it lands on a white canvas in dark mode. Let it size itself:
# forcing set_size_inches afterwards leaves dead canvas around the layout.
ink = plt.rcParams["text.color"]
res = sph.quicklook_fits(path, image=True, colormap="sph.deepsky",
                         color="white", offset_coords=True, field_size=34,
                         facecolor=plt.rcParams["figure.facecolor"],
                         axcolor=ink, info_color=ink,
                         norm=sph.make_norm(stretch="symlog", clip="manual",
                                            vmin=0, vmax=peak, a=5e-3))
fig = res.fig
```

## Catalogs & legends

### Catalog scatter
- guide: vectors
- api: plot_catalog
Drop any table on the sky and encode columns in marker color and size,
with a matched colorbar — column names are auto-detected.

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(6)
n = 500
cat = {"ra": rng.uniform(0, 360, n),
       "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
       "z": rng.random(n) ** 2 * 0.2,
       "flux": 10 ** rng.uniform(0, 2, n)}
fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.plot_catalog(ax, cat, colorby="z", sizeby="flux", size_scale="log",
                 cmap="sph.sunset", cbar=True, cbar_label="redshift")
```

### Stars in perceived colors
- guide: vectors
- api: color_index_to_rgb
- data: examples
Color a star catalog by each star's *perceived* color from a named color index
with `color_index_to_rgb` (hot stars blue-white, cool stars orange) — Gaia BP-RP
for the Pleiades here, sized by brightness, on a night-sky canvas.

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import skyplothelper as sph

# The Pleiades (M45) from a bundled Gaia catalog, colored by each star's real
# perceived color. color_index_to_rgb takes any named index; here Gaia BP-RP
# (sph.bp_rp_to_rgb is the equivalent shortcut for this one).
gaia = Table.read("examples/data/query_cache/gaia_m45.ecsv")
colors = sph.color_index_to_rgb(gaia["BP-RP"], index="BP-RP", saturation=1.0)
size = np.clip(200 * 10 ** (-0.4 * (gaia["Gmag"] - 6.0)), 2.0, 240.0)

fig = plt.figure(figsize=(6.4, 5.6))
ax = sph.make_wcs_frame(111, "TAN", center=(56.75, 24.12), fov_deg=2.7, fig=fig)
ax.set_facecolor("#05060a")   # a night-sky canvas, dark in both display modes
for c in (0, 1):              # light ticks/labels so they read on the black sky
    ax.coords[c].set_ticklabel(color="0.8", size=8)
    ax.coords[c].set_ticks(color="0.5")
ax.scatter(gaia["RA_ICRS"], gaia["DE_ICRS"], transform=ax.get_transform("world"),
           c=colors, s=size, lw=0, alpha=0.9, zorder=3)
ax.set_title("The Pleiades in perceived colors (color_index_to_rgb, BP-RP)",
             fontsize=11)
```

### Multi-channel legend
- guide: legends
- api: MultiLegend, plot_catalog
`MultiLegend` keys several visual channels at once — here marker size
(number of observations), shape (catalog class) and color (observing
band) — placed off-frame.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(8)
n = 300
ra = rng.uniform(0, 360, n)
dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
nobs = 10 ** rng.uniform(0, 3, n)
defining = rng.random(n) < 0.3
BANDS = {"S/X": "C0", "K": "C1", "Q": "C2"}
band = np.array(list(BANDS.values()))[rng.integers(0, 3, n)]

# The gallery renders this snippet in both modes, so pick the legend
# palette off the active figure background rather than hard-coding it.
from matplotlib.colors import to_rgb
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
fig, ax = sph.allsky_figure(projection="AIT", center=180)
for is_def, mk in [(False, "o"), (True, "^")]:
    m = defining == is_def
    cp = sph.plot_catalog(ax, {"ra": ra[m], "dec": dec[m], "nobs": nobs[m]},
                          sizeby="nobs", size_vlim=(1, 1000), size_scale="sqrt",
                          smin=6, smax=180, marker=mk, color=band[m], alpha=0.75)
# Three independent channels at once: size, shape and color.
(sph.MultiLegend(ax, palette=LP, loc="outside bottom", orientation="horizontal")
    .add_size_from(cp, values=[1, 10, 100, 1000], title="N obs")
    .add_shape("Class", {"Defining": "^", "Other": "o"})
    .add_color("Band", BANDS, swatch="marker")
    .draw())
```

### Channel-block catalog
- guide: legends
- api: MultiLegend
Every `MultiLegend` block kind on one figure — the marker/line channels
(color, shape, size, edge, fill, alpha, angle, line) and the specialty
swatches (hatch, region, sph reticle glyphs, a colorbar strip, text, and any
custom artist). A catalog of what you can key, not a template to fill.

```python
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
import skyplothelper as sph

# The gallery renders this snippet in both modes, so pick the legend palette
# off the active figure background rather than hard-coding it.
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
A = sph.CYCLE_PALETTES["atlas"]["colors"]
frame = plt.rcParams["axes.edgecolor"]
fig, (axA, axB) = plt.subplots(2, 1, figsize=(12, 4.6))
for a in (axA, axB):
    a.axis("off")

# Channels that vary a marker or line property.
(sph.MultiLegend(axA, loc="center", orientation="horizontal", block_sep=18, palette=LP)
    .add_color("Color", {"A": A[0], "B": A[1], "C": A[2]}, swatch="marker")
    .add_shape("Shape", {"disk": "o", "gal": "D", "star": "*"})
    .add_size("Size", values=[1, 100, 10000], smin=8, smax=230, scale="sqrt", fmt=".0f")
    .add_edge("Edge", {"secure": A[2], "flagged": A[3]})
    .add_fill("Fill", {"detected": "filled", "limit": "open"})
    .add_alpha("Alpha", values=[1, 5, 20], fmt=".0f")
    .add_orientation("Angle", {"0°": 0, "60°": 60, "120°": 120})
    .add_line("Line", {"fit": "--", "prior": ":"})
    .draw())

# Specialty swatch kinds — hatch, region, sph reticle glyphs, a colorbar
# strip, text, and any matplotlib artist you hand in.
star = Line2D([0], [0], marker=(6, 1, 0), markersize=12, linestyle="none",
              markerfacecolor=A[1], markeredgecolor=frame)
(sph.MultiLegend(axB, loc="center", orientation="horizontal", block_sep=18, palette=LP)
    .add_fill("Hatch", {"DES": "///", "LSST": "xxx"}, kind="patch", color=A[0])
    .add_region("Region", {"footprint": dict(fc=A[0], ec=A[0], alpha=0.35),
                           "mask": dict(fc=A[3], ec=A[3], hatch="//")})
    .add_glyph("Glyph", {"target": "reticle_circle", "mark": "crosshair"})
    .add_colorbar("Redshift", cmap="sph.deepsky", vmin=0, vmax=2, length=90, fmt=".1f")
    .add_text("Text", ["dashed = model"])
    .add_custom("Custom", {"my marker": star})
    .draw())
axA.set_title("Channels that vary a marker or line", fontsize=10, y=0.97)
axB.set_title("Specialty swatch kinds — hatch, region, sph glyphs, colorbar, text, custom",
              fontsize=10, y=0.97)
```

### Legend placement
- guide: legends
- api: MultiLegend
Where the key sits: anchored inside the axes, pushed into the figure
margin with an `"outside …"` preset, or at a free `(x, y)` anchor. The
outside presets are the all-sky selling point — the map keeps the whole
frame and the key sits beside it.

```python
import matplotlib.pyplot as plt
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(7)
n = 150
ra = rng.uniform(0, 360, n)
dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
survey_a = rng.random(n) < 0.55

# The gallery renders this snippet in both modes, so pick the legend
# palette off the active figure background rather than hard-coding it.
from matplotlib.colors import to_rgb
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
fig = plt.figure(figsize=(11, 3.0))
for i, (loc, orient) in enumerate([("lower right", "vertical"),
                                   ("outside bottom", "horizontal"),
                                   ("outside right", "vertical")]):
    ax = sph.make_wcs_frame(int(f"13{i + 1}"), "AIT", center=180)
    ax.scatter(ra[survey_a], dec[survey_a], s=10, color="C0", alpha=0.65,
               transform=ax.get_transform("world"))
    ax.scatter(ra[~survey_a], dec[~survey_a], s=10, color="C1", alpha=0.65,
               transform=ax.get_transform("world"))
    (sph.MultiLegend(ax, palette=LP, loc=loc, orientation=orient, fontsize=6)
        .add_color("survey", {"A": "C0", "B": "C1"}, swatch="marker")
        .draw())
    ax.set_title(f'loc="{loc}"', fontsize=8)
fig.subplots_adjust(wspace=0.35)
```

### Keying marker size
- guide: legends
- api: MultiLegend, plot_catalog
Two routes to a size key. `plot_catalog(size_legend=True)` draws its own
matplotlib key inside the axes — quick, when size is the only extra
dimension. `MultiLegend.add_size_from` reads the scaling off the same
plot, so swatches reproduce on-plot sizes, auto-picks round 1/2/5
representatives, and sits alongside other channels off-frame.

```python
import matplotlib.pyplot as plt
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(7)
n = 150
cat = {"ra": rng.uniform(0, 360, n),
       "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
       "n_obs": rng.integers(1, 60, n)}

# The gallery renders this snippet in both modes, so pick the legend
# palette off the active figure background rather than hard-coding it.
from matplotlib.colors import to_rgb
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
fig = plt.figure(figsize=(10, 3.2))
ax1 = sph.make_wcs_frame(121, "AIT", center=180)
sph.plot_catalog(ax1, cat, sizeby="n_obs", size_scale="sqrt", smin=6,
                 smax=170, color="C0", alpha=0.7, size_legend=True,
                 size_legend_num=3,
                 size_legend_kwargs=dict(loc="lower left", title="N obs",
                                         fontsize=7))
ax1.set_title("plot_catalog(size_legend=True)", fontsize=9)

ax2 = sph.make_wcs_frame(122, "AIT", center=180)
cp = sph.plot_catalog(ax2, cat, sizeby="n_obs", size_scale="sqrt", smin=6,
                      smax=170, color="C2", alpha=0.7)
(sph.MultiLegend(ax2, palette=LP, loc="outside bottom", orientation="horizontal",
                 fontsize=7)
    .add_size_from(cp, title="N obs")
    .add_shape("class", {"star": "o", "galaxy": "^"}, size=6)
    .draw())
ax2.set_title("MultiLegend.add_size_from(...)", fontsize=9)
fig.subplots_adjust(wspace=0.3)
```

### Standalone legend blocks
- guide: legends
- api: LegendBlock, MultiLegend
Every `add_*` wrapper has a block class behind it. Build
`LegendBlock`s directly — picking the swatch renderer with
`swatch_kind` (`line`, `patch`, `marker`, `region`, `text`, …) — and
attach them with `add_block`, mixing them with named glyph swatches so
one key describes curves, filled regions and reticles together.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph
from skyplothelper import LegendBlock

# The gallery renders this snippet in both modes, so pick the legend
# palette off the active figure background rather than hard-coding it.
from matplotlib.colors import to_rgb
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_great_circle(ax, pole_lon=0, pole_lat=90, frame="pole",
                     color="C1", ls="--", lw=2)
sph.add_great_circle(ax, pole_lon=90, pole_lat=30, frame="pole",
                     color="C2", ls=":", lw=2)
sph.add_geodesic_circle(ax, 300, 30, 25, facecolor="C0", alpha=0.35,
                        edgecolor="none")
sph.add_geodesic_circle(ax, 90, -25, 20, facecolor="C3", alpha=0.35,
                        edgecolor="none")
sph.add_reticle(ax, (180, 10), style="plus", size=14, color="0.35", lw=1.6)

lines = LegendBlock("Models", {"model A": dict(ls="--", lw=2, color="C1"),
                               "model B": dict(ls=":", lw=2, color="C2")},
                    swatch_kind="line")
regions = LegendBlock("Regions", {"observed": dict(facecolor="C0", alpha=0.35),
                                  "planned": dict(facecolor="C3", alpha=0.35)},
                      swatch_kind="patch")
(sph.MultiLegend(ax, palette=LP, loc="outside bottom", orientation="horizontal",
                 fontsize=7)
    .add_block(lines)
    .add_block(regions)
    .add_glyph("targets", {"primary": "reticle_plus"})
    .draw())
```

## Regions & footprints

### Compound region
- guide: regions
- api: CompoundRegion
Set algebra on the sphere: the galactic band — wrapping the whole sky and
crossing the antimeridian seam — with a cap *merged into* it and a square
punched out of the overlap, rendered as one visibly complex, seam-aware
region.

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
reg = (sph.CompoundRegion(ax)
       .add_frame_band(-13, 13, frame="galactic")            # the galactic band, across the seam
       .add_circle(lon=285, lat=-12, radius_deg=34)          # a cap that merges into the band
       .subtract_square(lon=276, lat=-20, size=15, angle=20))  # a square hole inside the overlap
reg.render(facecolor="C0", alpha=0.32)
reg.render_boundary(linewidth=1.2)
```

### Region membership
- guide: regions
- api: CompoundRegion
Point-in-region queries: `contains_points` classifies a catalog against
any compound region — here a cap with the galactic band removed, so
sources in the stripe fall *outside* — coloring members and non-members
distinctly.

```python
import numpy as np
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
reg = (sph.CompoundRegion(ax)
       .add_circle(lon=95, lat=12, radius_deg=42)
       .subtract_frame_band(-13, 13, frame="galactic"))
reg.render(facecolor="C0", alpha=0.14)
reg.render_boundary(color="C0", linewidth=1.3)

rng = np.random.default_rng(6)
lon = rng.uniform(20, 175, 700)
lat = rng.uniform(-40, 60, 700)
inside = reg.contains_points(lon, lat)   # boolean mask, one per source
tr = ax.get_transform("world")
ax.scatter(lon[~inside], lat[~inside], transform=tr, s=7, color="0.72", label="outside")
ax.scatter(lon[inside], lat[inside], transform=tr, s=15, color="#d1495b",
           edgecolor="w", lw=0.35, label="inside")
ax.legend(loc="lower right", fontsize=8)
```

### Masking data to a region
- guide: regions
- api: CompoundRegion, clip_to_land, clip_to_ocean
A region as a cookie-cutter: `clip()` masks any artist — an image, a data
field, a scatter — to a region's shape, so data shows only where it falls
*inside*. Here a smooth all-sky field is cut to a survey footprint (a box,
minus the galactic plane, with a hole punched). This is the sky counterpart
of the `clip_to_land` / `clip_to_ocean` helpers on Earth maps.

```python
import numpy as np
import skyplothelper as sph

# a smooth, large-scale all-sky field
lon = np.linspace(0, 360, 480)
lat = np.linspace(-90, 90, 240)
LON, LAT = np.meshgrid(lon, lat)
dlon = (LON - 150 + 180) % 360 - 180
field = np.exp(-(dlon**2 + (LAT - 25)**2) / (2 * 45**2)) + 0.3 * np.cos(np.radians(LAT))

fig, ax = sph.allsky_figure(projection="MOL", center=180)
footprint = (sph.CompoundRegion(ax)
             .add_lonlat_box(lat_min=-12, lat_max=70, lon_min=110, lon_max=260, frame="icrs")
             .subtract_frame_band(-25, 25, frame="galactic")   # avoid the plane
             .subtract_circle(180, 35, radius_deg=8))          # punch a hole
mesh = ax.pcolormesh(LON, LAT, field, transform=ax.get_transform("world"),
                     cmap="sph.deepsky", shading="gouraud")
footprint.clip(mesh)                       # mask the field to the footprint shape
footprint.render_boundary(color="white", linewidth=1.1)
```

### Region shapes
- guide: regions
- api: add_rectangle, add_ellipse, add_annulus, add_latitude_band
The shape family — rectangles, ellipses, annuli, and coordinate bands —
each seam-aware and correct at the poles.

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_rectangle(ax, lon=60, lat=35, width=44, height=26,
                  facecolor="C0", edgecolor="C0", alpha=0.35)
sph.add_ellipse(ax, lon=185, lat=-25, semi_major=32, semi_minor=15, angle=25,
                facecolor="C1", edgecolor="C1", alpha=0.35)
sph.add_annulus(ax, lon=300, lat=30, inner_radius=8, outer_radius=18,
                facecolor="C2", edgecolor="C2", alpha=0.4)
sph.add_latitude_band(ax, -10, 10, facecolor="C3", edgecolor="none", alpha=0.3)
```

### Survey footprints
- guide: overlays
- api: add_survey_footprint
Sky coverage of named surveys from the bundled, queryable footprint
catalog.

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="MOL", center=180)
sph.add_survey_footprint(ax, survey="des", label="DES", color="C0")
sph.add_survey_footprint(ax, survey="euclid", label="Euclid", color="C1")
ax.legend(loc="lower right")
```

### Tissot indicatrices
- guide: regions
- api: tissot
Equal-radius geodesic circles across the map make a projection's local
distortion visible at a glance.

```python
import numpy as np
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="MOL", center=180)
lons, lats = np.meshgrid(np.arange(30, 331, 60), np.arange(-60, 61, 30))
sph.tissot(ax, rad_deg=6, lons=lons.ravel(), lats=lats.ravel(),
           facecolor="C0", edgecolor="C0", alpha=0.4)
```

## HEALPix

### Source-density map
- guide: healpix
- api: sources_to_healpix_plot
A synthetic catalog binned into HEALPix pixels and rendered all-sky.

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(5)
lon = np.concatenate([rng.uniform(0, 360, 8000),
                      rng.normal(266, 14, 3000) % 360])
lat = np.concatenate([np.degrees(np.arcsin(rng.uniform(-1, 1, 8000))),
                      np.clip(rng.normal(-29, 10, 3000), -89, 89)])
sph.sources_to_healpix_plot(lon, lat, nside=32, cmap="sph.thicket")
```

### Sparse HEALPix map
- guide: healpix
- api: bin_data_sparse, plot_healpix_sparse
Only the occupied pixels, drawn as equal-area tiles — the efficient way
to render a partial-sky map at any resolution.

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(12)
lon = rng.normal(200, 25, 4000) % 360
lat = np.clip(rng.normal(20, 18, 4000), -89, 89)
ipix, vals = sph.bin_data_sparse(lon, lat, np.ones_like(lon),
                                 nside=16, statistic="count")
fig, ax = sph.allsky_figure(projection="MOL", center=180)
sph.plot_healpix_sparse(ipix, vals, nside=16, ax=ax, cmap="sph.dusk",
                        show_boundaries=True, boundary_color="w",
                        boundary_lw=0.3, set_extent=False)
```

### Smoothed HEALPix map
- guide: healpix
- api: bin_data_as_healpix, healpix_smooth, plot_healpix_map
Gaussian-smoothing a binned map on the sphere before rendering it across
any projection.

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(15)
lon = rng.normal(140, 30, 6000) % 360
lat = np.clip(rng.normal(-10, 22, 6000), -89, 89)
hp_arr, *_ = sph.bin_data_as_healpix(lon, lat, np.ones_like(lon), nside=32,
                                     statistic="count", blank_value=0)
sm = sph.healpix_smooth(hp_arr, sigma_deg=3.0)
fig, ax = sph.allsky_figure(projection="MOL", center=180)
sph.plot_healpix_map(sm, ax=ax, cmap="sph.deepsky")
```

### Layered HEALPix maps
- guide: healpix
- api: bin_data_as_healpix, healpix_smooth, plot_healpix_map, plot_healpix_sparse
Two maps on one frame — a smoothed density field with the occupied tiles of
a second, sparser catalog drawn over it.

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(19)
lon = rng.normal(150, 35, 9000) % 360
lat = np.clip(rng.normal(-5, 25, 9000), -89, 89)
base, *_ = sph.bin_data_as_healpix(lon, lat, np.ones_like(lon), nside=32,
                                   statistic="count", blank_value=0)
base = sph.healpix_smooth(base, sigma_deg=4.0)
fig, ax = sph.allsky_figure(projection="MOL", center=180)
sph.plot_healpix_map(base, ax=ax, cmap="sph.deepsky")
clon = rng.normal(150, 8, 400) % 360
clat = np.clip(rng.normal(-5, 8, 400), -89, 89)
ipix, vals = sph.bin_data_sparse(clon, clat, np.ones_like(clon),
                                 nside=32, statistic="count")
sph.plot_healpix_sparse(ipix, vals, nside=32, ax=ax, cmap="sph.sunset",
                        show_boundaries=True, boundary_color="w",
                        boundary_lw=0.2, set_extent=False)
```

## Globes & planets

### Celestial globe
- guide: globe
- api: make_globe_frame, plot_scatter_globe, add_surface_compass
A hemisphere of sky as a globe, with a coordinate graticule, catalog
points, and a surface compass drawn on the sphere itself.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(3)
fig = plt.figure(figsize=(4.6, 4.6))
ax = sph.make_globe_frame(111, center_LONdeg=180, center_LATdeg=30)
lon = rng.uniform(0, 360, 200)
lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 200)))
sph.plot_scatter_globe(ax, lon, lat, s=8, color="C0")
# A surface compass sits *on the sphere*, so its east/west follow the sky
# convention and the local graticule rather than screen axes.
sph.add_surface_compass(ax, 215, -5, size_deg=16, color="C1")
```

### Earth with surface features
- guide: globe
- api: make_planet_frame, plot_coastlines, plot_tectonic_plates
An orthographic Earth in the geographic convention, with coastlines and
tectonic-plate boundaries — no cartopy required.

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

fig = plt.figure(figsize=(4.6, 4.6))
ax = sph.make_planet_frame(111, center_LONdeg=-30, center_LATdeg=25)
sph.plot_coastlines(ax)
sph.plot_tectonic_plates(ax, color="C3", lw=0.8)
```

### Filled Earth features
- guide: globe
- api: plot_tectonic_plates, plot_land, clip_to_ocean
Beyond outlines, the Earth features also *fill*. `plot_tectonic_plates(fill=True)`
draws a plate choropleth — here colored from the built-in dual-mode
`REGION_PALETTE` — over a flat Mollweide world, with coastlines for reference.
`plot_land(lakes=True)`, `plot_rivers`, and `clip_to_ocean` fill the other layers
(see the Globe & Planet tutorial). The fills route through the same seam-aware
region machinery as `add_spherical_polygon`.

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import skyplothelper as sph

# a discrete choropleth colormap from the built-in dual-mode region palette
region_cmap = ListedColormap(sph.REGION_PALETTE)

fig = plt.figure(figsize=(6.4, 3.4))
ax = sph.make_planet_frame(111, projection="MOL", center_LONdeg=0,
                           grid=True, gridcolor="0.55", gridalpha=0.3)
sph.plot_tectonic_plates(ax, fill=True, cmap=region_cmap, alpha=0.9, edgecolor="0.2")
sph.plot_coastlines(ax, color="0.15", lw=0.5)
```

### Cartopy support
- guide: globe
- api: make_cartopy_frame
skyplothelper *also* supports [cartopy](https://scitools.org.uk/cartopy) as an
optional backend for geographic Earth maps — separate from its own planet globes
above. `make_cartopy_frame` builds a map in any cartopy projection with
coastline / land / ocean features in one call. (Needs the optional `cartopy`
dependency.)

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

# make_cartopy_frame gives an Earth map in any cartopy projection, with
# coastline / land / ocean features from one call. Four projections here.
fig = plt.figure(figsize=(11, 6.4))
specs = [
    ("robinson",     0,           "robinson"),
    ("mollweide",    0,           "mollweide"),
    ("orthographic", (-100, 35),  "orthographic (a globe)"),
    ("lambert_azimuthal", (10, 45), "lambert_azimuthal"),
]
for i, (proj, center, title) in enumerate(specs, start=1):
    ax = sph.make_cartopy_frame(subplotnumber=220 + i, projection=proj,
                                center=center, coastlines=True, land=True,
                                ocean=True, grid=True)
    ax.set_title(title, fontsize=10)
fig.suptitle("make_cartopy_frame - an Earth map in any projection", fontsize=13)
fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.03,
                    hspace=0.32, wspace=0.12)
```

### Textured planet
- guide: globe
- api: make_planet_frame, pseudofits_from_image, reproject_rgb_map, euler_to_fits_ortho
- data: examples
A solid body draped in its surface texture at its true axial tilt — Mars,
from a bundled map, on a planet frame.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

obl = sph.obliquities["mars"]
clon, clat, pole = sph.euler_to_fits_ortho(rotation=250, obliquity=obl,
                                           perspective=0)
fig = plt.figure(figsize=(4.6, 4.6))
ax = sph.make_planet_frame(111, body="mars", center_LONdeg=clon,
                           center_LATdeg=clat, lonpole=pole, Naxispix=500,
                           grid=False)
hdu = sph.pseudofits_from_image("examples/data/planet_maps/2k_mars.jpg", geo=True)
out_hdr = ax.wcs.to_header()
nx = round(ax.get_xlim()[1] - ax.get_xlim()[0])
ny = round(ax.get_ylim()[1] - ax.get_ylim()[0])
out_hdr["NAXIS1"], out_hdr["NAXIS2"] = nx, ny
bg = sph.reproject_rgb_map(hdu, out_hdr, shape_out=(ny, nx))
ax.imshow(np.nan_to_num(bg), zorder=-10)
# show_back=False: the dashed far-side graticule would otherwise draw on
# top of the surface raster, which reads as a glitch rather than depth.
sph.plot_ortho_grid(ax, front_color="0.75", front_lw=0.4, show_back=False)
```

### Nightshade
- guide: globe
- api: make_nightshade_blend, add_scale_bar_cylindrical
- data: examples
The day/night terminator for a given instant — a night-lights layer blended
onto the day map with a physical twilight falloff.

```python
import datetime as dt
import matplotlib.pyplot as plt
import skyplothelper as sph

day = plt.imread("examples/data/world.topo.bathy.200412.3x5400x2700.jpg")
night = plt.imread("examples/data/BlackMarble_2016_01deg.jpg").astype(float) / 255.0
when = dt.datetime(2024, 6, 21, 21, 0)
extent = [-180, 180, -90, 90]
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.imshow(day, extent=extent, origin="upper")
ax.imshow(sph.make_nightshade_blend(night, when, blend_sigma=60),
          extent=extent, origin="upper")
sph.add_scale_bar_cylindrical(ax, lat=45, body="earth", length_km=2000,
                              color="white", stroke_color="0.1")
ax.set_xticks([])
ax.set_yticks([])
```

### Nightshade on a globe
- guide: globe
- api: make_nightshade_blend, pseudofits_from_image, reproject_rgb_map
- data: examples
The same day/night terminator, draped on an orthographic Earth: composite
the blend in cylindrical space, hand the array straight to
`pseudofits_from_image`, and reproject it onto a planet frame.

```python
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skyplothelper as sph

day = plt.imread("examples/data/world.topo.bathy.200412.3x5400x2700.jpg")
night = plt.imread("examples/data/BlackMarble_2016_01deg.jpg").astype(float) / 255.0
when = dt.datetime(2024, 6, 21, 21, 0)

# Blend, then flatten day + night into ONE cylindrical raster (the blend's
# alpha carries the twilight falloff). The maps ship on different grids, so
# resample the day map onto the night map's before compositing.
shade = sph.make_nightshade_blend(night, when, blend_sigma=60)
h, w = shade.shape[:2]
day_r = np.asarray(Image.fromarray(day).resize((w, h), Image.BILINEAR)) / 255.0
alpha = shade[..., 3:4]
flat = np.clip(day_r * (1 - alpha) + shade[..., :3] * alpha, 0, 1)

# pseudofits_from_image takes the array directly — no temp file needed.
hdu = sph.pseudofits_from_image((flat * 255).astype("uint8"), geo=True)
fig = plt.figure(figsize=(4.6, 4.6))
ax = sph.make_planet_frame(111, center_LONdeg=-40, center_LATdeg=25,
                           Naxispix=500, grid=False)
out_hdr = ax.wcs.to_header()
nx = round(ax.get_xlim()[1] - ax.get_xlim()[0])
ny = round(ax.get_ylim()[1] - ax.get_ylim()[0])
out_hdr["NAXIS1"], out_hdr["NAXIS2"] = nx, ny
ax.imshow(np.nan_to_num(sph.reproject_rgb_map(hdu, out_hdr, shape_out=(ny, nx))),
          zorder=-10)
sph.plot_ortho_grid(ax, front_color="0.7", front_lw=0.4, show_back=False)
```

### Globe geodesics
- guide: globe
- api: great_circle_arc, highlight_great_circle, add_pole_rod
Great-circle arcs and a full highlighted great circle on a celestial globe,
with a pole rod marking the rotation axis.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(2)
fig = plt.figure(figsize=(4.6, 4.6))
ax = sph.make_globe_frame(111, center_LONdeg=200, center_LATdeg=35)
pts = [(150, 10), (210, 55), (270, 20), (320, 60)]
for (l1, b1), (l2, b2) in zip(pts, pts[1:]):
    lo, la = sph.great_circle_arc(l1, b1, l2, b2, n_pts=80)
    sph.plot_line_globe(ax, lo, la, color="C1", lw=1.8)
sph.highlight_great_circle(ax, inclination=60, node=200, color="C0", lw=2.0)
sph.add_pole_rod(ax)
```

### Baselines on a globe
- guide: vectors
- api: plot_baselines, make_planet_frame, plot_coastlines
A station network's great-circle baselines on an Earth globe rather than a flat
map: `plot_baselines` hides the arcs that dip behind the globe, drawing them as
faint dashes. A global VLBI array here — US, European, African, Asian, and
Australian stations.

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

# A global VLBI network - US (VLBA), plus stations in Europe, Africa, Asia, and
# Australia so some baselines cross to the far side of the globe.
SITES = {"MK": (-155.46, 19.80), "BR": (-119.68, 48.13), "PT": (-108.12, 34.30),
         "LA": (-106.25, 35.78), "NL": (-91.57, 41.77), "SC": (-64.58, 17.76),
         "EB": (6.88, 50.52), "HH": (27.69, -25.89),
         "TM": (121.14, 30.90), "CD": (133.81, -31.87)}

fig = plt.figure(figsize=(6.2, 6.2))
# An Earth globe (make_planet_frame) instead of a flat map: plot_baselines draws
# the network great circles and hides the far-side arcs automatically - the ones
# that dip behind the globe are drawn as faint dashes via back_hemisphere_*.
globe = sph.make_planet_frame(111, body="earth",
                              center_LONdeg=-40, center_LATdeg=25)
sph.plot_coastlines(globe, color="0.6")
sph.plot_baselines(globe, SITES, color="C1", linewidth=1.0, alpha=0.9,
                   marker_color="C0", site_label_fontsize=7,
                   back_hemisphere_linestyle=":", back_hemisphere_alpha=0.4)
globe.set_title("Global VLBI baselines on an Earth globe", fontsize=12)
```

## Cone diagrams

### Redshift cone
- guide: cone
- api: make_cone_frame, cone_scatter
A z–RA wedge with the observer at the apex.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(11)
fig = plt.figure(figsize=(5.2, 3.8))
ax = sph.make_cone_frame(111, angle_center=180, angle_half_width=35,
                         r_min=0, r_max=0.12, angle_label="R.A.", fig=fig)
n = 600
z = 0.12 * np.sqrt(rng.random(n))
ang = 180 + (rng.random(n) - 0.5) * 70
sph.cone_scatter(ax, ang, z, s=4, alpha=0.6, color="C0")
```

### Bowtie diagram
- guide: cone
- api: make_bowtie_frame, cone_scatter
A double-sided wedge — two opposing cones sharing the apex, for
back-to-back survey slices.

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(13)
top, bot = sph.make_bowtie_frame(
    angle_center=185, angle_half_width=42, r_min=0.0, r_max=0.13,
    angle_tick_spacing=15, r_tick_spacing=0.05)
fig = top.figure
# Both halves of the bowtie share the same angular center (185°); each
# points its radius the opposite way from the shared apex.
for ax in (top, bot):
    n = 500
    z = 0.13 * np.sqrt(rng.random(n))
    ang = 185 + (rng.random(n) - 0.5) * 78
    sph.cone_scatter(ax, ang, z, s=4, alpha=0.55, color="C0")
```

### Cone density
- guide: cone
- api: cone_hexbin, add_colorbar
Binning the wedge itself — a hexbin in cone space with a matched
colorbar.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(14)
fig = plt.figure(figsize=(5.2, 3.8))
ax = sph.make_cone_frame(111, angle_center=180, angle_half_width=35,
                         r_min=0, r_max=0.12, angle_label="R.A.", fig=fig)
n = 6000
z = 0.12 * np.sqrt(rng.random(n))
ang = 180 + (rng.random(n) - 0.5) * 70
hb = sph.cone_hexbin(ax, ang, z, gridsize=26, cmap="sph.dusk", mincnt=1)
sph.add_colorbar(hb, ax=ax, label="galaxies / hex", mode="simple", shrink=0.55)
```

## Vectors & motions

### VSH shift vectors
- guide: vectors
- api: vsh_field, plot_sky_vectors
Systematic position shifts across the sky, evaluated on a grid and drawn
as sky vectors — here a vector-spherical-harmonic **glide** field, the
Galactic-aberration signature. The same call draws any VSH term.

```python
import numpy as np
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
glon, glat = np.meshgrid(np.arange(0, 360, 20), np.arange(-75, 76, 15))
dlon, dlat = sph.vsh_field(glon.ravel(), glat.ravel(), {"D_3": 1.0})
# scale="auto" sizes the median arrow to a couple of degrees whatever the
# field's amplitude units are. A bare numeric scale here would be read in
# the default arcsec, making every arrow sub-pixel.
sph.plot_sky_vectors(ax, glon.ravel(), glat.ravel(), dlon, dlat,
                     scale="auto", auto_target_deg=9.0, color="C0")
```

### Displacement arrows
- guide: vectors
- api: plot_displacement
Individual epoch-to-epoch arrows whose shafts follow the great-circle
path — seam-aware, unlike a raw quiver.

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(21)
fig, ax = sph.allsky_figure(projection="AIT", center=180)
n = 60
lon1 = rng.uniform(0, 360, n)
lat1 = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
lon2 = lon1 + rng.normal(0, 6, n) / np.cos(np.radians(lat1))
lat2 = lat1 + rng.normal(0, 4, n)
sph.plot_displacement(ax, lon1, lat1, lon2, lat2, color="C1", geodesic=True)
```

### Co-visibility regions
- guide: vectors
- api: covisibility_region, covisibility_circles
Where two stations can see the sky at once — each station's cap, labeled
at its zenith in its own color, and their instantaneous intersection, as
renderable regions for one instant.

```python
import skyplothelper as sph

STATIONS = {"Wettzell": {"lat": 49.145, "lon": 12.878},
            "VLA": {"lat": 34.08, "lon": -107.62}}
TIME = "2026-07-02T07:00:00"
fig, ax = sph.allsky_figure(projection="AIT", center=180)
for name, col in [("Wettzell", "C0"), ("VLA", "C1")]:
    reg = sph.covisibility_region(ax, {name: STATIONS[name]}, TIME, el_min=15)
    reg.render(facecolor=col, alpha=0.22, edgecolor=col, lw=1.3)
    # Label each cap at its own center, in its own color, so the overlap
    # below is unambiguous about which station contributes which lobe.
    cap = sph.covisibility_circles({name: STATIONS[name]}, TIME, el_min=15)[0]
    ax.scatter(cap["center"].ra.deg, cap["center"].dec.deg, s=32, color=col,
               zorder=6, transform=ax.get_transform("world"))
    ax.annotate(name, ax.wcs.world_to_pixel_values(cap["center"].ra.deg,
                                                   cap["center"].dec.deg),
                xytext=(0, 9), textcoords="offset points", ha="center",
                fontsize=9, fontweight="bold", color=col, zorder=7)
both = sph.covisibility_region(ax, STATIONS, TIME, el_min=15)
both.render(facecolor="C3", alpha=0.5, edgecolor="C3", lw=1.6)
```

## Annotations & decorations

### Reticles & ruler
- guide: overlays
- api: add_reticle, Ruler
The four reticle styles marking targets, and a ruler measuring an
angular span with pixel-stable ticks.

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

fig = plt.figure(figsize=(4.8, 4.4))
ax = sph.make_wcs_frame(111, "TAN", center=(83.63, 22.01), fov_deg=0.5, fig=fig)
for i, style in enumerate(["plus", "x", "L", "circle"]):
    sph.add_reticle(ax, (83.78 - 0.1 * i, 22.16), style=style)
# add_to() is what draws; passing ax= as well double-adds and the ruler
# then renders nothing at all.
sph.Ruler((110, 110), (390, 330), pixscale_asec=3.6).add_to(ax)
```

### Constellation star chart
- guide: overlays
- api: add_constellation_boundaries, add_constellation_lines, add_constellation_labels
The IAU constellation kit — boundaries (precessed to ICRS), asterism
lines, and labels — over a wide field around Orion.

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

fig = plt.figure(figsize=(5.0, 4.6))
ax = sph.make_wcs_frame(111, "ARC", center=(83, 0), fov_deg=70, fig=fig)
sph.add_constellation_boundaries(ax, color="C7", lw=0.6)
sph.add_constellation_lines(ax, color="C1", lw=0.9)
sph.add_constellation_labels(ax)
```

### Beams & scale bars
- guide: overlays
- api: Beam, add_sizebar_asec, add_sizebar
- data: examples
The image-furnishing furniture on a VLBA 15 GHz image of 3C 84: a column of the
`Beam` styles (each `style=` manages its own outline, fill, crosshair, or hatch)
and two stacked scale bars carrying the *same* on-sky length in angular and in
physical units.

```python
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import skyplothelper as sph
from skyplothelper import Beam

hdu = fits.open("examples/data/0316+413.u.stacked.icd.fits")[0]
data, hdr = sph.squeeze_image(hdu.data, hdu.header)
wcs = WCS(hdr).celestial
fig = plt.figure(figsize=(5.4, 5.2))
ax = fig.add_subplot(111, projection=wcs)
ax.imshow(data, origin="lower", cmap="sph.deepsky",
          norm=sph.make_norm(stretch="symlog", clip="manual", vmin=0,
                             vmax=float(np.nanmax(data)), a=5e-3))
cx, cy = wcs.world_to_pixel_values(hdr["CRVAL1"], hdr["CRVAL2"])
half = 12.0 / (abs(hdr["CDELT2"]) * 3.6e6)          # +/-12 mas crop
ax.set_xlim(cx - half, cx + half)
ax.set_ylim(cy - half, cy + half)
ax.coords[0].set_ticklabel(size=8); ax.coords[1].set_ticklabel(size=8)
ax.coords[0].set_axislabel("RA (J2000)", fontsize=9)
ax.coords[1].set_axislabel("Dec (J2000)", fontsize=9)

# A column of the Beam styles up the left side, each labeled - each style=
# manages its own outline, fill, crosshair, or hatch pattern.
bx = cx - 0.66 * half
bmaj, bmin = 0.16 * half, 0.10 * half
styles = ["ellipse", "filled", "crosshair", "crosshairgrid", "hatch"]
ys = np.linspace(cy + 0.74 * half, cy - 0.74 * half, len(styles))
for style, by in zip(styles, ys):
    # Pass only ec - each style manages its own fill / crosshair / hatch, so the
    # five read as visibly different (outline, solid, crosshair, grid, hatch).
    Beam((bx, by), bmaj_pix=bmaj, bmin_pix=bmin, bpa_deg=30, style=style,
         ec="#7fdfff", lw=1.3, stroke_color="0.1", stroke_lw=2.0).add_to(ax)
    ax.text(bx + 0.15 * half, by, style, color="white", fontsize=8, va="center",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="0.1")])

# Two stacked scale bars: the same on-sky length labeled in angle and in physical
# units. NGC 1275 (z=0.0176) subtends ~0.36 pc/mas, so 5 mas ~ 1.8 pc.
sph.add_sizebar_asec(ax, hdr, 0.005, "5 mas", color="white", loc="lower right",
                     stroke_color="0.1", stroke_lw=2.0)
mas_to_px = 1.0 / (abs(hdr["CDELT2"]) * 3.6e6)
sph.add_sizebar(ax, 5 * mas_to_px, "1.8 pc", loc="lower right", borderpad=2.7,
                label_top=True, color="white", stroke_color="0.1", stroke_lw=2.0)
```

### BeamStack — combined-array beams
- guide: overlays
- api: BeamStack, Beam
Co-located beams for a combined-array observation, stacked at one position and
labeled for a single legend. Distinct sizes, position angles, and fills
(outline, hatch, two solid) keep the members legible even when nested.

```python
import matplotlib.pyplot as plt
import skyplothelper as sph
from skyplothelper import Beam, BeamStack

fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.set_aspect("equal")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("BeamStack - co-located beams from a combined array")

# Co-located beams for a combined-array observation. Each is a full Beam with its
# own size, position angle, and fill, so the members stay visually distinct even
# nested - outline, hatch, and two solid fills, at four different PAs.
BeamStack([
    Beam((50, 50), bmaj_pix=74, bmin_pix=52, bpa_deg=28,
         style="ellipse", ec="C0", lw=1.8, label="D config"),
    Beam((50, 50), bmaj_pix=50, bmin_pix=32, bpa_deg=-18,
         style="hatch", ec="C1", lw=1.5, label="C config"),
    Beam((50, 50), bmaj_pix=30, bmin_pix=19, bpa_deg=52,
         style="filled", fc="C2", ec="C2", alpha=0.55, label="B config"),
    Beam((50, 50), bmaj_pix=14, bmin_pix=10, bpa_deg=5,
         style="filled", fc="C3", ec="0.1", lw=1.0, label="A config (combined)"),
]).add_to(ax)
ax.legend(loc="upper right", fontsize=9, framealpha=0.6)
```

### Beam from a PSF fit
- guide: overlays
- api: Beam
Fit a `Beam` to a synthetic elliptical-Gaussian PSF with `Beam.from_psf_fit`,
draw its recovered FWHM ellipse on the image (an sph colormap here), and drop a
scale-matched PSF inset (asinh stretch, revealing the faint negative ring) with
`Beam.add_psf_inset`.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph
from skyplothelper import Beam

sph.set_style(base="structural")   # inward ticks on both axes

# A synthetic elliptical-Gaussian PSF (14 x 7 px FWHM, tilted 25 deg) with a
# faint negative sidelobe ring, on a padded field so the main view is zoomed out.
rng = np.random.default_rng(0)
nx = ny = 161
yy, xx = np.mgrid[0:ny, 0:nx]
cx = cy = (nx - 1) / 2.0
th = np.radians(25.0)
c, s = np.cos(th), np.sin(th)
xr = (xx - cx) * c + (yy - cy) * s
yr = -(xx - cx) * s + (yy - cy) * c
core = np.exp(-0.5 * ((xr / (7.0 / 2.3548)) ** 2 + (yr / (14.0 / 2.3548)) ** 2))
r = np.sqrt(xr ** 2 + yr ** 2)
ring = -0.08 * np.exp(-0.5 * ((r - 22.0) / 4.0) ** 2)
psf = core + ring + 0.004 * rng.standard_normal((ny, nx))

fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.imshow(psf, cmap="sph.deepsky", origin="lower")
ax.set_xlabel("x (pix)"); ax.set_ylabel("y (pix)")
ax.set_title("Beam.from_psf_fit + add_psf_inset")

# Fit a Beam to the PSF and draw its FWHM ellipse on the image; the fit recovers
# the shape and orientation, so the ellipse hugs the bright core.
beam = Beam.from_psf_fit(psf, xy=(cx, cy), style="crosshair", ec="white", lw=1.4,
                         crosshair_color="white", crosshair_lw=0.9,
                         stroke_color="0.1", stroke_lw=2.0)
beam.add_to(ax)

# A PSF inset at the SAME physical scale as the zoomed-out main view: pass a
# central crop sized to the inset's on-page fraction, asinh-stretched to reveal
# the faint ring the linear main view hides.
cr = 27
crop = psf[int(cy) - cr:int(cy) + cr + 1, int(cx) - cr:int(cx) + cr + 1]
inset = beam.add_psf_inset(ax, crop, size="34%", loc="upper left",
                           stretch="asinh", cmap="viridis", show_beam=True,
                           beam_kwargs={"style": "ellipse", "ec": "white", "lw": 1.3},
                           border_color="white", border_lw=1.1)
inset.set_xlabel("asinh PSF", fontsize=8)
inset.set_xticks([0, 25, 50]); inset.set_yticks([0, 25, 50])
inset.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True,
                  labelsize=6)
```

### Ruler styles
- guide: overlays
- api: Ruler
Every `Ruler` knob on one panel: ticks on both sides or just one, minor ticks,
varied tick lengths, and arrow / tick endcaps — each bar in a different unit
(arcsec, arcmin, and physical kpc / AU / Mpc via `convert=`) — plus a vertical
`Ruler` on the right standing in for a twin axis.

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import skyplothelper as sph
from skyplothelper import Ruler

NEUTRAL = mpl.rcParams["axes.edgecolor"]   # dark on light, light on dark

fig, ax = plt.subplots(figsize=(8.5, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Ruler - one bar, every knob")

# A stack of horizontal rulers, each self-labeled by its title, walking through
# the styling knobs: tick side, minor ticks, tick length, and endcap styles. A
# per-ruler pixscale_asec gives each a different physical scale and unit.
X0, X1 = 10, 90
rows = [
    dict(y=90, color="C0", pixscale_asec=0.5, tick_interval=10,
         title="default ticks (arcsec)"),
    dict(y=76, color="C1", pixscale_asec=1.875, tick_interval=30, tick_side="right",
         title="tick_side='right' (arcmin)"),
    dict(y=62, color="C2", pixscale_asec=0.5, tick_interval=10, fmt="%.0f",
         convert=dict(redshift=0.5, unit="kpc"), minor_ticks=4,
         title="minor_ticks=4 (kpc, z=0.5)"),
    dict(y=48, color="C3", pixscale_asec=0.5, tick_interval=10, fmt="%.0f",
         convert=dict(distance=100, distance_unit="pc", unit="au"), tick_length=9,
         minor_ticks=4, minor_tick_length=4, tick_side="left",
         title="long major + minor (AU @ 100 pc)"),
    dict(y=34, color="C4", pixscale_asec=5.0, tick_interval=100, fmt="%.1f",
         convert=dict(redshift=1.0, unit="Mpc"), endcap_style="arrow",
         endcaps="both", title="endcap_style='arrow' (Mpc, z=1)"),
    dict(y=20, color="C5", tick_interval=20, endcap_style="tick", tick_side="right",
         minor_ticks=2, title="endcap_style='tick' (pixels)"),
]
for r in rows:
    y = r.pop("y"); color = r.pop("color")
    Ruler((X0, y), (X1, y), ax=ax, color=color, lw=1.6,
          label_fontsize=8, title_fontsize=9, **r).add_to(ax)

# A vertical Ruler on the right, standing in for a twin y-axis: ticks point in,
# the title sits outside like a secondary axis label.
Ruler((96, 10), (96, 90), ax=ax, color=NEUTRAL, lw=1.6, pixscale_asec=0.5,
      tick_interval=10, tick_side="left", minor_ticks=4, label_side="left",
      label_fontsize=8, title="Ruler as a twin axis", title_side="right",
      title_fontsize=9).add_to(ax)
fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.05)
```

### Compasses & compass roses
- guide: globe
- api: add_compass_rose, add_surface_compass, add_scale_bar
Orientation furniture on a globe: a corner compass rose, an on-surface
compass that follows the sphere's orientation, a pole rod, and a
real-distance scale bar curved to the globe.

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

clon, clat, pole = sph.euler_to_fits_ortho(rotation=60, obliquity=23.44, perspective=10)
fig = plt.figure(figsize=(5.0, 5.0))
ax = sph.make_planet_frame(111, body="earth", center_LONdeg=clon, center_LATdeg=clat,
                           lonpole=pole, Naxispix=420)
sph.plot_coastlines(ax, color="0.6", lw=0.7)
sph.add_pole_rod(ax, color="C0")
# A corner rose (axes fraction), an on-surface compass following the sphere,
# and a real-distance scale bar curved to the globe.
sph.add_compass_rose(ax, x=0.14, y=0.84, size=34, style="simple", color="C3")
sph.add_surface_compass(ax, -30, 45, size_deg=13, style="star", color="C3")
sph.add_scale_bar(ax, lon_0=clon, lat_0=clat, body="earth", length_km=4000, color="C1")
```

## Markers

### Image-stamp markers
- guide: globe
- api: imscatter
- data: examples
Photographic icons placed as markers — the bundled Solar-System stamps
dropped onto a plain axes at graduated zoom.

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

# Every bundled stamp: solar-system bodies on the top row, facility icons
# on the bottom. zoom= is per-stamp, so mixed native resolutions still land
# at a common on-page size.
stamps = [("sun2_120pix", 0.42), ("Earth_Western_Hemisphere_120pix", 0.42),
          ("Mars_120pix", 0.42), ("FullMoon_240x240", 0.21),
          ("Jupiter_120pix", 0.42),
          ("RadioDish_250pix", 0.20), ("OpticalTelescope_250pix", 0.20),
          ("SpaceTelescope_250pix", 0.20), ("SMBH_250pix", 0.20),
          ("sun1_120pix", 0.42)]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
for i, (name, z) in enumerate(stamps):
    # plt.imread returns row 0 = TOP, but this style sets image.origin='lower'
    # (right for FITS, wrong for photographs), which flips the stamps. Reverse
    # the rows so they sit the way they were drawn.
    img = plt.imread(f"examples/data/icons/{name}.png")[::-1]
    sph.imscatter([i % 5], [-(i // 5)], img, ax=ax, zoom=z)
ax.set_xlim(-0.6, 4.6)
ax.set_ylim(-1.6, 0.6)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
```

### Instrument markers
- guide: overlays
- api: add_antenna_marker, add_telescope_marker, add_dome_marker
Procedural instrument glyphs that *aim* — an antenna and an optical
telescope solving their elevation from a shared target, and a dome whose
slit takes a compass bearing.

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
# Off to one side, so the two aimed instruments point at visibly different
# angles rather than both straight up.
target = (5.4, 8.7)
ax.scatter(*target, marker="*", s=280, color="C1", zorder=7)
style = dict(size=64, face_color="C0", edge_color="0.15",
             stroke_color="white", stroke_lw=2.0)
fig.canvas.draw()  # finalize the equal-aspect transform before solving angles
# The markers are base-anchored, so one shared y plants all three feet on the
# same ground line. aim_angles gives the on-screen tilt for the dish/tube, and
# box.anchors.sight_line_origin(phi) starts each sight-line inside the optics
# rather than at the pier foot, so the rays read as leaving the dish and tube.
for fn, xy, elev_kw in [(sph.add_antenna_marker, (2.4, 3.0), "dish_elev"),
                        (sph.add_telescope_marker, (8.4, 3.0), "tube_elev")]:
    phi = sph.aim_angles(ax, xy, target, target_coords="data")["aim_angle"]
    box = fn(ax, xy, rotation=0, **{elev_kw: phi}, **style)
    ox, oy = box.anchors.sight_line_origin(phi)
    ax.plot([ox, target[0]], [oy, target[1]], ls=(0, (1, 2.5)),
            color="0.6", lw=0.9, zorder=1)
# The dome doesn't tilt — its slit swings to a compass bearing instead.
sph.add_dome_marker(ax, (13.0, 3.0), slit_azim=-28, **style)
for x, name in [(2.4, "antenna"), (8.4, "telescope"), (13.0, "dome")]:
    ax.annotate(name, (x, 0.75), ha="center", fontsize=9)
ax.set_xticks([])
ax.set_yticks([])
```

### Instruments aimed at a source
- guide: overlays
- api: add_antenna_marker, add_telescope_marker
A mixed instrument array on a globe, each marker planted with its pedestal along
the local vertical (`aim_mode="planted"`, `globe_center=`) and its dish or tube
swung onto one shared source (`aim_at=`) — radio dishes and optical tubes here.

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import skyplothelper as sph
from matplotlib.colors import to_rgb

_r, _g, _b = to_rgb(mpl.rcParams["figure.facecolor"])
DARK = 0.299 * _r + 0.587 * _g + 0.114 * _b < 0.5
PAL = sph.ANNOTATION_PALETTES["dark" if DARK else "publication"]

CENTER = (-95.0, 20.0)
src = (-52.0, 60.0)                 # the shared celestial source, up and to the right
# A mixed instrument array - radio dishes and optical tubes - each planted on the
# globe (pedestal along its local vertical) and aimed at one source.
sites = [(-140, 35, "antenna"), (-120, 6, "telescope"), (-98, 46, "telescope"),
         (-72, 24, "antenna"), (-116, -18, "telescope"), (-84, -6, "antenna"),
         (-60, 4, "telescope")]

fig = plt.figure(figsize=(6.6, 6.2))
ax = sph.make_globe_frame(111, center_LONdeg=CENTER[0], center_LATdeg=CENTER[1],
                          projection="SIN", grid=True, Naxispix=360)
fig.canvas.draw()   # settle the transforms before the aim solver reads them
ax.plot(*ax.wcs.wcs_world2pix([src], 0)[0], marker="*", ms=18,
        color=PAL["label"], zorder=9)

style = dict(size=40, edge_color=PAL["text"], stroke_color=PAL["fig_bg"], stroke_lw=1.6)
makers = {"antenna": sph.add_antenna_marker, "telescope": sph.add_telescope_marker}
faces = {"antenna": PAL["accent"], "telescope": PAL["accent2"]}
for lon, lat, kind in sites:
    makers[kind](ax, (lon, lat), coord_type="world", aim_at=src, aim_mode="planted",
                 globe_center=CENTER, target_coords="world",
                 face_color=faces[kind], **style)
ax.set_title("A mixed array, planted on a globe, aimed at one source", fontsize=11)
```

### Geodesic baseline map
- guide: vectors
- api: plot_baselines
Great-circle baselines and self-labeling dish markers for any station
network on a plain lon/lat map — the VLBA here simply as a worked example.

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

VLBA = {"BR": (-119.68, 48.13), "OV": (-118.28, 37.23), "KP": (-111.61, 31.96),
        "PT": (-108.12, 34.30), "LA": (-106.25, 35.78), "FD": (-103.94, 30.63),
        "NL": (-91.57, 41.77), "HN": (-71.99, 42.93),
        "MK": (-155.46, 19.80), "SC": (-64.58, 17.76)}
fig, ax = plt.subplots(figsize=(5.4, 3.4))
sph.plot_baselines(ax, VLBA, color="C1", linewidth=0.8, alpha=0.85,
                   marker_color="C0", site_label_fontsize=7)
ax.set_xlim(-165, -60)
ax.set_ylim(10, 55)
ax.set_aspect("equal")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
```

## Styling

### Themes at a glance
- guide: styling
- api: set_style, style_context
One figure, four `set_style` themes: each panel forces a theme (and its data
palette) in a `style_context`, restyling the same sky field's background, frame,
grid, text, and colors. The base-preset and annotation-palette layers round out
the system — see the styling guide.

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

# Four themes, each forced in its own style_context so the specimen renders the
# same regardless of the page's light/dark mode. Each panel is the same little
# sky field, restyled: background, frame, grid, text, and the data-color cycle.
themes = [("publication", "uranometria"), ("twilight", "velvet"),
          ("dark_sky", "nightcap"), ("poster", "speakeasy")]
rng = np.random.default_rng(3)
groups = [(172.5, 5.5), (187.5, 5.5), (172.5, -5.5), (187.5, -5.5)]
pts = [(g[0] + rng.normal(0, 2.2, 55), g[1] + rng.normal(0, 2.2, 55)) for g in groups]

fig = plt.figure(figsize=(9.5, 8.4))
for i, (theme, pal) in enumerate(themes, start=1):
    with sph.style_context(base="standard", theme=theme, palette=pal):
        ax = sph.make_wcs_frame((2, 2, i), "TAN", center=(180, 0), fov_deg=22, fig=fig)
        cycle = sph.CYCLE_PALETTES[pal]["colors"]
        for g, (ra, dec) in enumerate(pts):
            ax.scatter(ra, dec, transform=ax.get_transform("world"), s=22, lw=0,
                       color=cycle[g % len(cycle)], alpha=0.9)
        ax.grid(True)
        # Label inside the axes on a card of the theme's own background, so it
        # reads in the theme's text color regardless of the page's light/dark mode.
        ax.text(0.035, 0.965, f"theme='{theme}'\npalette='{pal}'",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                bbox=dict(facecolor=plt.rcParams["axes.facecolor"], alpha=0.75,
                          edgecolor="none", boxstyle="round,pad=0.3"))
fig.suptitle("set_style - one sky field, four themes", fontsize=13)
fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.05,
                    hspace=0.28, wspace=0.22)
```
