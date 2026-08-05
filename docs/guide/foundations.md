# Foundations: SkyCoord, WCS & matplotlib transforms

skyplothelper is a thin, convenient layer over two libraries you'll keep
using directly: **matplotlib** (the drawing) and **astropy** (coordinates,
units, FITS, and the world coordinate system). You don't need to master
either to get a figure out of `sph`, but a working feel for the handful of
ideas below makes everything else on these pages click — and tells you what
to reach for when you drop below `sph` into raw matplotlib/astropy.

This page is a crash course in exactly those ideas. It assumes you can
already make a basic matplotlib plot (`fig, ax = plt.subplots()`,
`ax.plot(...)`); it does **not** assume you've plotted on the sky before.
Once it makes sense, {doc}`concepts` covers the conventions `sph` layers on
top, and the {doc}`getting started tutorial </tutorials/getting_started>`
puts it to work on real figures.

```python
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import skyplothelper as sph
```

## Units: numbers that carry their dimension

Astropy attaches a physical unit to a number, making a
[`Quantity`](https://docs.astropy.org/en/stable/units/index.html). You
multiply a value by a unit:

```python
10.5 * u.deg          # an angle
15 * u.arcmin         # 0.25 deg — convert with .to(u.deg)
2.3 * u.hourangle     # RA expressed in time units (1 hourangle = 15 deg)
```

Coordinates and angular sizes in astropy (and in some `sph` calls) are
`Quantity` objects rather than bare floats, because "10" is ambiguous —
degrees? arcminutes? hours? — and a unit removes the guesswork. Pull the
plain number back out with `.to(unit).value`, e.g.
`(15 * u.arcmin).to(u.deg).value` → `0.25`.

## `SkyCoord`: a position on the sky

[`SkyCoord`](https://docs.astropy.org/en/stable/coordinates/skycoord.html)
is astropy's representation of one sky position — or a whole array of them —
together with the **coordinate frame** it's expressed in (ICRS/equatorial,
Galactic, ecliptic, …). It's the standard currency for "where on the sky,"
and `sph` accepts it wherever a position is expected.

### Creating one

The constructor is deliberately flexible; the common forms:

```python
# From decimal degrees (ICRS is the default frame). Explicit is clearest:
c = SkyCoord(ra=10.684 * u.deg, dec=41.269 * u.deg)
c = SkyCoord(10.684, 41.269, unit="deg")            # positional (ra, dec)

# From sexagesimal strings — units inferred from the h/m/s, d/m/s markers:
c = SkyCoord("00h42m44.3s", "+41d16m09s")
# …or space-separated, where you must state the units (RA in hours here):
c = SkyCoord("00 42 44.3 +41 16 09", unit=(u.hourangle, u.deg))

# In another frame — use that frame's axis names:
g = SkyCoord(l=121.17 * u.deg, b=-21.57 * u.deg, frame="galactic")

# By name (needs internet; resolves via SIMBAD/NED):
m31 = SkyCoord.from_name("M31")

# A whole catalog at once — pass arrays:
cat = SkyCoord([10.68, 83.82] * u.deg, [41.27, -5.39] * u.deg)
```

> `sph` also wraps name resolution as {func}`~skyplothelper.resolve_name`
> (returns a `SkyCoord`) and offers offline sexagesimal parsing/formatting
> ({func}`~skyplothelper.sex2dec`, {func}`~skyplothelper.dec2sex`) if you'd
> rather not build a `SkyCoord` — see the {doc}`coordinates API
> </api/coordinates>`.

### Converting between frames

Ask any `SkyCoord` for its representation in another frame — the
transformation is exact and built in:

```python
c.galactic                 # attribute shortcut for the common frames
c.transform_to("galactic") # the general form (also 'fk5', 'ecliptic', …)
```

This is the *right* way to change coordinate systems — never convert by hand.
See astropy's
[transforming between systems](https://docs.astropy.org/en/stable/coordinates/transforming.html)
guide for the full frame graph. For array math without building `SkyCoord`
objects, `sph` mirrors the common conversions as plain functions —
{func}`~skyplothelper.convert_frame`, {func}`~skyplothelper.icrs_to_galactic`,
etc. ({doc}`concepts` covers when to use which).

Some frames describe *where you are standing*, not just where the target is.
Horizontal coordinates (altitude/azimuth) need an observing site and a time:

```python
from astropy.coordinates import AltAz, EarthLocation
from astropy.time import Time

site = EarthLocation(lat=19.82 * u.deg, lon=-155.47 * u.deg, height=4205 * u.m)
altaz = c.transform_to(AltAz(obstime=Time("2026-07-18 09:00"), location=site))
altaz.alt.deg, altaz.az.deg
```

### Reading values back out

Components come out as `Quantity` angles; ask for the unit you want:

```python
c.ra.deg     c.ra.hour     c.dec.deg          # equatorial
g.l.deg      g.b.deg                          # galactic (after .galactic)
c.to_string("hmsdms")                         # "00h42m44.3s +41d16m08.4s"
```

Longitudes wrap, and *where* they wrap matters when you plot: an all-sky map
centered at 0° wants longitudes on [-180, 180), one centered at 180° wants
[0, 360). Astropy's `Angle` handles it, and `sph` has `wrap_*` helpers for
plain arrays:

```python
from astropy.coordinates import Angle
Angle(350 * u.deg).wrap_at(180 * u.deg).deg      # -10.0
sph.wrap_pm180(lon_array)                        # same idea, on a numpy array
```

### Distances and matching

`SkyCoord` does spherical geometry for you — great-circle separations and
catalog cross-matching, correct across the poles and the 0h/24h wrap:

```python
sep = c1.separation(c2)                        # an Angle; sep.arcsec, sep.deg
idx, sep2d, _ = cat.match_to_catalog_sky(other_cat)   # nearest-neighbor match
```

(`sph` layers {func}`~skyplothelper.cone_search`,
{func}`~skyplothelper.crossmatch`, and {func}`~skyplothelper.region_search`
on top for catalog filtering — {doc}`queries`.) astropy's
[separations & matching](https://docs.astropy.org/en/stable/coordinates/matchsep.html)
page goes further.

## FITS files: data, header, and the WCS

A FITS file pairs an array with a header of metadata. Astropy reads it with
[`astropy.io.fits`](https://docs.astropy.org/en/stable/io/fits/index.html):

```python
with fits.open("image.fits") as hdul:
    hdu    = hdul[0]          # the primary HDU (a file can hold several)
    data   = hdu.data         # a plain numpy array, indexed [row, col] = [y, x]
    header = hdu.header       # a dict-like of keywords
```

Buried in that header are the keywords describing **where the pixels are on
the sky**. Handed to
[`astropy.wcs.WCS`](https://docs.astropy.org/en/stable/wcs/index.html), they
become a **World Coordinate System** — the mapping between *pixel* positions
`(i, j)` and *world* positions `(lon, lat)` under some projection:

```python
wcs = WCS(header)
wcs.pixel_to_world(512, 512)          # -> a SkyCoord
wcs.world_to_pixel(c)                 # -> pixel (x, y)
sph.describe_wcs(header)              # friendly human-readable summary
```

That WCS object is what makes an axes sky-aware, which is where matplotlib
comes in.

## Figures, axes, and saving

Matplotlib splits every plot into a **Figure** (the canvas) and one or more
**Axes** (a plotting area with its own coordinates). Two idioms:

```python
fig, ax = plt.subplots(figsize=(8, 5))      # the usual one-axes shortcut
fig = plt.figure(figsize=(8, 5))            # explicit, for custom layouts
ax  = fig.add_subplot(111, projection=wcs)  # <- a WCSAxes, from the WCS above
```

That `projection=wcs` argument is the raw-astropy way to get a sky-aware
axes; `sph`'s frame builders wrap it (and the header construction behind it)
so you rarely write it out. Drawing order is controlled by `zorder=` (higher
draws on top), which matters once you stack images, grids, and overlays.

Saving is `fig.savefig(...)`, and a few keywords do most of the work:

```python
fig.savefig("map.png", dpi=200,            # resolution
            bbox_inches="tight",           # crop surrounding whitespace
            transparent=True,              # no opaque background
            facecolor="white")             # or force one
```

In scripts (no display), matplotlib's non-interactive Agg backend is used
automatically when you only save; `plt.show()` is for interactive sessions.

## Transforms: what `transform=` means

Every matplotlib drawing call places its coordinates through a
**[transform](https://matplotlib.org/stable/users/explain/artists/transforms_tutorial.html)** —
the rule that maps *your* numbers to a spot on the canvas. You usually don't
name one, so you get the default, `ax.transData` (data coordinates). Two you
should know by name:

- **`ax.transData`** *(default)* — your data's own coordinate system. On an
  ordinary axes that's the x/y values; **on a sky frame it's *pixel*
  coordinates**, which is why raw `(RA, Dec)` don't land correctly without help.
- **`ax.transAxes`** — axes fractions, `(0,0)` bottom-left to `(1,1)`
  top-right, independent of the data. Handy for placing a label or legend in
  a fixed corner: `ax.text(0.02, 0.95, "N", transform=ax.transAxes)`.

The `transform=` keyword is how you say "interpret these numbers in *that*
system instead of the default." A sky frame adds one more — the **world**
transform — which is the whole point of the WCSAxes section below.

## rcParams and style sheets

Matplotlib keeps its defaults in **`plt.rcParams`**, a global dict-like of
several hundred settings consulted every time an artist is drawn. Changing
one changes every subsequent plot, so it's how you set a look *once* instead
of passing the same keywords to every call. A representative few:

| rcParam | Controls |
|---|---|
| `figure.figsize`, `figure.dpi` | default canvas size and resolution |
| `savefig.dpi`, `savefig.bbox` | output resolution; `'tight'` auto-crops |
| `font.family`, `font.size` | typeface and base text size |
| `axes.prop_cycle` | the default color cycle (`C0`, `C1`, …) |
| `image.cmap` | default colormap for `imshow` |
| `image.origin` | **`'upper'` by default** — FITS data usually wants `'lower'` |
| `lines.linewidth`, `legend.frameon` | line weight, legend box |
| `xtick.direction`, `ytick.direction` | tick geometry — *but see the caveat below* |

You can set them four ways, from most to least invasive:

```python
plt.rcParams["font.size"] = 12                    # one setting, globally
plt.rcParams.update({"figure.dpi": 150,           # several at once
                     "image.origin": "lower"})
plt.style.use("ggplot")                           # a whole style sheet
plt.rcdefaults()                                  # reset to matplotlib defaults
```

**Style sheets** are named bundles of rcParams. `plt.style.available` lists
the built-ins, and you can ship your own as a `.mplstyle` file and pass its
path. Prefer the **context-manager** forms when you only want the change to
apply to one figure — they restore the previous state afterwards:

```python
with plt.style.context("ggplot"):
    ...                                  # only this figure is styled
with plt.rc_context({"font.size": 8}):
    ...
```

`sph` builds on exactly this machinery: {func}`~skyplothelper.set_style`
composes three independent layers — a structural *base*, a light/dark
*theme*, and a color *palette* — into rcParams, and
{func}`~skyplothelper.style_context` is the scoped equivalent
({doc}`styling`).

```python
sph.set_style(base="publication", theme="dark_sky", palette="nightcap")
```

### The WCSAxes caveat

Here's the divergence worth knowing early: **a sky frame does not draw its
ticks the way an ordinary axes does.** astropy's WCSAxes renders ticks,
tick labels, and the graticule through its own `ax.coords` machinery rather
than through `ax.xaxis`/`ax.yaxis` — because on a curved projection a
"tick" isn't tied to a straight plot edge. The practical consequence:

- rcParams that describe **ticks** (`xtick.direction`, `xtick.major.size`,
  `xtick.labelsize`, …) largely **do not reach a sky frame**.
- Everything else still works normally — `figure.figsize`, `font.family`,
  `font.size`, `axes.prop_cycle`, `image.cmap`, `savefig.*`, and so on.

So if a `plt.rcParams` tweak seems to be ignored on a sky map, it's almost
certainly a tick setting. `sph`'s frame builders apply the tick side at
build time from their own arguments, and
{func}`~skyplothelper.style_wcs_axes` retrofits the current style onto an
already-built frame:

```python
sph.style_wcs_axes(ax, direction="in", major_size=6)
```

## WCSAxes: axes that understand the sky

```{image} /_static/features/catalog-scatter-light.png
:class: sph-plot plot-light dark-light
:alt: A catalog scattered onto a sky frame (light mode)
```
```{image} /_static/features/catalog-scatter-dark.png
:class: sph-plot plot-dark dark-light
:alt: A catalog scattered onto a sky frame (dark mode)
```
*{doc}`Catalog scatter </features/catalog-scatter>` — code in the Feature Gallery.*

A **[WCSAxes](https://docs.astropy.org/en/stable/visualization/wcsaxes/index.html)**
is a matplotlib `Axes` that has been handed a WCS. Because it knows the
projection, it can draw a *curved* coordinate graticule, label ticks in the
sky system, and — crucially — offer a transform that projects sky
coordinates onto the canvas for you:

```python
ax.get_transform("world")      # (lon, lat) in the frame's own system -> canvas
ax.get_transform("galactic")   # (l, b) even if the frame itself is ICRS
ax.get_transform("pixel")      # raw pixel coordinates (imshow's default)
```

So plotting a catalog on a sky frame is just an ordinary `scatter` with the
world transform supplied:

```python
ax.scatter(ra_deg, dec_deg, transform=ax.get_transform("world"))
```

That one keyword — `transform=ax.get_transform("world")` — is the bridge
between plain matplotlib and the sky, and you'll see it throughout these
docs. Note the asymmetry it implies: **images go on in pixel coordinates**
(`ax.imshow(data)` needs no `transform=`, and the WCS makes the sky grid line
up over it), while **sky coordinates need the world transform**.

The same asymmetry governs **zooming**: because a sky frame's data coordinates
are pixels, `ax.set_xlim`/`set_ylim` take *pixels*, not degrees. To set the
view in world coordinates use the view helpers —
{func}`~skyplothelper.set_extent` (a `[lon_min, lon_max, lat_min, lat_max]`
box), {func}`~skyplothelper.zoom_to` (fit the view to a set of points), or
{func}`~skyplothelper.set_view` (a center and an angular width) — which frame a
region on any projection.

Ticks and grids are reached through `ax.coords` — per-axis objects with their
own API (`ax.coords[0].set_ticks(...)`, `.set_major_formatter(...)`,
`.grid(...)`). That's the machinery behind the rcParams caveat above, and
{doc}`ticks` covers what `sph` layers on top.

## Displaying image data

`ax.imshow` maps array values to colors through three knobs worth
understanding together:

```python
im = ax.imshow(data, origin="lower",        # FITS row order (see below)
               cmap="viridis",              # value -> color mapping
               vmin=lo, vmax=hi)            # clip limits
fig.colorbar(im, ax=ax, label="Jy/beam")
```

- **`origin`** — matplotlib defaults to `'upper'` (row 0 at the top, an
  image-processing convention), but FITS arrays are indexed from the
  bottom-left, so astronomy nearly always wants **`origin="lower"`**. A
  vertically flipped image is usually this.
- **`cmap`** — see matplotlib's
  [colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html);
  `sph` also registers its own set under an `sph.` prefix
  (`cmap="sph.deepsky"`, {doc}`styling`).
- **`vmin`/`vmax` or `norm=`** — astronomical images span huge dynamic range,
  so a linear scale between the extremes usually shows nothing. Pass a
  `norm=` instead: matplotlib offers `LogNorm`, `PowerNorm`, …, and astropy's
  [visualization](https://docs.astropy.org/en/stable/visualization/index.html)
  module composes an *interval* (how limits are chosen) with a *stretch* (the
  curve between them) via `ImageNormalize`.

`sph` wraps that composition in one call —
{func}`~skyplothelper.make_norm` (plus {func}`~skyplothelper.auto_stretch`
and {func}`~skyplothelper.list_stretches`), and
{func}`~skyplothelper.quicklook_plot` does the whole
load-stretch-display-colorbar sequence at once ({doc}`images`):

```python
norm = sph.make_norm(stretch="asinh", data=data)   # interval + stretch
ax.imshow(data, origin="lower", norm=norm, cmap="sph.deepsky")
```

## How skyplothelper builds on all this

`sph` doesn't replace any of the above — it removes the boilerplate and adds
the sky-specific care that raw matplotlib lacks:

| Doing it by hand | With skyplothelper |
|---|---|
| Hand-write a FITS header, build `WCS`, `add_subplot(projection=wcs)` | {func}`~skyplothelper.make_wcs_frame` / {func}`~skyplothelper.allsky_figure` — 32 projections, sensible defaults |
| `ax.scatter(ra, dec, transform=ax.get_transform("world"))` | the same — or {func}`~skyplothelper.plot_catalog` for sizing/coloring/legends |
| A great circle or region drawn with `ax.plot` **streaks across the map** at the ±180° seam | {func}`~skyplothelper.add_great_circle`, {func}`~skyplothelper.add_spherical_polygon`, … split/clip the seam correctly |
| `ImageNormalize(interval=…, stretch=…)` then `imshow` then `colorbar` | {func}`~skyplothelper.make_norm`, {func}`~skyplothelper.quicklook_plot`, {func}`~skyplothelper.add_colorbar` |
| rcParams by hand, tick styling through `ax.coords[...]` | {func}`~skyplothelper.set_style` / {func}`~skyplothelper.style_wcs_axes` |

The load-bearing point: **an `sph` frame *is* a real `WCSAxes`**, so
everything you know about matplotlib and astropy still applies. Drop to raw
`ax.plot(..., transform=ax.get_transform("world"))` whenever a helper doesn't
cover your case — just remember the seam caveat, which is the one thing the
`sph.add_*`/`plot_*` helpers handle that a bare `ax.plot` does not
({doc}`concepts` has the details).

```python
# The whole stack in five lines:
fig, ax = sph.allsky_figure(projection="AIT", center=180)   # WCS + WCSAxes
cat = SkyCoord.from_name("M31")                             # a position
ax.scatter(cat.ra.deg, cat.dec.deg,                        # plain matplotlib…
           transform=ax.get_transform("world"))            # …via the world transform
sph.add_constellation_boundaries(ax)                       # sph does the sky-aware parts
```

## Going deeper

External references, when you want the full story:

- **astropy** — [coordinates &
  `SkyCoord`](https://docs.astropy.org/en/stable/coordinates/index.html),
  [units](https://docs.astropy.org/en/stable/units/index.html),
  [FITS I/O](https://docs.astropy.org/en/stable/io/fits/index.html),
  [WCS](https://docs.astropy.org/en/stable/wcs/index.html),
  [WCSAxes](https://docs.astropy.org/en/stable/visualization/wcsaxes/index.html),
  [visualization &
  stretches](https://docs.astropy.org/en/stable/visualization/index.html)
- **matplotlib** — [transforms
  tutorial](https://matplotlib.org/stable/users/explain/artists/transforms_tutorial.html),
  [customizing with
  rcParams & style sheets](https://matplotlib.org/stable/users/explain/customizing.html),
  [colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)

And within these docs: {doc}`concepts` for the `sph`-wide conventions,
{doc}`frames` for the full frame-builder tour, {doc}`styling` for the
three-layer style system, {doc}`images` for the display stack, and the
{doc}`getting started tutorial </tutorials/getting_started>` for a runnable
first figure.
