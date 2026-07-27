# Core concepts & conventions

A handful of ideas and defaults run through the entire package. This page
collects them in one place so the other guide pages — and your own code —
can build on them without surprises. Five minutes here saves most of the
"why is my map mirrored?" class of confusion later.

```python
import skyplothelper as sph
```

The whole public API is re-exported at the top level: `sph.make_wcs_frame`,
`sph.add_great_circle`, `sph.plot_healpix_allsky`, and so on. Subpackages
exist (and organize the {doc}`API reference <../api/index>`), but you never
need to import them individually — the one exception is the interactive
backend, conventionally imported as `import skyplothelper.plotly as sphpl`.

## Everything starts from a frame

```{image} /_static/features/all-sky-frame-with-overlays-light.png
:class: sph-plot plot-light dark-light
:alt: An all-sky frame carrying coordinate overlays (light mode)
```
```{image} /_static/features/all-sky-frame-with-overlays-dark.png
:class: sph-plot plot-dark dark-light
:alt: An all-sky frame carrying coordinate overlays (dark mode)
```
*{doc}`All-sky frame with overlays </features/all-sky-frame-with-overlays>` — code in the Feature Gallery.*

A *frame* in skyplothelper is a matplotlib axes wired to a world coordinate
system, so that sky coordinates land in the right place on the canvas. With
one exception (the {doc}`cone frames <cone>`, which are polar wedges rather
than sky maps), frames are real astropy
{class}`~astropy.visualization.wcsaxes.WCSAxes` — anything astropy's WCSAxes
documentation says you can do, you can also do here.

Frames come in three shapes, and the builders select among them:

| Shape | Looks like | Typical projections | Builder shortcuts |
|---|---|---|---|
| **elliptical** | full-sky oval | AIT, MOL | {func}`~skyplothelper.allsky_figure` |
| **circular** | globe / hemisphere disk | SIN, ARC, STG, ZEA | {func}`~skyplothelper.make_globe_frame`, {func}`~skyplothelper.make_planet_frame` |
| **rectangular** | bounded field | TAN, CAR, MER, conics | {func}`~skyplothelper.offset_figure` |

{func}`~skyplothelper.make_wcs_frame` is the master builder underneath all of
these — every shape, every projection, every option. The shortcuts are
conveniences for the common cases. Once built, a frame is just an axes:
pass it to any `sph.add_*` / `sph.plot_*` helper, or to plain matplotlib
calls via `ax.plot(..., transform=ax.get_transform("world"))`.

```python
# The same frame three ways, increasingly explicit:
fig, ax = sph.allsky_figure(projection="MOL", center=180)
ax = sph.make_wcs_frame(111, "MOL", center=180)
ax = sph.make_wcs_frame(111, "mollweide", center_lon=180, center_lat=0)
```

Projection names accept FITS codes (`"AIT"`, `"TAN"`, ...) or friendly
aliases (`"hammer"`, `"gnomonic"`, ...); `sph.list_projections()` prints the
full table. See {doc}`frames` for the complete tour.

## The conventions

These defaults apply to **every** frame builder unless noted, and most
confusion traces back to one of them.

### Longitude direction: astronomical (east-left) by default

Sky maps and Earth maps disagree about which way longitude runs. Looking
*up* at the sky, east is to the **left** when north is up; looking *down*
at the Earth, east is to the right. skyplothelper is an astronomy package,
so every frame defaults to the astronomical convention:

- `direction='sky'` (aliases: `'astro'`, `'astronomical'`) — the default
  everywhere; east is to the left.
- `direction='geo'` (aliases: `'earth'`, `'geographic'`, `'cartographic'`) — east
  to the right, for terrestrial / planetary maps.

If an Earth map looks mirrored, this is why. The fix is **not** to flip
axes by hand — pass `direction='geo'`, or better, use
{func}`~skyplothelper.make_planet_frame`, which bundles the geographic
direction with the body-fixed coordinate frame in one call:

```python
# Earth globe, geographic orientation, the easy way:
ax = sph.make_planet_frame(111, center_LONdeg=-75, center_LATdeg=20)
```

(The cartopy backend, {func}`~skyplothelper.make_cartopy_frame`, also
defaults geographic, since cartopy is a mapping library.)

### Coordinate system: ICRS by default

`frame='ICRS'` is the default sky frame; `'galactic'`, `'ecliptic'`, and
`'supergalactic'` are accepted everywhere a frame is. This sets the
coordinate system *of the frame itself* — the gridlines, the tick labels,
where (lon, lat) inputs land. Converting data between systems is separate
and explicit:

```python
glon, glat = sph.convert_frame(ra, dec, from_frame="icrs", to_frame="galactic")
```

{func}`~skyplothelper.convert_frame` is the general converter; named
shortcuts like {func}`~skyplothelper.icrs_to_galactic` exist too, alongside
sexagesimal parsing/formatting, {func}`~skyplothelper.angulardistance`, and
the `wrap_*` longitude helpers — the whole coordinate/math toolkit is in the
{doc}`coordinates API reference </api/coordinates>` (it has no separate
guide page).

You can also draw a *second* coordinate system over an existing frame
(galactic gridlines on an ICRS map, say) without converting anything — see
{doc}`ticks` for coordinate overlays.

### Tick units: hours for RA, degrees for everything else

`lon_units='auto'` labels longitude in **hours** on equatorial frames (the
RA convention) and in **degrees** on geographic/planetary frames. Pass
`'degrees'` or `'hours'` to override. Latitude is always degrees.

### Centering

`center=` takes a longitude in degrees (all-sky maps), a `(lon, lat)` tuple,
a `SkyCoord`, or — wherever names make sense — anything the resolver
understands. All-sky figures default to `center=180`, which puts 0h RA at
the right edge. Passing a `(lon, lat)` tilts most all-sky projections to an
oblique aspect; a few projections constrain this (the HEALPix/quad-cube
family stays equatorial, conics center on their standard parallel) — see
{doc}`frames` for the per-projection rules.

## Projection, clipping & rendering

Every overlay, region, and decoration follows the same path from sky
coordinates to a drawn artist. Knowing the shape of that pipeline makes it easy
to predict how any helper behaves — and to find where a specific one does its
work.

**The pipeline, end to end:**

1. **Project.** Sky `(lon, lat)` go through one primitive,
   {func}`~skyplothelper.project`, shared by the matplotlib and plotly
   backends. You rarely call it directly — `x, y = sph.project(lon, lat,
   projection="AIT", center=180)` is there for custom work — but every helper
   routes through it.
2. **Handle the seam.** Geometry that crosses the projection's *antimeridian*
   (the ±180° wrap from the center) is split so it leaves one frame edge and
   re-enters the other instead of streaking across the map. Closed regions are
   clipped against the seam; open curves are broken into segments. You never
   pre-wrap your own coordinates.
3. **Clip to the frame.** The result is trimmed to the projection's visible
   boundary — the oval of an all-sky map, the disk of a globe, the rectangle of
   a field — so nothing spills outside the frame outline.
4. **Render.** The clipped geometry becomes a backend artist: a matplotlib
   patch/line, or a plotly SVG path / scatter trace.

How aggressively steps 2–3 run is the **`clip=`** knob that every closed-region
renderer accepts (`'auto'` → `'d3'` full spherical clipping, down through
`'simple'` / `'none'`); the options and when to reach for each are tabulated in
{doc}`regions`.

### One core, two backends

The compute — steps 1–3 — is **shared**, so a region looks the same whether you
draw it on a static matplotlib frame or an interactive plotly figure. A
backend-agnostic *projector* adapter owns the vetted antimeridian-clip →
project → stitch → frame-clip sequence; each backend only plugs in its own
projection primitive, frame outline, and final render step. (The matplotlib
side additionally overrides the polygon path to repair pixel-space pole and
jump cases the shared pipeline can't model; the plotly side runs the shared
pipeline directly on top of `sph.project()`.) The upshot: most `sph.add_*`
helpers have a `sphpl.add_*` twin that produces the same geometry
({doc}`plotly`) — what you learn once applies twice.

### Which route does a helper take?

A high-level map of how the main families reach the page, so you can place — and
trace — any given task:

| What you're drawing | Helpers | Seam / clip route |
|---|---|---|
| **Closed regions & polygons** | {func}`~skyplothelper.add_spherical_polygon`, {func}`~skyplothelper.add_rectangle` / {func}`~skyplothelper.add_square` / {func}`~skyplothelper.add_ellipse` / {func}`~skyplothelper.add_annulus`, {func}`~skyplothelper.add_geodesic_circle`, {func}`~skyplothelper.tissot` | the `clip=` polygon pipeline (antimeridian clip → project → frame clip → filled patch) |
| **Bands** | {func}`~skyplothelper.add_latitude_band`, {func}`~skyplothelper.add_longitude_band`, {func}`~skyplothelper.add_great_circle_band`, {func}`~skyplothelper.add_lonlat_box`, {func}`~skyplothelper.add_frame_band` | the same `clip=` polygon pipeline |
| **Compound (set-algebra) regions** | {class}`~skyplothelper.CompoundRegion` | each piece through the polygon pipeline, combined with shapely set ops, then clipped to the frame |
| **Survey footprints** | {func}`~skyplothelper.add_survey_footprint` | routes through the band / polygon pipeline (`clip=`) |
| **Open curves** | {func}`~skyplothelper.add_great_circle`, {func}`~skyplothelper.add_plane_overlay` | polyline antimeridian *split* (segmented, not area-clipped) |
| **HEALPix / raster data** | {func}`~skyplothelper.plot_healpix_map`, {func}`~skyplothelper.projection_gallery` | data-cell seam masking + boundary clip ({func}`~skyplothelper.mask_seam_crossing_quads`, {func}`~skyplothelper.clip_to_projection_boundary`) — not `clip=` |
| **Frame edge & graticule** | the frame builders ({doc}`frames`) | the projection's boundary curve + gridline backfill |

> **Note:** `clip=` governs the *vector* region/band families (the first four
> rows). Raster data (a HEALPix `pcolormesh`) is cleaned with the seam-mask +
> boundary-clip pair instead, and open curves are *split* rather than
> area-clipped — so reach for `clip=` on regions and bands, and the masking
> helpers when you hand-roll a `pcolormesh` ({doc}`healpix`).

## Optional dependencies fail softly

The core install needs numpy, matplotlib, astropy, shapely (the spherical
set-algebra geometry), and healpy (HEALPix — auto-skipped on Windows, which
has no healpy wheel; the HEALPix paths there raise an informative error while
everything else works). Heavier or backend-specific features sit behind
optional extras (cartopy, scipy, astroquery, reproject, plotly, dash) — see
{doc}`../installation`. Nothing is stripped from the package when an extra is
missing: calling an unavailable feature raises an informative `ImportError`
naming the extra to install, and everything else keeps working.

## Discovering what's available

Each registry has an enumerator, useful interactively:

```python
sph.list_projections()          # projection codes, aliases, shapes
sph.list_surveys()              # survey-footprint catalog
sph.list_constellations()       # IAU constellation names
sph.list_stretches()            # image-stretch names
sph.list_skyview_surveys()      # downloadable SkyView surveys
sph.list_cartopy_projections()  # cartopy backend projections
sph.describe_wcs(header)        # friendly summary of any WCS/FITS header
```

## Styling: rcParams and the WCSAxes caveat

{func}`~skyplothelper.set_style` composes three independent layers — a
structural base (tick geometry, legend, fonts), a light/dark *theme*
(backgrounds, foregrounds), and a *palette* (the data-color cycle):

```python
sph.set_style(base="standard", theme="dark_sky", palette="nightcap")
```

One caveat worth knowing early: **WCSAxes ignores most `xtick.*`/`ytick.*`
rcParams** — astropy routes tick control through its own `ax.coords` API
instead. skyplothelper's frame builders account for this at creation time,
and {func}`~skyplothelper.style_wcs_axes` retrofits the current style onto
an already-built frame. If you ever wonder why `plt.rcParams` tweaks aren't
reaching your sky frame, that's the mechanism. {doc}`styling` has the full
story.

## Where things live

The top-level namespace is flat, but the package (and the API reference) is
organized by subsystem:

| Subsystem | What's in it | Guide page |
|---|---|---|
| `wcs_frame`, `projections`, `figures` | frame builders, projection registry, `project()` | {doc}`frames` |
| `core.coords` & math | frame conversions, sexagesimal parsing/formatting, angular distance, longitude wrapping | {doc}`/api/coordinates` |
| `ticks`, `grid`, `coord_overlay` | tick formats, grid styling, second coordinate systems | {doc}`ticks` |
| `overlays` | planes, footprints, constellations, beams, rulers, reticles, markers | {doc}`overlays` |
| `geometry` | spherical regions + `CompoundRegion` set algebra | {doc}`regions` |
| `images`, `core.fits_utils` | stretch/normalize, quicklook, reprojection, header utilities | {doc}`images` |
| `healpix` | binning, plotting, queries, resolution changes | {doc}`healpix` |
| `globe` | globe views of sky & solid bodies, Earth/planet maps, nightshade, insets | {doc}`globe` |
| `cone` | z-RA wedge diagrams | {doc}`cone` |
| `queries` | name resolution, catalog and image services | {doc}`queries` |
| `data_plots`, `vsh`, `visibility` | vector fields, spherical harmonics, co-visibility | {doc}`vectors` |
| `style` | themes, palettes, publication presets | {doc}`styling` |
| `plotly` | interactive backend + Dash FITS viewer | {doc}`plotly` |

## Common pitfalls

- **Mirrored Earth map** → the frame is in the astronomical east-left
  default. Use {func}`~skyplothelper.make_planet_frame` or
  `direction='geo'`; don't flip axes manually.
- **RA ticks in degrees (or vice versa)** → set `lon_units=` explicitly;
  `'auto'` decides by coordinate frame, not by your preference.
- **rcParams not affecting a sky frame** → WCSAxes bypasses tick rcParams;
  use {func}`~skyplothelper.style_wcs_axes` or the builder arguments.
- **A line streaking across the whole map** → almost always a raw
  `ax.plot(..., transform=ax.get_transform("world"))` crossing the seam.
  The `sph.add_*`/`plot_*` helpers handle the split; use them, or break the
  array at the antimeridian yourself.
- **An oblique tilt that "doesn't take"** → a few projections constrain the
  aspect: the HEALPix/quad-cube family is equatorial-only (`center_lat`
  ignored) and conics center on their standard parallel. That's by design,
  not a bug — see {doc}`frames` for the per-projection rules.

**New here?** The {doc}`getting started tutorial </tutorials/getting_started>`
puts these conventions to work — your first frames, working with coordinates,
a first taste of overlays and tick formatting — with runnable figures.
