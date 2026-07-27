# Frames & projections

Every skyplothelper plot starts with a *frame*: a matplotlib axes wired to a
world coordinate system so that sky positions land where the projection says
they should. This page covers building frames — the master builder, the
convenience wrappers, the 32 available projections and how to choose among
them — plus the utilities for offset (tangent-plane) coordinates and
synthetic WCS headers. The conventions that apply to every builder
(`direction=`, `frame=`, `lon_units=`, centering) are covered once in
{doc}`concepts` and not repeated here.

```python
import skyplothelper as sph
import matplotlib.pyplot as plt
```

## The master builder: `make_wcs_frame`

{func}`~skyplothelper.make_wcs_frame` builds any frame the package supports —
all-sky ovals, globe disks, bounded fields — and returns a ready-to-draw
WCSAxes. A minimal call is just a subplot position and a projection:

```python
ax = sph.make_wcs_frame(111, "AIT", center=180)
```

Its arguments fall into three groups:

**What sky goes on the canvas** — `projection=` (FITS code or alias),
`center=` / `center_lon=` / `center_lat=`, `frame=` (`'ICRS'`,
`'galactic'`, ...), `direction=`, and for the projections that need them,
`lonpole=`/`latpole=` and the conic `pv2_1=`/`pv2_2=` parameters (sensible
defaults are supplied, so conics work out of the box).

**How much of it** — for bounded fields, set the field of view with
`fov_deg=` (or, FITS-style, `cdelt=` + `npix=`). All-sky projections ignore
these and show the whole sphere. `shape=` overrides the frame outline
(elliptical / circular / rectangular) when you want something other than
the projection's natural boundary.

**How it's dressed** — `grid=`, `gridcolor=`, `gridalpha=`,
`lon_spacing=`/`lat_spacing=` (`'auto'` picks round values),
`lon_units=`, `tick_style=`, `tick_rotation=`, and `auto_fontsize=`. These
are creation-time conveniences; everything is re-tunable afterward (see
{doc}`ticks`).

Two practically useful extras: `fig=` targets an existing figure (for
multi-panel layouts), and `return_hdr=True` also returns the synthesized
FITS header, handy when downstream code (reprojection, HEALPix rasterizing)
needs the WCS itself.

```python
# Galactic-frame Mollweide with a coarser grid, on an existing figure
fig = plt.figure(figsize=(9, 5))
ax = sph.make_wcs_frame(111, "MOL", frame="galactic", center=0,
                        lon_spacing=30, lat_spacing=15, fig=fig)
```

## Convenience builders

For the everyday cases, one call makes the figure and the frame together:

| Builder | Makes | Notes |
|---|---|---|
| {func}`~skyplothelper.allsky_figure` | full-sky elliptical frame | returns `(fig, ax)`; `style=` applies a tick/label preset |
| {func}`~skyplothelper.offset_figure` | tangent-plane field on a target | `center=` + `fov_deg=`; offset (relative) coordinates |
| {func}`~skyplothelper.make_globe_frame` | orthographic celestial globe | see {doc}`globe` |
| {func}`~skyplothelper.make_planet_frame` | Earth/planet globe, geographic convention | see {doc}`globe` |
| {func}`~skyplothelper.make_cone_frame` | z–RA wedge (not a WCS frame) | see {doc}`cone` |
| {func}`~skyplothelper.make_cartopy_frame` | cartopy GeoAxes | terrestrial maps with cartopy's feature stack |
| {func}`~skyplothelper.projection_gallery` | grid of frames across projections | quick visual comparison |

```python
fig, ax = sph.allsky_figure(projection="AIT", center=180)

# 12-arcmin TAN field on the Crab Nebula, offset coordinates
fig, ax = sph.offset_figure(center=(83.63, 22.01), fov_deg=0.2)
```

## The projections

```{image} /_static/features/projection-gallery-light.png
:class: sph-plot plot-light dark-light
:alt: The same all-sky field rendered in many projections (light mode)
```
```{image} /_static/features/projection-gallery-dark.png
:class: sph-plot plot-dark dark-light
:alt: The same all-sky field rendered in many projections (dark mode)
```
*{doc}`Projection gallery </features/projection-gallery>` — code in the Feature Gallery.*

{func}`~skyplothelper.list_projections` prints the full registry — 27 FITS projections
(everything astropy/wcslib supports) plus five classic compromise
projections implemented as custom matplotlib frames (Robinson,
Kavrayskiy VII, Eckert IV, Winkel Tripel, McBryde–Thomas). Each entry lists
its aliases, natural frame shape, and whether it can show the full sky.
The ones you'll probably encounter the most:

| Code | Name | Character | Typical use |
|---|---|---|---|
| `AIT` | Hammer–Aitoff | equal-area, elliptical | the default all-sky map |
| `MOL` | Mollweide | equal-area, elliptical | all-sky, straight parallels |
| `CAR` | Plate Carrée | equirectangular | simple lon/lat grids, Earth maps |
| `SFL` | Sanson–Flamsteed | equal-area, sinusoidal | all-sky with straight parallels |
| `TAN` | Gnomonic | tangent plane | fields, FITS images, interferometry |
| `SIN` | Slant orthographic | globe view | hemispheres, tilted globes |
| `STG` | Stereographic | conformal disk | wide fields with low shape distortion |
| `ZEA` | Lambert azimuthal | equal-area disk | polar caps, hemisphere statistics |
| `MER` | Mercator | conformal, cylindrical | low-latitude strips |
| `COE` ... | conics | between cylindrical and azimuthal | mid-latitude regions |

Quick guidance on choosing: use an **equal-area** projection (AIT, MOL, SFL,
CEA, ZEA) whenever the *density* of things matters — source counts, survey
coverage, HEALPix maps — so a deg² covers the same canvas area everywhere.
Use a **conformal** projection (TAN, STG, MER) when local *shapes* matter —
imaging fields, morphology. The **compromise** projections (Robinson,
Winkel Tripel, ...) trade a little of both for looks, which is exactly what
you want in outreach figures. When in doubt:
{func}`~skyplothelper.projection_gallery` renders your choice of projections
side by side.

```python
sph.projection_gallery(projections=["AIT", "MOL", "SFL", "robinson"], center=180)
```

All 27 FITS projections plus the five compromise frames render as proper
all-sky maps — with a complete graticule and a drawn boundary outline — and
the interrupted / "oddball" ones (HEALPix `HPX`/`XPH`, Bonne `BON`, polyconic
`PCO`, the quad-cubes, and the conics) clip data to their visible region so
nothing bleeds past the frame.

## Centering, aspect & projection constraints

`center=` and the conventions behind it are covered in {doc}`concepts`; a few
projection-specific constraints are worth knowing when you go beyond the
common set (all are spelled out in the {func}`~skyplothelper.make_wcs_frame`
docstring):

- **Oblique aspect.** Most all-sky projections honor `center=(lon, lat)` to
  tilt the map off the equator. The exceptions are the pole-tiled
  **HEALPix/quad-cube** projections (`HPX`, `XPH`, `TSC`, `CSC`, `QSC`), which
  stay equatorial — `center_lat` is ignored for them.
- **Conics** (`COD`/`COE`/`COO`/`COP`) are all-sky by default, centered on
  their standard parallel (set via `pv2_1`, default 45°); `center_lat` doesn't
  apply. `COO`/`COP` clip the divergent far pole. Pass `fov_deg=` for a zoomed
  regional view instead.
- **Quad-cubes** read cleanest at a face-aligned `center_lon` of 0/90/180/270.
- **Bonne / polyconic** (`BON`, `PCO`) accept an oblique `center_lat`
  mathematically, but the boundary outline overflows under a latitude shift —
  prefer longitude shifts for clean frames.

Two finite-resolution caveats for raster overlays on interrupted projections:
pcolormesh/HEALPix data can't completely fill the extreme corners of
`HPX`/`XPH`/cube frames (thin edge gaps that shrink as `nside` grows), and
`PCO`'s overlapping lobes double-value data beyond |lon| ~ 90°. Use
{func}`~skyplothelper.clip_to_projection_boundary` to clip a custom data
artist to the visible region (the built-in plotters already do this).

### Tick placement on odd frames

`make_wcs_frame`'s `tick_style=` controls where coordinate ticks land:
`'auto'` (default) routes the interrupted projections to legible in-frame
central crosshair labels; `'boundary'` places ticks on the projection's true
edge (the HEALPix diamond, the conic wedge, the Bonne cardioid, …) rather than
the canvas rectangle; `'in_frame'` and `'native'` force the other two modes.

## Offset coordinates & tangent-plane fields

Zoomed-in fields usually want *relative* coordinates — arcsec or arcmin from
a reference position — rather than absolute RA/Dec.
{func}`~skyplothelper.offset_figure` builds this directly. For finer
control, the underpinnings are public:

- {func}`~skyplothelper.WCS_to_offsetWCS` — convert an absolute WCS into an
  offset WCS about a reference point.
- {func}`~skyplothelper.offset_coord_WCS` — the offset-coordinate WCS for a
  given center and scale.
- {func}`~skyplothelper.apply_boundary_labels` — label a frame's boundary
  with offset-style tick labels, oriented `perpendicular` (default),
  `parallel`, or `horizontal` to the edge.

The offset tick styles themselves (arcsec offsets, VLBI hybrid
absolute+offset labeling) live with the rest of the tick machinery — see
{doc}`ticks`.

## Drawing data on a frame

A WCSAxes plots in *pixel* coordinates by default, so raw `ax.plot(ra, dec)`
lands in the wrong place. skyplothelper mirrors the everyday matplotlib
plotting methods as module-level functions that take **sky** coordinates
instead — a {class}`~astropy.coordinates.SkyCoord` or plain `(lon, lat)` in
degrees — project them through the frame, honor a `frame=` conversion, and
split lines at the antimeridian so nothing streaks across an all-sky map:

```python
import numpy as np
import skyplothelper as sph

ax = sph.make_wcs_frame(111, "AIT", center=0)

lon = np.linspace(-170, 170, 60)
lat = 30 * np.sin(np.radians(lon))
sph.plot(ax, lon, lat, frame="galactic", color="C1")   # a great-circle-ish track
sph.scatter(ax, [45, 120], [10, -20], s=40)            # points
sph.text(ax, 0, 0, "GC", frame="galactic", ha="center")
```

The full set — {func}`~skyplothelper.plot`, {func}`~skyplothelper.scatter`,
{func}`~skyplothelper.errorbar`, {func}`~skyplothelper.step`,
{func}`~skyplothelper.fill`/{func}`~skyplothelper.fill_between`,
{func}`~skyplothelper.text`/{func}`~skyplothelper.annotate`,
{func}`~skyplothelper.contour`/{func}`~skyplothelper.contourf`/{func}`~skyplothelper.tricontourf`,
{func}`~skyplothelper.pcolormesh`, {func}`~skyplothelper.hist2d` — each forwards
its `**kwargs` straight to the matplotlib method of the same name. For a custom
artist not in the list, draw it yourself against
{func}`~skyplothelper.world_transform`, the `(lon, lat) → display` transform:

```python
ax.plot(lon, lat, transform=sph.world_transform(ax))
```

The higher-level catalog and vector plotters ({doc}`vectors`) are built on top
of these; reach for the passthroughs when you want a plain matplotlib call to
simply understand sky coordinates.

## Synthetic headers & frame utilities

Sometimes you need a WCS without having data yet — for layout planning,
testing, or rasterizing onto a target grid:

- {func}`~skyplothelper.dummy_allsky_hdr`, {func}`~skyplothelper.dummy_ortho_hdr`,
  {func}`~skyplothelper.dummy_offset_hdr`, {func}`~skyplothelper.dummy_standard_hdr`
  — ready-made FITS headers for each frame family.
- {func}`~skyplothelper.get_frame_class` — the matplotlib frame class
  (elliptical, circular, ...) registered for a projection.
- {func}`~skyplothelper.clip_to_frame` — clip *all* data artists on a
  frame to its curved boundary (handy after adding artists outside the
  helpers); {func}`~skyplothelper.clip_to_projection_boundary` clips a
  *single* artist to the projection's visible region (the per-artist
  variant the interrupted-projection plotters use).
- {func}`~skyplothelper.describe_wcs` — print a readable summary of any
  WCS or FITS header.

### The projection primitive

Underneath every overlay sits one function, {func}`~skyplothelper.project`,
mapping sky `(lon, lat)` to canvas `(x, y)` for a given projection/center —
the primitive shared by the matplotlib and plotly backends ({doc}`concepts`).
You rarely call it directly, but it's public for custom work, alongside the
lower-level {func}`~skyplothelper.project_to_canvas` and the HEALPix-specific
{func}`~skyplothelper.healpix_to_canvas` ({doc}`healpix`).

## Pitfalls

- **A field that's all sky, or an all-sky map that's a postage stamp** —
  on the elliptical all-sky projections (AIT, MOL, …) `fov_deg=`/`cdelt=`/
  `npix=` don't crop; they always show the full sphere. The conics are the
  exception: they're all-sky by default but switch to a zoomed regional
  view when you pass `fov_deg=`.
- **Conic projections erroring in other tools** — conics genuinely require
  PV parameters; skyplothelper supplies a usable default and exposes
  `pv2_1=`/`pv2_2=` when you need a specific standard parallel.
- **Mirrored or "backwards" maps** — longitude direction, not the
  projection. See {doc}`concepts`.
- **Mixing offset and absolute thinking** — an `offset_figure` frame labels
  positions *relative to the center*; overlay helpers still take absolute
  sky coordinates and project correctly. Don't pre-subtract the center
  yourself.

The full builder/utility listing is in the {doc}`API reference
<../api/frames>`.

**See also:** {doc}`images` (reproject imagery onto a frame),
{doc}`styling` (the `style=` tick/label presets), {doc}`healpix`
(HEALPix rasters and the HPX pole-lock).

**Tutorial:** {doc}`A tour of projections </tutorials/projections>` walks
through the projection gallery, choosing among frame types, and the
longitude conventions with worked figures.
