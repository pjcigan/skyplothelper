# Globe & planet plots

The globe subsystem renders orthographic views — the sphere shown as a
disk. That covers two equally common pictures: the **celestial sphere** as
a hemisphere of sky (the dome above an observatory, a survey hemisphere,
one pole's worth of catalog), and **solid bodies** — Earth, Moon, planets —
seen from outside, at any orientation including a physically tilted,
rotating Earth. Under the hood these are SIN-projection WCS frames, so
everything from the rest of the package (overlays, regions, HEALPix maps)
draws onto them; what this page adds is the orientation machinery,
hemisphere-aware plotting, Earth/planet surface features, and
globe-specific decorations.

```python
import skyplothelper as sph
import matplotlib.pyplot as plt
```

## Building a globe

```{image} /_static/features/celestial-globe-light.png
:class: sph-plot plot-light dark-light
:alt: An orthographic globe view of the sky with its graticule (light mode)
```
```{image} /_static/features/celestial-globe-dark.png
:class: sph-plot plot-dark dark-light
:alt: An orthographic globe view of the sky with its graticule (dark mode)
```
*{doc}`Celestial globe </features/celestial-globe>` — code in the Feature Gallery.*

```python
fig = plt.figure()
# grid=False: let plot_ortho_grid draw the graticule (its front/back styling)
# instead of the builder's default grid, to avoid a faint double graticule.
ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=23.44, grid=False)
sph.plot_ortho_grid(ax)
```

{func}`~skyplothelper.make_globe_frame` builds a celestial globe centered
on whatever point of sky you choose (`center_LONdeg`, `center_LATdeg`) —
center it on your observatory's zenith for a "what's up tonight"
hemisphere, on a celestial pole for a polar-cap view, or on a target
region for a hemisphere of context around it.

For solid bodies, {func}`~skyplothelper.make_planet_frame` is the right
entry point: it selects the **geographic** longitude convention and the
body-fixed coordinate system in one call (`body='earth'` by default), so
continents come out un-mirrored. This is the one place the package's
astro east-left default trips people most — see {doc}`concepts`.

```python
ax = sph.make_planet_frame(111, center_LONdeg=-75, center_LATdeg=20)
```

{func}`~skyplothelper.plot_ortho_grid` draws the globe graticule with
independent styling for the front and back hemispheres (the dotted
"see-through" back is a nice touch for wireframe looks).
{func}`~skyplothelper.highlight_great_circle` emphasizes any great circle
traced *completely* around the globe — front solid, far side dashed at a
matched color and weight (the color defaults to the theme's frame color).
Specify the circle by its `pole`, two `points` on it, or an orbital
`inclination`/`node`. {func}`~skyplothelper.highlight_meridian_tracer` is
the convenience special case for a meridian (meridian + antimeridian, over
both poles).

### Tilted orientations

A physically *tilted* globe — the obliquity-inclined, spinning Earth as
seen from its orbital plane — is an important special case with dedicated
machinery:

- {func}`~skyplothelper.euler_to_fits_ortho` converts the *physical*
  rotation state — `(rotation, obliquity, perspective)` Euler angles, the
  intuitive parameters for a spinning tilted planet — into the
  `(center_lon, center_lat, lonpole)` values the frame builders take.
  Vectorized, which makes rotation *sequences* (animation frames) one
  call. Perspective (precession) is only visible with a tilted pole — at
  `obliquity=0` it simply adds to `rotation`.
  {func}`~skyplothelper.quaternion_to_fits_ortho` is the quaternion
  counterpart (with a `scalar_first=` order knob); it agrees with the
  Euler form for the same orientation.
- {class}`~skyplothelper.TiltedEarthFrame` is an astropy coordinate frame
  with those three Euler angles as attributes — use it as a coordinate
  overlay to draw a tilted graticule on any frame.
- {func}`~skyplothelper.make_globe_angles` generates *sequences* of
  orientation angles (spin, nutation, precession over `n_steps`) for
  animating a rotating, nutating, or precessing globe. Its default output
  is exactly the `(center_lon, center_lat, lonpole)` triples the frame
  builders take, so each step feeds straight into
  {func}`~skyplothelper.make_globe_frame` /
  {func}`~skyplothelper.make_planet_frame` (the `euler_to_fits_ortho`
  conversion happens inside). The bundled `obliquities` and `rot_periods`
  tables (axial tilts and spin periods for solar-system bodies, next to
  `planet_radii`) supply physically motivated rates.

```{tip}
**Generating the angle series.** For "one clean full rotation (or
precession) over the whole clip," reach for the *whole-animation totals*
instead of hand-computed per-step rates: `spin_total=` / `prec_total=`
set the total sweep in degrees, and `nut_cycles=` the number of nutation
oscillations, over all `n_steps`. Each is endpoint-exclusive, so
`spin_total=360` loops seamlessly. (Pass a total **or** its per-step
`*_rate`, never both.)

    lons, lats, poles = sph.make_globe_angles([0, 24, 45], 120, spin_total=360.)

Precession here drives the **perspective** (third Euler / `psi`) angle.
Under skyplothelper's per-frame frame re-aiming that reads on screen as
the pole *precessing* around the sky; the same series fed to a
fixed-camera 3-D engine instead spins the body about its already-tilted
pole (the pole itself stays put) — worth knowing if you ever port these
angles to an external 3-D renderer.
```

## Plotting on a globe

The far side of a globe is the classic trap — a naive
`ax.plot(..., transform=...)` happily draws right through the sphere. The
globe plotters are hemisphere-aware:

- {func}`~skyplothelper.plot_scatter_globe`,
  {func}`~skyplothelper.plot_line_globe` — points and polylines, far-side
  portions masked (or restyled).
- {func}`~skyplothelper.plot_pcolormesh_globe`,
  {func}`~skyplothelper.plot_contour_globe` — gridded fields on the
  visible hemisphere.
- {func}`~skyplothelper.imscatter_globe` (and
  {func}`~skyplothelper.imscatter` / {func}`~skyplothelper.imscatter_rotated`)
  — scatter *image stamps* (icons, thumbnails) at sky positions instead of
  markers, hemisphere-aware on a globe. `zoom=` takes an array as readily as a
  scalar, sizing each stamp individually — the raster counterpart of
  `scatter(s=...)`.
- {func}`~skyplothelper.orthographic_visibility` — the underlying "is
  this point on the visible side?" test, public for your own logic, with
  {func}`~skyplothelper.orthographic_forward` /
  {func}`~skyplothelper.orthographic_inverse` for direct projection math.

### Aiming & mirroring image stamps

{func}`~skyplothelper.imscatter_rotated` can point an icon *at a target*:
pass `aim_at=` (a position or `SkyCoord`; mutually exclusive with `rotations=`)
together with `rest_angle=` — the icon's **native boresight**, the direction
the un-rotated image already points, in degrees counter-clockwise from
screen-right. The rotation is then solved as `aim_angle - rest_angle`. This is
the raster twin of the instrument markers' aiming, and
{func}`~skyplothelper.aim_angles` exposes the same solver ({doc}`overlays`).
Measured `rest_angle` values for the bundled icons live in
`examples/data/README.md` — the source of truth (the radio dish, for instance,
points at 130°). One asymmetry to note: `target_coords=` defaults to `'data'`
here (matching this function's own `x`/`y`), whereas `aim_angles` defaults to
`'display'`.

Two similar-sounding knobs are unrelated. `imscatter_rotated`'s `flip=` mirrors
an icon horizontally when the target lies on its far side, so an aimed icon
doesn't roll past vertical and read upside-down. `imscatter_globe` instead
mirrors stamps by a **hemisphere rule** — which side of the globe a point falls
on. Same word, different mechanism. (`imscatter_globe` also names the
upper-right-icon assumption it has always made, as `rest_angle=45.0`.)

The spherical-geodesy helpers underneath are general-purpose:
{func}`~skyplothelper.great_circle_distance`,
{func}`~skyplothelper.great_circle_arc`, {func}`~skyplothelper.midpoint`,
{func}`~skyplothelper.initial_bearing`,
{func}`~skyplothelper.destination_point`,
{func}`~skyplothelper.small_circle`.

## Earth & planet surfaces

```{image} /_static/features/earth-with-surface-features-light.png
:class: sph-plot plot-light dark-light
:alt: An Earth globe with coastlines and surface features (light mode)
```
```{image} /_static/features/earth-with-surface-features-dark.png
:class: sph-plot plot-dark dark-light
:alt: An Earth globe with coastlines and surface features (dark mode)
```
*{doc}`Earth with surface features </features/earth-with-surface-features>` — code in the Feature Gallery.*

**Vector features** — coastlines, filled land, lakes, rivers, tectonic
plates, and time zones — draw from small data files fetched **once per
environment** with {func}`~skyplothelper.prepare_earth_data` (Natural Earth
via the optional `cartopy` extra, plus the Bird 2003 plate polygons; see
{doc}`../installation`). They are not shipped with the package:

```python
sph.prepare_earth_data()          # one-time; needs the cartopy extra + network
sph.plot_coastlines(ax)
sph.plot_land(ax, lakes=True)     # filled land, lakes punched out as holes
sph.plot_rivers(ax)
sph.plot_tectonic_plates(ax, color="tab:red")
sph.plot_time_zones(ax)
```

The **area** features ({func}`~skyplothelper.plot_land`,
{func}`~skyplothelper.plot_lakes`, filled plates) fill through the same
region machinery as {func}`~skyplothelper.add_spherical_polygon`, so they
work on the flat all-sky projections and the custom Robinson/Eckert frames
as well as on the globe. {func}`~skyplothelper.plot_tectonic_plates` takes
`fill=True` for filled plates — one color, a categorical map, or a
`values=`-driven **choropleth** (the general
{func}`~skyplothelper.choropleth` helper does the same for any list of
rings). {func}`~skyplothelper.clip_to_land` /
{func}`~skyplothelper.clip_to_ocean` mask any artist — an image drape, a
scatter — to the coastline.

skyplothelper's Earth maps aim to make whole-globe views and simple
planetary plots look good with little setup; for heavy terrestrial
cartography (fine-resolution features, national borders, filled land/ocean
at scale, GIS queries) reach for the cartopy backend below.

({func}`~skyplothelper.load_boundary_data` and
{func}`~skyplothelper.fetch_boundary_data` are the lower-level load / mirror-
download helpers; {func}`~skyplothelper.plot_boundaries_globe` /
{func}`~skyplothelper.plot_boundaries_ortho` draw arbitrary boundary
datasets, and {func}`~skyplothelper.split_segments` breaks polylines at
the visibility horizon.)

**Raster maps** — any equirectangular texture (NASA Blue Marble, Moon and
Mars mosaics, ...) becomes a globe surface via
{func}`~skyplothelper.pseudofits_from_image`, which wraps the image in a
synthetic WCS so the standard reprojection machinery can drape it. It takes
a file path or an array already in memory, so a raster you *computed* —
a composite, a model map, a blended day/night frame — drapes the same way a
file does, without a round trip through disk.

**Nightshade** — {func}`~skyplothelper.make_nightshade_blend` blends a
day raster against night (darkened, or a night-lights image) for a given
`date`. The default `blend='elevation'` mode computes the actual solar
elevation across the surface, giving a physical terminator with twilight
falloff; a softer cosmetic `'gaussian'` mode is available when you want
a stylized look.

```python
shaded = sph.make_nightshade_blend(day_rgb, date="2026-06-21 18:00")
```

## The cartopy backend

For terrestrial maps that want cartopy's feature stack (coastlines,
borders, land/ocean fills, the full projection library), build a cartopy
`GeoAxes` instead of a WCS globe: {func}`~skyplothelper.make_cartopy_frame`
and the one-call {func}`~skyplothelper.cartopy_figure`, with
{func}`~skyplothelper.list_cartopy_projections` enumerating the options.
Like {func}`~skyplothelper.make_planet_frame`, these default to the
geographic (east-right) convention. (Requires the `cartopy` extra; the
trio is listed in the {doc}`utilities API reference </api/utilities>`.)

## Decorations & insets

Globe-specific furniture: {func}`~skyplothelper.add_compass_rose` (a fixed
N/E/S/W rose in screen space) versus
{func}`~skyplothelper.add_surface_compass` (planted *on* the surface at a
`(lon, lat)`, so it foreshortens with the globe — `style=` picks a two-tone
`'star'` rose, a connected `'arrow'` N+E frame, or geodesic `'lines'`
arms), {func}`~skyplothelper.add_checkered_border` (surveyor-style
alternating border), {func}`~skyplothelper.add_pole_rod` (the axis rod
through the poles, for orientation at a glance), and
distance scale bars — {func}`~skyplothelper.add_scale_bar` with
`_cylindrical` and `_curved_parallel` variants that follow a parallel at
the chosen latitude, in real distance units via the body radius
(`planet_radii` covers the solar system).

Inset axes connect a globe to a zoom:

```python
inset = sph.reproject_inset_axes(ax, [0.65, 0.05, 0.3, 0.3],
                                 projection="TAN", center=(-80, 25), size=8)
sph.mark_inset_axes(ax, inset)
sph.connect_inset_axes(ax, inset)
```

{func}`~skyplothelper.reproject_inset_axes` builds a *reprojected* child
frame (different projection, different scale) and inherits the parent's
on-screen longitude direction by default, so a geographic parent never
gets a mirrored inset. The connector lines accept a curvature control for
when straight connectors would cut through the marked region.

## Pitfalls

- **Mirrored continents** → you built a celestial frame for terrestrial
  data; use {func}`~skyplothelper.make_planet_frame` (see
  {doc}`concepts`).
- **Tracks drawn through the planet** → raw matplotlib calls aren't
  hemisphere-aware; use the `plot_*_globe` family or mask with
  {func}`~skyplothelper.orthographic_visibility`.
- **`plot_coastlines` (or `plot_land` / `plot_tectonic_plates` / …)
  complaining about missing data** → run
  {func}`~skyplothelper.prepare_earth_data` once per environment to fetch
  and cache the vector Earth data (needs the `cartopy` extra).
- **Scale bar length looks wrong on another body** → pass the body so the
  radius lookup matches (`planet_radii` keys); a Mars km is not an Earth
  degree.

Baseline networks between ground stations
({func}`~skyplothelper.plot_baselines`) and co-visibility regions are
covered with the other station-network tools in {doc}`vectors`. Full
listing: {doc}`API reference <../api/globe>`.

**See also:** {doc}`frames` — globe frames are SIN-projection WCSAxes, so
the centering/aspect rules and the rest of the frame toolkit apply here too.

**Tutorials:** {doc}`Globe & planet plotting </tutorials/globe_plots>`
(tilted-Earth orientation, raster planet maps, Earth features, nightshade,
and globe decorations); {doc}`Insets & zoom axes </tutorials/insets_and_zoom>`
for the inset/zoom machinery; and {doc}`Animations </tutorials/animations>`
for rotating globes and the advancing terminator.
