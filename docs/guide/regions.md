# Regions & spherical geometry

skyplothelper draws sky regions — circles, rectangles, ellipses, bands,
polygons, and arbitrary set-algebraic combinations of them — as
*projection-aware* patches: edges follow the curved geometry of the sphere,
boundaries that cross the projection's antimeridian are clipped and closed
correctly (including regions that enclose a pole), and the same region
renders identically on the matplotlib and plotly backends. This page covers
the three layers of the system and the shared keywords that control them.

The region set algebra runs on shapely, a core dependency of skyplothelper,
so regions work out of the box — no optional extras.

```python
import skyplothelper as sph
fig, ax = sph.allsky_figure(projection="AIT", center=180)
```

## Three layers

**1. Vertex constructors** compute boundary coordinates and hand them to
you — no drawing. Use these when you want the raw `(lon, lat)` outline for
your own machinery: {func}`~skyplothelper.geodesic_circle`,
{func}`~skyplothelper.rectangle`, {func}`~skyplothelper.ellipse`.

**2. Renderers** draw a single region onto an axes in one call — the
`add_*` family: {func}`~skyplothelper.add_geodesic_circle`,
{func}`~skyplothelper.add_spherical_polygon`,
{func}`~skyplothelper.add_rectangle`, {func}`~skyplothelper.add_square`,
{func}`~skyplothelper.add_ellipse`, {func}`~skyplothelper.add_annulus`,
the band helpers ({func}`~skyplothelper.add_latitude_band`,
{func}`~skyplothelper.add_longitude_band`,
{func}`~skyplothelper.add_great_circle_band`,
{func}`~skyplothelper.add_frame_band`), and
{func}`~skyplothelper.add_lonlat_box`. {func}`~skyplothelper.tissot`
belongs here too — it draws a grid of equal-radius geodesic circles
(Tissot-style indicatrices) to visualize a projection's distortion.

**3. `CompoundRegion`** combines shapes with set algebra — union,
intersection, difference, symmetric difference — and renders the result as
a single patch, holes included. This is the layer for "inside survey A but
outside the galactic plane" masks.

```python
# Layer 2: one call per region
sph.add_geodesic_circle(ax, lon=266.4, lat=-29.0, radius_deg=20,
                        color="tab:orange", alpha=0.3)
sph.add_frame_band(ax, -10, 10, frame="galactic", alpha=0.2)
```

## The shared renderer surface

The closed-region renderers share a common set of keywords, so one mental
model covers all of them.

### `clip=` — how the projection seam is handled

Every closed-region renderer accepts `clip=`, selecting how the boundary is
clipped against the projection's antimeridian and frame edge:

| `clip=` | Behavior |
|---|---|
| `'auto'` (default) | the helper's principled default — `'d3'` (full spherical clipping) for the closed-region patch helpers |
| `'d3'` | full spherical clipping; handles wrap-around and pole-enclosing regions |
| `'project_shape'` | a faster path, with a few known edge artifacts |
| `'simple'` | raw vertex projection, no clipping |
| `'none'` | skip clipping entirely — raw vertices, no antimeridian handling |

Leave it on `'auto'` unless you're trading robustness for speed on
thousands of small shapes, or debugging a seam.

### `backend=` — what kind of artist is produced

`'patch'` (a filled `PathPatch`) is the default and, for most helpers, the
only option. Helpers with genuinely multiple renderings accept more:
{func}`~skyplothelper.add_frame_band` takes `'contour'`, and the HEALPix
renderers take `'pcolormesh'`/`'imshow'` (see {doc}`healpix`). Singular and
plural spellings are interchangeable.

### `resolution=` — boundary sampling

Every renderer takes `resolution=`, the number of points along the boundary
(for vertex-list shapes, per edge). Defaults are sensible; raise it if a
large region's edge looks faceted in a strongly curved projection.

### `complement=` — fill the outside

Most shape renderers accept `complement=True` to fill everything *except*
the region — the quick way to draw exclusion zones and avoidance masks.

### `stroke_color=` — legibility over imagery

The closed-region helpers — the band and box helpers
({func}`~skyplothelper.add_latitude_band`,
{func}`~skyplothelper.add_longitude_band`,
{func}`~skyplothelper.add_great_circle_band`,
{func}`~skyplothelper.add_lonlat_box`, {func}`~skyplothelper.add_frame_band`),
the shape helpers ({func}`~skyplothelper.add_spherical_polygon`,
{func}`~skyplothelper.add_geodesic_circle`, and the rectangle/ellipse
family), {func}`~skyplothelper.tissot`, and
{meth}`~skyplothelper.CompoundRegion.render` — take the package's usual
`stroke_color=`/`stroke_lw=` pair, outlining their edges so a region stays
readable across a bright background.

### Geodesic vs. linear edges

On a sphere, "straight line between two vertices" is ambiguous: a great
circle, or a straight line in lon/lat space? For small shapes the
difference is invisible; for survey-scale polygons it is not.
{func}`~skyplothelper.add_spherical_polygon` exposes the choice as
`geodesic=` — `'auto'` (decide per edge by length), `True` (always
great-circle), `False` (always linear, i.e. edges follow graticule lines).
Surveys whose footprints are defined by constant-RA/Dec boundaries want
`False`; physical regions on the sky generally want `True` or `'auto'`.

```python
sph.add_spherical_polygon(ax, lons=[30, 80, 80, 30], lats=[10, 10, 45, 45],
                          geodesic=False, alpha=0.3)   # graticule-bounded
```

## Compound regions: set algebra on the sphere

```{image} /_static/features/compound-region-light.png
:class: sph-plot plot-light dark-light
:alt: A cap minus an inner hole minus the galactic plane (light mode)
```
```{image} /_static/features/compound-region-dark.png
:class: sph-plot plot-dark dark-light
:alt: A cap minus an inner hole minus the galactic plane (dark mode)
```
*{doc}`Compound region </features/compound-region>` — code in the Feature Gallery.*

{class}`~skyplothelper.CompoundRegion` is built against an axes (it needs
the frame's projection to do its planar geometry) and then composed through
four verb families — `add_*` (union), `subtract_*` (difference),
`intersect_*`, and `xor_*` — over the shape vocabulary (`circle`,
`ellipse`, `annulus`, `rectangle`, `square`, `polygon`, `lonlat_box`,
`latitude_band`, `longitude_band`, `frame_band`, `great_circle_band`).
`add_`/`subtract_` cover all eleven shapes; `intersect_`/`xor_` cover most
of them (a few band shapes are union/difference-only — see the
{doc}`API reference <../api/geometry>` for the exact method list).
Calls chain:

```python
region = (
    sph.CompoundRegion(ax)
    .add_circle(lon=180, lat=30, radius_deg=25)        # start: a cap
    .subtract_circle(lon=180, lat=30, radius_deg=8)    # punch a hole
    .subtract_frame_band(-10, 10, frame="galactic")    # avoid the plane
)
region.render(facecolor="teal", alpha=0.3)
region.render_boundary(color="teal", lw=1.5)
```

Beyond rendering, a region is a queryable object:

- {meth}`~skyplothelper.CompoundRegion.contains_points` /
  `contains_point` — membership tests for catalogs ("which of my sources
  fall in the survey?").
- {meth}`~skyplothelper.CompoundRegion.area_frac` and `solid_angle` — sky
  coverage of the region (frame fraction, and an approximate `sq_deg` / `sr`).
- {meth}`~skyplothelper.CompoundRegion.centroid`,
  `representative_point`, and `bounds` — *where* the region is: the area
  centroid, a point guaranteed to lie **inside** it, and its lon/lat bounding
  box. Prefer `representative_point` for placing a label — a centroid can fall
  in a hole or between disjoint lobes, whereas this always lands on the region.
  {func}`~skyplothelper.zoom_to` accepts a region directly and frames it via
  those bounds. (These three need a FITS-WCS frame.)
- {attr}`~skyplothelper.CompoundRegion.label` +
  {meth}`~skyplothelper.CompoundRegion.annotate` — name a region, then drop the
  name at `representative_point` in one call (`region.annotate(ax)`); producers
  like the co-visibility builders set `label` for you.
- {meth}`~skyplothelper.CompoundRegion.expand` / `contract` — grow or
  shrink by an angular margin (buffer zones).
- {meth}`~skyplothelper.CompoundRegion.complement` — invert the region;
  `is_empty` — sanity check after aggressive intersections.
- {meth}`~skyplothelper.CompoundRegion.union` / `intersection` /
  `difference` / `symmetric_difference` — combine two whole regions (as
  opposed to the `add_`/`subtract_` shape verbs, which fold one shape in at
  a time).
- {meth}`~skyplothelper.CompoundRegion.clip` — mask arbitrary artists (an
  image drape, a scatter) to the region; the Earth wrappers
  {func}`~skyplothelper.clip_to_land` / {func}`~skyplothelper.clip_to_ocean`
  are this applied to the coastline.
- {meth}`~skyplothelper.CompoundRegion.from_points` (convex/concave hull of
  a scatter) and {meth}`~skyplothelper.CompoundRegion.to_healpix_mask` /
  `from_healpix_mask` bridge regions to point sets and HEALPix maps.
- {meth}`~skyplothelper.CompoundRegion.render` fills the region and returns
  its artists — the fill `PathPatch`es *and* the boundary `Line2D`s — so a
  rendered region can be removed cleanly; {meth}`~skyplothelper.CompoundRegion.render_boundary`
  strokes just its outline; the underlying shapely geometry is on the
  `.geometry` attribute for custom analysis.

Region overlays have a coordinated default color palette —
`REGION_PALETTE` (an ordered list) and `REGION_PALETTE_NAMED` (by name) —
for when several regions share a map; the survey-footprint catalog is
discoverable via {func}`~skyplothelper.list_surveys` /
{func}`~skyplothelper.survey_keys` ({doc}`overlays`).

Regions (and all the layer-2 shape helpers) render on every frame family —
the FITS all-sky and field projections, orthographic globes, and the
custom non-FITS projections (Robinson, Eckert, Winkel Tripel, Kavrayskiy,
McBryde). The projection seam and pole handling are shared across all of
them, so a wrap-straddling shape or a polar cap fills correctly regardless
of the frame.

The same `CompoundRegion` also works on the interactive backend: build it
against a plotly figure with `sphpl.make_compound_region(fig)` and render
with `sphpl.add_compound_region(fig, region)` — holes render correctly
there too. See {doc}`plotly`.

## Visualizing projection distortion

```{image} /_static/features/tissot-indicatrices-light.png
:class: sph-plot plot-light dark-light
:alt: Tissot indicatrices showing projection distortion (light mode)
```
```{image} /_static/features/tissot-indicatrices-dark.png
:class: sph-plot plot-dark dark-light
:alt: Tissot indicatrices showing projection distortion (dark mode)
```
*{doc}`Tissot indicatrices </features/tissot-indicatrices>` — code in the Feature Gallery.*

{func}`~skyplothelper.tissot` drops a lattice of equal-radius geodesic
circles across the frame. Where they render as identical circles the
projection is locally faithful; where they stretch into ellipses you can
read the distortion directly. It's one line and worth doing once for any
projection you're about to commit a paper figure to:

```python
fig, ax = sph.allsky_figure(projection="MOL")
sph.tissot(ax, rad_deg=8, alpha=0.25)
```

## Pitfalls

- **A region's edge looks kinked at the map edge** — that's the seam
  closure working as intended for a region that crosses the antimeridian;
  if it looks *wrong*, check that `clip='auto'` hasn't been overridden.
- **Polar regions in AIT/MOL** — projections that pinch at the poles
  compress pole-adjacent regions visually. The geometry is correct; for a
  better *view* of a polar region, render it on a ZEA/ARC polar frame.
- **A survey footprint that bulges** — its boundary was probably defined
  along constant RA/Dec; use `geodesic=False` so edges follow the
  graticule instead of great circles.
- **Faceted curves on big regions** — raise `resolution=`.

The full listing is in the {doc}`API reference <../api/geometry>`; survey
footprints and constellation polygons — which render through this same
machinery — are covered in {doc}`overlays`.

**See also:** {doc}`concepts` §"Projection, clipping & rendering" for the
shared pipeline the `clip=` modes above plug into, and {doc}`healpix` for
region → pixel membership queries.

**Tutorial:** {doc}`Regions & spherical polygons </tutorials/regions>` works
through simple vs. spherical polygons, coordinate-plane bands, Tissot
indicatrices, expand/contract, and compound set-algebra regions.
