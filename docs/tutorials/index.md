# Tutorials

```{toctree}
:hidden:

getting_started
projections
decorating_frames
overlay_grids
fits_images
annotations
regions
insets_and_zoom
healpix_workflows
cone_bowtie
globe_plots
markers
catalogs
vector_fields
constellations
interactive_plotly
animations
styling
```

Where the {doc}`user guide <../guide/index>` explains what each subsystem
does and how the pieces fit together, the tutorials are *worked journeys* —
runnable notebooks that build complete figures step by step, with real
data and the reasoning along the way. Every figure on these pages is the
notebook's actual output.

```{tip}
New here? Start with {doc}`Getting started <getting_started>` and
{doc}`A tour of projections <projections>`, then follow whichever thread you
need — each notebook stands on its own and links back to the matching
{doc}`guide <../guide/index>` page for the reference details.
```

## The series

**{doc}`Getting started <getting_started>`** — your first frames (a zoomed
field and an all-sky map), the conventions that trip people up, working with
coordinates, a first taste of overlays and tick formatting, and a preview of
themes and palettes.

**{doc}`A tour of projections <projections>`** — the full projection gallery,
choosing among frame types, reprojecting images between coordinate systems,
and the longitude-direction conventions for sky vs. Earth maps.

**{doc}`Decorating frames <decorating_frames>`** — grid styling and
highlighted gridlines, tick placement (frame edge vs. in-frame), minor ticks,
label rotation, and the publication presets.

**{doc}`Overlay coordinate grids <overlay_grids>`** — second coordinate
systems over a frame: galactic graticules on equatorial maps, overlay ticks
and labels, and how this compares with astropy's native overlay machinery.

**{doc}`FITS images & quicklook <fits_images>`** — displaying raster data: the
interval/stretch scaling stack, colorbars (placement modes and styling) that
read true values, σ-spaced contours, one-call quicklook figures, signed-data
symmetric-log stretches, and multi-band RGB composites.

**{doc}`Annotations & overlays <annotations>`** — beams (single, stacked,
inset), scale bars, compasses, band labels, instrument markers, rulers
(geodesic, kpc conversion, twin-axis tricks), and reticles.

**{doc}`Regions & spherical polygons <regions>`** — simple vs. spherical
polygons, coordinate-plane bands, Tissot indicatrices, region
expand/contract, compound set-algebra regions, membership testing, and linear
vs. geodesic edges.

**{doc}`Insets & zoom axes <insets_and_zoom>`** — the one-call zoom inset,
marked regions and connector lines, controlling inset placement, circular
insets on globes and all-sky maps, overview/locator insets, and orientation
indicators.

**{doc}`HEALPix workflows <healpix_workflows>`** — binning catalogs, sparse
plotting, circle/polygon queries, resolution changes and smoothing, and the
same map across projections.

**{doc}`Cone & bowtie plots <cone_bowtie>`** — redshift-survey wedge diagrams:
building and orienting the cone, plotting catalogs as points, tracks, and
density, the double-sided bowtie for two-cap surveys, and the twin radial axis
that pairs redshift with comoving distance.

**{doc}`Globe & planet plotting <globe_plots>`** — tilted-Earth orientation
with Euler angles, raster planet maps, Earth features, nightshade day/night
blending, baseline networks, and globe decorations (checkered borders, compass
roses, distance scale bars).

**{doc}`Markers: rotatable & image stamps <markers>`** — direction-aware
markers that point (antennas, telescopes, domes), image stamps (planets, the
Sun, instrument photos) with the imscatter family, markers on globes, and
telescope-network site maps.

**{doc}`Catalogs: querying, plotting & searching <catalogs>`** — one-call
catalog plotting with size/color encoding, name resolution and SIMBAD/NED/VizieR
lookups, image cutouts under your data, cone searches and cross-matching,
survey-membership tests, and observation planning.

**{doc}`Vector fields & sky kinematics <vector_fields>`** — proper-motion and
displacement fields, vector spherical harmonics, and station co-visibility
regions.

**{doc}`Constellations <constellations>`** — boundaries, asterisms, labels,
and single-constellation highlighting (with pointers to dedicated star-chart
tools for richer planetarium-style charts).

**{doc}`Interactive plotting <interactive_plotly>`** — the plotly backend
tour: projections, overlays, hover data, the FITS viewer, and slider-driven
visualizations.

**{doc}`Animations <animations>`** — rotating planets, the advancing
nightshade terminator, and time-evolving co-visibility.

**{doc}`Themes, palettes & fonts <styling>`** — the whole-figure look system:
the base/theme/palette layers, color-vision-safe cycle palettes, annotation
palettes for finder charts, portable font stacks with paired math fonts,
styling sky frames, and building a reusable house style.

## Example data

The tutorials use a small set of reference datasets — a VLBA FITS image,
all-sky panoramas, Earth and planet textures, marker icons — hosted in the
repository's
[`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data)
directory (not bundled with the pip install; see
{doc}`../installation` for details).
