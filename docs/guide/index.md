# User guide

The user guide explains skyplothelper subsystem by subsystem. Each page
describes what that part of the package does, how it relates to the rest,
short examples of the common operations, and the caveats worth knowing —
a searchable reference for *what exists and how it fits together*. For
extended, end-to-end worked examples, see the {doc}`tutorials
<../tutorials/index>`.

If you're new, start with the {doc}`quickstart <../quickstart>`. If the
astropy/matplotlib substrate is also new — `SkyCoord`, WCSAxes, the
`transform=` keyword — read {doc}`foundations` first; then {doc}`concepts`,
which covers the conventions (longitude direction, coordinate frames, units)
that every other page builds on.

```{toctree}
:maxdepth: 1

../quickstart
foundations
concepts
frames
ticks
overlays
regions
images
healpix
globe
cone
queries
vectors
legends
styling
plotly
extending
```

## Orientation

**Foundations.** {doc}`foundations` is a crash course in the astropy +
matplotlib layer `sph` sits on — units, `SkyCoord`, matplotlib's `transform=`,
and astropy's WCS/WCSAxes — for readers new to sky/WCS-aware plotting.
{doc}`concepts` then explains the package-wide model: every plot starts from a
*frame* (a WCSAxes wired to a sky projection), everything else draws onto
frames through one shared projection pipeline, and a handful of conventions
(astro east-left longitude, ICRS default, auto tick units) apply everywhere.
{doc}`frames` covers building those frames across ~30 projections.

**Dressing the frame.** {doc}`ticks` (tick formats, grids, second coordinate
overlays), {doc}`overlays` (beams, rulers, reticles, compasses, scale bars,
coordinate planes, survey footprints, constellations), and {doc}`styling`
(themes, palettes, publication presets) control how a plot reads.

**Putting data on the sky.** {doc}`images` (stretching, quicklook, and
reprojection of FITS and RGB images), {doc}`healpix` (binning catalogs and
rendering HEALPix maps), {doc}`regions` (geodesic circles, bands, polygons,
and set-algebraic compound regions), and {doc}`vectors` (proper motions,
displacement fields, vector spherical harmonics, station co-visibility).
{doc}`legends` builds compact multi-channel keys — one block per encoded
dimension — for the busy maps those tools produce.

**Specialized frames.** {doc}`globe` (orthographic globe views of the sky
and of solid bodies — including tilted-Earth orientation — with surface
features, nightshade, and inset axes) and {doc}`cone` (cosmology z-RA
wedge diagrams, single or double-sided).

**Beyond matplotlib.** {doc}`queries` (name resolution, SIMBAD/NED/VizieR,
sky-survey image downloads) and {doc}`plotly` (the interactive web-export
backend and Dash FITS viewer).

**Extending it.** {doc}`extending` is the contract for writing your own
overlay — the three primitives (frame, projection, theme color) the bundled
decorations use, on both backends — so an extension behaves like a built-in.
