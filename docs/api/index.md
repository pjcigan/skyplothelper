# API reference

The full public API is re-exported from the top-level package, so everything
documented here is reachable as `skyplothelper.<name>` (conventionally
`sph.<name>`). The reference is grouped by subsystem; each page lists its public
functions and classes with auto-generated signatures from the source.

```{toctree}
:maxdepth: 2

frames
coordinates
fits
ticks
overlays
images
healpix
geometry
globe
cone
science
legends
utilities
plotly
```

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Frames & figures
:link: frames
:link-type: doc
`make_wcs_frame`, figure builders, the projection registry, and `project`.
:::

:::{grid-item-card} Coordinates & math
:link: coordinates
:link-type: doc
Frame conversions, sexagesimal helpers, angular distances, wrapping utilities.
:::

:::{grid-item-card} FITS headers
:link: fits
:link-type: doc
Header coordinate grids, pixel/sky conversions, beam parameters, diagnostics.
:::

:::{grid-item-card} Ticks & labels
:link: ticks
:link-type: doc
RA/Dec tick formatters, offset ticks, VLBI hybrid labels, curved lon ticks.
:::

:::{grid-item-card} Overlays
:link: overlays
:link-type: doc
Planes, survey footprints, constellations, beams, rulers, reticles, annotations.
:::

:::{grid-item-card} Images
:link: images
:link-type: doc
Clip / stretch / normalize, quicklook figures, reprojection.
:::

:::{grid-item-card} HEALPix
:link: healpix
:link-type: doc
Binning, plotting across frames, queries, smoothing, and resolution changes.
:::

:::{grid-item-card} Geometry & regions
:link: geometry
:link-type: doc
Geodesic circles, bands, polygons, Tissot indicatrices, and `CompoundRegion`.
:::

:::{grid-item-card} Globes & planets
:link: globe
:link-type: doc
Orthographic globes, Earth-feature overlays, nightshade, inset axes.
:::

:::{grid-item-card} Cone frames
:link: cone
:link-type: doc
Cosmological cone / pie-wedge plots, twin radial axes, bowtie frames.
:::

:::{grid-item-card} Science modules
:link: science
:link-type: doc
Mutual sky visibility (co-visibility) and vector spherical harmonics.
:::

:::{grid-item-card} Legends
:link: legends
:link-type: doc
`MultiLegend` and the per-channel blocks: color, shape, size, fill, and more.
:::

:::{grid-item-card} Utilities
:link: utilities
:link-type: doc
Styling, grids, catalog plots, the cartopy backend, queries, and constants.
:::

:::{grid-item-card} Interactive (plotly)
:link: plotly
:link-type: doc
The plotly export backend, overlays, and the Dash FITS viewer.
:::

::::
