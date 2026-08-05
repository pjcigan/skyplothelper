# Changelog

All notable changes to skyplothelper are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0]

This release adds a world-coordinate way to set a frame's field of view,
a few region-introspection conveniences, and a layered co-visibility plot.
Every existing entry point is unchanged.

### Added
- **Set the view in degrees.** A skyplothelper frame is a WCS axes, so
  `ax.set_xlim`/`set_ylim` take *pixels*. The new view helpers set the view in
  world coordinates on any projection: `set_extent(ax, [lon0, lon1, lat0, lat1])`
  (exact on rectilinear frames, a bounding box on curved ones), `zoom_to(ax, lon,
  lat, pad=...)` (fit to a set of points, a `SkyCoord`, or a `CompoundRegion`),
  `set_view(ax, center, fov)` (center + angular width), and `set_xlim` / `set_ylim`
  shortcuts. Also available as `ax.sky_*` methods.
- **`CompoundRegion` label & anchor.** `representative_point()` returns a
  `(lon, lat)` guaranteed *inside* the region (a better label anchor than
  `centroid`, which can fall in a hole); a `.label` attribute plus
  `annotate(ax, text=None)` drops that name at the anchor in one call.
- **Layered co-visibility coverage.** `covisibility_coverage(target, stations)`
  builds and draws one colored region per coverage count *k* — disjoint
  exactly-*k* bands (`mode='exactly'`, a coverage choropleth) or nested ≥*k*
  shells (`mode='atleast'`), optionally labeled with *k*.

### Changed
- `covisibility_region` / `covisibility_circles` now default `time=None`,
  meaning the current instant (`Time.now()`), so you can see the co-visible sky
  — or its overall sky fraction — without constructing a `Time`.
  `covisibility_region` also sets a default `.label` (`"Co-visible"`).

[1.2.0]: https://github.com/pjcigan/skyplothelper/releases/tag/v1.2.0

## [1.1.0]

This release extends the planet / geographic side of skyplothelper to reuse the
same WCS and spherical-region machinery as the celestial tools: flat (non-globe)
planet projections, filled geographic overlays, longitude-West labeling, and a
set of region-machinery additions (region-to-region set algebra, region masking,
non-FITS fills, and catalog / DS9 / CRTF interop). Every existing entry point is
unchanged unless a new opt-in keyword is passed.

### Added
- **Flat planet frames.** `make_planet_frame(projection=...)` builds flat world
  maps (plate carrée / Mollweide / Robinson / Eckert / …) that carry the full sph
  machinery — lon/lat coordinate input, regions, overlays, baselines — not just
  the SIN globe. `projection='SIN'` (the default) keeps the orthographic globe
  unchanged.
- **Filled geographic overlays.** `plot_land` (with `lakes=True` to punch lakes
  out as true holes), `plot_lakes`, `plot_rivers`, and
  `plot_tectonic_plates(fill=True)` — the plates as a single color, a categorical
  map, or a `values=`-driven **choropleth**. `plot_time_zones` draws the
  UTC-offset meridians. A general `choropleth` helper colors any list of rings by
  value. The bundled 110 m Natural Earth / Bird (2003) plate data is fetched on
  demand with `prepare_earth_data`.
- **Region masking.** `CompoundRegion.clip(artists, complement=)` and
  `clip_path()` mask any matplotlib artist (image, scatter, quiver, contour) to a
  region's shape; `clip_to_land` / `clip_to_ocean` are the Earth-map conveniences.
- **Region-to-region set algebra.** `CompoundRegion.union` / `intersection` /
  `difference` / `symmetric_difference` combine two independently-built regions.
- **More region construction + interop.** `CompoundRegion.from_points`
  (convex / concave footprint from a scatter), `from_polygons` (batch),
  `to_healpix_mask` / `from_healpix_mask`, `to_ds9` / `to_crtf` / `to_regions`
  and `from_ds9` / `from_crtf` / `from_regions`, plus `centroid` / `bounds`
  properties.
- **Longitude-West labeling.** `lon_west=True` on the frame builders labels
  longitude westward (e.g. `71°W`) — labels only; the underlying data and the map
  orientation are unchanged. `lon_west_to_east` / `lon_east_to_west` converters
  handle west-longitude input.
- **Region fills on non-FITS projections.** The region fill / clip pipeline now
  works on the custom Robinson / Eckert IV / Winkel Tripel / Kavrayskiy VII /
  McBryde frames, not only FITS projections.
- **Stroke on region shapes.** `stroke_color` / `stroke_lw` on the region shape
  helpers and `CompoundRegion.render`, matching the other decoration helpers.
- `make_globe_frame` / `make_planet_frame` accept `fig=` plus a subplot spec, so a
  SIN globe drops cleanly into a multi-panel grid.

### Fixed
- **Filled regions on a globe** now clip to the visible hemisphere (a
  cartopy-style domain clip) instead of chording across the disk or filling the
  complement — a land mass, plate, or survey cap that spills past the limb fills
  correctly. A region that encloses the whole visible hemisphere fills the whole
  disk, and the ±180 meridian cut of a pole-enclosing region no longer draws a
  spurious seam under an edge color.
- Set-algebra holes (e.g. a symmetric difference) render correctly on the
  radians-scale non-FITS frames instead of being welded shut by the render-time
  seam cleanup.

[1.1.0]: https://github.com/pjcigan/skyplothelper/releases/tag/v1.1.0


## [1.0.2]

### Fixed
- Tutorial notebook code cells now syntax-highlight in the ReadTheDocs build.
  The notebooks use the `ipython3` Pygments lexer, which ships with IPython
  (not bundled in Pygments); `ipython` was missing from the docs build
  environment, so those cells fell back to plain, unhighlighted text. Added
  `ipython` to the docs requirements and the `[docs]` extra. No change to the
  package or its runtime dependencies.

[1.0.2]: https://github.com/pjcigan/skyplothelper/releases/tag/v1.0.2


## [1.0.1]

### Performance
- Constellation boundary overlays render dramatically faster: the plotly path
  ~25× (a ~25 s call is now ~1 s) and the matplotlib path ~10×. `project()`
  now memoizes the per-projection WCS (which also speeds up any code that
  projects many small batches against the same frame), and
  `add_constellation_boundaries` draws all chords in a single artist.

### Changed
- `add_constellation_boundaries` returns a one-element list containing a single
  `matplotlib.collections.LineCollection` of all boundary chords (previously a
  list of individual `Line2D` artists). Iterate `get_segments()` for
  per-segment access.

### Fixed
- Compatibility with numpy 2.0 (`ndarray.ptp()` / `np.trapz` removals) and
  Python 3.14 (annotation rendering in the generated `llms-full.txt`).

[1.0.1]: https://github.com/pjcigan/skyplothelper/releases/tag/v1.0.1


## [1.0.0]

Initial release.

skyplothelper is an astronomy visualization toolkit built on matplotlib and
astropy WCSAxes. A single `import skyplothelper as sph` provides:

- WCS frame builders for all-sky, globe, and field plots (`make_wcs_frame`,
  `allsky_figure`, `offset_figure`, `make_globe_frame`, `make_planet_frame`),
  plus a registry of FITS and non-FITS projections.
- Sky-coordinate tick formatting, coordinate-system conversions, and FITS
  header utilities.
- Overlays and annotations: coordinate planes, survey footprints, IAU
  constellations, beams, rulers, reticles, scale bars, and compasses.
- Spherical-region geometry — geodesic circles, rectangles, ellipses, bands,
  polygons, and set-algebraic `CompoundRegion`.
- Tilted-globe (orthographic) frames with Earth-feature overlays and a
  day/night nightshade blend.
- Cone (z-RA wedge) frames for cosmology diagrams.
- HEALPix binning, plotting, and queries; FITS image quicklook and
  reprojection; catalog queries (SIMBAD / NED / SkyView / VizieR).
- An interactive plotly export backend with a Dash-based FITS image viewer.
- Vector spherical harmonics and mutual sky-visibility (co-visibility) regions.

Optional features are gated behind per-feature install extras and fail
gracefully with informative messages when a dependency is absent.

[1.0.0]: https://github.com/pjcigan/skyplothelper/releases/tag/v1.0.0
