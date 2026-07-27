# HEALPix

HEALPix divides the sphere into equal-area pixels — the natural
pixelization for all-sky data, because "counts per pixel" means the same
thing at the pole as at the equator. skyplothelper covers the plotting
side of the HEALPix workflow: binning catalogs into maps, rendering maps
into any frame the package can build, spatial queries, and resolution
changes. (HEALPix support is built in — healpy is a core dependency, so it
works out of the box; the one exception is Windows, which has no healpy
wheel, where the HEALPix paths raise an informative error.)

```python
import skyplothelper as sph
```

## From catalog to map

```{image} /_static/features/source-density-map-light.png
:class: sph-plot plot-light dark-light
:alt: A HEALPix source-density map binned from a catalog (light mode)
```
```{image} /_static/features/source-density-map-dark.png
:class: sph-plot plot-dark dark-light
:alt: A HEALPix source-density map binned from a catalog (dark mode)
```
*{doc}`Source-density map </features/source-density-map>` — code in the Feature Gallery.*

{func}`~skyplothelper.bin_data_as_healpix` bins `(lon, lat)` samples plus a
data column into a dense HEALPix array, with `statistic=` selecting the
aggregation (`'mean'`, `'sum'`, etc.); for pure source counts with no data
column, use {func}`~skyplothelper.sources_to_healpix_bins` (below):

```python
# returns (hpxmap, lon_centers, lat_centers, values) — the last three are
# the pcolormesh-ready mesh for plotting
hpxmap, lon_c, lat_c, vals = sph.bin_data_as_healpix(
    ras, decs, fluxes, nside=64, statistic="mean")
```

`statistic=` takes a preset string (`'mean'`, `'sum'`, `'median'`,
`'min'`/`'max'`, `'std'`, `'count'`) or a **callable** that receives the
1-D array of finite values in each cell and returns a scalar — e.g.
`statistic=lambda v: np.percentile(v, 90)` for a 90th-percentile map, or a
robust biweight / trimmed-mean estimator. The same contract holds for
{func}`~skyplothelper.bin_data_sparse` and
{func}`~skyplothelper.image_to_healpix`.

For source *counts*, the shortcut pair
{func}`~skyplothelper.sources_to_healpix_bins` /
{func}`~skyplothelper.sources_to_healpix_plot` goes from a position list to
a binned map (and a plot) directly. {func}`~skyplothelper.auto_nside`
picks the nside whose pixels are at or finer than a target angular
resolution (e.g. `resolution_arcmin=`), returning `(nside, actual_res)`;
{func}`~skyplothelper.bin_data_sparse` keeps only the occupied pixels — the
right representation when a small catalog would leave most of a dense map
empty.

`nside` must be a power of two; each step doubles the resolution
(nside 64 ≈ 0.9° pixels). Maps use healpy's RING ordering by default, with
`nest=` available throughout.

## From image to map

{func}`~skyplothelper.image_to_healpix` bins a FITS *image* into a HEALPix
map — each pixel's value goes into the HEALPix cell at its sky position
(per-pixel world coordinates from the WCS). It is a *binning* reprojection,
not a flux-interpolating one, so it fits the common case where the HEALPix
resolution is comparable to or coarser than the image; for flux-conserving
reprojection reach for `reproject.reproject_to_healpix`. Input is flexible —
a 2-D array (with a header/WCS), an image HDU or HDUList, a FITS path, or a
`(data, header)` tuple:

```python
# All-sky image → dense map, everything automatic
hpx = sph.image_to_healpix("allsky.fits")     # HDU / path / array all OK
sph.plot_healpix_map(hpx, ax=ax)              # nside inferred from length

# Rebin an equatorial image onto a galactic map at a chosen resolution
hpx = sph.image_to_healpix(data, header, nside="5arcmin", frame="galactic")
```

`nside='auto'` (default) matches the image's pixel scale; you can also pass
an explicit power-of-two int or a target resolution (an astropy `Quantity`
or a string like `'30arcsec'`). `frame=` transforms the pixels onto a
`'galactic'`/`'icrs'`/`'ecliptic'` grid first; `statistic=` chooses the
aggregation (any preset string or callable, as above);
`return_counts=True` also returns a coverage map for masking thin cells:

```python
hpx, counts = sph.image_to_healpix(data, header, return_counts=True)
hpx[counts < 3] = np.nan          # drop under-sampled cells
```

**Dense vs sparse return.** With `sparse='auto'` (default), an image that
fills most of the sky returns a plain full-sky array — ready for
{func}`~skyplothelper.plot_healpix_map` directly — while a small high-res
field returns a {class}`~skyplothelper.HealpixBins` NamedTuple (`pixels`,
`values`, `nside`, `counts`), avoiding a giant mostly-empty allocation. The
sparse form *carries its `nside`*, because sparse pixel indices don't encode
it (a high index only sets a lower bound), so plot via its attributes:

```python
r = sph.image_to_healpix(hdu, sparse=True)    # HealpixBins
sph.plot_healpix_sparse(r.pixels, r.values, r.nside, ax=ax)
```

A *dense* map's length is exactly `12·nside²`, so its resolution is
recoverable — {func}`~skyplothelper.nside_from_array` exposes that (and the
dense plotters use it internally, which is why they take no `nside`).

## Rendering maps

- {func}`~skyplothelper.healpix_allsky_figure` — the one-call form: builds
  the figure *and* an all-sky frame, renders the map, and adds a colorbar,
  returning a {class}`~skyplothelper.HealpixResult` named tuple
  (`fig`, `ax`, `mappable`, `colorbar`). Like `healpy.mollview`, but a real
  WCSAxes you can keep drawing on.
- {func}`~skyplothelper.plot_healpix_allsky` — renders an all-sky map onto
  an *existing* WCSAxes (returns the mappable); use it when you built the
  frame yourself.
- {func}`~skyplothelper.plot_healpix_map` — also renders onto an existing
  frame, but for any projection, any coordinate system, full-sky or a
  `lonlatlims=` sub-window. This is how the same map lands on an AIT oval,
  a tilted globe, and a TAN field.
- {func}`~skyplothelper.plot_healpix_sparse` — draws only occupied pixels
  as individual polygons (`backend=`, `show_boundaries=`); the honest
  rendering for sparse catalogs.

```python
result = sph.healpix_allsky_figure(hpx, projection="AIT")
result.colorbar.set_label("mean flux")

# Same map, different frame (build the frame, then plot onto it):
ax2 = sph.make_globe_frame(111, center_LONdeg=180, center_LATdeg=30)
sph.plot_healpix_map(hpx, ax=ax2)
```

Dense maps are rasterized onto the frame's pixel grid; `xyres_pix=`
controls that raster's resolution (raise it for large print figures).

On projections with a seam or facet edges, cells that would bridge across
the seam are blanked automatically (`mask_seams=True`) so a map doesn't
smear a band across the figure;
{func}`~skyplothelper.mask_seam_crossing_quads` is the underlying helper if
you build a pcolormesh yourself, and is opt-out where it's applied. One
finite-resolution caveat: on the interrupted HEALPix/quad-cube projections
(`HPX`, `XPH`, the cubes), raster data can't completely fill the extreme
corners — thin edge gaps that shrink as `nside` (and `xyres_pix`) grow.

## Spatial queries & pixel geometry

- {func}`~skyplothelper.healpix_circle_query` /
  {func}`~skyplothelper.healpix_polygon_query` — which pixels fall inside
  a disk or polygon (the building block for "is this source in the
  masked area?" logic).
- {func}`~skyplothelper.healpix_pixel_corners` — corner coordinates of
  given pixels, for custom drawing.
- {func}`~skyplothelper.healpix_to_celestial` — sample a HEALPix array onto
  a lon/lat meshgrid (the array the renderers hand to `pcolormesh`).
- {func}`~skyplothelper.healpix_to_canvas` — the lower-level frame-conversion
  entry point the plotters build on; this is also where a map in one
  coordinate system is rotated onto a frame in another (the `frame=`
  argument of {func}`~skyplothelper.healpix_allsky_figure` routes through
  it).

## Resolution & smoothing

{func}`~skyplothelper.healpix_upgrade` / {func}`~skyplothelper.healpix_downgrade`
move between nside levels; {func}`~skyplothelper.healpix_smooth` applies
beam smoothing; {func}`~skyplothelper.healpix_combine` merges maps. The
usual workflow for noisy count maps is bin fine → smooth → downgrade.

## Pitfalls

- **Use an equal-area projection for density maps** — that's the whole
  point of HEALPix. AIT, MOL, SFL, ZEA preserve the counts-per-area
  reading; TAN and MER visually distort it.
- **RING vs. NESTED ordering** — everything here defaults to RING
  (`nest=False`), matching healpy's default. A map that renders as
  scrambled tiles is almost always an ordering mismatch.
- **A mostly-empty dense map** — use the sparse pathway
  ({func}`~skyplothelper.bin_data_sparse` +
  {func}`~skyplothelper.plot_healpix_sparse`) instead of binning a small
  catalog into millions of NaN pixels.
- **Galactic map on an equatorial frame looks rotated** — it *is*; give
  {func}`~skyplothelper.healpix_allsky_figure` the map's own `frame=`
  (e.g. `frame="galactic"`) so it's rotated onto the displayed frame, or
  build the frame in the map's system.

```{note}
**Two things share the name "HEALPix."** The *pixelization* (an
equal-area tiling of the sphere — equal-area, not equal-*shape*: pixels
cover equal solid angle but their outlines distort toward the poles) is
independent of the *projections* `HPX`/`XPH` (the diamond / butterfly
layouts, which are pole-locked — see {doc}`frames`). A HEALPix map can be
drawn in *any* projection; you don't have to use the HPX projection to
plot a HEALPix array.
```

Full listing: {doc}`API reference <../api/healpix>`.

**See also:** {doc}`regions` (region → pixel membership queries),
{doc}`frames` (the HPX/XPH projections and pole-lock), {doc}`queries`
(getting catalogs to bin).

**Tutorial:** {doc}`HEALPix workflows </tutorials/healpix_workflows>` covers
binning catalogs, sparse plotting, circle and polygon queries, resolution
changes and smoothing, and drawing one map across several projections.
