# Catalogs & queries

Light wrappers around astroquery for the lookups that punctuate a
plotting session — resolving a name to coordinates, grabbing a catalog
around a position, pulling a survey image to put under your data. They
are deliberately thin conveniences for plotting workflows; for serious
catalog work, use astroquery directly. (Requires the `query` extra:
`pip install skyplothelper[query]`. Everything here talks to remote
services, so expect network latency and occasional outages.)

```python
import skyplothelper as sph
```

## Name resolution

{func}`~skyplothelper.resolve_name` turns an object name into
coordinates — the output drops straight into any `center=`:

```python
coord = sph.resolve_name("M87")
fig, ax = sph.offset_figure(center=coord, fov_deg=0.3)
```

{func}`~skyplothelper.resolve_names` is the batched form; it returns the
resolved coordinates *and* the list of names that failed, with
`on_error=` choosing between warning and raising — so one typo in a
target list doesn't take down the batch. Both accept `service='simbad'`
or `'ned'`.

## Catalog & object queries

- {func}`~skyplothelper.query_simbad` /
  {func}`~skyplothelper.query_ned` — look up an object, or search a
  region by passing a coordinate plus `radius=`.
- {func}`~skyplothelper.search_vizier` — a cone search against any
  VizieR catalog by its identifier, with column selection and a row
  limit. The center (`coord=`) accepts an object name, a `SkyCoord`, or a
  bare `(ra, dec)` degree tuple:

```python
table = sph.search_vizier("I/350/gaiaedr3", coord, radius=10)
sph.plot_catalog(ax, table, ra_col="RA_ICRS", dec_col="DE_ICRS")
```

```{note}
VizieR truncates server-side at `row_limit` (default 5000) without saying
so, which quietly turns a crowded field into a partial one. If a result
comes back at exactly the limit, `search_vizier` warns that the catalog is
probably incomplete — raise `row_limit=`, or pass `row_limit=-1` for no
limit.
```

Once you hold a table, {func}`~skyplothelper.plot_catalog` drops it onto a
frame directly: column names auto-detect (or set `ra_col`/`dec_col`, or the
frame-neutral `lon_col`/`lat_col`); `frame=` converts on the way in
(galactic `l`/`b` onto an equatorial map); and `colorby=`/`sizeby=` —
each with a `color_scale=`/`size_scale=` and `cmap_range=` — encode extra
table columns, returning a {class}`~skyplothelper.CatalogPlot` when a
colorbar is drawn. Full treatment in {doc}`vectors`.

## Survey images

- {func}`~skyplothelper.download_skyview` — fetch a cutout from any
  SkyView survey as `(data, header)`, ready for {doc}`quicklook
  <images>` or reprojection. {func}`~skyplothelper.list_skyview_surveys`
  enumerates the survey names — they must match exactly.
- {func}`~skyplothelper.overlay_cutout` — the one-call version: fetch a
  cutout *and* lay it under an existing frame's data
  (`zorder=0`, grayscale by default):

```python
sph.overlay_cutout(ax, coord, survey="DSS2 Red")
```

- {func}`~skyplothelper.download_hips` — the HiPS equivalent: a cutout
  from the CDS HiPS2FITS service (`download_hips(coord, hips_id, size=,
  pixels=)`), reaching the many all-sky surveys published as HiPS rather
  than through SkyView.

{func}`~skyplothelper.download_skyview` caches its results (`cache=True` by
default) so re-running a notebook doesn't re-download.

## Checking survey membership

"Which of my targets fall inside survey X?" doesn't need a web query:
the bundled survey footprints ({doc}`overlays`) render through the
region machinery, and regions are queryable with `contains_points` —
see {doc}`regions`.

To filter a *whole* catalog offline (rather than test points one by one),
{func}`~skyplothelper.region_search` returns the rows inside any region —
including a {class}`~skyplothelper.CompoundRegion`;
{func}`~skyplothelper.cone_search` does the same for a circular field, and
{func}`~skyplothelper.crossmatch` finds nearest-neighbor counterparts
against a reference catalog. All three are type-preserving (a `Table` in
gives a `Table` out, a `DataFrame` a `DataFrame`).

## Pitfalls

- **`ImportError` on any of these** — install the `query` extra; the
  rest of the package doesn't need it.
- **A SkyView survey name that "doesn't exist"** — the names are exact
  strings (`'DSS2 Red'`, not `'dss2-red'`); check
  {func}`~skyplothelper.list_skyview_surveys`.
- **One bad name killing a batch resolve** — that's what
  `resolve_names(..., on_error='warn')` is for; collect the failures it
  returns instead of try/excepting each name.
- **Hammering services in a loop** — these are courtesy wrappers around
  shared community services; cache results (the image fetchers do by
  default) and batch where possible.

Full listing: {doc}`API reference <../api/utilities>`.

**See also:** {doc}`foundations` — how skyplothelper accepts the sky
coordinates these lookups return (a `SkyCoord` drops straight onto any frame).

**Tutorial:** {doc}`Catalogs: querying, plotting & searching </tutorials/catalogs>`
covers one-call catalog plotting, name resolution and SIMBAD/NED/VizieR
lookups, image cutouts under your data, cone searches and cross-matching, and
survey-membership tests.
