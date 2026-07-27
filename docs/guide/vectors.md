# Vectors & sky kinematics

Tools for *motion on the celestial sphere* — proper-motion and
displacement fields, the vector-spherical-harmonic (VSH) machinery of
reference-frame analysis, and the station-network geometry tools
(baselines and mutual-visibility regions). Everything renders through the
shared projection pipeline, so arrows and arcs wrap correctly at the map
seam in any frame.

```python
import skyplothelper as sph
fig, ax = sph.allsky_figure(projection="AIT", center=180)
```

## Vector & displacement fields

```{image} /_static/features/displacement-arrows-light.png
:class: sph-plot plot-light dark-light
:alt: Displacement arrows across a sky field (light mode)
```
```{image} /_static/features/displacement-arrows-dark.png
:class: sph-plot plot-dark dark-light
:alt: Displacement arrows across a sky field (dark mode)
```
*{doc}`Displacement arrows </features/displacement-arrows>` — code in the Feature Gallery.*

{func}`~skyplothelper.plot_sky_vectors` draws a quiver-style field of
(dlon, dlat) arrows at sky positions — proper motions, residuals,
tectonic velocities:

```python
res = sph.plot_sky_vectors(ax, lon, lat, dlon, dlat, scale=50)
```

`scale=` exaggerates the (usually tiny) physical displacements to visible
length. It multiplies each vector's magnitude — read in `units=` (arcsec by
default) — and that product is the arrow's on-sky length, converted to degrees
for plotting. So `scale=1` draws a unit-magnitude vector one *unit* long: one
degree with `units='deg'`, one milliarcsecond with `units='mas'`. The resulting
angular length is absolute — it doesn't change with the projection, the pixel
scale, or `npix` — so a given `scale=`/`units=` reads the same on an all-sky map
and on a zoomed field. (`scale='auto'` is the automatic counterpart: it sizes
the *median* arrow to `auto_target_deg` degrees, on any frame.) Add a
reference key so readers can decode the exaggeration —
`plot_sky_vectors` returns a {class}`~skyplothelper.SkyVectorResult`
(`quiver`, `colorbar`, `scale`, `deg_per_pix`), and
{func}`~skyplothelper.sky_quiverkey` draws the key for you in physical units,
so you never touch the frame's pixel scale:

```python
res = sph.plot_sky_vectors(ax, lon, lat, dlon, dlat, units="mas", scale="auto")
sph.sky_quiverkey(res, ax, 0.9, 0.05, 1000, "1000 mas/yr", units="mas")
```

This matters because the arrows are stored in the quiver's own coordinates —
the frame's *pixels*, not the sky degrees `scale=` is expressed in — so a raw
`ax.quiverkey` fed a degree value would draw a key short by the pixel scale.
`sky_quiverkey` reads the resolved `scale` and `deg_per_pix` off the result and
converts for you (which also means `scale='auto'` keys work without your ever
seeing the number `'auto'` picked — `res.scale` reads it back if you want it).

For magnitude-coded arrows,
`color_by_magnitude=True` with `cmap=` and `add_colorbar=True` populates
`result.colorbar` (label it with `result.colorbar.set_label(...)`).

{func}`~skyplothelper.plot_displacement` draws individual epoch-1 →
epoch-2 arrows whose shafts follow the great-circle path between the
positions (`geodesic=True`, seam-aware) — better than a uniform quiver
when you have few sources, large displacements, or want per-arrow
styling.

{func}`~skyplothelper.plot_catalog` is the companion for plain catalog
scatter: hand it any table with sky-coordinate columns (`ra_col=`/`dec_col=`,
or the frame-neutral `lon_col=`/`lat_col=`; common column names are
recognized automatically, case-insensitively, so VizieR spellings like
`RA_ICRS`/`DE_ICRS` are picked up without naming the columns) and it
lands on the frame correctly. `frame=` converts the input on the way in, so
a galactic catalog (`l`, `b`) drops straight onto an equatorial map with
`frame='galactic'`. Encode extra dimensions with `colorby=`/`sizeby=`.
`color_scale=` takes `'linear'`/`'sqrt'`/`'log'` or a matplotlib `Normalize`;
`size_scale=` takes those same names or a callable that maps the raw column to
the transformed array. Use `cmap_range=` to truncate the colormap, an optional
`size_legend=`, and `cbar=True`. When a colorbar is drawn it returns a
{class}`~skyplothelper.CatalogPlot` named tuple (`scatter`, `colorbar`,
tuple-unpackable); otherwise just the scatter artist. When you encode more than
one dimension at once — say color *and* size *and* shape — build a compact
per-channel key with {class}`~skyplothelper.MultiLegend`; see {doc}`legends`.

```python
sc, cb = sph.plot_catalog(ax, table, colorby="redshift", color_scale="log",
                          sizeby="mass", size_scale="sqrt", cbar=True)
```

To color a star catalog by its *perceived* color rather than a colormap,
{func}`~skyplothelper.teff_to_rgb` turns an effective temperature into
per-point RGB — hot stars blue-white, cool stars orange-red — and
{func}`~skyplothelper.color_index_to_rgb` does the same from a named color
index (Johnson `B-V`, Gaia `BP-RP`, SDSS/PS1 `g-r`, or 2MASS `J-K`), with
{func}`~skyplothelper.bv_to_rgb` and {func}`~skyplothelper.bp_rp_to_rgb` as the
thin shortcuts for the two common ones. Each index resolves to a temperature,
so a star reads the same color whichever one you have — but reach for
`bp_rp_to_rgb` on Gaia rather than `bv_to_rgb(bp_rp)`, which over-reddens.
They're a tristimulus integral (so a Sun-temperature star is white, not green
despite its peak intensity being near the green part of the spectrum)
and don't encode brightness, so map magnitude to size or alpha separately:

```python
ax.scatter(cat["ra"], cat["dec"], transform=ax.get_transform("world"),
           c=sph.bp_rp_to_rgb(cat["bp_rp"]), s=20)
```

Missing photometry (a non-finite input color) comes back as a masked RGB row,
so `np.isfinite(colors).all(axis=1)` flags stars to drop or mark.

## Vector spherical harmonics

```{image} /_static/features/vsh-shift-vectors-light.png
:class: sph-plot plot-light dark-light
:alt: A vector spherical harmonic glide field (light mode)
```
```{image} /_static/features/vsh-shift-vectors-dark.png
:class: sph-plot plot-dark dark-light
:alt: A vector spherical harmonic glide field (dark mode)
```
*{doc}`VSH shift vectors </features/vsh-shift-vectors>` — code in the Feature Gallery.*

The VSH tools here are **forward-model only** — evaluate and apply a given
VSH parameter set; *fitting* a VSH model to data is out of scope.

The VSH basis describes systematic vector fields on the sphere — the
standard language of reference-frame comparisons. The parameter vector
(`VSH_PARAM_NAMES`) covers the three rotations `R_1..R_3`, three glides
`D_1..D_3` (the Galactic-aberration signature lives here), and the ten
degree-2 electric/magnetic terms. Parameters go in as a sequence or,
more readably, a dict — anything omitted is zero:

```python
dlon, dlat = sph.vsh_field(lon, lat, {"D_3": 5.8})   # glide toward the pole
sph.plot_sky_vectors(ax, lon, lat, dlon, dlat, scale=20)
```

{func}`~skyplothelper.vsh_shift_sources` applies the field as actual
position shifts (with a `scale=` exaggeration) — useful for
before/after visualizations — and {func}`~skyplothelper.vsh_shift_frame`
applies it to the frame instead of the sources.

## Station networks

**Baselines** — {func}`~skyplothelper.plot_baselines` draws great-circle
arcs between ground stations on any map: flat lon/lat axes, CAR/AIT WCS
frames, or orthographic globes (with far-side clipping and an optional
dashed "hidden line" back-hemisphere style — `back_hemisphere_markers=True`
also draws the far-side site markers, at `back_hemisphere_alpha=`). Sites go
in as a dict of `(lon, lat)`, `(name, lon, lat)` tuples, or astropy
`EarthLocation`s; `pairs=` selects which baselines, `show_lengths=True`
labels each at its midpoint in your choice of unit:

```python
sites = {"VLA": (-107.62, 34.08), "GBT": (-79.84, 38.43),
         "Effelsberg": (6.88, 50.52)}
sph.plot_baselines(ax, sites, show_lengths=True)
```

**Co-visibility** — which sky can several stations see *simultaneously*?
Purely geometric (elevation-limit) answers, returned as renderable,
queryable {doc}`regions <regions>`:

- {func}`~skyplothelper.covisibility_circles` — each station's
  visibility cap at a given time, as drawable circle specs.
- {func}`~skyplothelper.covisibility_region` — the instantaneous mutual
  region for a target: all stations, or at least `min_stations=` of
  them, above `el_min=`.
- {func}`~skyplothelper.covisibility_duration_band` — the declination
  band visible to the network for at least `min_hours=` per day —
  the long-exposure version of the same question.

Because the results are `CompoundRegion`s, they render on either backend
and answer membership queries (`contains_points`) — "which calibrators
are co-visible right now" is a one-liner. For real observation planning
(scheduling, slew, weather), these pair with the `obsplanning` package;
this is the geometry layer.

## Pitfalls

- **Quivering with raw matplotlib** — `ax.quiver` on a WCSAxes knows
  nothing about the seam or the cos(lat) convergence of meridians; the
  helpers here exist precisely for that.
- **Forgetting the scale key** — exaggerated vectors without a printed
  scale are a figure-referee magnet; let {func}`~skyplothelper.sky_quiverkey`
  draw it (a raw `ax.quiverkey` fed a degree value renders short by the pixel
  scale, since the arrows live in the quiver's pixel units).
- **VSH parameter ordering** — when passing a bare sequence, the order
  is exactly `VSH_PARAM_NAMES`; the dict form sidesteps the ordering trap
  entirely.
- **Treating co-visibility as a schedule** — it's geometry above an
  elevation limit, not an observability calculation with sun avoidance
  and slew limits.

Full listing: {doc}`API reference <../api/science>` (VSH, co-visibility)
and {doc}`globe API <../api/globe>` (baselines).

**See also:** {doc}`concepts` (the frame conventions these coordinates
follow), {doc}`globe` (co-visibility regions and baselines drawn on a
globe), and {doc}`plotly` (the interactive vector renderer and the VSH
slider demo).

**Tutorial:** {doc}`Vector fields & sky kinematics </tutorials/vector_fields>`
covers proper-motion and displacement fields, vector spherical harmonics, and
station co-visibility regions. (For `plot_catalog` scatter, see the
{doc}`catalogs tutorial </tutorials/catalogs>`.)
