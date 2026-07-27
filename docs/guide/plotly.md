# Interactive plots (plotly)

`skyplothelper.plotly` is a second rendering backend: the same projections
and overlay vocabulary, drawn onto interactive
[plotly](https://plotly.com/python/) figures — pan, zoom, hover tooltips,
and single-file HTML export that anyone can open in a browser. It shares
the projection pipeline with the matplotlib side, so geometry (wrap seams,
polar closures, geodesics) comes out identical; what changes is the output
medium. (Requires the `plotly` extra; the FITS viewer app additionally
needs `dash`.)

```python
import skyplothelper.plotly as sphpl

fig = sphpl.make_figure(projection="AIT", center=180)
sphpl.add_constellation_boundaries(fig)
sphpl.add_plane_overlay(fig, plane="ecliptic")
sphpl.add_scatter(fig, ras, decs)
fig.show()                       # or fig.write_html("skymap.html")
```

## The figure & the parity model

{func}`~skyplothelper.plotly.make_figure` takes the same frame arguments
as the matplotlib builders — `projection=`, `center=`, `frame=`,
`direction=`, `lon_units=` — plus figure-level `theme=` (`'light'` /
`'dark'`), `width=`/`height=`, and grid visibility. The figure remembers
its projection setup, so subsequent `add_*` calls don't need it repeated
(every helper still accepts explicit overrides).

Most matplotlib helpers have a same-named twin here:

| | matplotlib | plotly |
|---|---|---|
| data | `ax.scatter` / `plot_healpix_*` | `add_scatter`, `add_healpix`, `add_healpix_sparse`, `add_sky_vectors` |
| lines & planes | `add_great_circle`, `add_plane_overlay` | same names |
| regions | `add_geodesic_circle`, `add_spherical_polygon`, bands, `add_lonlat_box` | same names |
| constellations | boundaries / lines / labels / polygon | same names |
| decorations | `Reticle`, `Ruler`, frame edge, coordinate labels | `add_reticle`, `add_ruler`, `add_frame_edge`, `add_coord_labels` |
| compound regions | `CompoundRegion(ax)` | `make_compound_region(fig)` + `add_compound_region` |
| legends | `MultiLegend` blocks | `add_legend(fig, blocks)` |

The [multi-channel legend](legends.md) blocks are backend-agnostic: build the
same `ColorBlock` / `SizeBlock` / … and hand them to
{func}`~skyplothelper.plotly.add_legend`, which renders them as native plotly
legend entries (color, shape, size, edge, open/solid fill, alpha, orientation,
line, and a real colorbar for `ColorbarBlock`; hatch, region, text, custom, and
glyph blocks are matplotlib-only and skip with a warning).

One structural difference: where the matplotlib side offers *classes* with
mutating methods (`Reticle`, `Ruler`), the plotly side offers *functions*
— plotly's added shapes are effectively immutable, so an
add-and-reconfigure object API wouldn't map honestly. Configure at call
time instead. A few overlays also gain backend-specific options — e.g.
`add_sky_vectors` accepts `shaft_color='match'` to color each arrow's shaft to
match its (per-magnitude) arrowhead color when `color_by_magnitude=True` (or a
numeric `color=` array) is in use, which the matplotlib `plot_sky_vectors`
doesn't.

The shared projection primitive is exposed here too as
{func}`~skyplothelper.plotly.project` — the building block for custom
hover/zoom callbacks that need sky → canvas coordinates.

## Hover data

Interactivity is the point, and the overlays lean into it: scatter points
carry coordinate readouts via `hovertemplate=` (fully customizable in
plotly's template syntax), HEALPix tiles report their pixel index and
value, and constellation polygons can name themselves on hover. Most
overlay helpers accept `hover=` to enable, customize, or disable their
tooltip.

```python
sphpl.add_scatter(fig, ras, decs,
                  hovertemplate="RA %{customdata[0]:.3f}°<br>"
                                "Dec %{customdata[1]:.3f}°")
```

{func}`~skyplothelper.plotly.add_healpix` also takes `line_color=` for the tile
outlines, which otherwise match each tile's own fill.

## Regions & compound regions

The region helpers mirror their matplotlib semantics — seam-aware,
pole-safe, `resolution=` sampling — rendered as SVG paths. Set-algebraic
regions cross the backend boundary cleanly:

```python
region = (sphpl.make_compound_region(fig)
          .add_circle(180, 30, 25)
          .subtract_frame_band(-10, 10, frame="galactic"))
sphpl.add_compound_region(fig, region)
```

It's the same `CompoundRegion` class as in {doc}`regions` (built against a
plotly-figure projector instead of an axes), so the full verb families and
query methods — `contains_points`, `area_frac`, `expand` — work
identically.

### An interactive region explorer

To sweep a region across a parameter — grow an avoidance radius, slide a band,
watch which catalog sources fall in or out — you write **one factory** that
maps parameters to a region, and hand it to either of two renderers:

```python
def make_region(radius):
    return sphpl.make_compound_region(fig).add_circle(180, 30, radius)
```

{func}`~skyplothelper.plotly.add_region_slider` is the turnkey one. It
precomputes a region for each step, draws them as a plotly slider, and (given a
catalog) recolors each source inside/outside per step. The result is
**self-contained** — it renders in the Sphinx build, on nbviewer, and in a
`write_html` export with **no running kernel**, so it is the right choice for a
docs page or a figure you send someone:

```python
fig = sphpl.make_figure(projection="AIT", center=180)
sphpl.add_region_slider(fig, make_region,
                        [{"radius": r} for r in range(10, 51, 5)],
                        catalog=my_skycoords, slider_label="avoidance radius")
```

{func}`~skyplothelper.plotly.compound_region_states` is the primitive beneath
it — pure compute, no plotly state — returning one dict per step
(`params`, `fill_path`, `outline_path`, `outline_x`/`outline_y`, `contains`,
`contains_int`, `n_inside`, `n_outside`). Reach for it when you want to drive
your own UI or read the containment counts directly.

For a genuinely *live* explorer — drag a slider and re-run the set algebra and
containment test in Python on every frame — the optional
{mod}`~skyplothelper.plotly.dash_region` module packages a Dash app
(`region_explorer_app`, plus `register_region_callback` to wire your own), the
region-side counterpart of `dash_fits`. It needs a kernel (local, Binder, or
Colab) and the `dash` extra (`skyplothelper[dash]`), so unlike the slider it
does not embed in static pages.

```{note}
**Why precomputed steps rather than a continuous client-side slider.** Set
algebra and point-in-region tests run in Python; reproducing them in the
browser would mean porting the whole boundary tracer and containment engine to
JavaScript. So the embeddable path bakes a fixed set of states, and the live
path keeps the Python in the loop via Dash — pick by whether you need a
self-contained file or real-time dragging.

Two details worth stating precisely: the slider ships each source's containment
as a **0/1 integer** through a two-stop colorscale, with the catalog
coordinates sent **once** (not a fresh per-source color list every step); and
containment is **canvas-space point-in-polygon against the figure's projector**,
so it matches exactly what is drawn — seam handling and all — rather than a
separate sphere-containment test. A `SkyCoord` catalog is classified in its own
frame; a bare `(lon, lat)` is taken in the figure's display frame.
```

A live-rendered slider is in the
{doc}`interactive plotting tutorial </tutorials/interactive_plotly>`.

## The FITS viewer

A WCS-aware interactive image stack:

- {func}`~skyplothelper.plotly.make_fits_figure` +
  {func}`~skyplothelper.plotly.add_fits_image` — display an image with
  correct sky axes and the full stretch/clip vocabulary from
  {doc}`images` (`stretch=`, `clip=`, percentile bounds):

  ```python
  fig = sphpl.make_fits_figure(wcs)
  sphpl.add_fits_image(fig, data, wcs, stretch="asinh", clip="zscale")
  ```

- {func}`~skyplothelper.plotly.add_fits_scatter` — catalog points on the
  image; {func}`~skyplothelper.plotly.make_fits_compound_region` —
  regions on FITS axes.
- {func}`~skyplothelper.plotly.fits_ticks_for_range` and
  {func}`~skyplothelper.plotly.beam_shape_for_range` — stateless helpers
  that recompute ticks and the beam ellipse for the current zoom; these
  are the building blocks if you're wiring your own pan/zoom callbacks.
- The `dash_fits` module packages all of it as a ready-to-run Dash app —
  pan, zoom, restretch, and inspect a FITS image in the browser.

## Sharing & export

`fig.write_html("map.html")` produces a single self-contained file —
the natural hand-off to collaborators. Static images
(`fig.write_image("map.png")`) go through plotly's kaleido engine.
Interactive figures embed directly in notebooks and webpages — which is
how the {doc}`interactive plotting tutorial </tutorials/interactive_plotly>`
presents them.

## Pitfalls

- **`ImportError` on import** — the backend needs the `plotly` extra;
  only the Dash viewer app needs `dash`.
- **Huge HEALPix maps feel sluggish** — a dense all-sky map at high nside
  is a lot of SVG; downgrade the map or use the sparse renderer for the
  occupied area.
- **Looking for `Reticle`-style objects** — the plotly side is
  function-shaped by design (see above); there's nothing to mutate after
  the call.
- **Styling expectations from matplotlib** — rcParams and
  {doc}`styling` themes don't reach plotly; use `make_figure(theme=...)`
  and plotly's own layout options.

Full listing: {doc}`API reference <../api/plotly>`.

**See also:** {doc}`concepts` — the coordinate conventions (longitude
direction, centering, frames) the plotly backend mirrors from the matplotlib side.

**Tutorial:** {doc}`Interactive plotting </tutorials/interactive_plotly>` tours
the backend end to end — projections, overlays, hover data, the FITS viewer,
and slider-driven visualizations.
