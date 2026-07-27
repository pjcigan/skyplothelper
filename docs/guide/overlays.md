# Overlays & annotations

The furniture of a sky figure: coordinate planes, survey footprints,
constellations, beams, rulers, reticles, instrument markers, compasses,
scale bars. Everything here takes the axes as its first argument, projects
through the shared pipeline (so seams and curvature are handled), and most
of it has an interactive twin in the {doc}`plotly backend <plotly>`. Many
helpers accept `stroke_color=`/`stroke_lw=` — a thin stroke behind text and
lines, the classic cartographic trick for keeping annotations legible on
busy backgrounds.

```python
import skyplothelper as sph
fig, ax = sph.allsky_figure(projection="AIT", center=180)
```

## Coordinate planes & great circles

{func}`~skyplothelper.add_plane_overlay` draws the galactic, ecliptic, or
supergalactic plane on any frame, with optional `parallels=` (e.g. ±10°
bounding lines) drawn in a secondary style:

```python
sph.add_plane_overlay(ax, plane="galactic", parallels=[-10, 10])
sph.add_plane_overlay(ax, plane="ecliptic", color="orange")
```

The bounding parallels take their own styling via `parallel_lw=` and
`parallel_color=`, so they can read as a lighter companion to the plane itself.

{func}`~skyplothelper.add_great_circle` is the general form — any great
circle, specified by its pole (`pole_lon=`, `pole_lat=`, in any `frame=`),
with `lat_offset=` for small circles parallel to it.

## Survey footprints

```{image} /_static/features/survey-footprints-light.png
:class: sph-plot plot-light dark-light
:alt: Survey footprints drawn on an all-sky map (light mode)
```
```{image} /_static/features/survey-footprints-dark.png
:class: sph-plot plot-dark dark-light
:alt: Survey footprints drawn on an all-sky map (dark mode)
```
*{doc}`Survey footprints </features/survey-footprints>` — code in the Feature Gallery.*

{func}`~skyplothelper.add_survey_footprint` draws the sky coverage of a
named survey from the bundled catalog ({func}`~skyplothelper.list_surveys`
enumerates the keys; {func}`~skyplothelper.survey_keys` returns them
programmatically). Footprints render through the {doc}`region machinery
<regions>`, so they're seam-aware and accept the shared `clip=` keyword:

```python
sph.add_survey_footprint(ax, survey="sdss", label="SDSS")
sph.add_survey_footprint(ax, survey="des", color="tab:purple", fill=False)
```

To test which of *your* sources fall inside a footprint, build the survey
as a region and use `contains_points` — the same point-in-region query works
against any {class}`~skyplothelper.CompoundRegion`, so a catalog splits cleanly
into members and non-members:

```{image} /_static/features/region-membership-light.png
:class: sph-plot plot-light dark-light
:alt: Sources classified inside versus outside a compound region (light mode)
```
```{image} /_static/features/region-membership-dark.png
:class: sph-plot plot-dark dark-light
:alt: Sources classified inside versus outside a compound region (dark mode)
```
*{doc}`Region membership </features/region-membership>` — code in the Feature Gallery; full treatment in {doc}`regions`.*

## Constellations

```{image} /_static/features/constellation-star-chart-light.png
:class: sph-plot plot-light dark-light
:alt: A star chart with constellation boundaries and figure lines (light mode)
```
```{image} /_static/features/constellation-star-chart-dark.png
:class: sph-plot plot-dark dark-light
:alt: A star chart with constellation boundaries and figure lines (dark mode)
```
*{doc}`Constellation star chart </features/constellation-star-chart>` — code in the Feature Gallery.*

IAU constellation overlays as cartographic decoration:

- {func}`~skyplothelper.add_constellation_boundaries` — the official
  boundary segments (precessed to ICRS).
- {func}`~skyplothelper.add_constellation_lines` — asterism
  (connect-the-dots) figures.
- {func}`~skyplothelper.add_constellation_labels` — names or
  abbreviations (`labels='abbr'`), with per-constellation placement
  tuning built in; restrict to a subset via `constellations=` (list the
  valid abbreviations with {func}`~skyplothelper.list_constellations`).
- {func}`~skyplothelper.add_constellation_polygon` — fill one named
  constellation as a region.

```python
sph.add_constellation_boundaries(ax)
sph.add_constellation_labels(ax, labels="abbr")
```

These are deliberately *chart decorations*, not a planetarium engine —
for magnitude-scaled stars, deep-sky objects, and proper-motion-accurate
charts, reach for dedicated tools like skyfield or starplot.

## Beams

```{image} /_static/features/beams-scale-bars-light.png
:class: sph-plot plot-light dark-light
:alt: Beam ellipses and a scale bar on an image (light mode)
```
```{image} /_static/features/beams-scale-bars-dark.png
:class: sph-plot plot-dark dark-light
:alt: Beam ellipses and a scale bar on an image (dark mode)
```
*{doc}`Beams & scale bars </features/beams-scale-bars>` — code in the Feature Gallery.*

Synthesized-beam ellipses for radio and other PSF-bearing images, as a
class with constructors for each starting point:

```python
beam = sph.Beam.from_header(hdr, ax=ax)        # BMAJ/BMIN/BPA from FITS
beam = sph.Beam.from_arcsec(0.35, 0.12, bpa_deg=20, pixscale_asec=0.1)
```

`Beam.from_psf_fit` fits the beam from a PSF image, and
{class}`~skyplothelper.BeamStack` overlays several beams (multi-band
figures) in one anchored box. Position angles follow the FITS `BPA`
convention (degrees east of north); the class converts to and from
matplotlib's angle convention for you. If you just need the *numbers*
rather than a drawn ellipse, the header helpers
{func}`~skyplothelper.beampars_asec_fromhdr`,
{func}`~skyplothelper.pixperbeam_from_hdr`, and
{func}`~skyplothelper.pixperbeam_from_pars` extract beam sizes and
pixels-per-beam from a FITS header (see {doc}`images`). Over busy imagery a
beam takes the usual `stroke_color=`/`stroke_lw=` pair, and
{meth}`~skyplothelper.Beam.set_stroke` adds or changes that stroke on an
already-drawn beam.

## Rulers & reticles

```{image} /_static/features/reticles-ruler-light.png
:class: sph-plot plot-light dark-light
:alt: Reticles marking targets alongside a ruler (light mode)
```
```{image} /_static/features/reticles-ruler-dark.png
:class: sph-plot plot-dark dark-light
:alt: Reticles marking targets alongside a ruler (dark mode)
```
*{doc}`Reticles & ruler </features/reticles-ruler>` — code in the Feature Gallery.*

{class}`~skyplothelper.Ruler` draws an angular measurement bar between two
points, with pixel-stable ticks, automatic or explicit tick intervals,
optional geodesic (great-circle) paths, and `label_unit='auto'` promoting
across the full angular range (deg → arcmin → arcsec → mas → μas → nas),
resolved once per ruler so all ticks share one unit:

```python
sph.Ruler((x0, y0), (x1, y1), ax=ax, pixscale_asec=0.004).add_to(ax)
```

`Ruler` is a two-step artist: constructing it (with `ax=`) sets up the
coordinate projection and pixel scale, and `.add_to(ax)` is what actually
draws — building a `Ruler` without adding it is a silent no-op.

{class}`~skyplothelper.Reticle` / {func}`~skyplothelper.add_reticle` mark
targets in four styles (`'plus'`, `'x'`, `'L'`, `'circle'`), with
automatic label-side selection:

```python
sph.add_reticle(ax, (83.63, 22.01), style="L", label="Crab")
```

One deliberate asymmetry worth knowing: a plain numeric tuple means a
**sky position** to a reticle ("I have a target at this RA/Dec") but a
**pixel position** to a ruler ("measure between two points I see on the
image") — each matches its canonical use. Both accept explicit overrides
when you want the other convention.

## Instrument markers

```{image} /_static/features/instrument-markers-light.png
:class: sph-plot plot-light dark-light
:alt: Procedurally drawn antenna, telescope, and dome site markers (light mode)
```
```{image} /_static/features/instrument-markers-dark.png
:class: sph-plot plot-dark dark-light
:alt: Procedurally drawn antenna, telescope, and dome site markers (dark mode)
```
*{doc}`Instrument markers </features/instrument-markers>` — code in the Feature Gallery.*

Procedurally drawn site markers — no image files needed, every part
colorable: {func}`~skyplothelper.add_antenna_marker` (radio dish, with
pointable elevation), {func}`~skyplothelper.add_telescope_marker`
(refractor on a tripod), {func}`~skyplothelper.add_dome_marker`
(observatory dome with positionable slit). For *image-based* markers
(photos, planet stamps), use the imscatter family in {doc}`globe`.

The antenna and telescope markers can **aim at a target**: pass
`aim_at=(x, y)` or a `SkyCoord` and the dish/tube (and mount rotation) are
solved to point there — `aim_mode='aimed'` swings the whole sprite onto the
source (the "array on source" look), `'planted'` keeps the pier along the
local vertical and only tilts the dish/tube. ({func}`~skyplothelper.add_dome_marker`
deliberately has no `aim_at=`: a dome aims through its slit, via `slit_azim=`.)
{func}`~skyplothelper.aim_angles` exposes the solver directly; its raster
counterpart is `imscatter_rotated(aim_at=...)`, which applies the same
`aim_angle - rest_angle` recipe to image stamps ({doc}`globe`). The two
markers don't share a rotation convention — the antenna bowl's on-screen angle
counts the mount rotation twice, the telescope tube once — so always let
`aim_angles` do the geometry rather than hand-rolling it. Call
`fig.canvas.draw()` once first (the solver needs valid display transforms).
The {doc}`markers tutorial </tutorials/markers>` works through aimed vs.
planted, the flip-behind-the-horizon handling, and the static-icon recipes.

All three markers can label themselves in one call: `label=` with
`label_side=` (`'auto'` by default), `label_offset=`, `label_color=`,
`label_fontsize=`, and `label_kwargs=` for any remaining text properties. The
returned `AnchoredOffsetbox` carries the text as `.label_artist` (`None` when
unlabeled), and removing the marker removes its label along with it.

## Compasses, scale bars & figure annotations

- {func}`~skyplothelper.add_compass` — N/E direction indicator
  (`loc='lower left'` or an `(x, y)` position).
- {func}`~skyplothelper.add_sizebar_asec` — angular scale bar sized from
  the image header; {func}`~skyplothelper.add_sizebar` is the
  general-units form. (Globe *distance* scale bars — km rather than
  arcsec — live in {doc}`globe`.)
- {func}`~skyplothelper.add_colorbar` — a colorbar matched to the image
  height on fixed-aspect WCS/image axes (where plain `plt.colorbar`
  mis-sizes to the bbox): `sph.add_colorbar(im, ax=ax, label="Jy/beam")`.
  `mode=` picks the placement — `'divider'` (default; matches the image
  and reserves space, no neighbor overlap), `'inset'` (floats beside
  without shrinking the image), or `'simple'` (a thin wrapper over the plain
  `plt.colorbar`). `location=` puts the bar on any side and moves its ticks and
  label outward — `sph.add_colorbar(im, ax=ax, location="left", label="Jy/beam")`
  — taking precedence over `orientation` (left/right ⇒ vertical, top/bottom ⇒
  horizontal). On an axes that already owns a locator (an `ImageGrid` or
  {func}`~skyplothelper.channel_map` panel), `mode='divider'` falls back to
  `'inset'` with a warning rather than breaking the layout. The bar carries
  **adaptive minor ticks** by default (`minor_ticks='auto'`) — an even
  subdivision on a linear bar, `1/2/3/5 × 10ᵏ` across the occupied decades on a
  compressed (log / asinh / symlog) one; pass `minor_ticks=False` for the
  bare-matplotlib look (no minor ticks), or a sequence of positions / a
  `Locator` to place them yourself. The major-tick **labels** are
  matplotlib's by default; `tick_format='auto'` makes their precision follow
  the displayed range — a 0–3 Jy bar reads `0.5 1.0 …` instead of a collapsed
  `0 1 2 3` — and a format string (`'%.3f'` or `'{x:.3f}'`) or a `Formatter`
  sets it explicitly. It is opt-in (default `None`) because it rewrites every
  label rather than adding to the bar. Both knobs share one implementation
  with {func}`~skyplothelper.quicklook_plot` (its `cbar_minor_ticks` /
  `cbar_format`), so the convenience path and the general colorbar render
  identically. On a hard-to-read
  colormap, `stroke_color=`/`stroke_lw=`
  add a legibility stroke behind the ticks, axis label, and frame
  (`stroke_targets=` — `'both'` (default), `'ticks'`, or `'spine'` — selects
  which). For full control — or several bars on one axes — pass `cax=`
  your own `ax.inset_axes([...])`: the bar draws there, bypassing the
  auto-placement modes (the stroke/zorder polish still applies). Two
  colormapped scatter sets, say, get one `add_colorbar(sc, cax=…)` each
  (or `mode='simple'` auto-stacks repeated same-side bars).
- {func}`~skyplothelper.add_contour_overlay` — line *or* filled contours
  drawn directly from world-coordinate `(lon, lat, values)` data (or a
  second image, reprojected onto the frame if its WCS differs).
- {func}`~skyplothelper.add_bandlabels` — corner labels for multi-panel
  band/epoch figures, returning the text artists it drew (and taking
  `stroke_color=`/`stroke_lw=`, `zorder=`, and any further text properties);
  {func}`~skyplothelper.add_axis_inlay` — a compact
  *orientation indicator* (a small wireframe of the projection outline with
  arrows showing the longitude/latitude axis directions), not a content
  inset. For a real zoom/inset axes, see the inset machinery in
  {doc}`globe`.

## Pitfalls

- **`add_sizebar_asec` and `Beam.from_header` need header metadata** —
  pixel scale for the former, `BMAJ`/`BMIN`/`BPA` for the latter. With
  arrays and no header, use the explicit-units constructors.
- **Labels vanishing into a busy background** — set `stroke_color=`
  (usually white or the background color) rather than reaching for boxes.
- **A "constellation chart" that needs real stars** — that's planetarium
  territory; pair these overlays with skyfield/starplot data rather than
  expecting a star catalog here.
- **Ruler vs. reticle coordinate conventions** — see the note above
  before debugging "my ruler is in the wrong place."

Full listing: {doc}`API reference <../api/overlays>`. Second coordinate
grids and overlay ticks are in {doc}`ticks`; region-style footprints in
{doc}`regions`.

**See also:** {doc}`images` (beam / colorbar / contour annotations on
images) and {doc}`globe` (the inset/zoom machinery and `add_compass_rose`).

**Tutorial:** {doc}`Annotations & overlays </tutorials/annotations>` builds
these onto a real figure — beams, scale bars, compasses, band labels,
instrument markers, rulers, and reticles.
