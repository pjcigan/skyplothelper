# Ticks, grids & labels

How a sky frame *reads*: sexagesimal vs. decimal tick labels, offset
coordinates for zoomed fields, grid styling, and second coordinate systems
drawn over the first. Everything here operates on an existing frame — the
builders apply sensible defaults at creation, and these helpers re-tune
afterward. (For tick *color/size/direction* as part of a global look, see
{doc}`styling`; this page is about formats and structure.)

```python
import skyplothelper as sph
fig, ax = sph.allsky_figure(projection="AIT", center=180)
```

## Tick label formats

{func}`~skyplothelper.format_ticklabels` is the one call for label
appearance — a named `style=` plus overrides for the longitude/latitude
formats, separators, size, color, and an optional legibility stroke:

```python
sph.format_ticklabels(ax, style="publication")
sph.format_ticklabels(ax, lon_sep=("ʰ", "ᵐ", "ˢ"), simplify=True)
```

`style=` is a whole vocabulary of presets, not just `'publication'`:
`'letter'`, `'casa'`, `'latex'`, `'compact'`, `'minimal'`,
`'decimal'`/`'decimal_plain'`, `'allsky_hours'`/`'allsky_deg'`, the
`'offset'`/`'offset_arcsec'`/`'offset_arcmin'` family, and
`'vlbi'`/`'anchored_offset'` (the offset and anchored-offset styles are also
reachable through the dedicated appliers below).

`simplify=True` (the default) drops redundant trailing fields (so a round
hour ticks as `14ʰ`, not `14ʰ00ᵐ00ˢ`). Separator presets — superscript
h/m/s and °′″, plain letters, colons, IAU spacing, LaTeX — are collected
in the `SEPARATORS` constant, keyed like `'hms_full'`, `'dms_colon'`,
`'deg_symbol'`. The default `'hms_full'` renders the superscripts via
**mathtext** (`$^\mathregular{h}$`…), which displays in any font; a
literal-Unicode variant lives under `'hms_unicode'` (it can tofu in fonts
like Arial that lack those glyphs). Underneath sit the per-backend workers
({func}`~skyplothelper.format_WCS_ticklabels`,
{func}`~skyplothelper.format_mpl_ticklabels`) and the RA formatters
({class}`~skyplothelper.RAlabelformatter`,
{func}`~skyplothelper.RAlabellist`) if you're wiring labels manually.

Style one axis at a time with `which=`: `'lon'` (or `'ra'`, `'longitude'`,
`'l'`, …) or `'lat'` (or `'dec'`, `'b'`, …) limits the whole call to that
axis and leaves the other's format, separator, rotation, and axis label
untouched — case-insensitive, the same alias vocabulary the highlight and
overlay tools use. The default `'both'` (also `'all'`) touches both. For
finer control inside one call, the per-axis overrides act independently:
`lon_sep`/`lat_sep` (separators), `lon_rotation`/`lat_rotation` (angles), and
a `stroke_lw`/`stroke_color` legibility stroke.

## Offset & anchored-offset ticks

```{image} /_static/features/offset-coordinate-ticks-light.png
:class: sph-plot plot-light dark-light
:alt: A field labeled in offset coordinates (light mode)
```
```{image} /_static/features/offset-coordinate-ticks-dark.png
:class: sph-plot plot-dark dark-light
:alt: A field labeled in offset coordinates (dark mode)
```
*{doc}`Offset coordinate ticks </features/offset-coordinate-ticks>` — code in the Feature Gallery.*

High-magnification fields read better in *relative* coordinates.
{func}`~skyplothelper.offset_figure` builds an offset frame from scratch
({doc}`frames`); these helpers convert the labeling of an existing frame:

```{note}
**Which formatter?** {func}`~skyplothelper.format_ticklabels` (above) is the
*absolute* celestial formatter — sexagesimal or decimal, and celestial-only by
design (it raises `Invalid format: hh:mm:ss` on a non-celestial frame). For
*relative* arcsec/mas/μas labels reach for `apply_offset_ticks` instead: it
works directly on an ordinary celestial field — `offset_figure(center,
fov_deg=…)` then `apply_offset_ticks(ax, ...)` — with no synthetic
linear-offset WCS header (that header is what makes `format_ticklabels` fail).
```

- {func}`~skyplothelper.apply_offset_ticks` — offsets about a reference
  position (defaults to the frame center). `unit=` takes the angular set
  (`deg`/`arcmin`/`arcsec`/`mas`/`μas`) or `'auto'`, which walks down that
  set to match the field of view. It sizes the tick spacing from the
  *current* (possibly cropped) axes view, so re-call it after a
  `set_xlim`/`set_ylim` crop to lay finer ticks in the zoomed window — the
  spacing is not re-derived automatically on an `xlim` change. A `spacing=`
  override (an astropy `Quantity` in any angular unit, or a per-axis pair)
  pins the interval explicitly, independent of the label unit.
  `show_unit=False` drops the unit suffix from each tick (bare `+400`, `0`,
  `-200`) — the tidy choice when the axis title already carries the unit. The
  `color=` and `stroke_color=`/`stroke_lw=` styling reaches both the tick
  labels *and* the axis label, so a recolored offset frame reads as one piece
  (`fontsize=` stays tick-only).
- {func}`~skyplothelper.apply_anchored_offset` — mark one absolute
  *anchor* tick with offset ticks around it, so a reader gets absolute
  position and relative scale from the same axis (particularly useful for
  VLBI scales, but it works at any scale — e.g. an all-sky plot
  anchored on a named source). `unit=` accepts the angular set
  (`deg`/`arcmin`/`arcsec`/`mas`/`μas`) or `'auto'` (FOV-derived; default
  `'mas'`); per-part precision and a compact rotated variant for μas-scale
  fields are also exposed. `spacing=` takes an astropy `Quantity` in any
  angular unit (e.g. `2*u.arcmin`) or a per-axis pair, independent of the
  label unit — offset *display* units are angular by design (radians and
  time-of-RA are deliberately excluded; specify those via a `Quantity`
  `spacing=`). The anchor label
  format is set by `anchor_format=`: `'sexagesimal'` (default, HMS/DMS),
  `'decimal'` (decimal degrees, `ref_precision` places — handy for
  galactic `l`/`b`), or a callable `f(value_deg) -> str` for a fully
  custom label (e.g. a source name). Only the anchor changes; the offset
  ticks are unaffected.

```python
fig, ax = sph.offset_figure(center=(196.6, -10.6), fov_deg=0.002)
sph.apply_anchored_offset(ax, unit="mas")
```

These appliers are backed by the public formatter classes
{class}`~skyplothelper.OffsetFormatter` and
{class}`~skyplothelper.AnchoredOffsetFormatter`, which you can attach to a
coordinate axis directly for full control.

## Grid control

The frame builders draw a graticule (axis grid) by default (`grid=`, 
`gridcolor=`, `gridalpha=`); afterward:

- {func}`~skyplothelper.style_grid` — restyle the whole graticule:
  color, alpha, width, linestyle, or a legibility stroke for grids over
  imagery.
- {func}`~skyplothelper.highlight_gridline` /
  {func}`~skyplothelper.highlight_gridlines` — emphasize specific curves
  without touching the rest:

```python
sph.highlight_gridline(ax, 0, coord="lat", color="tab:red", lw=1.5)   # equator
```

```{tip}
Over imagery, all three frame elements can carry a legibility stroke via
the same `stroke_color=`/`stroke_lw=` pair: **labels** through
{func}`~skyplothelper.format_ticklabels`, **grid lines** through
{func}`~skyplothelper.style_grid`, and **tick marks** through
{func}`~skyplothelper.style_wcs_axes` ({doc}`styling`). Set `stroke_lw`
a little above the element's own line width.
```

## Second coordinate systems

```{image} /_static/features/second-coordinate-grid-light.png
:class: sph-plot plot-light dark-light
:alt: A galactic graticule overlaid on an equatorial frame (light mode)
```
```{image} /_static/features/second-coordinate-grid-dark.png
:class: sph-plot plot-dark dark-light
:alt: A galactic graticule overlaid on an equatorial frame (dark mode)
```
*{doc}`Second coordinate grid </features/second-coordinate-grid>` — code in the Feature Gallery.*

Drawing another system's graticule *over* the frame — galactic curves on
an ICRS map — without converting any data:

- {func}`~skyplothelper.add_second_grid` — the one-call form: pick an
  `overlay_frame=` and a contrasting style, optionally with ticks and
  labels.
- {class}`~skyplothelper.CoordinateOverlay` /
  {func}`~skyplothelper.add_coord_overlay` — the configurable layer:
  choose exactly which meridians/parallels are drawn (`lon_vals=`,
  `lat_vals=`) and style each family independently. Left unset, the
  spacing adapts to the field of view — a zoomed cross-frame overlay
  (e.g. galactic lines on a few-degree equatorial field) gets nice
  1/2/5° values that fall inside the view, while all-sky and globe
  frames keep the standard 30°/15° graticule.
- {func}`~skyplothelper.add_overlay_ticks` — ticks and labels for the
  overlay system. The key option is *where*: `lon_at=`/`lat_at=`
  place them on the frame `'boundary'` or along an in-frame gridline —
  the trick for labeling an overlay system inside the map where the
  frame edge belongs to the primary system. Style the two families
  independently with `lon_tick_kwargs`/`lat_tick_kwargs` and
  `lon_label_kwargs`/`lat_label_kwargs` (merged over the shared
  `tick_kwargs`/`label_kwargs`), add a `stroke_lw`/`stroke_color` for
  legibility, and set the label angle *relative to the gridline* with a
  `rotate=` in `label_kwargs`: `'tangent'` (along the line),
  `'tangent_upright'`, `'tangent_perp'` (across it — tangent + 90°, kept
  readable-upright), the general `'tangent+N'`/`'tangent-N'`,
  `'horizontal'`, a fixed angle, or a callable.
- {func}`~skyplothelper.add_graticule_overlay` — an alias of
  {func}`~skyplothelper.add_coord_overlay` (same function, alternate name).

```python
sph.add_second_grid(ax, overlay_frame="galactic",
                    color="tab:red", alpha=0.4)
```

astropy's own `ax.get_coords_overlay(...)` does the core of this too —
the skyplothelper layer adds the curve selection, per-family styling,
in-frame tick placement, and seam handling that the tutorials lean on.
Both can coexist on one frame.

For globes and pseudocylindrical projections,
{func}`~skyplothelper.add_curved_lon_ticks` places longitude labels
*along a parallel* (e.g. the equator) where they follow the curved
graticule — with back-hemisphere hiding on globe frames.

## Choosing a labeling tool

Three tools put coordinate labels on a frame, and their jobs genuinely
differ — which is why it helps to see them side by side:

| Capability | {func}`~skyplothelper.format_ticklabels` | {func}`~skyplothelper.add_overlay_ticks` | {func}`~skyplothelper.apply_boundary_labels` |
|---|---|---|---|
| **What it labels** | astropy's native edge ticks | custom labels on *any* curve | labels along the frame boundary |
| **Placement** | frame spines only | boundary / axis / `lat=N` / `lon=N` / custom curve | boundary edge (left / right / both) |
| **Move lon labels onto an interior parallel** | ✗ | ✓ `lon_at='lat=N'` | ✗ |
| **Offset (arcsec/mas) labels** | → {func}`~skyplothelper.apply_offset_ticks` | ✗ | ✗ |
| **Single- / per-axis styling** | ✓ `which=`, `lon_*`/`lat_*` | ✓ `lon_*`/`lat_*` kwargs | one coord per call (`coord_index=`) |
| **Rotation modes** | fixed angle (`rotation`, `lon/lat_rotation`) | `tangent` / `tangent_upright` / `tangent_perp` / `tangent±N` / `horizontal` / float / callable | `perpendicular` / `parallel` / `horizontal` |
| **Separators** | `lon_sep`/`lat_sep` | `label_kwargs={'sep': …}` (+ per-axis) | n/a |
| **Stroke** | `stroke_lw`/`stroke_color` | `stroke_lw`/`stroke_color` | `stroke_lw`/`stroke_color` |

```{note}
**Two rotation vocabularies.** {func}`~skyplothelper.apply_boundary_labels`
names orientation *relative to the boundary* (`orient='parallel'` runs the
label along the boundary), while {func}`~skyplothelper.add_overlay_ticks` and
{meth}`CoordinateOverlay.render_labels <skyplothelper.CoordinateOverlay.render_labels>`
name it *relative to the gridline tangent* (`rotate='tangent'` runs the label
along the gridline). So `apply_boundary_labels(orient='parallel')` is the same
look as `add_overlay_ticks(..., label_kwargs={'rotate': 'tangent_perp'})`.
```

A common request — putting only the longitude labels onto an interior
parallel, upright — falls out of `add_overlay_ticks`:

```python
# Move ONLY the longitude labels onto the lat=-30 parallel, upright:
sph.add_overlay_ticks(ax, lon_at="lat=-30", lat_at=None,
                      suppress_default="lon",
                      label_kwargs={"rotate": "tangent_perp", "sep": "plain"})
```

## Label sizing

{func}`~skyplothelper.auto_size_ticklabels` shrinks tick labels to fit
the figure scale (with a floor); builders apply it via
`auto_fontsize=True`, and it's callable directly after layout changes.

## Pitfalls

- **rcParams tick settings not taking effect** — WCSAxes routes ticks
  through `ax.coords`, not rcParams; use these helpers or
  {func}`~skyplothelper.style_wcs_axes` ({doc}`concepts`).
- **Offset labels on an absolute frame** — `apply_offset_ticks` relabels
  in place; for a genuinely offset *WCS* (so pixel math is relative
  too), build with {func}`~skyplothelper.offset_figure`.
- **Overlay ticks colliding with primary labels** — move the overlay's
  ticks in-frame (`lon_at=`/`lat_at=` on a gridline) instead of stacking
  both systems on the boundary.
- **An anchor tick crowding its neighbors** — the anchored-offset style
  manages this spacing; if you're hand-rolling labels with the raw
  formatters, leave the anchor more room than you think it needs.

Full listing: {doc}`API reference <../api/ticks>`. The second-coordinate
overlay and grid-styling helpers used above are documented in the
{doc}`overlays API reference </api/overlays>`.

**Tutorials:** {doc}`Decorating frames </tutorials/decorating_frames>` (grid
styling, tick placement, minor ticks, label rotation, the publication presets)
and {doc}`Overlay coordinate grids </tutorials/overlay_grids>` (second
coordinate systems drawn over a frame).
