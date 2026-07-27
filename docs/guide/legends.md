# Multi-channel legends

A single sky map often encodes several data dimensions at once — marker
*shape*, *color*, *size*, *fill*, *opacity*, *orientation*, and *line style*
can each carry a different meaning. Explaining all of that with matplotlib's
built-in `Axes.legend` is awkward: it flattens everything into one list of
proxy handles you hand-build yourself, with nothing to keep the shape swatches
visually distinct from the colored data, to group channels, or to place the
key off the axes frame — the usual need on an all-sky plot.

{class}`~skyplothelper.MultiLegend` is built for exactly this. You describe the
legend one *channel* at a time and it lays the channels out as separate,
labeled blocks — grayscale where it should be, mode-aware, and placeable in the
figure margin. It works on any matplotlib `Axes`, not just WCSAxes, so it is
useful for any busy figure.

```{seealso}
Section 2 of the {doc}`catalogs tutorial </tutorials/catalogs>`, *Legends for
multiple dimensions*, is the worked companion to this page — it builds the
same ideas up figure by figure, starting from a naive `ax.legend()` and
showing what each step fixes. Its subsections mirror the ones below, so most
headings here have an illustrated counterpart there.
```

## The channel-block model

```{image} /_static/features/multi-channel-legend-light.png
:class: sph-plot plot-light dark-light
:alt: A legend with separate blocks keying size, color and shape (light mode)
```
```{image} /_static/features/multi-channel-legend-dark.png
:class: sph-plot plot-dark dark-light
:alt: A legend with separate blocks keying size, color and shape (dark mode)
```
*{doc}`Multi-channel legend </features/multi-channel-legend>` — code in the Feature Gallery.*

A `MultiLegend` is a stack of **blocks**, one per visual channel. Each block
maps one channel (say, marker size) to one data meaning (say, number of
observations) and knows how to draw its own swatches. You build the legend
fluently — every `add_*` method returns the legend, so calls chain — and
finish with {meth}`~skyplothelper.MultiLegend.draw`:

```python
import skyplothelper as sph

(sph.MultiLegend(ax, loc="lower right")
    .add_color("Target", {"DDO 69": "purple", "DDO 70": "C0",
                          "DDO 75": "green", "DDO 210": "#b0a0d0"}, ncol=2)
    .add_shape("Sample", {"DGS": "o", "KINGFISH": "D", "Galliano+08": "^"})
    .add_fill("Measurement", {"Dust–gas": "filled", "Dust–HI": "open"})
    .add_line("Fit", {r"$X_{CO,MWG}$": ":", r"$X_{CO,Z}$": "-."})
    .draw())
```

Each `add_*` call takes a title and its **entries** — a `{label: value}` dict
(insertion-ordered) or a list of `(label, value)` pairs. The `value` is
whatever that channel varies: a color, a marker, a line style. For a *combined*
encoding (one entry varying several props at once), pass a full style dict as
the value and add it through {meth}`~skyplothelper.MultiLegend.add_custom` or a
bare {class}`~skyplothelper.LegendBlock` (see [below](#standalone-blocks)).

## Channels

```{image} /_static/features/channel-block-catalog-light.png
:class: sph-plot plot-light dark-light
:alt: A catalog of every MultiLegend channel-block kind on one figure (light mode)
```
```{image} /_static/features/channel-block-catalog-dark.png
:class: sph-plot plot-dark dark-light
:alt: A catalog of every MultiLegend channel-block kind on one figure (dark mode)
```
*{doc}`Channel-block catalog </features/channel-block-catalog>` — code in the Feature Gallery; each channel keyed on one figure in the {doc}`catalogs tutorial </tutorials/catalogs>`, so you can see how they read together.*

Every wrapper below is a thin convenience over one generic block; they differ
only in which property the entries vary and how the swatch is drawn.

| Method | Channel | Typical use |
|--------|---------|-------------|
| {meth}`~skyplothelper.MultiLegend.add_color` | face (or edge) color | an independent category — galaxy, survey, class |
| {meth}`~skyplothelper.MultiLegend.add_shape` | marker shape | literature sample, source type |
| {meth}`~skyplothelper.MultiLegend.add_size` / `add_size_from` | graduated marker size | a continuous quantity — counts, magnitude, aperture |
| {meth}`~skyplothelper.MultiLegend.add_edge` | marker edge color | a second category layered on the face color |
| {meth}`~skyplothelper.MultiLegend.add_fill` | open / solid (or `hatch`) | measurement type, detection vs limit |
| {meth}`~skyplothelper.MultiLegend.add_alpha` | graduated opacity | density, confidence |
| {meth}`~skyplothelper.MultiLegend.add_orientation` | marker rotation | position angle, polarization |
| {meth}`~skyplothelper.MultiLegend.add_line` | line style *or* width | fits, model curves, significance |
| {meth}`~skyplothelper.MultiLegend.add_region` | translucent filled patch | survey footprint, exclusion zone |
| {meth}`~skyplothelper.MultiLegend.add_colorbar` | continuous-color strip | a compact gradient inside the stack |
| {meth}`~skyplothelper.MultiLegend.add_glyph` | named sph glyph | reticle / instrument shapes ([glyphs](#named-glyph-swatches)) |
| {meth}`~skyplothelper.MultiLegend.add_text` | free note | "dashed = model" |
| {meth}`~skyplothelper.MultiLegend.add_custom` | any matplotlib artist | an escape hatch for anything above |

A few knobs worth knowing:

- **Color swatch shape** — `add_color` draws a filled color *chip*
  (`swatch="patch"`, the default) because a colored *circle* would wrongly
  imply "this color applies to circular markers." When color genuinely is tied
  to a marker (the data really are circles), pass `swatch="marker"` for a
  colored marker glyph, or `swatch="line"` for a thick color segment. Color the
  marker *edge* instead of its face with `target="edge"`.
- **Line style vs width** — `add_line` varies the dash pattern by default; pass
  `vary="lw"` for a line-*width* key.
- **Fill vs hatch** — `add_fill` entries are `"filled"`/`"open"` by default;
  pass `kind="patch"` and hatch strings (`"//"`, `"xx"`) for textured
  categories.
- **Per-block layout** — any block takes `ncol=` to wrap its entries into
  columns (the 2×2 color grid above), independent of the others.

## Placement

```{image} /_static/features/legend-placement-light.png
:class: sph-plot plot-light dark-light
:alt: Inside, off-frame and free-anchor legend placements side by side (light mode)
```
```{image} /_static/features/legend-placement-dark.png
:class: sph-plot plot-dark dark-light
:alt: Inside, off-frame and free-anchor legend placements side by side (dark mode)
```
*{doc}`Legend placement </features/legend-placement>` — code in the Feature Gallery; worked walkthrough in **Placing the legend** ({doc}`catalogs tutorial </tutorials/catalogs>`).*

`loc=` accepts three kinds of value:

- **Inside the axes** — matplotlib's usual location names (`"lower right"`,
  `"upper left"`, `"center"`, …), anchored in axes coordinates.
- **In the figure margin** — an `"outside …"` preset. All twelve combinations
  exist: the edges (`"outside right"`, `"outside bottom"`, …) and the corners
  (`"outside lower right"`, `"outside top left"`, …). This is the all-sky
  selling point — the map fills the frame and the key sits beside it.
- **A free anchor** — `loc=(x, y)` read in `coords="axes"` (default) or
  `coords="figure"`.

Because blocks are self-contained, you can attach several independent
`MultiLegend` instances to one axes to split keys across corners. An off-frame
legend overflows the figure by default; pass `reserve=True` to shrink the host
axes and open margin room for it.

```python
(sph.MultiLegend(ax, loc="outside bottom", orientation="horizontal")
    .add_size_from(cat, values=[1, 5, 10, 20, 50], title="N obs")
    .add_color("ICRF3", {"Defining": "orange", "Other": "C0"}, swatch="marker")
    .draw())
```

`orientation="horizontal"` lays the *blocks* side by side (natural for a
bottom-margin key); `orientation="vertical"` (the default) stacks them.

## Two ways to key marker size

```{image} /_static/features/keying-marker-size-light.png
:class: sph-plot plot-light dark-light
:alt: A plot_catalog size key beside a MultiLegend size block (light mode)
```
```{image} /_static/features/keying-marker-size-dark.png
:class: sph-plot plot-dark dark-light
:alt: A plot_catalog size key beside a MultiLegend size block (dark mode)
```
*{doc}`Keying marker size </features/keying-marker-size>` — code in the Feature Gallery; worked walkthrough in **Two ways to key marker size** ({doc}`catalogs tutorial </tutorials/catalogs>`).*

Marker size gets special treatment because the legend swatches must match the
plotted sizes exactly — a size key that disagrees with the scatter is worse
than none.

- **Quick, single-channel** — {func}`~skyplothelper.plot_catalog` will draw its
  own size key for you: `plot_catalog(ax, cat, sizeby="n_obs",
  size_legend=True)`. Use it when size is the *only* extra dimension.
- **Multi-channel** — when size is one channel among several, add it to a
  `MultiLegend` with
  {meth}`~skyplothelper.MultiLegend.add_size_from`, handing it the
  `plot_catalog` result. It reads the exact size scaling off that plot (see
  {meth}`SizeBlock.from_catalog <skyplothelper.SizeBlock.from_catalog>`) so the
  swatches reproduce on-plot sizes, and — with no `values=` given — auto-picks
  round 1/2/5-decade representatives (1, 5, 10, 50, …) spanning the data
  instead of raw min/mean/max.

```python
cat = sph.plot_catalog(ax, df, sizeby="n_obs", size_scale="sqrt",
                       smin=8, smax=400)
sph.MultiLegend(ax, loc="outside bottom").add_size_from(cat).draw()
```

`add_size_from` (and `SizeBlock.from_catalog`) require the plot to have been
drawn with `sizeby=` — that is what records the scaling for the key to recover.
When several `plot_catalog` calls share one scatter (e.g. one call per marker
shape), give them a common `size_vlim=(lo, hi)` so they share a single
raw→size scale and one key describes them all. Keep the key `values=` within
that displayed `size_vlim` range — a value above the upper bound extrapolates
past the clip and draws a swatch larger than any marker on the plot (the
auto path, with no `values=`, stays inside the range for you).

## Automatic niceties

*Illustrated: **What MultiLegend handles for you** in the {doc}`catalogs tutorial </tutorials/catalogs>` — the grayscale-neutral and mode-aware behavior shown against a hand-built key.*

`MultiLegend` handles the fiddly conventions so you don't have to:

- **Grayscale-neutral categories.** A `shape`, `size`, `fill`, or
  `orientation` block sitting alongside a `color` block turns neutral gray
  automatically, so "shape means one thing, color means another" reads at a
  glance. This is the single biggest fix to hand-built multi-channel legends.
  Override any block's tone by passing an explicit `color=`.
- **Mode-aware styling.** Pass `palette=` (an annotation palette name or dict)
  and the text, frame, and background colors follow it, staying legible on
  light or dark themes. Explicit `text_color` / `frame_color` / `facecolor`
  override it. Let the palette track the surrounding page or figure: a legend
  built with a fixed light palette (`"publication"`) renders as a bright card
  on a dark background, so pick it off the active theme — e.g.
  `palette="dark" if is_dark else "publication"`.
- **Stroke.** `stroke_color=` / `stroke_lw=` add an outline to the legend text
  and swatches — the same stroke convention as every other decoration — for
  legibility over a busy background.

## Standalone blocks

```{image} /_static/features/standalone-legend-blocks-light.png
:class: sph-plot plot-light dark-light
:alt: A legend mixing line, patch and glyph blocks over a sky map (light mode)
```
```{image} /_static/features/standalone-legend-blocks-dark.png
:class: sph-plot plot-dark dark-light
:alt: A legend mixing line, patch and glyph blocks over a sky map (dark mode)
```
*{doc}`Standalone legend blocks </features/standalone-legend-blocks>` — code in the Feature Gallery; see also the {doc}`catalogs tutorial </tutorials/catalogs>`.*

Each `add_*` wrapper has a matching class — {class}`~skyplothelper.ColorBlock`,
{class}`~skyplothelper.ShapeBlock`, {class}`~skyplothelper.SizeBlock`, and so on
— that you can build directly and attach with
{meth}`~skyplothelper.MultiLegend.add_block`. Underneath them all is one generic
{class}`~skyplothelper.LegendBlock` whose entries are arbitrary style dicts and
whose `swatch_kind` picks the renderer (`marker`, `line`, `patch`, `region`,
`text`, `custom`). Reach for it when you want a fully *combined* encoding in a
single entry:

```python
from skyplothelper import LegendBlock

block = LegendBlock("Models", {"model A": dict(ls="--", lw=2, color="C1"),
                               "model B": dict(ls=":", lw=1, color="C2")},
                    swatch_kind="line")
sph.MultiLegend(ax).add_block(block).draw()
```

(This is also the mechanism behind `add_custom`, which wraps whatever matplotlib
artist you hand it.)

## Named glyph swatches

The reticle shapes double as legend glyphs. {meth}`~skyplothelper.MultiLegend.add_glyph`
draws a registered glyph by name (`"reticle_plus"`, `"crosshair"`, `"target"`,
…) using the real glyph geometry, so the swatch and the plotted reticle can't
drift apart. Register your own with
{func}`~skyplothelper.register_glyph`, and list what's available with
{func}`~skyplothelper.list_glyphs`.

## Interactive (plotly) backend

The block classes are backend-agnostic. On the interactive side,
{func}`sphpl.add_legend <skyplothelper.plotly.add_legend>` renders the *same*
blocks as native plotly legend entries (grouped, invisible named traces),
including graduated size and alpha keys that plotly has no native equivalent
for. Build the blocks exactly as above and pass them in:

```python
import skyplothelper.plotly as sphpl

sphpl.add_legend(fig, [sph.ColorBlock("Class", {"A": "C0", "B": "C1"}),
                       sph.SizeBlock("N obs", values=[1, 10, 100])])
```

Plotly covers color, shape, size, edge, open/solid fill, alpha, orientation,
and line, and turns a `ColorbarBlock` into a real plotly colorbar. The
matplotlib-only refinements — hatch, translucent region patches, free text,
custom artists, and named glyphs — are skipped with a warning. See
{doc}`plotly` for the interactive backend as a whole.

---

For putting the catalogs these legends describe onto the sky, see
{doc}`vectors` (`plot_catalog` and the vector-field tools). The full
`MultiLegend` API and every block class are in the {doc}`legends API
reference </api/legends>`.
