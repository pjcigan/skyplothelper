# Styling & themes

A coherent look across a paper, talk, or tutorial series should be one
call, not a per-figure ritual. skyplothelper's styling is four
independent, composable layers — apply any one alone or all together, and
each accepts either a bundled preset or your own custom definition:

| Layer | Sets | Setter | Custom input |
|---|---|---|---|
| **base** | structural rcParams: tick geometry, minor ticks, legend (+ a default font) | {func}`~skyplothelper.set_base_style` | your own rcParams dict |
| **theme** | background/foreground coordination: light vs. dark | {func}`~skyplothelper.set_theme` | any matplotlib style name, or a dict |
| **palette** | the data-color cycle | {func}`~skyplothelper.set_palette` | an explicit color list |
| **font** | the font-family stack (+ paired math fontset) | {func}`~skyplothelper.set_font` | an explicit family stack, or a dict |

```python
import skyplothelper as sph

sph.set_style(base="standard", theme="dark_sky", palette="nightcap")  # all three
sph.set_palette("speakeasy")                                       # just one layer
```

Layers apply in order (base → theme → palette → font → any extra rcParams
passed to {func}`~skyplothelper.set_style`), so a later layer wins on
shared keys — `font=` overrides whatever font the base preset set. {func}`~skyplothelper.style_context` is the same composition as a
context manager — the right tool for notebook cells, since the previous
rcParams are restored on exit:

```python
with sph.style_context(theme="dark_sky", palette="velvet"):
    fig, ax = sph.allsky_figure()
    ...
# rcParams restored here
```

## Custom looks

Every layer takes user-supplied definitions, so a fully custom house
style is just three objects you define once and reuse everywhere:

```python
MY_BASE = {"xtick.direction": "out", "font.size": 11,
           "legend.frameon": False}
MY_THEME = {"figure.facecolor": "#1d1c1a", "axes.facecolor": "#262522",
            "text.color": "#d9d5c5", "axes.labelcolor": "#d9d5c5",
            "xtick.color": "#9a958a", "ytick.color": "#9a958a",
            "grid.color": "#403e39"}
MY_COLORS = ["#B98E3E", "#A35D4C", "#41736B", "#67809F"]

sph.set_style(base=MY_BASE, theme=MY_THEME, palette=MY_COLORS)
```

`set_theme` also resolves **matplotlib's built-in style names** — anything
in `matplotlib.style.available` (`'ggplot'`, `'bmh'`,
`'seaborn-v0_8-darkgrid'`, ...) — so existing matplotlib looks drop
straight in:

```python
sph.set_theme("ggplot")
sph.set_style(theme="seaborn-v0_8-darkgrid", palette="atlas")
```

(skyplothelper's own preset names win if a name ever exists in both
registries.)

```{note}
The figures below follow the navbar **plot-color** toggle (A → L → D, next
to the site light/dark switch), so you can preview a light look while
reading in dark mode, or vice versa. They are generated straight from the
live `BASE_PRESETS` / `CYCLE_PALETTES` / `ANNOTATION_PALETTES` dicts
(`docs/make_style_gallery.py`), so they never drift from the package.
```

## Base presets

The structural layer ({func}`~skyplothelper.set_base_style`, or `base=` to
{func}`~skyplothelper.set_style`). Eight named presets in `BASE_PRESETS`,
each tuned for a medium — same data, different line-weight hierarchy,
ticks, legend, and font stack:

::::{grid} 1 2 2 4
:gutter: 2

:::{grid-item-card} `standard`
```{image} /_static/style/preset-standard-light.png
:class: sph-plot plot-light dark-light
:alt: standard base preset (light)
```
```{image} /_static/style/preset-standard-dark.png
:class: sph-plot plot-dark dark-light
:alt: standard base preset (dark)
```
+++
Default. Opinionated, batteries-included general look.
:::

:::{grid-item-card} `structural`
```{image} /_static/style/preset-structural-light.png
:class: sph-plot plot-light dark-light
:alt: structural base preset (light)
```
```{image} /_static/style/preset-structural-dark.png
:class: sph-plot plot-dark dark-light
:alt: structural base preset (dark)
```
+++
Color/font-*agnostic*: structural nudges only; your colors and fonts untouched.
:::

:::{grid-item-card} `journal`
```{image} /_static/style/preset-journal-light.png
:class: sph-plot plot-light dark-light
:alt: journal base preset (light)
```
```{image} /_static/style/preset-journal-dark.png
:class: sph-plot plot-dark dark-light
:alt: journal base preset (dark)
```
+++
Thin lines, fine ticks, serif/cm, 600 dpi — print.
:::

:::{grid-item-card} `press`
```{image} /_static/style/preset-press-light.png
:class: sph-plot plot-light dark-light
:alt: press base preset (light)
```
```{image} /_static/style/preset-press-dark.png
:class: sph-plot plot-dark dark-light
:alt: press base preset (dark)
```
+++
Bolder sans-serif, framed legend — reproduction.
:::

:::{grid-item-card} `poster`
```{image} /_static/style/preset-poster-light.png
:class: sph-plot plot-light dark-light
:alt: poster base preset (light)
```
```{image} /_static/style/preset-poster-dark.png
:class: sph-plot plot-dark dark-light
:alt: poster base preset (dark)
```
+++
Large type, heavy lines — viewing at distance.
:::

:::{grid-item-card} `tufte`
```{image} /_static/style/preset-tufte-light.png
:class: sph-plot plot-light dark-light
:alt: tufte base preset (light)
```
```{image} /_static/style/preset-tufte-dark.png
:class: sph-plot plot-dark dark-light
:alt: tufte base preset (dark)
```
+++
Minimal-ink serif, no top/right spines — restraint.
:::

:::{grid-item-card} `screen`
```{image} /_static/style/preset-screen-light.png
:class: sph-plot plot-light dark-light
:alt: screen base preset (light)
```
```{image} /_static/style/preset-screen-dark.png
:class: sph-plot plot-dark dark-light
:alt: screen base preset (dark)
```
+++
Slightly heavier so hairlines don't vanish on displays; lower dpi.
:::

:::{grid-item-card} `minimalist`
```{image} /_static/style/preset-minimalist-light.png
:class: sph-plot plot-light dark-light
:alt: minimalist base preset (light)
```
```{image} /_static/style/preset-minimalist-dark.png
:class: sph-plot plot-dark dark-light
:alt: minimalist base preset (dark)
```
+++
Frameless, tickless — splash / title images **only**, not data plots.
:::
::::

Fonts are applied as *stacks* (the first installed face wins — see
[Fonts](#fonts) for why that matters for the ′ ″ marks); serif presets pair
`mathtext.fontset='cm'`, sans presets `'stixsans'`.

## Themes

The light/dark coordination layer ({func}`~skyplothelper.set_theme`) —
backgrounds and foregrounds, holding the structure fixed (same `standard`
base + `uranometria` cycle below, so only the theme changes):

::::{grid} 1 2 4 4
:gutter: 2

:::{grid-item-card} `publication`
```{image} /_static/style/theme-publication.png
:class: sph-plot
:alt: publication theme
```
+++
Clean white.
:::

:::{grid-item-card} `poster`
```{image} /_static/style/theme-poster.png
:class: sph-plot
:alt: poster theme
```
+++
White, larger type.
:::

:::{grid-item-card} `twilight`
```{image} /_static/style/theme-twilight.png
:class: sph-plot
:alt: twilight theme
```
+++
Violet-tinged dark.
:::

:::{grid-item-card} `dark_sky`
```{image} /_static/style/theme-dark_sky.png
:class: sph-plot
:alt: dark_sky theme
```
+++
Near-black night sky.
:::
::::

Beyond these, `set_theme` also takes any matplotlib built-in style name or
a custom dict (see [Custom looks](#custom-looks) above).

## Cycle palettes

The data-color cycle ({func}`~skyplothelper.set_palette`). Muted sets in
`CYCLE_PALETTES`, designed to hold up at full opacity *and* as alpha fills;
adjacent colors are separated in lightness, so the sequences survive
color-vision-deficiency simulation and grayscale printing.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} `speakeasy`
```{image} /_static/style/cycle-speakeasy-light.png
:class: sph-plot plot-light dark-light
:alt: speakeasy palette (light)
```
```{image} /_static/style/cycle-speakeasy-dark.png
:class: sph-plot plot-dark dark-light
:alt: speakeasy palette (dark)
```
+++
Dual-mode.
:::

:::{grid-item-card} `atlas`
```{image} /_static/style/cycle-atlas-light.png
:class: sph-plot plot-light dark-light
:alt: atlas palette (light)
```
```{image} /_static/style/cycle-atlas-dark.png
:class: sph-plot plot-dark dark-light
:alt: atlas palette (dark)
```
+++
Dual-mode.
:::

:::{grid-item-card} `uranometria`
```{image} /_static/style/cycle-uranometria-light.png
:class: sph-plot plot-light dark-light
:alt: uranometria palette (light)
```
```{image} /_static/style/cycle-uranometria-dark.png
:class: sph-plot plot-dark dark-light
:alt: uranometria palette (dark)
```
+++
Dual-mode.
:::

:::{grid-item-card} `letterpress`
```{image} /_static/style/cycle-letterpress-light.png
:class: sph-plot plot-light dark-light
:alt: letterpress palette (light)
```
```{image} /_static/style/cycle-letterpress-dark.png
:class: sph-plot plot-dark dark-light
:alt: letterpress palette (dark)
```
+++
Light-only.
:::

:::{grid-item-card} `nightcap`
```{image} /_static/style/cycle-nightcap-light.png
:class: sph-plot plot-light dark-light
:alt: nightcap palette (light)
```
```{image} /_static/style/cycle-nightcap-dark.png
:class: sph-plot plot-dark dark-light
:alt: nightcap palette (dark)
```
+++
Dark-only.
:::

:::{grid-item-card} `velvet`
```{image} /_static/style/cycle-velvet-light.png
:class: sph-plot plot-light dark-light
:alt: velvet palette (light)
```
```{image} /_static/style/cycle-velvet-dark.png
:class: sph-plot plot-dark dark-light
:alt: velvet palette (dark)
```
+++
Dark-only.
:::
::::

## Annotation palettes

`ANNOTATION_PALETTES` are coordinated *role* palettes — 12 roles:
`fig_bg`, `ax_bg`, `text`, `text2` (secondary text), `label`, `compass`,
`grid`, `grid2` (minor grid), `frame`, `stars`, `accent`, `accent2` — for
finder-chart style figures where the annotation *hierarchy*, not just the
data colors, carries the design. Each palette is inherently light or dark.

The one thing to know, because it is not guessable from the signature:
{func}`~skyplothelper.style_annotation` **styles the axes chrome**
(backgrounds, frame, ticks, grid) **and returns the resolved role dict so
you color your own artists with it.** It is not ambient state that overlays
consult behind the scenes — you apply the roles yourself:

```python
pal = sph.style_annotation(ax, "night")   # styles the chrome, returns the dict
ax.scatter(ra, dec, color=pal["stars"], transform=ax.get_transform("world"))
ax.text(x, y, "M31", color=pal["label"])  # you pick which role each artist takes
```

The bundled palettes:

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} `parchment`
```{image} /_static/style/annot-parchment.png
:class: sph-plot
:alt: parchment annotation palette
```
+++
Light — cream cartographic paper.
:::

:::{grid-item-card} `publication`
```{image} /_static/style/annot-publication.png
:class: sph-plot
:alt: publication annotation palette
```
+++
Light — data-ink minimalism on white.
:::

:::{grid-item-card} `dark`
```{image} /_static/style/annot-dark.png
:class: sph-plot
:alt: dark annotation palette
```
+++
Dark.
:::

:::{grid-item-card} `night`
```{image} /_static/style/annot-night.png
:class: sph-plot
:alt: night annotation palette
```
+++
Dark — deep night sky.
:::

:::{grid-item-card} `denim`
```{image} /_static/style/annot-denim.png
:class: sph-plot
:alt: denim annotation palette
```
+++
Dark — warm charcoal (the docs dark figures use this).
:::
::::

## Color defaults, and a theme-safe figure

Most skyplothelper decorations take `color=` (and `stroke_color=`) with a
default of `None` meaning **"follow the current theme."** The resolution
order is uniform: an explicit argument wins; otherwise the color follows
sph's style layer (the matplotlib rcParams that `set_style`/`set_theme`
write — `text.color` for ink, `axes.facecolor` for the stroke behind it);
otherwise a sensible historical literal. So a compass, ruler, reticle,
plane overlay, or sky label drawn with defaults comes out dark on a light
theme and light on a dark one, with no per-figure color-wrangling:

```python
with sph.style_context(theme="dark_sky"):
    fig, ax = sph.allsky_figure(projection="AIT")
    sph.add_compass(ax)          # ink follows text.color -> light, automatically
    sph.add_plane_overlay(ax, "galactic")
```

```{note}
This uniformity is a property of skyplothelper's *own* decorations. When
you drop to **raw matplotlib**, `color=None` is not uniformly theme-aware —
it means something different per artist, which is a common source of a
stray blue line on an otherwise-coordinated figure:

| Call | `color=None` (or omitted) resolves to |
|---|---|
| `ax.text(...)` | `rcParams["text.color"]` — theme-aware ✅ |
| `ax.plot(...)` | `rcParams["lines.color"]`, whose default is `"C0"` — **the first cycle color, i.e. blue**, not the theme ink |
| `ax.scatter(...)` | the next **property-cycle** color — also cycle, not ink |

To draw a *line* in the theme ink, pass it explicitly —
`ax.plot(..., color=plt.rcParams["text.color"])` — or use the sph verb
({func}`~skyplothelper.plot`), which applies the frame-aware defaults for
you. (This is matplotlib behavior, not sph's, so it holds for any figure.)
```

Putting it together — a dark figure with correct data colors, correct
decoration colors, and a legible legend is just the three layers plus the
palette's own role dict where you hand-draw:

```python
with sph.style_context(base="standard", theme="dark_sky", palette="nightcap"):
    fig, ax = sph.allsky_figure(projection="AIT")
    ax.scatter(ra, dec, transform=ax.get_transform("world"))  # cycle -> palette
    sph.add_compass(ax)                                       # None -> theme ink
    pal = sph.style_annotation(ax, "night")                   # chrome + role dict
    ax.legend(facecolor=pal["ax_bg"], edgecolor=pal["frame"],
              labelcolor=pal["text"])                         # legible on dark
```

## Image colormaps

A curated set of image colormaps for astronomical data, registered under
the `sph.` prefix on import (the cmocean/cmasher approach), so they drop
into any `cmap=`:

```python
ax.imshow(img, cmap="sph.deepsky")            # registry string
ax.imshow(img, cmap=sph.get_colormap("deepsky"))   # fetch the object
ax.imshow(img, cmap=sph.colormaps.deepsky)    # attribute access (cmocean-style)
ax.imshow(img, cmap="sph.sunset_r")           # every name reverses as _r
```

Three ways to reach a map: the registered **string** (`"sph.deepsky"`,
usable wherever a `cmap=` is accepted since the maps register on import);
{func}`~skyplothelper.get_colormap` to fetch the `Colormap` **object** (the
`sph.` prefix and `_r` suffix are both optional there); or **attribute
access** on the module (`sph.colormaps.deepsky`, cmocean-style, `_r` too)
for readers who prefer objects over strings.

The twelve linear maps have monotonic luminance (safe for images — no
false ridges), while the six `sph.diff_*` maps are diverging, for
residual/difference data. {func}`~skyplothelper.list_colormaps` enumerates
them and {func}`~skyplothelper.show_colormaps` renders the swatch below.
They are
namespaced under `sph.` so they never clobber matplotlib's or another
package's colormaps. For a full worked demo, see the
{doc}`FITS images tutorial </tutorials/fits_images>`.

```{image} /_static/style/colormaps-light.png
:class: sph-plot plot-light dark-light
:alt: bundled sph. image colormaps as labeled gradient swatches
:width: 100%
```
```{image} /_static/style/colormaps-dark.png
:class: sph-plot plot-dark dark-light
:alt: bundled sph. image colormaps as labeled gradient swatches
:width: 100%
```

## Fonts

The font layer ({func}`~skyplothelper.set_font`, or `font=` on
{func}`~skyplothelper.set_style`) sets the font-family stack and a paired
math fontset. Presets are *stacks*, not single faces, so a look degrades to
the next available family rather than dropping straight to DejaVu.

One thing worth knowing, because it is easy to assume otherwise: matplotlib
resolves a family stack by picking the first **installed** entry and drawing
the whole string with it — there is no per-glyph fallback. A face listed at
the end of a stack is therefore unreachable whenever anything ahead of it is
installed, so a trailing "safety net" font cannot rescue a missing glyph.
That matters here because skyplothelper emits ′ and ″ in coordinate labels,
and several common faces (Calibri, Carlito) don't carry them.

So the guarantee is enforced at apply time instead: `set_font` checks whether
the face that would actually win can render those marks and, if it can't,
promotes the DejaVu bundled with matplotlib to the front. That holds on any
machine and font set rather than depending on which faces happen to ship with
an OS. The repair applies to the workhorse `sans`/`serif`/`mono` presets;
a display or handwriting preset keeps its face and warns instead, since
choosing a decorative font is a deliberate trade — reach for
`separator='hms_letter'` there if the marks matter.

```python
sph.set_font("journal")                  # TeX Gyre Termes + cm math
sph.set_font("web", math="stixsans")     # sans + explicit math fontset
sph.set_font(["EB Garamond", "TeX Gyre Termes"])   # an explicit stack
sph.set_style(base="journal", font="journal")      # as the fourth layer
```

`math='auto'` (the default) pairs the math fontset to the family —
serif→`cm`, sans→`stixsans`, handwriting/mono→`dejavusans`. The bundled
presets in `FONT_PRESETS`:

| Preset | Description |
|---|---|
| `journal` | Times-metric serif (TeX Gyre Termes) — polished ApJ/MNRAS journal look. |
| `talk` | Palatino-like serif (TeX Gyre Pagella) — warm and open; talks & posters. |
| `tufte` | Avant-Garde geometric sans (TeX Gyre Adventor) — Gill Sans nod for Tufte-style charts. |
| `web` | Helvetica-metric sans (TeX Gyre Heros / Nimbus Sans / Liberation Sans) — clean and modern for screens & docs. |
| `classical` | Cinzel monumental caps over a serif body — classical headers (needs `register=`). |
| `sketch` | Hand-drawn stack (xkcd Script / Patrick Hand / Caveat) — informal explainers (needs `register=`); pairs with `plt.xkcd()`. |
| `mono` | Monospace stack (IBM Plex Mono) — tables, code, fixed-width tick labels. |

`classical` and `sketch` rely on faces matplotlib doesn't ship; register
them first with `register=` (a file, directory, or list) — otherwise the
preset degrades to its tier-1 fallback:

```python
sph.set_font("classical", register="Cinzel-Regular.ttf")
```

{data}`~skyplothelper.MONO_STACK` is a separate per-artist monospace stack
for fixed-width readouts (`ax.text(..., family=sph.MONO_STACK)`), distinct
from the figure-wide `mono` preset above. The sample strip below shows the
serif / sans / monospace stacks:

```{image} /_static/style/fonts-light.png
:class: sph-plot plot-light dark-light
:alt: serif / sans-serif / monospace font-stack samples
:width: 100%
```
```{image} /_static/style/fonts-dark.png
:class: sph-plot plot-dark dark-light
:alt: serif / sans-serif / monospace font-stack samples
:width: 100%
```

## Styling sky frames (the WCSAxes caveat)

astropy's WCSAxes ignores most `xtick.*`/`ytick.*` rcParams — tick
direction, size, and color are controlled through `ax.coords`, not the rc
machinery. The package bridges this in two ways:

- Frame builders consult the current style at creation time, so frames
  made *after* `set_style()` come out styled.
- {func}`~skyplothelper.style_wcs_axes` retrofits the current (or
  explicitly passed) tick/label/frame/grid styling onto an already-built
  WCSAxes. It also takes `stroke_lw=`/`stroke_color=` (opt-in) for a
  legibility stroke on the native tick marks over imagery — the tick-mark
  member of the stroke trio with {func}`~skyplothelper.style_grid` (grid
  lines) and {func}`~skyplothelper.format_ticklabels` (labels). On
  elliptical all-sky frames (`AIT`/`MOL`/…), `direction=` also drives the
  curved-boundary Dec ticks skyplothelper draws there (astropy can't), so
  an inward-tick base — or `direction='in'` — points them inward.
- {func}`~skyplothelper.apply_frame_stroke` is the one-call shortcut for the
  common case: it strokes the frame edge and the tick marks together, so a
  frame stays legible drawn over a bright image without styling each piece.

```python
sph.set_style(theme="dark_sky", palette="nightcap")
fig, ax = sph.allsky_figure(projection="AIT")   # styled at creation
sph.style_wcs_axes(other_ax)                    # retrofit an older frame
```

## Pitfalls

- **rcParams tweaks not reaching a sky frame** → the WCSAxes caveat
  above; use {func}`~skyplothelper.style_wcs_axes` or set the style
  before building the frame.
- **A palette looks muddy after switching themes** → palettes carry a
  design mode; `letterpress` is light-only and `nightcap`/`velvet`
  dark-only. The dual-mode sets (`speakeasy`, `atlas`, `uranometria`)
  are safe either way.
- **Styles leaking between notebook figures** → prefer
  {func}`~skyplothelper.style_context` over the setters in notebooks;
  the setters intentionally change the global rcParams.
- **`set_theme` vs. annotation palette `'dark'`** → two different
  namespaces: `set_theme('dark_sky')` sets rcParams; 
  `style_annotation(ax, 'dark')` applies role colors to one axes.

Per-element control — tick label formats, grid styling, highlighted
gridlines — is covered in {doc}`ticks`. Full listing: {doc}`API reference
<../api/utilities>`.

**See also:** {doc}`plotly` — the interactive backend has its own
light/dark theming (`make_figure(theme=...)`), separate from these rcParams.

**Tutorial:** {doc}`Themes, palettes & fonts </tutorials/styling>` puts the
whole look system to work — the base/theme/palette/font layers, CVD-safe
cycles, annotation palettes, and building a reusable house style. For
per-element tuning, see {doc}`Decorating frames </tutorials/decorating_frames>`.
