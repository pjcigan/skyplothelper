# Tick label styles & conventions

```{image} /_static/features/tick-label-styles-conventions-light.png
:class: sph-plot plot-light dark-light
:alt: Tick label styles & conventions (light mode)
```


```{image} /_static/features/tick-label-styles-conventions-dark.png
:class: sph-plot plot-dark dark-light
:alt: Tick label styles & conventions (dark mode)
```


One call — `format_ticklabels(ax, style=...)` — swaps the whole labeling
convention (sexagesimal superscripts, a colon convention, decimal degrees, or
relative offsets); on a curved frame `make_wcs_frame(tick_rotation=...)` sets
whether the labels ride the gridline tangent or lie flat.

Guide: {doc}`/guide/ticks` — API: {py:obj}`~skyplothelper.format_ticklabels` · {py:obj}`~skyplothelper.make_wcs_frame`

## Code

```python
import skyplothelper as sph
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 7.2))
# One call - sph.format_ticklabels(ax, style=...) - swaps the whole labeling
# convention: sexagesimal superscripts, a colon convention, decimal degrees, or
# relative offsets. Only the last needs a matched (mas-scale) field of view.
presets = [
    ("publication", None),   # sexagesimal h/m/s superscripts (the default)
    ("casa",        None),   # colon convention, 05:42:00
    ("decimal",     None),   # decimal degrees
    ("offset_mas",  4e-4),   # relative offsets in mas (a +/-40 mas field)
]
for i, (style, cdelt_asec) in enumerate(presets, start=1):
    kw = dict(projection="TAN", center=(180.0, 30.0), fig=fig)
    if cdelt_asec is not None:
        kw.update(cdelt=cdelt_asec / 3600.0, npix=200)
    ax = sph.make_wcs_frame((2, 3, i), **kw)
    sph.format_ticklabels(ax, style=style)
    ax.set_title("style=%r" % style, fontsize=9)
# On a curved frame the labels ride the gridline tangent by default, or lie
# flat - set once at build time with tick_rotation=.
for j, rot in enumerate(["tangent", "horizontal"], start=5):
    ax = sph.make_wcs_frame((2, 3, j), projection="SIN", center=(60.0, 35.0),
                            fig=fig, tick_rotation=rot)
    ax.set_title("tick_rotation=%r" % rot, fontsize=9)
fig.suptitle("Tick label styles & conventions", fontsize=13)
fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.05,
                    hspace=0.45, wspace=0.40)
```
