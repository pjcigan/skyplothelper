# Themes at a glance

```{image} /_static/features/themes-at-a-glance-light.png
:class: sph-plot plot-light dark-light
:alt: Themes at a glance (light mode)
```


```{image} /_static/features/themes-at-a-glance-dark.png
:class: sph-plot plot-dark dark-light
:alt: Themes at a glance (dark mode)
```


One figure, four `set_style` themes: each panel forces a theme (and its data
palette) in a `style_context`, restyling the same sky field's background, frame,
grid, text, and colors. The base-preset and annotation-palette layers round out
the system — see the styling guide.

Guide: {doc}`/guide/styling` — API: {py:obj}`~skyplothelper.set_style` · {py:obj}`~skyplothelper.style_context`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

# Four themes, each forced in its own style_context so the specimen renders the
# same regardless of the page's light/dark mode. Each panel is the same little
# sky field, restyled: background, frame, grid, text, and the data-color cycle.
themes = [("publication", "uranometria"), ("twilight", "velvet"),
          ("dark_sky", "nightcap"), ("poster", "speakeasy")]
rng = np.random.default_rng(3)
groups = [(172.5, 5.5), (187.5, 5.5), (172.5, -5.5), (187.5, -5.5)]
pts = [(g[0] + rng.normal(0, 2.2, 55), g[1] + rng.normal(0, 2.2, 55)) for g in groups]

fig = plt.figure(figsize=(9.5, 8.4))
for i, (theme, pal) in enumerate(themes, start=1):
    with sph.style_context(base="standard", theme=theme, palette=pal):
        ax = sph.make_wcs_frame((2, 2, i), "TAN", center=(180, 0), fov_deg=22, fig=fig)
        cycle = sph.CYCLE_PALETTES[pal]["colors"]
        for g, (ra, dec) in enumerate(pts):
            ax.scatter(ra, dec, transform=ax.get_transform("world"), s=22, lw=0,
                       color=cycle[g % len(cycle)], alpha=0.9)
        ax.grid(True)
        # Label inside the axes on a card of the theme's own background, so it
        # reads in the theme's text color regardless of the page's light/dark mode.
        ax.text(0.035, 0.965, f"theme='{theme}'\npalette='{pal}'",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                bbox=dict(facecolor=plt.rcParams["axes.facecolor"], alpha=0.75,
                          edgecolor="none", boxstyle="round,pad=0.3"))
fig.suptitle("set_style - one sky field, four themes", fontsize=13)
fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.05,
                    hspace=0.28, wspace=0.22)
```
