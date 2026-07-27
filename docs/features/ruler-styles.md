# Ruler styles

```{image} /_static/features/ruler-styles-light.png
:class: sph-plot plot-light dark-light
:alt: Ruler styles (light mode)
```


```{image} /_static/features/ruler-styles-dark.png
:class: sph-plot plot-dark dark-light
:alt: Ruler styles (dark mode)
```


Every `Ruler` knob on one panel: ticks on both sides or just one, minor ticks,
varied tick lengths, and arrow / tick endcaps — each bar in a different unit
(arcsec, arcmin, and physical kpc / AU / Mpc via `convert=`) — plus a vertical
`Ruler` on the right standing in for a twin axis.

Guide: {doc}`/guide/overlays` — API: {py:obj}`~skyplothelper.Ruler`

## Code

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import skyplothelper as sph
from skyplothelper import Ruler

NEUTRAL = mpl.rcParams["axes.edgecolor"]   # dark on light, light on dark

fig, ax = plt.subplots(figsize=(8.5, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Ruler - one bar, every knob")

# A stack of horizontal rulers, each self-labeled by its title, walking through
# the styling knobs: tick side, minor ticks, tick length, and endcap styles. A
# per-ruler pixscale_asec gives each a different physical scale and unit.
X0, X1 = 10, 90
rows = [
    dict(y=90, color="C0", pixscale_asec=0.5, tick_interval=10,
         title="default ticks (arcsec)"),
    dict(y=76, color="C1", pixscale_asec=1.875, tick_interval=30, tick_side="right",
         title="tick_side='right' (arcmin)"),
    dict(y=62, color="C2", pixscale_asec=0.5, tick_interval=10, fmt="%.0f",
         convert=dict(redshift=0.5, unit="kpc"), minor_ticks=4,
         title="minor_ticks=4 (kpc, z=0.5)"),
    dict(y=48, color="C3", pixscale_asec=0.5, tick_interval=10, fmt="%.0f",
         convert=dict(distance=100, distance_unit="pc", unit="au"), tick_length=9,
         minor_ticks=4, minor_tick_length=4, tick_side="left",
         title="long major + minor (AU @ 100 pc)"),
    dict(y=34, color="C4", pixscale_asec=5.0, tick_interval=100, fmt="%.1f",
         convert=dict(redshift=1.0, unit="Mpc"), endcap_style="arrow",
         endcaps="both", title="endcap_style='arrow' (Mpc, z=1)"),
    dict(y=20, color="C5", tick_interval=20, endcap_style="tick", tick_side="right",
         minor_ticks=2, title="endcap_style='tick' (pixels)"),
]
for r in rows:
    y = r.pop("y"); color = r.pop("color")
    Ruler((X0, y), (X1, y), ax=ax, color=color, lw=1.6,
          label_fontsize=8, title_fontsize=9, **r).add_to(ax)

# A vertical Ruler on the right, standing in for a twin y-axis: ticks point in,
# the title sits outside like a secondary axis label.
Ruler((96, 10), (96, 90), ax=ax, color=NEUTRAL, lw=1.6, pixscale_asec=0.5,
      tick_interval=10, tick_side="left", minor_ticks=4, label_side="left",
      label_fontsize=8, title="Ruler as a twin axis", title_side="right",
      title_fontsize=9).add_to(ax)
fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.05)
```
