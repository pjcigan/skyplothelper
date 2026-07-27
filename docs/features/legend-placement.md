# Legend placement

```{image} /_static/features/legend-placement-light.png
:class: sph-plot plot-light dark-light
:alt: Legend placement (light mode)
```


```{image} /_static/features/legend-placement-dark.png
:class: sph-plot plot-dark dark-light
:alt: Legend placement (dark mode)
```


Where the key sits: anchored inside the axes, pushed into the figure
margin with an `"outside …"` preset, or at a free `(x, y)` anchor. The
outside presets are the all-sky selling point — the map keeps the whole
frame and the key sits beside it.

Guide: {doc}`/guide/legends` — API: {py:obj}`~skyplothelper.MultiLegend`

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(7)
n = 150
ra = rng.uniform(0, 360, n)
dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
survey_a = rng.random(n) < 0.55

# The gallery renders this snippet in both modes, so pick the legend
# palette off the active figure background rather than hard-coding it.
from matplotlib.colors import to_rgb
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
fig = plt.figure(figsize=(11, 3.0))
for i, (loc, orient) in enumerate([("lower right", "vertical"),
                                   ("outside bottom", "horizontal"),
                                   ("outside right", "vertical")]):
    ax = sph.make_wcs_frame(int(f"13{i + 1}"), "AIT", center=180)
    ax.scatter(ra[survey_a], dec[survey_a], s=10, color="C0", alpha=0.65,
               transform=ax.get_transform("world"))
    ax.scatter(ra[~survey_a], dec[~survey_a], s=10, color="C1", alpha=0.65,
               transform=ax.get_transform("world"))
    (sph.MultiLegend(ax, palette=LP, loc=loc, orientation=orient, fontsize=6)
        .add_color("survey", {"A": "C0", "B": "C1"}, swatch="marker")
        .draw())
    ax.set_title(f'loc="{loc}"', fontsize=8)
fig.subplots_adjust(wspace=0.35)
```
