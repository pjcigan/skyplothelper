# Keying marker size

```{image} /_static/features/keying-marker-size-light.png
:class: sph-plot plot-light dark-light
:alt: Keying marker size (light mode)
```


```{image} /_static/features/keying-marker-size-dark.png
:class: sph-plot plot-dark dark-light
:alt: Keying marker size (dark mode)
```


Two routes to a size key. `plot_catalog(size_legend=True)` draws its own
matplotlib key inside the axes — quick, when size is the only extra
dimension. `MultiLegend.add_size_from` reads the scaling off the same
plot, so swatches reproduce on-plot sizes, auto-picks round 1/2/5
representatives, and sits alongside other channels off-frame.

Guide: {doc}`/guide/legends` — API: {py:obj}`~skyplothelper.MultiLegend` · {py:obj}`~skyplothelper.plot_catalog`

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(7)
n = 150
cat = {"ra": rng.uniform(0, 360, n),
       "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
       "n_obs": rng.integers(1, 60, n)}

# The gallery renders this snippet in both modes, so pick the legend
# palette off the active figure background rather than hard-coding it.
from matplotlib.colors import to_rgb
LP = ("dark" if sum(to_rgb(plt.rcParams["figure.facecolor"])) / 3 < 0.5
      else "publication")
fig = plt.figure(figsize=(10, 3.2))
ax1 = sph.make_wcs_frame(121, "AIT", center=180)
sph.plot_catalog(ax1, cat, sizeby="n_obs", size_scale="sqrt", smin=6,
                 smax=170, color="C0", alpha=0.7, size_legend=True,
                 size_legend_num=3,
                 size_legend_kwargs=dict(loc="lower left", title="N obs",
                                         fontsize=7))
ax1.set_title("plot_catalog(size_legend=True)", fontsize=9)

ax2 = sph.make_wcs_frame(122, "AIT", center=180)
cp = sph.plot_catalog(ax2, cat, sizeby="n_obs", size_scale="sqrt", smin=6,
                      smax=170, color="C2", alpha=0.7)
(sph.MultiLegend(ax2, palette=LP, loc="outside bottom", orientation="horizontal",
                 fontsize=7)
    .add_size_from(cp, title="N obs")
    .add_shape("class", {"star": "o", "galaxy": "^"}, size=6)
    .draw())
ax2.set_title("MultiLegend.add_size_from(...)", fontsize=9)
fig.subplots_adjust(wspace=0.3)
```
