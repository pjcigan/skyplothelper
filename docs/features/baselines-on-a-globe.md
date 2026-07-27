# Baselines on a globe

```{image} /_static/features/baselines-on-a-globe-light.png
:class: sph-plot plot-light dark-light
:alt: Baselines on a globe (light mode)
```


```{image} /_static/features/baselines-on-a-globe-dark.png
:class: sph-plot plot-dark dark-light
:alt: Baselines on a globe (dark mode)
```


A station network's great-circle baselines on an Earth globe rather than a flat
map: `plot_baselines` hides the arcs that dip behind the globe, drawing them as
faint dashes. A global VLBI array here — US, European, African, Asian, and
Australian stations.

Guide: {doc}`/guide/vectors` — API: {py:obj}`~skyplothelper.plot_baselines` · {py:obj}`~skyplothelper.make_planet_frame` · {py:obj}`~skyplothelper.plot_coastlines`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

# A global VLBI network - US (VLBA), plus stations in Europe, Africa, Asia, and
# Australia so some baselines cross to the far side of the globe.
SITES = {"MK": (-155.46, 19.80), "BR": (-119.68, 48.13), "PT": (-108.12, 34.30),
         "LA": (-106.25, 35.78), "NL": (-91.57, 41.77), "SC": (-64.58, 17.76),
         "EB": (6.88, 50.52), "HH": (27.69, -25.89),
         "TM": (121.14, 30.90), "CD": (133.81, -31.87)}

fig = plt.figure(figsize=(6.2, 6.2))
# An Earth globe (make_planet_frame) instead of a flat map: plot_baselines draws
# the network great circles and hides the far-side arcs automatically - the ones
# that dip behind the globe are drawn as faint dashes via back_hemisphere_*.
globe = sph.make_planet_frame(111, body="earth",
                              center_LONdeg=-40, center_LATdeg=25)
sph.plot_coastlines(globe, color="0.6")
sph.plot_baselines(globe, SITES, color="C1", linewidth=1.0, alpha=0.9,
                   marker_color="C0", site_label_fontsize=7,
                   back_hemisphere_linestyle=":", back_hemisphere_alpha=0.4)
globe.set_title("Global VLBI baselines on an Earth globe", fontsize=12)
```
