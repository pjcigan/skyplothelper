# Geodesic baseline map

```{image} /_static/features/geodesic-baseline-map-light.png
:class: sph-plot plot-light dark-light
:alt: Geodesic baseline map (light mode)
```


```{image} /_static/features/geodesic-baseline-map-dark.png
:class: sph-plot plot-dark dark-light
:alt: Geodesic baseline map (dark mode)
```


Great-circle baselines and self-labeling dish markers for any station
network on a plain lon/lat map — the VLBA here simply as a worked example.

Guide: {doc}`/guide/vectors` — API: {py:obj}`~skyplothelper.plot_baselines`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

VLBA = {"BR": (-119.68, 48.13), "OV": (-118.28, 37.23), "KP": (-111.61, 31.96),
        "PT": (-108.12, 34.30), "LA": (-106.25, 35.78), "FD": (-103.94, 30.63),
        "NL": (-91.57, 41.77), "HN": (-71.99, 42.93),
        "MK": (-155.46, 19.80), "SC": (-64.58, 17.76)}
fig, ax = plt.subplots(figsize=(5.4, 3.4))
sph.plot_baselines(ax, VLBA, color="C1", linewidth=0.8, alpha=0.85,
                   marker_color="C0", site_label_fontsize=7)
ax.set_xlim(-165, -60)
ax.set_ylim(10, 55)
ax.set_aspect("equal")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
```
