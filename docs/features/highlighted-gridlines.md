# Highlighted gridlines

```{image} /_static/features/highlighted-gridlines-light.png
:class: sph-plot plot-light dark-light
:alt: Highlighted gridlines (light mode)
```


```{image} /_static/features/highlighted-gridlines-dark.png
:class: sph-plot plot-dark dark-light
:alt: Highlighted gridlines (dark mode)
```


Emphasizing individual meridians and parallels — a colormapped family of
parallels and a few accented meridians — without redrawing the frame.

Guide: {doc}`/guide/ticks` — API: {py:obj}`~skyplothelper.highlight_gridlines`

## Code

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="MOL", center=180)
sph.highlight_gridlines(ax, lat_values=list(range(-60, 61, 30)),
                        lat_cmap="sph.dusk", lw=2.2)
sph.highlight_gridlines(ax, lon_values=[0, 90, 180, 270], color="C1", lw=1.6)
```
