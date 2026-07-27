# Region shapes

```{image} /_static/features/region-shapes-light.png
:class: sph-plot plot-light dark-light
:alt: Region shapes (light mode)
```


```{image} /_static/features/region-shapes-dark.png
:class: sph-plot plot-dark dark-light
:alt: Region shapes (dark mode)
```


The shape family — rectangles, ellipses, annuli, and coordinate bands —
each seam-aware and correct at the poles.

Guide: {doc}`/guide/regions` — API: {py:obj}`~skyplothelper.add_rectangle` · {py:obj}`~skyplothelper.add_ellipse` · {py:obj}`~skyplothelper.add_annulus` · {py:obj}`~skyplothelper.add_latitude_band`

## Code

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_rectangle(ax, lon=60, lat=35, width=44, height=26,
                  facecolor="C0", edgecolor="C0", alpha=0.35)
sph.add_ellipse(ax, lon=185, lat=-25, semi_major=32, semi_minor=15, angle=25,
                facecolor="C1", edgecolor="C1", alpha=0.35)
sph.add_annulus(ax, lon=300, lat=30, inner_radius=8, outer_radius=18,
                facecolor="C2", edgecolor="C2", alpha=0.4)
sph.add_latitude_band(ax, -10, 10, facecolor="C3", edgecolor="none", alpha=0.3)
```
