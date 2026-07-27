# Compasses & compass roses

```{image} /_static/features/compasses-compass-roses-light.png
:class: sph-plot plot-light dark-light
:alt: Compasses & compass roses (light mode)
```


```{image} /_static/features/compasses-compass-roses-dark.png
:class: sph-plot plot-dark dark-light
:alt: Compasses & compass roses (dark mode)
```


Orientation furniture on a globe: a corner compass rose, an on-surface
compass that follows the sphere's orientation, a pole rod, and a
real-distance scale bar curved to the globe.

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.add_compass_rose` · {py:obj}`~skyplothelper.add_surface_compass` · {py:obj}`~skyplothelper.add_scale_bar`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

clon, clat, pole = sph.euler_to_fits_ortho(rotation=60, obliquity=23.44, perspective=10)
fig = plt.figure(figsize=(5.0, 5.0))
ax = sph.make_planet_frame(111, body="earth", center_LONdeg=clon, center_LATdeg=clat,
                           lonpole=pole, Naxispix=420)
sph.plot_coastlines(ax, color="0.6", lw=0.7)
sph.add_pole_rod(ax, color="C0")
# A corner rose (axes fraction), an on-surface compass following the sphere,
# and a real-distance scale bar curved to the globe.
sph.add_compass_rose(ax, x=0.14, y=0.84, size=34, style="simple", color="C3")
sph.add_surface_compass(ax, -30, 45, size_deg=13, style="star", color="C3")
sph.add_scale_bar(ax, lon_0=clon, lat_0=clat, body="earth", length_km=4000, color="C1")
```
