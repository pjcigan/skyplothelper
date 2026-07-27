# Region membership

```{image} /_static/features/region-membership-light.png
:class: sph-plot plot-light dark-light
:alt: Region membership (light mode)
```


```{image} /_static/features/region-membership-dark.png
:class: sph-plot plot-dark dark-light
:alt: Region membership (dark mode)
```


Point-in-region queries: `contains_points` classifies a catalog against
any compound region — here a cap with the galactic band removed, so
sources in the stripe fall *outside* — coloring members and non-members
distinctly.

Guide: {doc}`/guide/regions` — API: {py:obj}`~skyplothelper.CompoundRegion`

## Code

```python
import numpy as np
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
reg = (sph.CompoundRegion(ax)
       .add_circle(lon=95, lat=12, radius_deg=42)
       .subtract_frame_band(-13, 13, frame="galactic"))
reg.render(facecolor="C0", alpha=0.14)
reg.render_boundary(color="C0", linewidth=1.3)

rng = np.random.default_rng(6)
lon = rng.uniform(20, 175, 700)
lat = rng.uniform(-40, 60, 700)
inside = reg.contains_points(lon, lat)   # boolean mask, one per source
tr = ax.get_transform("world")
ax.scatter(lon[~inside], lat[~inside], transform=tr, s=7, color="0.72", label="outside")
ax.scatter(lon[inside], lat[inside], transform=tr, s=15, color="#d1495b",
           edgecolor="w", lw=0.35, label="inside")
ax.legend(loc="lower right", fontsize=8)
```
