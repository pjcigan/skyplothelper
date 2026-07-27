# Compound region

```{image} /_static/features/compound-region-light.png
:class: sph-plot plot-light dark-light
:alt: Compound region (light mode)
```


```{image} /_static/features/compound-region-dark.png
:class: sph-plot plot-dark dark-light
:alt: Compound region (dark mode)
```


Set algebra on the sphere: the galactic band — wrapping the whole sky and
crossing the antimeridian seam — with a cap *merged into* it and a square
punched out of the overlap, rendered as one visibly complex, seam-aware
region.

Guide: {doc}`/guide/regions` — API: {py:obj}`~skyplothelper.CompoundRegion`

## Code

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="AIT", center=180)
reg = (sph.CompoundRegion(ax)
       .add_frame_band(-13, 13, frame="galactic")            # the galactic band, across the seam
       .add_circle(lon=285, lat=-12, radius_deg=34)          # a cap that merges into the band
       .subtract_square(lon=276, lat=-20, size=15, angle=20))  # a square hole inside the overlap
reg.render(facecolor="C0", alpha=0.32)
reg.render_boundary(linewidth=1.2)
```
