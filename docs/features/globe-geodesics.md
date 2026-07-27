# Globe geodesics

```{image} /_static/features/globe-geodesics-light.png
:class: sph-plot plot-light dark-light
:alt: Globe geodesics (light mode)
```


```{image} /_static/features/globe-geodesics-dark.png
:class: sph-plot plot-dark dark-light
:alt: Globe geodesics (dark mode)
```


Great-circle arcs and a full highlighted great circle on a celestial globe,
with a pole rod marking the rotation axis.

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.great_circle_arc` · {py:obj}`~skyplothelper.highlight_great_circle` · {py:obj}`~skyplothelper.add_pole_rod`

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import skyplothelper as sph

rng = np.random.default_rng(2)
fig = plt.figure(figsize=(4.6, 4.6))
ax = sph.make_globe_frame(111, center_LONdeg=200, center_LATdeg=35)
pts = [(150, 10), (210, 55), (270, 20), (320, 60)]
for (l1, b1), (l2, b2) in zip(pts, pts[1:]):
    lo, la = sph.great_circle_arc(l1, b1, l2, b2, n_pts=80)
    sph.plot_line_globe(ax, lo, la, color="C1", lw=1.8)
sph.highlight_great_circle(ax, inclination=60, node=200, color="C0", lw=2.0)
sph.add_pole_rod(ax)
```
