# Instruments aimed at a source

```{image} /_static/features/instruments-aimed-at-a-source-light.png
:class: sph-plot plot-light dark-light
:alt: Instruments aimed at a source (light mode)
```


```{image} /_static/features/instruments-aimed-at-a-source-dark.png
:class: sph-plot plot-dark dark-light
:alt: Instruments aimed at a source (dark mode)
```


A mixed instrument array on a globe, each marker planted with its pedestal along
the local vertical (`aim_mode="planted"`, `globe_center=`) and its dish or tube
swung onto one shared source (`aim_at=`) — radio dishes and optical tubes here.

Guide: {doc}`/guide/overlays` — API: {py:obj}`~skyplothelper.add_antenna_marker` · {py:obj}`~skyplothelper.add_telescope_marker`

## Code

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import skyplothelper as sph
from matplotlib.colors import to_rgb

_r, _g, _b = to_rgb(mpl.rcParams["figure.facecolor"])
DARK = 0.299 * _r + 0.587 * _g + 0.114 * _b < 0.5
PAL = sph.ANNOTATION_PALETTES["dark" if DARK else "publication"]

CENTER = (-95.0, 20.0)
src = (-52.0, 60.0)                 # the shared celestial source, up and to the right
# A mixed instrument array - radio dishes and optical tubes - each planted on the
# globe (pedestal along its local vertical) and aimed at one source.
sites = [(-140, 35, "antenna"), (-120, 6, "telescope"), (-98, 46, "telescope"),
         (-72, 24, "antenna"), (-116, -18, "telescope"), (-84, -6, "antenna"),
         (-60, 4, "telescope")]

fig = plt.figure(figsize=(6.6, 6.2))
ax = sph.make_globe_frame(111, center_LONdeg=CENTER[0], center_LATdeg=CENTER[1],
                          projection="SIN", grid=True, Naxispix=360)
fig.canvas.draw()   # settle the transforms before the aim solver reads them
ax.plot(*ax.wcs.wcs_world2pix([src], 0)[0], marker="*", ms=18,
        color=PAL["label"], zorder=9)

style = dict(size=40, edge_color=PAL["text"], stroke_color=PAL["fig_bg"], stroke_lw=1.6)
makers = {"antenna": sph.add_antenna_marker, "telescope": sph.add_telescope_marker}
faces = {"antenna": PAL["accent"], "telescope": PAL["accent2"]}
for lon, lat, kind in sites:
    makers[kind](ax, (lon, lat), coord_type="world", aim_at=src, aim_mode="planted",
                 globe_center=CENTER, target_coords="world",
                 face_color=faces[kind], **style)
ax.set_title("A mixed array, planted on a globe, aimed at one source", fontsize=11)
```
