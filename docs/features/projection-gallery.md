# Projection gallery

```{image} /_static/features/projection-gallery-light.png
:class: sph-plot plot-light dark-light
:alt: Projection gallery (light mode)
```


```{image} /_static/features/projection-gallery-dark.png
:class: sph-plot plot-dark dark-light
:alt: Projection gallery (dark mode)
```


The same all-sky map through several projections at once — the full FITS
set plus non-FITS cartographers' projections like Robinson.

Guide: {doc}`/guide/frames` — API: {py:obj}`~skyplothelper.projection_gallery` · {py:obj}`~skyplothelper.bin_data_as_healpix`

## Code

```python
import numpy as np
import skyplothelper as sph

rng = np.random.default_rng(4)
# A uniform sky plus a galactic-plane overdensity, so the demo map has
# real structure to carry through each projection.
# Center the overdensity on the gallery center (180) so the zenithal
# panels — which only ever show a hemisphere — have structure in frame.
lon = np.concatenate([rng.uniform(0, 360, 40000),
                      rng.normal(180, 22, 30000) % 360])
lat = np.concatenate([np.degrees(np.arcsin(rng.uniform(-1, 1, 40000))),
                      np.clip(rng.normal(-12, 12, 30000), -89, 89)])
demo, *_ = sph.bin_data_as_healpix(lon, lat, np.ones_like(lon),
                                   nside=16, statistic="count", blank_value=0)
# A spread across the families: elliptical and pseudocylindrical all-sky,
# a cylindrical, the HEALPix quad-cube, and the zenithal projections (which
# by nature show only part of the sky).
fig, axes = sph.projection_gallery(
    demo, projections=["AIT", "MOL", "robinson", "CAR", "HPX", "SFL",
                       "TAN", "SIN", "ZEA"],
    center=180, ncols=3, cmap="sph.deepsky")
# A thin graticule on each panel — the whole point of the comparison is how
# each projection deforms the coordinate grid, which a bare map can't show.
# The map has to be pushed below the gridlines (WCSAxes draws its grid at a
# fixed stage, so the image would otherwise cover it), and the tick labels
# are dropped: unreadable at this size and they clutter the curved frames.
for ax in np.ravel(axes):
    for art in list(ax.collections) + list(ax.images):
        art.set_zorder(-10)
    try:
        for c in (0, 1):
            ax.coords[c].grid(True, color="0.85", linewidth=0.6, alpha=0.95,
                              linestyle="-")
            ax.coords[c].set_ticklabel_visible(False)
            ax.coords[c].set_axislabel("")
    except Exception:          # non-WCS frames (robinson & co.)
        sph.style_grid(ax, color="0.85", lw=0.6, alpha=0.95, ls="-")

# Some frames draw their in-frame labels as separate tagged artists, which
# set_ticklabel_visible doesn't reach — delete them after a draw.
fig.canvas.draw()
for ax in np.ravel(axes):
    for art in list(ax.lines) + list(ax.texts) + list(ax.collections):
        if getattr(art, "_sph_overlay_ticklabel", False):
            art.remove()
    for txt in list(ax.texts):
        txt.remove()
```
