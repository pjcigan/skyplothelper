# Insets — a zoom and a circular cutout

```{image} /_static/features/insets-a-zoom-and-a-circular-cutout-light.png
:class: sph-plot plot-light dark-light
:alt: Insets — a zoom and a circular cutout (light mode)
```


```{image} /_static/features/insets-a-zoom-and-a-circular-cutout-dark.png
:class: sph-plot plot-dark dark-light
:alt: Insets — a zoom and a circular cutout (dark mode)
```


Two detail views on one all-sky catalog map: a rectangular tangent-plane zoom
(boxed and connected) resolves a dense knot, and a circular orthographic cutout
drops the NOIRLab Milky Way panorama with the catalog scattered over it, marked
by a circle and its two tangent connectors.

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.reproject_inset_axes` · {py:obj}`~skyplothelper.mark_inset_axes` · {py:obj}`~skyplothelper.connect_inset_axes`

## Code

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skyplothelper as sph
from astropy.visualization.wcsaxes.frame import EllipticalFrame

# Mode-aware decoration colors (the gallery renders one snippet light + dark).
from matplotlib.colors import to_rgb
_r, _g, _b = to_rgb(mpl.rcParams["figure.facecolor"])
DARK = 0.299 * _r + 0.587 * _g + 0.114 * _b < 0.5
PAL = sph.ANNOTATION_PALETTES["dark" if DARK else "publication"]
MARK = PAL["accent2"]        # rectangular-zoom marker + connectors, on-theme
WARM = "#F5B301"             # amber reads over the (always-dark) Milky Way image

# A synthetic all-sky source catalog in galactic coords, plus a dense knot for
# the rectangular zoom to resolve.
rng = np.random.default_rng(11)
n = 620
l = rng.uniform(-180, 180, n)
b = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
KNOT = (-58.0, 26.0)
nk = 120
lk = KNOT[0] + rng.normal(0, 1.1, nk) / np.cos(np.radians(KNOT[1]))
bk = KNOT[1] + rng.normal(0, 1.1, nk)
cat = {"l": np.r_[l, lk], "b": np.r_[b, bk],
       "z": 10 ** rng.uniform(-2.0, -0.8, n + nk),
       "bright": 10 ** rng.uniform(0, 2, n + nk)}

# A smooth density field peaking on the knot, contoured in yellow - showing an
# inset is a normal Axes that takes the same overlays as the parent.
gl, gb = np.meshgrid(np.linspace(-180, 180, 361), np.linspace(-90, 90, 181))
_dl = ((gl - KNOT[0] + 180) % 360 - 180) * np.cos(np.radians(gb))
dens = np.exp(-0.5 * ((_dl / 5.0) ** 2 + ((gb - KNOT[1]) / 5.0) ** 2))

def science(target, smin, smax):
    sc = sph.plot_catalog(target, cat, lon_col="l", lat_col="b", frame="galactic",
                          colorby="z", sizeby="bright", cmap="sph.dusk",
                          size_scale="sqrt", smin=smin, smax=smax,
                          vmin=0.01, vmax=0.16, alpha=0.85)
    sph.add_contour_overlay(target, gl, gb, dens, levels=[0.3, 0.6],
                            colors="gold", linewidths=1.4, alpha=0.9)
    return sc

# Parent: all-sky galactic map of the catalog.
panorama, pano_hdr = sph.load_sky_image(
    "examples/data/Allsky_noirlab2430b_1280x640.jpg", frame="galactic", center=0)
fig = plt.figure(figsize=(13, 6.5))
ax = sph.make_wcs_frame(111, projection="AIT", center=0, frame="galactic", fig=fig)
science(ax, smin=4, smax=55)
ax.set_title("Two insets on one all-sky field: a rectangular zoom and a circular cutout",
             fontsize=12)

# Right: rectangular TAN zoom on the knot (marked + connected).
zoom = sph.reproject_inset_axes(ax, rect=[0.70, 0.10, 0.28, 0.40], transform="figure",
                                projection="TAN", center=KNOT, size=(14, 12))
science(zoom, smin=16, smax=110)
for c in (0, 1):
    zoom.coords[c].axislabels.set_visible(False)
sph.mark_inset_axes(ax, zoom, edgecolor=MARK, linewidth=1.8)
sph.connect_inset_axes(ax, zoom, color=MARK, linewidth=1.3, corners='diagonal')

# Left: circular SIN inset on a second (galactic-plane) patch, showing the
# NOIRLab panorama with the catalog scattered over it.
PATCH = (48.0, 0.0)
circ = sph.reproject_inset_axes(ax, rect=[0.03, 0.10, 0.26, 0.52], transform="figure",
                                projection="SIN", center=PATCH, size=45, npix=1000,
                                frame_class=EllipticalFrame)
circ.imshow(sph.reproject_background(panorama, pano_hdr, circ), origin="lower")
science(circ, smin=8, smax=70)
for c in (0, 1):
    circ.coords[c].set_ticklabel(color="0.9")
    circ.coords[c].axislabels.set_visible(False)
sph.mark_inset_axes(ax, circ, style="circle", center=PATCH, radius=20,
                    edgecolor=WARM, linewidth=1.8)
sph.connect_inset_axes(ax, circ, color=WARM, linewidth=1.2)
fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.04)
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
