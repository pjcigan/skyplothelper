# Cartopy support

```{image} /_static/features/cartopy-support-light.png
:class: sph-plot plot-light dark-light
:alt: Cartopy support (light mode)
```


```{image} /_static/features/cartopy-support-dark.png
:class: sph-plot plot-dark dark-light
:alt: Cartopy support (dark mode)
```


skyplothelper *also* supports [cartopy](https://scitools.org.uk/cartopy) as an
optional backend for geographic Earth maps — separate from its own planet globes
above. `make_cartopy_frame` builds a map in any cartopy projection with
coastline / land / ocean features in one call. (Needs the optional `cartopy`
dependency.)

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.make_cartopy_frame`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

# make_cartopy_frame gives an Earth map in any cartopy projection, with
# coastline / land / ocean features from one call. Four projections here.
fig = plt.figure(figsize=(11, 6.4))
specs = [
    ("robinson",     0,           "robinson"),
    ("mollweide",    0,           "mollweide"),
    ("orthographic", (-100, 35),  "orthographic (a globe)"),
    ("lambert_azimuthal", (10, 45), "lambert_azimuthal"),
]
for i, (proj, center, title) in enumerate(specs, start=1):
    ax = sph.make_cartopy_frame(subplotnumber=220 + i, projection=proj,
                                center=center, coastlines=True, land=True,
                                ocean=True, grid=True)
    ax.set_title(title, fontsize=10)
fig.suptitle("make_cartopy_frame - an Earth map in any projection", fontsize=13)
fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.03,
                    hspace=0.32, wspace=0.12)
```
