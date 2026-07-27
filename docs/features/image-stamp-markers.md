# Image-stamp markers

```{image} /_static/features/image-stamp-markers-light.png
:class: sph-plot plot-light dark-light
:alt: Image-stamp markers (light mode)
```


```{image} /_static/features/image-stamp-markers-dark.png
:class: sph-plot plot-dark dark-light
:alt: Image-stamp markers (dark mode)
```


Photographic icons placed as markers — the bundled Solar-System stamps
dropped onto a plain axes at graduated zoom.

Guide: {doc}`/guide/globe` — API: {py:obj}`~skyplothelper.imscatter`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

# Every bundled stamp: solar-system bodies on the top row, facility icons
# on the bottom. zoom= is per-stamp, so mixed native resolutions still land
# at a common on-page size.
stamps = [("sun2_120pix", 0.42), ("Earth_Western_Hemisphere_120pix", 0.42),
          ("Mars_120pix", 0.42), ("FullMoon_240x240", 0.21),
          ("Jupiter_120pix", 0.42),
          ("RadioDish_250pix", 0.20), ("OpticalTelescope_250pix", 0.20),
          ("SpaceTelescope_250pix", 0.20), ("SMBH_250pix", 0.20),
          ("sun1_120pix", 0.42)]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
for i, (name, z) in enumerate(stamps):
    # plt.imread returns row 0 = TOP, but this style sets image.origin='lower'
    # (right for FITS, wrong for photographs), which flips the stamps. Reverse
    # the rows so they sit the way they were drawn.
    img = plt.imread(f"examples/data/icons/{name}.png")[::-1]
    sph.imscatter([i % 5], [-(i // 5)], img, ax=ax, zoom=z)
ax.set_xlim(-0.6, 4.6)
ax.set_ylim(-1.6, 0.6)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
