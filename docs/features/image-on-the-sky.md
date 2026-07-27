# Image on the sky

```{image} /_static/features/image-on-the-sky-light.png
:class: sph-plot plot-light dark-light
:alt: Image on the sky (light mode)
```


```{image} /_static/features/image-on-the-sky-dark.png
:class: sph-plot plot-dark dark-light
:alt: Image on the sky (dark mode)
```


Resampling a sky panorama onto a curved projection — the all-sky NOIRLab
image reprojected onto a galactic Aitoff frame.

Guide: {doc}`/guide/images` — API: {py:obj}`~skyplothelper.load_sky_image` · {py:obj}`~skyplothelper.reproject_background`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

pano = "examples/data/Allsky_noirlab2430b_1280x640.jpg"
img, hdr = sph.load_sky_image(pano, frame="galactic", center=0)
fig = plt.figure(figsize=(6.4, 3.4))
ax = sph.make_wcs_frame(111, "AIT", center=0, frame="galactic",
                        npix=(1200, 600), fig=fig)
ax.imshow(sph.reproject_background(img, hdr, ax))
sph.add_plane_overlay(ax, plane="ecliptic", color="C1", lw=1.0)
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
