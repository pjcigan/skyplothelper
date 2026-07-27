# Data cube channel maps

```{image} /_static/features/data-cube-channel-maps-light.png
:class: sph-plot plot-light dark-light
:alt: Data cube channel maps (light mode)
```


```{image} /_static/features/data-cube-channel-maps-dark.png
:class: sph-plot plot-dark dark-light
:alt: Data cube channel maps (dark mode)
```


`channel_map` turns a spectral-line cube (the DDO 70 HI sub-cube) into a
uniform panel grid on one shared normalization — velocity labels, the shared
colorbar, and sparse ticks all handled in one call.

Guide: {doc}`/guide/images` — API: {py:obj}`~skyplothelper.channel_map` · {py:obj}`~skyplothelper.ChannelMapResult`

## Code

```python
import skyplothelper as sph

res = sph.channel_map("examples/data/ddo70_hi_subcube.fits",
                      channels=9, ncols=3, cmap="sph.dusk")
fig = res.fig
```

```{note}
This example uses a file from the repository's [`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data) directory (not bundled with the pip install) — see the README there for provenance and credits.
```
