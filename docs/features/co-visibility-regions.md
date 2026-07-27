# Co-visibility regions

```{image} /_static/features/co-visibility-regions-light.png
:class: sph-plot plot-light dark-light
:alt: Co-visibility regions (light mode)
```


```{image} /_static/features/co-visibility-regions-dark.png
:class: sph-plot plot-dark dark-light
:alt: Co-visibility regions (dark mode)
```


Where two stations can see the sky at once — each station's cap, labeled
at its zenith in its own color, and their instantaneous intersection, as
renderable regions for one instant.

Guide: {doc}`/guide/vectors` — API: {py:obj}`~skyplothelper.covisibility_region` · {py:obj}`~skyplothelper.covisibility_circles`

## Code

```python
import skyplothelper as sph

STATIONS = {"Wettzell": {"lat": 49.145, "lon": 12.878},
            "VLA": {"lat": 34.08, "lon": -107.62}}
TIME = "2026-07-02T07:00:00"
fig, ax = sph.allsky_figure(projection="AIT", center=180)
for name, col in [("Wettzell", "C0"), ("VLA", "C1")]:
    reg = sph.covisibility_region(ax, {name: STATIONS[name]}, TIME, el_min=15)
    reg.render(facecolor=col, alpha=0.22, edgecolor=col, lw=1.3)
    # Label each cap at its own center, in its own color, so the overlap
    # below is unambiguous about which station contributes which lobe.
    cap = sph.covisibility_circles({name: STATIONS[name]}, TIME, el_min=15)[0]
    ax.scatter(cap["center"].ra.deg, cap["center"].dec.deg, s=32, color=col,
               zorder=6, transform=ax.get_transform("world"))
    ax.annotate(name, ax.wcs.world_to_pixel_values(cap["center"].ra.deg,
                                                   cap["center"].dec.deg),
                xytext=(0, 9), textcoords="offset points", ha="center",
                fontsize=9, fontweight="bold", color=col, zorder=7)
both = sph.covisibility_region(ax, STATIONS, TIME, el_min=15)
both.render(facecolor="C3", alpha=0.5, edgecolor="C3", lw=1.6)
```
