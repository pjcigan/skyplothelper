# Instrument markers

```{image} /_static/features/instrument-markers-light.png
:class: sph-plot plot-light dark-light
:alt: Instrument markers (light mode)
```


```{image} /_static/features/instrument-markers-dark.png
:class: sph-plot plot-dark dark-light
:alt: Instrument markers (dark mode)
```


Procedural instrument glyphs that *aim* — an antenna and an optical
telescope solving their elevation from a shared target, and a dome whose
slit takes a compass bearing.

Guide: {doc}`/guide/overlays` — API: {py:obj}`~skyplothelper.add_antenna_marker` · {py:obj}`~skyplothelper.add_telescope_marker` · {py:obj}`~skyplothelper.add_dome_marker`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
# Off to one side, so the two aimed instruments point at visibly different
# angles rather than both straight up.
target = (5.4, 8.7)
ax.scatter(*target, marker="*", s=280, color="C1", zorder=7)
style = dict(size=64, face_color="C0", edge_color="0.15",
             stroke_color="white", stroke_lw=2.0)
fig.canvas.draw()  # finalize the equal-aspect transform before solving angles
# The markers are base-anchored, so one shared y plants all three feet on the
# same ground line. aim_angles gives the on-screen tilt for the dish/tube, and
# box.anchors.sight_line_origin(phi) starts each sight-line inside the optics
# rather than at the pier foot, so the rays read as leaving the dish and tube.
for fn, xy, elev_kw in [(sph.add_antenna_marker, (2.4, 3.0), "dish_elev"),
                        (sph.add_telescope_marker, (8.4, 3.0), "tube_elev")]:
    phi = sph.aim_angles(ax, xy, target, target_coords="data")["aim_angle"]
    box = fn(ax, xy, rotation=0, **{elev_kw: phi}, **style)
    ox, oy = box.anchors.sight_line_origin(phi)
    ax.plot([ox, target[0]], [oy, target[1]], ls=(0, (1, 2.5)),
            color="0.6", lw=0.9, zorder=1)
# The dome doesn't tilt — its slit swings to a compass bearing instead.
sph.add_dome_marker(ax, (13.0, 3.0), slit_azim=-28, **style)
for x, name in [(2.4, "antenna"), (8.4, "telescope"), (13.0, "dome")]:
    ax.annotate(name, (x, 0.75), ha="center", fontsize=9)
ax.set_xticks([])
ax.set_yticks([])
```
