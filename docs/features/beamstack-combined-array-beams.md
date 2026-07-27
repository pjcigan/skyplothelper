# BeamStack — combined-array beams

```{image} /_static/features/beamstack-combined-array-beams-light.png
:class: sph-plot plot-light dark-light
:alt: BeamStack — combined-array beams (light mode)
```


```{image} /_static/features/beamstack-combined-array-beams-dark.png
:class: sph-plot plot-dark dark-light
:alt: BeamStack — combined-array beams (dark mode)
```


Co-located beams for a combined-array observation, stacked at one position and
labeled for a single legend. Distinct sizes, position angles, and fills
(outline, hatch, two solid) keep the members legible even when nested.

Guide: {doc}`/guide/overlays` — API: {py:obj}`~skyplothelper.BeamStack` · {py:obj}`~skyplothelper.Beam`

## Code

```python
import matplotlib.pyplot as plt
import skyplothelper as sph
from skyplothelper import Beam, BeamStack

fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.set_aspect("equal")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("BeamStack - co-located beams from a combined array")

# Co-located beams for a combined-array observation. Each is a full Beam with its
# own size, position angle, and fill, so the members stay visually distinct even
# nested - outline, hatch, and two solid fills, at four different PAs.
BeamStack([
    Beam((50, 50), bmaj_pix=74, bmin_pix=52, bpa_deg=28,
         style="ellipse", ec="C0", lw=1.8, label="D config"),
    Beam((50, 50), bmaj_pix=50, bmin_pix=32, bpa_deg=-18,
         style="hatch", ec="C1", lw=1.5, label="C config"),
    Beam((50, 50), bmaj_pix=30, bmin_pix=19, bpa_deg=52,
         style="filled", fc="C2", ec="C2", alpha=0.55, label="B config"),
    Beam((50, 50), bmaj_pix=14, bmin_pix=10, bpa_deg=5,
         style="filled", fc="C3", ec="0.1", lw=1.0, label="A config (combined)"),
]).add_to(ax)
ax.legend(loc="upper right", fontsize=9, framealpha=0.6)
```
