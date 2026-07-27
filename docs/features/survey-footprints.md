# Survey footprints

```{image} /_static/features/survey-footprints-light.png
:class: sph-plot plot-light dark-light
:alt: Survey footprints (light mode)
```


```{image} /_static/features/survey-footprints-dark.png
:class: sph-plot plot-dark dark-light
:alt: Survey footprints (dark mode)
```


Sky coverage of named surveys from the bundled, queryable footprint
catalog.

Guide: {doc}`/guide/overlays` — API: {py:obj}`~skyplothelper.add_survey_footprint`

## Code

```python
import skyplothelper as sph

fig, ax = sph.allsky_figure(projection="MOL", center=180)
sph.add_survey_footprint(ax, survey="des", label="DES", color="C0")
sph.add_survey_footprint(ax, survey="euclid", label="Euclid", color="C1")
ax.legend(loc="lower right")
```
