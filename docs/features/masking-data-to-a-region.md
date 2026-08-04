# Masking data to a region

```{image} /_static/features/masking-data-to-a-region-light.png
:class: sph-plot plot-light dark-light
:alt: Masking data to a region (light mode)
```


```{image} /_static/features/masking-data-to-a-region-dark.png
:class: sph-plot plot-dark dark-light
:alt: Masking data to a region (dark mode)
```


A region as a cookie-cutter: `clip()` masks any artist — an image, a data
field, a scatter — to a region's shape, so data shows only where it falls
*inside*. Here a smooth all-sky field is cut to a survey footprint (a box,
minus the galactic plane, with a hole punched). This is the sky counterpart
of the `clip_to_land` / `clip_to_ocean` helpers on Earth maps.

Guide: {doc}`/guide/regions` — API: {py:obj}`~skyplothelper.CompoundRegion` · {py:obj}`~skyplothelper.clip_to_land` · {py:obj}`~skyplothelper.clip_to_ocean`

## Code

```python
import numpy as np
import skyplothelper as sph

# a smooth, large-scale all-sky field
lon = np.linspace(0, 360, 480)
lat = np.linspace(-90, 90, 240)
LON, LAT = np.meshgrid(lon, lat)
dlon = (LON - 150 + 180) % 360 - 180
field = np.exp(-(dlon**2 + (LAT - 25)**2) / (2 * 45**2)) + 0.3 * np.cos(np.radians(LAT))

fig, ax = sph.allsky_figure(projection="MOL", center=180)
footprint = (sph.CompoundRegion(ax)
             .add_lonlat_box(lat_min=-12, lat_max=70, lon_min=110, lon_max=260, frame="icrs")
             .subtract_frame_band(-25, 25, frame="galactic")   # avoid the plane
             .subtract_circle(180, 35, radius_deg=8))          # punch a hole
mesh = ax.pcolormesh(LON, LAT, field, transform=ax.get_transform("world"),
                     cmap="sph.deepsky", shading="gouraud")
footprint.clip(mesh)                       # mask the field to the footprint shape
footprint.render_boundary(color="white", linewidth=1.1)
```
