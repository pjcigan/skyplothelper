# Quickstart

A single `import skyplothelper as sph` gives you the whole sky-plot workflow.
The recipes below cover the five most common starting points.

```python
import skyplothelper as sph
import matplotlib.pyplot as plt
```

## 1. All-sky plot with overlays

`allsky_figure` builds an elliptical all-sky frame in any supported projection.
Add coordinate-plane overlays, IAU constellation boundaries, and survey
footprints on top.

```python
fig, ax = sph.allsky_figure(projection="AIT", center=180)
sph.add_plane_overlay(ax, plane="ecliptic", color="orange")
sph.add_constellation_boundaries(ax)
sph.add_survey_footprint(ax, survey="sdss")
plt.show()
```

## 2. A tangent-plane (TAN) field

`offset_figure` builds a rectangular field centered on a target with a given
field of view, in offset (tangent-plane) coordinates. This is what you
typically work in when you are displaying a single FITS image field of view.

```python
fig, ax = sph.offset_figure(center=(83.63, 22.01), fov_deg=0.2)
sph.add_compass(ax)
plt.show()
```

## 3. A (tilted) globe

`make_globe_frame` builds an orthographic globe; tilt it with Euler-angle
center longitude/latitude, then add a graticule and a compass rose.

```python
fig = plt.figure()
ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=23.44, grid=False)
sph.plot_ortho_grid(ax)
sph.add_compass_rose(ax)
plt.show()
```

## 4. A cosmology cone (pie wedge)

`make_cone_frame` builds a z-RA wedge for cone diagrams; the radial axis is
redshift and the angular axis is a sky coordinate.

```python
fig = plt.figure()
ax = sph.make_cone_frame(
    111, angle_center=180, angle_half_width=30,
    r_min=0, r_max=0.15,                          # redshift range
    angle_label="R.A.", fig=fig,
)
sph.cone_scatter(ax, galaxy_ras, galaxy_redshifts, s=3)
plt.show()
```

## 5. A HEALPix all-sky map

`healpix_allsky_figure` renders a HEALPix array into a fresh all-sky figure
(with a colorbar), returning a `HealpixResult` (`fig`, `ax`, `mappable`,
`colorbar`).

```python
result = sph.healpix_allsky_figure(my_hpx_map, projection="AIT")
result.colorbar.set_label("value")
plt.show()
```

## 6. An interactive (plotly) map

The plotly backend mirrors the same API for pan/zoom/hover figures and
single-file HTML export.

```python
import skyplothelper.plotly as sphpl

fig = sphpl.make_figure(projection="AIT", center=180)
sphpl.add_constellation_boundaries(fig)
sphpl.add_scatter(fig, ras, decs)
fig.show()                       # or fig.write_html("skymap.html")
```

## 7. Plot a queried catalog

Resolve a name, pull a catalog around it, and drop it on a frame
(requires the `query` extra).

```python
coord = sph.resolve_name("M87")
fig, ax = sph.offset_figure(center=coord, fov_deg=0.5)
table = sph.search_vizier("I/350/gaiaedr3", coord, radius=10)
sph.plot_catalog(ax, table, ra_col="RA_ICRS", dec_col="DE_ICRS", colorby="Gmag")
plt.show()
```

## Discovering what's available

Each registry has a `list_*` helper that enumerates its options:

```python
sph.list_projections()          # FITS + non-FITS frame projections
sph.list_surveys()              # survey-footprint catalog
sph.list_constellations()       # IAU constellations
sph.list_stretches()            # image-stretch names
sph.list_cartopy_projections()  # cartopy backend projections
```

And `sph.describe_wcs(header)` prints a friendly summary of any WCS / FITS
header.

Continue with the {doc}`user guide <guide/index>` for a tour of each subsystem,
start from {doc}`core concepts <guide/concepts>` for the conventions behind
every recipe above, follow the {doc}`tutorials <tutorials/index>` for worked
end-to-end walkthroughs (begin with {doc}`getting started <tutorials/getting_started>`),
or jump to the {doc}`API reference <api/index>`.
