# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive Plotting with Plotly
#
# Every tutorial in this series so far has produced *pictures*. This one
# produces *conversations*: sky maps you hover for identities and
# coordinates, zoom into, spin with the mouse, and steer with sliders —
# then save as a single HTML file anyone can open in a browser.
#
# The engine is `skyplothelper.plotly` (imported as `sphpl`), a second
# rendering backend that shares the projection pipeline with the matplotlib
# side. The geometry — wrap seams, pole closures, geodesics — comes out
# identical; what changes is the medium. Most helpers are same-named twins
# of their matplotlib counterparts, so the tour below will feel familiar:
# planes and circles, regions and set algebra, HEALPix, constellations,
# vector fields, FITS images. Along the way we cover what only this medium
# can do — hover data, a legend that takes a chart apart layer by layer, a
# globe you spin with the mouse, sliders that need no Python behind them —
# and what it costs (an interactive figure *carries its data with it*, so
# we watch figure weight as we go).
#
# Everything here needs only the `plotly` extra (`pip install
# skyplothelper[plotly]`); the closing Dash viewer additionally wants
# `dash`. The example data — the Messier catalog, the Hipparcos naked-eye
# stars, the ICRF3 defining sources, the Galactic-center S-star orbits, and a
# VLBA image of 3C 84 — ships with the repository.
#
# > **Note:** the figures on this page are the committed notebook outputs,
# > live without any kernel — that trick is `pio.renderers.default =
# > "notebook_connected"` in the first cell, and it's yours to reuse
# > (section 14).
#
# ## Contents
#
# 1. [A sky map that answers back](#1.-A-sky-map-that-answers-back)
# 2. [The interactive figure and the parity model](#2.-The-interactive-figure-and-the-parity-model)
# 3. [Hover data](#3.-Hover-data)
# 4. [Lines, planes, and decorations](#4.-Lines,-planes,-and-decorations)
# 5. [Regions and set algebra](#5.-Regions-and-set-algebra)
# 6. [HEALPix maps](#6.-HEALPix-maps)
# 7. [Constellations](#7.-Constellations)
# 8. [Vector fields and the VSH explorer](#8.-Vector-fields-and-the-VSH-explorer)
# 9. [Orbits around the black hole](#9.-Orbits-around-the-black-hole)
# 10. [The sky in deep time](#10.-The-sky-in-deep-time)
# 11. [A drag-rotate globe](#11.-A-drag-rotate-globe)
# 12. [The FITS viewer](#12.-The-FITS-viewer)
# 13. [A spectral-cube viewer](#13.-A-spectral-cube-viewer)
# 14. [Sharing and export](#14.-Sharing-and-export)
# 15. [Putting it together](#15.-Putting-it-together)
# 16. [Where to go next](#16.-Where-to-go-next)

# %% [markdown]
# ## 1. A sky map that answers back

# %%
import warnings

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

import skyplothelper as sph
import skyplothelper.plotly as sphpl

# A saved notebook must carry its own plotly wiring: the
# "notebook_connected" renderer embeds each figure's data plus a small
# loader into the notebook itself, so the figures stay fully interactive
# wherever the notebook is viewed (these docs, GitHub, nbviewer) — no
# running Python needed.
pio.renderers.default = "notebook_connected"

# base='structural' applies just the structural style layer (frame and tick
# geometry) to the *matplotlib* figures in this notebook. Plotly does not
# read matplotlib rcParams — its styling happens per-figure (more on this
# in section 2).
sph.set_style(base="structural")

# Data series pull from the 'uranometria' cycle palette (dual-mode: reads on
# both light and dark pages); decoration colors for the dark plotly figures
# come from the 'dark' annotation palette.
C = sph.CYCLE_PALETTES["uranometria"]["colors"]
PAL = sph.ANNOTATION_PALETTES["dark"]


# sph's bundled colormaps (sph.list_colormaps()) are matplotlib objects;
# plotly wants an explicit list of color stops instead. This three-line
# bridge converts any of them — the lo/hi trim clips a colormap's near-white
# or near-black ends when they'd vanish against the page.
def plotly_scale(name, lo=0.0, hi=1.0, n=32):
    colors = sph.get_colormap(name)(np.linspace(lo, hi, n))
    return [[i / (n - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
            for i, (r, g, b, _a) in enumerate(colors)]


# An interactive figure stores its data as text, and full float64 precision
# is wasted on pixels and map coordinates alike. Rounding costs nothing
# visible and roughly halves a figure's weight — see section 14.
def lighten_heatmap(fig, image_decimals=3, value_decimals=2):
    """Round a FITS figure's image and hover arrays in place."""
    for trace in fig.data:
        if trace.type == "heatmap":
            trace.z = np.round(np.asarray(trace.z, dtype=float), image_decimals)
            if trace.customdata is not None:
                trace.customdata = np.round(
                    np.asarray(trace.customdata, dtype=float), value_decimals)
            trace.x = np.round(np.asarray(trace.x, dtype=float), 3)
            trace.y = np.round(np.asarray(trace.y, dtype=float), 3)
    return fig


def round_traces(fig, decimals=2):
    """Round every trace's canvas coordinates in place (big overlays)."""
    for trace in fig.data:
        if getattr(trace, "x", None) is not None and trace.type == "scatter":
            trace.x = np.round(np.asarray(trace.x, dtype=float), decimals)
            trace.y = np.round(np.asarray(trace.y, dtype=float), decimals)
    return fig

# %% [markdown]
# Here are the 110 objects of the Messier catalog, tinted by family —
# galaxies, clusters, and nebulae & friends. **Hover** one for its name and
# coordinates. **Drag a box** to zoom into the Virgo cluster's swarm.
# **Double-click** to reset:

# %%
messier = Table.read("../../examples/data/messier.csv")

# Object families, worked out once and reused (and colored) consistently
# through the whole tutorial — galaxies, clusters, everything else.
vmag = np.array([f"{v:.1f}" if np.isfinite(v) else "—" for v in messier["vmag"]])
custom = np.stack([messier["name"], messier["otype"], vmag,
                   messier["ra_deg"], messier["dec_deg"]], axis=-1)
otype = np.asarray(messier["otype"])
galaxy_types = {"Galaxy", "GinPair", "GtowardsCl", "GtowardsGroup", "AGN",
                "LINER", "Seyfert", "Seyfert2", "StarburstG", "HIIG"}
family = np.where(np.isin(otype, list(galaxy_types)), "galaxies",
                  np.where(np.char.find(otype, "Cluster") >= 0, "clusters",
                           "nebulae & other"))
FAMILY_COLOR = {"galaxies": C[0], "clusters": C[2], "nebulae & other": C[5]}
point_color = [FAMILY_COLOR[f] for f in family]

fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark",
    width=880, height=500,
    title="The Messier catalog — hover a point, zoom into a cluster",
)
sphpl.add_constellation_lines(fig, rank_max=1, color=PAL["accent"],
                              width=0.9, opacity=0.7)
# A soft shaded Milky Way band under the plane line gives the map depth.
sphpl.add_frame_band(fig, -10, 10, frame="galactic", color=PAL["grid"],
                     fillcolor="rgba(140,180,255,0.10)", opacity=1, width=1.0)
sphpl.add_plane_overlay(fig, plane="galactic", color=PAL["accent2"],
                        width=1.5, hover=True)
sphpl.add_scatter(
    fig, messier["ra_deg"], messier["dec_deg"],
    text=messier["name"],
    hovertemplate="<b>%{text}</b><br>RA %{customdata[0]:.2f}°, "
                  "Dec %{customdata[1]:.2f}°<extra></extra>",
    marker=dict(size=8, color=point_color, opacity=0.95, line=dict(width=0)),
)
fig.data[-1].update(showlegend=False)     # the key below names the colors
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30, color=PAL["label"])
sphpl.add_frame_edge(fig, color=PAL["grid"])
# A static key for the family colors. It's one scatter with per-point colors,
# so there's nothing to toggle — section 3 splits the same families into real
# traces and gets a clickable legend out of it.
sphpl.add_legend(fig, [sph.ColorBlock("Object type", FAMILY_COLOR,
                                      swatch="marker")])
fig.update_layout(legend=dict(x=0.99, xanchor="right", y=0.02, yanchor="bottom",
                              font=dict(size=10), itemclick=False,
                              itemdoubleclick=False))
fig.show()

# %% [markdown]
# One import, one figure call, and a handful of overlays — the same vocabulary as
# the matplotlib side, on a map that answers back. The rest of this
# tutorial is that pattern, one family at a time.
#
# > **Note:** every live figure in this tutorial uses `theme="dark"`. A
# > plotly figure is a self-contained interactive panel — it keeps its own
# > background rather than following this page's light/dark toggle, and a
# > dark sky reads naturally on both. (`theme="light"` is the default and
# > works everywhere the same way.)

# %% [markdown]
# ## 2. The interactive figure and the parity model
#
# The opener's scaffold call was `make_figure()`, the backend's entry
# point. It takes the same frame arguments you already know from the
# matplotlib builders — `projection=`, `center=`, `lat_center=`, `frame=`,
# `direction=`, `lon_units=` — plus figure-level knobs: `theme=` (`'light'`
# / `'dark'`), `width=`/`height=`, grid visibility and spacing
# (`show_grid=`, `grid_lon_spacing=`, `grid_lat_spacing=`), and a
# `title=`. The figure
# *remembers its projection setup*, which is why none of the opener's
# `add_*` calls had to repeat it (every helper still accepts explicit
# overrides).
#
# ### The same map in matplotlib
#
# The point of the backend is *parity*: the same projection pipeline drives
# both sides, so geometry — wrap seams, pole closures, geodesics — comes out
# identical, and switching medium is a change of a few calls, not a rewrite:

# %%
fig, ax = sph.allsky_figure(projection="AIT", center=180, figsize=(8.2, 4.4))
ax.scatter(messier["ra_deg"], messier["dec_deg"],
           transform=ax.get_transform("world"),
           s=18, color=C[2], zorder=5)
ax.set_title("The same 110 objects, matplotlib backend")
plt.show()

# %% [markdown]
# Most helpers have a same-named twin on the other side:
#
# | family | matplotlib (`sph`) | plotly (`sphpl`) |
# |---|---|---|
# | figure / frame | `make_wcs_frame`, `allsky_figure` | `make_figure` |
# | points | `ax.scatter(..., transform=...)` | `add_scatter` |
# | lines & planes | `add_great_circle`, `add_plane_overlay` | same names |
# | regions | `add_geodesic_circle`, `add_spherical_polygon`, `add_frame_band`, `add_great_circle_band`, `add_lonlat_box` | same names |
# | set algebra | `CompoundRegion(ax)` | `make_compound_region(fig)` + `add_compound_region(fig, region)` |
# | HEALPix | `plot_healpix_map`, `plot_healpix_sparse` | `add_healpix`, `add_healpix_sparse` |
# | constellations | `add_constellation_boundaries` / `_lines` / `_labels` / `_polygon` | same names |
# | vector fields | `plot_sky_vectors` | `add_sky_vectors` |
# | decorations | `Reticle`, `Ruler` (classes) | `add_reticle`, `add_ruler` (functions) |
# | FITS images | `plot_fits_image` and friends | `make_fits_figure` + `add_fits_image` |
# | projection math | `sph.project` | `sphpl.project` (the same function) |
#
# Two differences worth knowing up front:
#
# - **Everything is a function here, configured when you call it.** Where
#   matplotlib offers adjustable objects (`Reticle.set_size()`,
#   `Ruler.remove()`), a shape added to a plotly figure can't be modified
#   afterward — so there is no object to hold onto. Pass all the styling
#   as arguments in the call itself.
# - **Styling doesn't cross between the backends.** `set_style()`, themes,
#   palettes, and rcParams reach only the matplotlib side. On the plotly
#   side, use `make_figure(theme=...)` and per-call `color=`/`width=`/
#   `opacity=` arguments (plus plotly's own `fig.update_layout(...)` for
#   anything else).
#
# ### Steering the frame
#
# The frame arguments do exactly what they do on the matplotlib side. Here's
# the same catalog on a Galactic-frame Mollweide, with a coarser graticule —
# note the data goes in as `l`/`b` now, because coordinates are always in the
# *frame's own* system:

# %%
gl, gb = sph.icrs_to_galactic(messier["ra_deg"], messier["dec_deg"])

fig = sphpl.make_figure(
    projection="MOL", center=0, frame="galactic", theme="dark",
    width=820, height=440,
    grid_lon_spacing=60, grid_lat_spacing=30,
    title="Galactic-frame Mollweide, 60° × 30° graticule",
)
sphpl.add_scatter(
    fig, gl, gb, name="Messier",
    hovertemplate="l: %{customdata[0]:.2f}°<br>b: %{customdata[1]:.2f}°<extra></extra>",
    marker=dict(size=7, color=C[2], opacity=0.9),
)
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30)
sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.show()

# %% [markdown]
# Two helpers earn a place in almost every figure. `add_coord_labels()`
# writes longitude/latitude labels along the graticule (the frame itself
# ships label-free — hover already reports coordinates, so labels are opt-in
# decoration), and `add_frame_edge()` traces the projection's silhouette so
# the map ends with a clean rim instead of fading out at the last gridline.
#
# ### Not just all-sky ovals
#
# The whole projection registry from the
# [Tour of Projections](projections.ipynb) is available here — `projection=`
# takes the same codes. An orthographic globe (`"SIN"`) is just a frame like
# any other: give it a `center=`/`lat_center=` to say which hemisphere faces
# you, and the overlays curve onto it correctly. Here's the sky as seen
# looking straight at the Galactic center:

# %%
fig = sphpl.make_figure(
    projection="SIN", center=266.4, lat_center=-29.0, theme="dark",
    width=560, height=560,
    grid_lon_spacing=30, grid_lat_spacing=30,
    title="An orthographic globe, centered on the Galactic center",
)
sphpl.add_plane_overlay(fig, plane="galactic", color=PAL["accent"],
                        width=2.0, name="Galactic plane", hover=True)
sphpl.add_geodesic_circle(fig, 266.4, -29.0, 25.0, fill=True,
                          fillcolor="rgba(140,180,255,0.18)", color="#8FB3F0",
                          width=1.4, opacity=1,
                          name="25° around the Galactic center", hover=True)
sphpl.add_scatter(
    fig, messier["ra_deg"], messier["dec_deg"],
    text=messier["name"],
    hovertemplate="<b>%{text}</b><br>RA %{customdata[0]:.2f}°, "
                  "Dec %{customdata[1]:.2f}°<extra></extra>",
    marker=dict(size=6, color=C[2], opacity=0.9),
)
# PAL["frame"] (not the darker PAL["grid"]) so the globe's limb reads as a
# clean rim rather than dissolving into the background.
sphpl.add_frame_edge(fig, color=PAL["frame"], width=1.5)
fig.show()

# %% [markdown]
# Only the facing hemisphere is drawn — points and curves behind the globe
# are simply not projectable, so they vanish on their own, no masking
# needed. Watch the galactic plane run off the limb and stop. Zenithal
# frames (`"SIN"`, `"TAN"`, `"ZEA"`, `"ARC"`, ...), pseudocylindricals
# (`"MOL"`, `"AIT"`, `"SFL"`), and the rest of the registry all behave this
# way.
#
# > **Note:** filled shapes work on a globe as long as they fit the visible
# > hemisphere — the geodesic circle above, and the set-algebra regions of
# > [section 5](#5.-Regions-and-set-algebra), which clip cleanly to the near
# > side. The one thing that stays outline-only on a globe is the *direct*
# > `add_frame_band`; reach for a compound region when you want a filled
# > band here.
#
# > **Note:** this globe is a *projection* — a static viewpoint you can pan
# > and zoom, like every other figure here. For a globe you can grab and
# > **spin**, see [section 11](#11.-A-drag-rotate-globe), which borrows
# > plotly's own geographic engine to do it.

# %% [markdown]
# ## 3. Hover data
#
# Interactivity is the point, and hover is where it pays off first. Every
# `add_scatter` point ships with an RA/Dec readout by default; the
# `hovertemplate=` argument takes any [plotly template
# string](https://plotly.com/python/hover-text-and-formatting/), and
# `customdata=` supplies the per-point columns the template refers to.
# Here the Messier catalog answers with its name, type, and magnitude —
# and, split into one trace per object family, it gains an interactive
# legend as a bonus (**click** a legend entry to hide that family,
# **double-click** to isolate it):

# %%
# family / custom were built in section 1; here we reuse them, splitting the
# catalog into one trace per family so each gets a legend entry and hover.
fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark",
    width=840, height=460,
    title="Hover for details; click the legend to toggle a family",
)
for fam, color in FAMILY_COLOR.items():
    sel = family == fam
    sphpl.add_scatter(
        fig, messier["ra_deg"][sel], messier["dec_deg"][sel],
        name=f"{fam} ({sel.sum()})",
        customdata=custom[sel],
        hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                       "V = %{customdata[2]}<br>"
                       "RA %{customdata[3]:.3f}°, Dec %{customdata[4]:.3f}°"
                       "<extra></extra>"),
        marker=dict(size=7, color=color, opacity=0.9),
    )
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30)
sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.update_layout(showlegend=True,
                  legend=dict(x=0.99, xanchor="right", y=0.02))
fig.show()

# %% [markdown]
# Toggle the clusters off and the galaxies' clustering toward Virgo jumps
# out. Two details worth copying: the `<extra></extra>` tail suppresses
# plotly's secondary hover box (the floating trace-name tag), and the
# figure's legend is off by default — flip it on with
# `fig.update_layout(showlegend=True)` when your traces carry `name=`s.
#
# The overlay helpers speak hover too, through a lighter-weight `hover=`
# argument with a shared contract: `False` (silent, the default for most
# decorations), `True` (report the overlay's `name=` and coordinates), or a
# **string** used as the hover text verbatim. You'll see it on nearly every
# helper in the sections ahead.

# %% [markdown]
# ## 4. Lines, planes, and decorations
#
# The decoration vocabulary from the matplotlib side carries straight over.
# Reference planes and great circles first — `add_plane_overlay()` knows the
# named planes (`'galactic'`, `'ecliptic'`, `'supergalactic'`), each with a
# sensible default color, and brackets its plane with `parallels=` small
# circles; `add_great_circle()` draws the equator of any frame, or any
# custom great circle via `frame='pole'` with `pole_lon=`/`pole_lat=`:

# %%
fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark",
    width=820, height=460,
    title="Reference planes and circles — hover each curve for its identity",
)
sphpl.add_plane_overlay(fig, plane="galactic",
                        color=PAL["accent"], width=2.0,
                        parallels=[-30, 30], hover=True)
sphpl.add_plane_overlay(fig, plane="ecliptic", width=1.5, hover=True)
sphpl.add_plane_overlay(fig, plane="supergalactic", width=1.5, hover=True)
# A custom great circle from its pole — here the debris plane of the
# Sagittarius dwarf, whose tidal stream wraps the sky on a near-polar
# great circle (pole from Majewski et al. 2003).
sphpl.add_great_circle(fig, frame="pole", pole_lon=123.6, pole_lat=-59.5,
                       color=C[5], width=1.5,
                       name="Sagittarius stream plane", hover=True)
sphpl.add_geodesic_circle(fig, 10.68, 41.27, 20.0,
                          color=C[0], width=1.5,
                          fill=True, fillcolor="rgba(120,160,255,0.22)",
                          name="20° around M31", hover=True)
sphpl.add_frame_edge(fig, color=PAL["grid"])
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30)
fig.show()

# %% [markdown]
# Every curve is seam-aware: the galactic plane crosses the wrap edge of this
# `center=180` frame and comes out as cleanly broken segments, not a smear
# across the canvas — and the M31 circle, straddling the edge, splits into
# its two rim arcs. Zoom into a crossing to check. The `parallels=[-30, 30]`
# small circles bracketing the galactic plane come from the same machinery
# as `add_great_circle(lat_offset=...)` — any parallel of any frame.
#
# ### Reticles and rulers on a field
#
# For zoomed fields, the plotly side has the same pointing and measuring
# decorations as the matplotlib `Reticle` and `Ruler` classes — as functions.
# Frame a field by passing `make_figure` an `fov_deg=` (the field's width in
# degrees) along with the `center`/`lat_center` you want at the middle:

# %%
# Bright stars of Orion, for context.
orion = {
    "Betelgeuse": (88.793, 7.407), "Rigel": (78.634, -8.202),
    "Bellatrix": (81.283, 6.350), "Mintaka": (83.002, -0.299),
    "Alnilam": (84.053, -1.202), "Alnitak": (85.190, -1.943),
    "Saiph": (86.939, -9.670),
}
names = list(orion)
s_ra = np.array([orion[n][0] for n in names])
s_dec = np.array([orion[n][1] for n in names])

fig = sphpl.make_figure(
    projection="SIN", center=84, lat_center=-2, fov_deg=22, theme="dark",
    width=640, height=640,
    title="Orion — reticle styles, and a ruler with real on-sky ticks",
)

sphpl.add_scatter(fig, s_ra, s_dec, name="Orion stars",
                  customdata=np.stack([names, s_ra, s_dec], axis=-1),
                  hovertemplate="<b>%{customdata[0]}</b><br>"
                                "RA %{customdata[1]:.2f}°, Dec %{customdata[2]:.2f}°"
                                "<extra></extra>",
                  marker=dict(size=9, color=C[1], opacity=0.95))

# A reticle marks a target without covering it. All four styles, one per star:
sphpl.add_reticle(fig, 83.82, -5.39, style="plus", size=16, label="M42")
sphpl.add_reticle(fig, 81.283, 6.350, style="circle", size=14,
                  color=C[3], label="Bellatrix")
sphpl.add_reticle(fig, 84.053, -1.202, style="x", size=14,
                  color=C[4], label="Alnilam")
sphpl.add_reticle(fig, 86.939, -9.670, style="L", size=16,
                  color=C[0], label="Saiph")

# A ruler measures along the sky — ticks and labels are computed in angular
# units on the sphere, not in screen pixels.
sphpl.add_ruler(fig, *orion["Betelgeuse"], *orion["Rigel"],
                tick_interval=300, label_unit="arcmin",
                endcap_style="tick", title="Betelgeuse → Rigel",
                color=PAL["label"])
# placement="canvas" pins the RA/Dec labels to the axis edges — a zoomed
# field's projection silhouette (the default "frame" anchor) is off-screen.
sphpl.add_coord_labels(fig, placement="canvas", lon_spacing=10, lat_spacing=10,
                       color=PAL["label"])
fig.show()

# %% [markdown]
# The ruler's `label_unit='auto'` picks a sensible angular unit for the
# span (degrees down to μas as you zoom); here it's pinned to arcminutes
# with a tick every 300′. The four reticle styles — `'plus'`, `'x'`, `'L'`, and
# `'circle'` — are all on the field above, each with `size=`, `gap=`,
# `rotation=`, and stroke options, plus a `label=` with automatic label-side
# placement — the same vocabulary as the matplotlib classes, minus the
# mutability.
#
# > **Note:** `fov_deg` frames the field for you by calling `sphpl.project()`
# > — the shared sky → canvas primitive, listed in the parity table — to work
# > out the axis range. When you build your own pan/zoom or hover callbacks,
# > `project()` is the piece you reach for directly.

# %% [markdown]
# ## 5. Regions and set algebra
#
# The region family renders sky-aware filled shapes: latitude bands of any
# frame, great-circle corridors, lon/lat boxes, and free-form spherical
# polygons. All of them handle the wrap edge and the poles for you:

# %%
fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark",
    width=820, height=460,
    title="The region vocabulary — every shape is seam- and pole-aware",
)
# One styling habit for all filled overlays: keep the trace opacity at 1
# and put the transparency in the fill's rgba alpha — otherwise the two
# multiply and fills wash out.
sphpl.add_frame_band(fig, -10, 10, frame="galactic",
                     color=PAL["accent"], fillcolor="rgba(140,180,255,0.30)",
                     opacity=1, name="Galactic band |b| < 10°", hover=True)
sphpl.add_lonlat_box(fig, -20, 20, 30, 90, frame="galactic",
                     color=C[5], fillcolor="rgba(220,140,255,0.28)",
                     opacity=1, name="Galactic box 30° < l < 90°", hover=True)
sphpl.add_great_circle_band(fig, ra_pole=280.0, dec_pole=62.0, half_width=6,
                            color=C[3], fillcolor="rgba(255,180,100,0.28)",
                            opacity=1, name="Orbit corridor (custom pole)",
                            hover=True)
sphpl.add_spherical_polygon(fig,
                            lons=[320, 355, 350, 315], lats=[-55, -50, -25, -30],
                            color=C[0], fillcolor="rgba(120,220,180,0.28)",
                            opacity=1, name="Survey footprint", hover=True)
sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.show()

# %% [markdown]
# `add_spherical_polygon()` connects vertices with true geodesics when the
# edges are long (`geodesic='auto'`, switching over at 10°) — so a "rectangle"
# of sky keeps its spherical shape instead of cutting flat chords. And the
# corridor is `add_great_circle_band()` — a band of given `half_width=`
# around any custom great circle (a satellite orbit's ground track on the
# sky, say), defined by its `ra_pole=`/`dec_pole=`.
#
# ### Compound regions
#
# Set algebra crosses the backend boundary intact. `make_compound_region(fig)`
# returns the *same* `CompoundRegion` class you know from the regions
# tutorial — built against the plotly figure's projection instead of a
# matplotlib axes — so the full verb families (`add_*`, `subtract_*`,
# `intersect_*`) and the query methods work identically:

# %%
fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark",
    width=820, height=460,
    title="Set-algebra regions — holes render as true holes",
)

# The Zone of Avoidance, minus a Galactic-Center cutout.
# hover_anchor='area' (the default): the tooltip appears anywhere over the fill.
zoa = (sphpl.make_compound_region(fig)
       .add_frame_band(-10, 10, frame="galactic")
       .subtract_circle(266.417, -28.936, 10))
sphpl.add_compound_region(fig, zoa,
                          color=PAL["accent"],
                          fillcolor="rgba(140,180,255,0.35)", opacity=1,
                          name="Galactic band, GC excised",
                          hover=True, hover_anchor="area")

# A survey footprint with two bright-star masks punched out.
# hover_anchor='point': one tooltip target inside the region instead.
survey = (sphpl.make_compound_region(fig)
          .add_polygon(lons=[210, 250, 250, 210], lats=[35, 35, 55, 55])
          .subtract_circle(220, 45, 4)
          .subtract_circle(240, 50, 3))
sphpl.add_compound_region(fig, survey,
                          color=C[3],
                          fillcolor="rgba(255,180,100,0.35)", opacity=1,
                          name="Footprint minus star masks (point hover)",
                          hover=True, hover_anchor="point")

sphpl.add_frame_edge(fig, color=PAL["grid"])
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30)
fig.show()

# %% [markdown]
# Sweep the cursor across the two regions and you'll see the difference.
# The blue band answers **anywhere inside its fill** (`hover_anchor='area'`,
# the default) — plotly's `hoveron='fills'` under the hood. The orange
# footprint answers only at **one point** near its middle
# (`hover_anchor='point'`). Area hover is what you almost always want; the
# point anchor is there for slivers and tiny regions where a whole-area
# target would swallow the things drawn on top of it.

# %% [markdown]
# Under the hood this is still a full `CompoundRegion`, so the analysis
# methods come along for free — which Messier objects sit in the Zone of
# Avoidance?

# %%
inside = zoa.contains_points(messier["ra_deg"], messier["dec_deg"])
print(f"Band covers {zoa.area_frac:.1%} of the sky "
      f"and contains {inside.sum()} of {len(messier)} Messier objects:")
print(", ".join(messier["name"][inside]))

# %% [markdown]
# ### The same region, on any projection
#
# The region math lives on the sphere; only the *projector* changes with the
# figure. So the identical Zone of Avoidance — the galactic band with the
# Galactic-center hole punched out — renders just as happily on an
# orthographic globe, clipped to the visible hemisphere:

# %%
fig = sphpl.make_figure(
    projection="SIN", center=266.4, lat_center=-29.0, theme="dark",
    width=560, height=560, grid_lon_spacing=30, grid_lat_spacing=30,
    title="The Zone of Avoidance, set algebra intact, on a globe",
)
# Rebuilt against this figure's projector — same verbs, same geometry.
gzoa = (sphpl.make_compound_region(fig)
        .add_frame_band(-10, 10, frame="galactic")
        .subtract_circle(266.417, -28.936, 10))
sphpl.add_compound_region(fig, gzoa, color=PAL["accent"],
                          fillcolor="rgba(140,180,255,0.28)", opacity=1,
                          name="Galactic band, GC excised", hover=True)
sphpl.add_scatter(fig, messier["ra_deg"], messier["dec_deg"],
                  text=messier["name"],
                  hovertemplate="<b>%{text}</b><br>RA %{customdata[0]:.2f}°, "
                                "Dec %{customdata[1]:.2f}°<extra></extra>",
                  marker=dict(size=6, color=C[2], opacity=0.9))
sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.show()

# %% [markdown]
# The band wraps across the near side and stops at the limb; the hole is a
# real hole; hover still works over the fill. Nothing about the region
# changed — only the projection it was drawn through, which is the whole
# point of the shared compute pipeline.

# %% [markdown]
# ### Making it live: a region that grows
#
# Every region so far has been *static*. But `contains_points` is just a
# function of the region, and the region is just a function of its parameters —
# so if we precompute a handful of sizes, a slider can grow the region and
# re-sort the catalog at every step, with no Python behind the page (the same
# precompute-and-replay trick as the
# [VSH explorer](#8.-Vector-fields-and-the-VSH-explorer)).
#
# Here's a real use for it. A VLBI astrometric catalog — the `usno2025a`
# reference sources, with the 303 ICRF3 *defining* sources drawn as diamonds —
# against a growing **exclusion zone**: a galactic-latitude band (the Milky
# Way's plane is a poor place to anchor a reference frame) unioned with
# avoidance cones around two bright, confusing radio sources (Cygnus A,
# Centaurus A) and the Sun. Drag the band wider and three things happen at once
# — the zone **grows**, the cones are **swallowed** one by one as the
# set-algebra union absorbs them, and every source **re-sorts** into excluded
# (amber) or usable (grey):

# %%
vlbi = pd.read_csv("../../examples/data/usno2025a_vlbi.csv")
v_ra, v_dec = vlbi["ra_deg"].to_numpy(), vlbi["dec_deg"].to_numpy()
is_def = vlbi["defining"].to_numpy().astype(bool)          # 303 ICRF3 defining
vx, vy = (np.round(a, 1) for a in
          sphpl.project(v_ra, v_dec, projection="AIT", center=180))

# Three avoidance cones at staggered galactic latitudes, so a widening band
# reaches them one after another. The Sun is frozen at one date ~30° off the
# plane — a real scheduling constraint you can't observe through.
CYG_A, CEN_A = (299.868, 40.734), (201.365, -43.019)
_sun = get_sun(Time("2026-05-16"))
SUN = (float(_sun.ra.deg), float(_sun.dec.deg))

fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark", width=900, height=520,
    title="A VLBI reference frame's exclusion zone — drag the band wider")

# This is an ICRS frame, but the band lives in galactic coordinates, so a faint
# galactic plane marks where the growing band is centered.
_i = len(fig.data)
sphpl.add_plane_overlay(fig, plane="galactic", color=PAL["grid"],
                        width=1.0, opacity=0.6)
for _t in fig.data[_i:]:
    _t.showlegend = False          # keep the plane out of the compound legend


# The factory takes fig as a keyword (defaulting to this figure) so the SAME
# function drives both the static states below and the live Dash app at the end.
def exclusion_zone(band_hw, fig=fig):
    """|b| < band_hw, unioned with three avoidance cones."""
    return (sphpl.make_compound_region(fig)
            .add_frame_band(-band_hw, band_hw, frame="galactic")
            .add_circle(*CYG_A, 8).add_circle(*CEN_A, 8).add_circle(*SUN, 18))


WIDTHS = [4, 10, 16, 24, 32, 40]
states = sphpl.compound_region_states(
    fig, exclusion_zone, [dict(band_hw=w) for w in WIDTHS], catalog=(v_ra, v_dec))

# The region is two layout shapes — a fill and an outline — the slider swaps by
# path. (`add_region_slider` is the turnkey version; we go a level down here
# only to split the catalog into two marker traces.)
AMBER, GREY = "#E8A33D", "#5A6070"
shape0 = len(fig.layout.shapes)
fig.add_shape(type="path", path=states[0]["fill_path"], layer="below",
              fillcolor="rgba(140,180,255,0.16)", line=dict(width=0))
fig.add_shape(type="path", path=states[0]["outline_path"], layer="below",
              fillcolor="rgba(0,0,0,0)", line=dict(color="#8FB3F0", width=1.2))

# Two marker traces: standard (small circles) and ICRF3-defining (diamonds).
# Color is a 0/1 containment code through a two-stop scale — far lighter than
# shipping thousands of color strings per step (see section 14).
SCALE = [[0.0, GREY], [1.0, AMBER]]
tr = {}
for key, mask, sym, sz in [("std", ~is_def, "circle", 3.0),
                           ("def", is_def, "diamond", 6.0)]:
    role = "standard" if key == "std" else "ICRF3 defining"
    cd = np.column_stack([vlbi["iers_name"].to_numpy()[mask],
                          v_ra[mask], v_dec[mask]])
    fig.add_trace(go.Scattergl(
        x=vx[mask], y=vy[mask], mode="markers", name=role, customdata=cd,
        hovertemplate="<b>%{customdata[0]}</b> (" + role + ")<br>"
                      "RA %{customdata[1]:.2f}°, Dec %{customdata[2]:.2f}°"
                      "<extra></extra>",
        marker=dict(size=sz, symbol=sym, color=states[0]["contains_int"],
                    colorscale=SCALE, cmin=0, cmax=1,
                    line=dict(width=0.4, color="#0a0a14")),
        showlegend=False))
    tr[key] = len(fig.data) - 1

# A live count that updates with the slider.
ann0 = len(fig.layout.annotations)
fig.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper",
                   xanchor="left", yanchor="top", showarrow=False,
                   font=dict(size=13, color=PAL["label"]),
                   text=f"excluded: {states[0]['n_inside']} / {len(vlbi)}")

# One update step per width: swap both shape paths, both marker color codes,
# and the count.
steps = []
for w, st in zip(WIDTHS, states):
    ci = np.asarray(st["contains_int"])
    steps.append(dict(
        method="update", label=f"{w}°",
        args=[{"marker.color": [ci[~is_def].tolist(), ci[is_def].tolist()]},
              {f"shapes[{shape0}].path": st["fill_path"],
               f"shapes[{shape0 + 1}].path": st["outline_path"],
               f"annotations[{ann0}].text":
                   f"excluded: {st['n_inside']} / {len(vlbi)}"},
              [tr["std"], tr["def"]]]))

# A compound key in one call: color = containment, shape = catalog role.
# `add_legend` renders the channels plotly's own legend can't — a categorical
# color key and a distinct-shape key — as a single grouped legend.
sphpl.add_legend(fig, [
    sph.ColorBlock("Exclusion", {"excluded": AMBER, "usable": GREY},
                   swatch="marker"),
    sph.ShapeBlock("Catalog", {"standard": "o", "ICRF3 defining": "D"},
                   color="#C8CCD4", size=100),   # ShapeBlock size is marker area
])
sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.update_layout(
    xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
    margin=dict(r=150),
    legend=dict(x=1.01, xanchor="left", y=0.5, yanchor="middle",
                font=dict(size=10), itemclick=False, itemdoubleclick=False),
    sliders=[dict(active=0, steps=steps, x=0.06, len=0.74, pad=dict(t=40),
                  currentvalue=dict(prefix="|b| < ", font=dict(size=13)))])
fig.show()

# %% [markdown]
# Three set-algebra merges play out as you drag: Cygnus A (nearly *in* the
# plane) is absorbed almost at once, Centaurus A around a band of ~20°, and the
# solar cone last — each join is the union quietly swallowing a cone, and the
# "excluded" count jumps as the sources caught between band and cone flip. All
# the compute — the union geometry *and* the `contains_points` re-sort — ran
# once at build time in `compound_region_states()`; the page just replays it.
#
# > **Note:** the slider replays six precomputed widths — smooth enough to read
# > as growth, but discrete, and *set algebra is what keeps it discrete*. A
# > plain `|b| < w` band could be recomputed live in the browser from a
# > one-line formula, but a union of arbitrary regions needs the full geometry
# > engine, which only runs in Python. For a genuinely *continuous* slider, run
# > the notebook on a live kernel and flip the switch below.

# %%
# A continuous, live version — needs a running kernel and `dash` (does nothing
# on this static page or on nbviewer, which have no kernel; run the notebook
# locally or on Binder to try it). It reuses the very same `exclusion_zone`
# factory, now driven by a slider with no precomputed steps at all:
RUN_DASH = False
if RUN_DASH:
    from skyplothelper.plotly.dash_region import region_explorer_app

    app = region_explorer_app(
        (v_ra, v_dec), exclusion_zone, params={"band_hw": (2, 45, 1)},
        projection="AIT", center=180, theme="dark",
        inside_color=AMBER, outside_color=GREY,
        marker_by=np.where(is_def, "ICRF3 defining", "standard"),
        title="VLBI exclusion zone — live")
    app.run(debug=True)          # → http://127.0.0.1:8050, drag continuously

# %% [markdown]
# ## 6. HEALPix maps
#
# HEALPix maps render as true tile polygons — each pixel is a clickable,
# hoverable shape that knows its index and value. Here's the classic
# all-sky dipole pattern (the shape of the CMB dipole, at the real apex
# direction) on a Galactic-frame Mollweide — hover any tile for its
# pixel index and amplitude:

# %%
nside = 4
l_pix, b_pix = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
apex = hp.ang2vec(264.02, 48.25, lonlat=True)          # CMB dipole apex (l, b)
dipole_mK = 3.36 * (hp.ang2vec(l_pix, b_pix, lonlat=True) @ apex)

fig = sphpl.make_figure(
    projection="MOL", center=0, frame="galactic", theme="dark",
    width=880, height=470,
    title="Dipole sky at nside=4 — hover a tile for ipix and value",
)
sphpl.add_healpix(fig, dipole_mK, nside,
                  colorscale=plotly_scale("sph.diff_blueorange", 0.12, 0.88),
                  vmin=-3.36, vmax=3.36,      # symmetric, so 0 sits on white
                  tile_resolution=2,
                  hover_format=("l {lon:.1f}°, b {lat:.1f}°<br>"
                                "ΔT = {value:+.2f} mK<br>ipix {ipix}"),
                  add_colorbar=True, cbar_title="ΔT (mK)")
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30,
                       lat_exterior=True, color=PAL["label"])
sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.show()

# %% [markdown]
# The tiles follow the projection: edges curve with the graticule, tiles
# straddling the wrap edge split cleanly, and the polar tiles close through
# the pole. Four knobs did the styling work. `colorscale=` takes a plotly
# colorscale name or an explicit color list (this one is sph's bundled
# `sph.diff_blueorange` diverging map through the `plotly_scale` bridge,
# trimmed away from its darkest ends), paired with a symmetric
# `vmin`/`vmax` so zero lands on the neutral middle. `hover_format=` is a
# template with `{lon}`, `{lat}`, `{value}`, and `{ipix}` slots, so the
# tooltip speaks your data's units. `add_colorbar=True` (with `cbar_title=`)
# attaches the key — worth knowing *why* it's opt-in: each tile is painted a
# flat fill color, which carries no colorscale of its own, so the bar has to
# ride along on a companion trace. And `tile_resolution=` controls how
# densely each tile's edges are sampled, which also makes it your
# figure-size dial (see the sizing note below).
#
# ### Sparse maps
#
# For data that only occupy part of the sky, `add_healpix_sparse()` takes
# *pixel indices + values* and draws only the occupied tiles. Tiles are just
# another overlay, so the rest of the vocabulary layers around them
# normally — here the galactic band is shaded behind Messier object counts
# per pixel, and `colorbar_kwargs=` moves the key beneath the map:

# %%
pix = hp.ang2pix(8, np.asarray(messier["ra_deg"]),
                 np.asarray(messier["dec_deg"]), lonlat=True)
occupied, counts = np.unique(pix, return_counts=True)

fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark",
    width=860, height=520,
    title="Messier objects per nside=8 pixel — only occupied tiles drawn",
)
sphpl.add_frame_band(fig, -10, 10, frame="galactic",
                     color=PAL["accent"], fillcolor="rgba(140,180,255,0.18)",
                     opacity=1, width=1.0,
                     name="Galactic band |b| < 10°", hover=True)
sphpl.add_healpix_sparse(fig, occupied, counts, nside=8,
                         colorscale=plotly_scale("sph.lagoon", 0.25, 0.95),
                         tile_resolution=2, opacity=0.9,
                         hover_format=("RA {lon:.1f}°, Dec {lat:.1f}°<br>"
                                       "{value:.0f} objects<br>ipix {ipix}"),
                         add_colorbar=True, cbar_title="objects per pixel",
                         colorbar_kwargs=dict(orientation="h", x=0.5,
                                              xanchor="center", y=-0.02,
                                              yanchor="top", len=0.55,
                                              thickness=13))
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30,
                       lat_exterior=True, color=PAL["label"])
sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.show()

# %% [markdown]
# The Virgo cluster's galaxy swarm lights up immediately — and hovering the
# brightest tile tells you exactly which pixel to query in your analysis.
# Notice how few Messier objects fall inside the shaded galactic band: Messier
# was hunting comets, and the Milky Way's dust hides the galaxies that make up
# most of his catalog.
#
# > **Note:** every HEALPix tile is an SVG polygon, so a dense all-sky map at
# > high nside is a *lot* of geometry — an embedded nside=64 map would be
# > tens of megabytes and would make the browser struggle. Interactive maps
# > shine at modest nside (≤ 16) or with the sparse renderer; for
# > high-resolution HEALPix imaging, use the matplotlib renderers from the
# > [HEALPix Workflows](healpix_workflows.ipynb) tutorial, which rasterize
# > to pixels instead.
#
# ## 7. Constellations
#
# The constellation family also mirrors the matplotlib side:
# `add_constellation_lines()` (stick figures, with the same `rank_max=`
# prominence cut), `add_constellation_labels()`, and
# `add_constellation_polygon()` — which turns a constellation's IAU boundary
# into a filled, *hover-named* region:

# %%
fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark",
    width=820, height=460,
    title="Constellations — hover a shaded region for its name",
)
sphpl.add_constellation_lines(fig, rank_max=2, color=C[2], opacity=0.7)
sphpl.add_constellation_labels(fig, labels="abbr", color=PAL["label"])
for con, cname, fill in [("Ori", "Orion", "rgba(120,160,255,0.35)"),
                         ("UMi", "Ursa Minor", "rgba(255,180,100,0.40)"),
                         ("Cru", "Crux", "rgba(180,255,120,0.35)")]:
    sphpl.add_constellation_polygon(fig, con, step_deg=1.0, name=cname,
                                    fillcolor=fill, opacity=1, hover=True)
sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.show()

# %% [markdown]
# Ursa Minor is here deliberately: it encloses the celestial pole, the
# classic stress test for polygon closure — the fill wraps over the top of
# the frame correctly instead of leaking down the map.
#
# > **Note:** the full IAU boundary network (`add_constellation_boundaries()`)
# > works here too, but its vertex data are heavy — about 2.2 MB of embedded
# > coordinates. Rounding those to two decimals cuts it to 0.9 MB with no
# > visible change (see the `round_traces()` helper, and section 14), which
# > is exactly what the next figure does. The general lesson: an interactive
# > figure *carries its data with it*.
#
# ### A star chart you can take apart
#
# Which brings us to the payoff. Plotly's legend isn't just a key — it's a
# control panel: **click** an entry to hide that layer, **double-click** to
# show it alone. So the same constellation vocabulary, laid over the 4,992
# naked-eye stars of the Hipparcos catalog, becomes a chart the reader can
# disassemble. Peel the boundaries away to see the asterisms breathe; hide
# everything but the stars and the galactic band.
#
# The one wrinkle: several helpers (`add_frame_band` here) draw a layer with
# *more than one* trace — a fill plus its edges. Tie them together with a
# shared `legendgroup` and `groupclick="togglegroup"`, and one click moves
# the whole layer:

# %%
stars = pd.read_csv("../../examples/data/hipparcos_bright_pm.csv")

# Classic star-chart scaling: marker area grows as (m_lim - V)^2.
star_sizes = np.clip(0.32 * (6.5 - stars.Vmag.values) ** 2, 0.8, 17.0)


def legend_layer(fig, first, name, hover=False):
    """Turn every trace added since index `first` into one legend entry."""
    for trace in fig.data[first:]:
        trace.update(legendgroup=name, showlegend=False)
        if not hover:
            trace.update(hoverinfo="skip")
    fig.data[first].update(name=name, showlegend=True)


fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark", width=980, height=580,
    title="Every naked-eye star — click a legend entry to add or remove a layer")

i = len(fig.data)
sphpl.add_frame_band(fig, -10, 10, frame="galactic", color="#8FB3F0",
                     fillcolor="rgba(140,180,255,0.16)", opacity=1, width=1.0,
                     name="Galactic band", hover=True)
legend_layer(fig, i, "Galactic band", hover=True)

i = len(fig.data)
sphpl.add_constellation_boundaries(fig, color="#6E7686", width=0.6, opacity=0.6)
legend_layer(fig, i, "IAU boundaries")

i = len(fig.data)
sphpl.add_constellation_lines(fig, rank_max=1, color=PAL["accent"],
                              width=1.0, opacity=0.9)
legend_layer(fig, i, "Asterisms")

i = len(fig.data)
sphpl.add_constellation_labels(fig, labels="abbr", color=PAL["label"])
legend_layer(fig, i, "Names")

i = len(fig.data)
sphpl.add_scatter(
    fig, stars.RAICRS.values, stars.DEICRS.values, name="Naked-eye stars",
    customdata=np.stack([stars.RAICRS, stars.DEICRS, stars.Vmag], axis=-1),
    hovertemplate="V = %{customdata[2]:.2f}<br>RA %{customdata[0]:.2f}°, "
                  "Dec %{customdata[1]:.2f}°<extra></extra>",
    marker=dict(size=star_sizes, color=PAL["stars"], opacity=0.9,
                line=dict(width=0)))
fig.data[i].update(legendgroup="Naked-eye stars")

sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.update_layout(showlegend=True,
                  legend=dict(x=0.5, xanchor="center", y=-0.02, yanchor="top",
                              orientation="h", font=dict(size=11),
                              groupclick="togglegroup"))
round_traces(fig)        # 2.9 MB → 1.3 MB, no visible change
fig.show()

# %% [markdown]
# Hover any star for its magnitude. Zoom into Orion and watch the asterism
# resolve into individual stars. The whole chart — 4,992 stars, 88
# boundaries, the bright asterisms, their labels, and a galactic band —
# rides in about 1.3 MB, because every coordinate got rounded to two
# decimals on the way out.

# %% [markdown]
# ## 8. Vector fields and the VSH explorer
#
# `add_sky_vectors()` is the plotly twin of `plot_sky_vectors` — arrow fields
# whose directions and lengths live on the sphere, drawn correctly through
# the projection. It expects the `Δα·cosδ` convention for `dlon=` (the
# default `cos_dec=True`), takes its magnitudes in your `units=`
# (`'deg'` … `'mas'`, `'uas'`), sets arrow length via `scale=` (degrees
# of map per unit of magnitude, or `'auto'`), and anchors each arrow at its
# data point by the `pivot=` (`'middle'`, `'tail'`, or `'tip'`).
#
# Here's a *real* systematic pattern: the solar system accelerates around
# the Galactic center, which aberrates every quasar position by a few
# μas/yr — a pure *glide* toward the Galactic center. Evaluating that glide
# at the positions of the 303 ICRF3 defining sources, with
# magnitude-colored arrows (`color_by_magnitude=True`, plus the
# plotly-specific `shaft_color='match'` so shafts follow their heads) in
# sph's bundled `sph.sunset` colormap — trimmed to its saturated body, the
# same arrow recipe the [Vector Fields](vector_fields.ipynb) tutorial uses:

# %%
icrf3 = Table.read("../../examples/data/icrf3_defining.csv")

# Galactocentric-acceleration glide: ~5.8 μas/yr toward the Galactic center.
gc_ra, gc_dec = np.radians(266.417), np.radians(-28.936)
glide = 5.8 * np.array([np.cos(gc_dec) * np.cos(gc_ra),
                        np.cos(gc_dec) * np.sin(gc_ra),
                        np.sin(gc_dec)])
dlon, dlat = sph.vsh_field(icrf3["ra_deg"], icrf3["dec_deg"],
                           {"D_1": glide[0], "D_2": glide[1], "D_3": glide[2]})

fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark",
    width=820, height=470,
    title="Galactocentric-acceleration glide at the ICRF3 defining sources",
)
sphpl.add_sky_vectors(fig, icrf3["ra_deg"], icrf3["dec_deg"], dlon, dlat,
                      units="uas", scale="auto", auto_target_deg=5,
                      arrow_size=6, width=1.2,
                      color_by_magnitude=True, shaft_color="match",
                      cmap=plotly_scale("sph.sunset", 0.12, 0.85),
                      add_colorbar=True,
                      cbar_title="μas / yr", hover=True)
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30)
sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.show()

# %% [markdown]
# The arrows converge on the Galactic center and shrink toward it and its
# antipode (a glide is largest 90° from its axis) — and hovering any arrow
# reports its anchor, magnitude, and position angle. For what these
# patterns *mean* and how to fit them to data, see the
# [Vector Fields & Sky Kinematics](vector_fields.ipynb) tutorial; this
# section is about the rendering.
#
# ### The VSH explorer
#
# There is no live Python running behind this page — so anything a control does
# has to be baked into the figure ahead of time. That's exactly what
# plotly's built-in menus and sliders are for: each slider position holds a
# ready-made set of arrow coordinates (a plotly `restyle` step), and each
# dropdown entry swaps in both the arrows and the matching slider. The
# result is a self-contained VSH mode explorer that stays fully interactive
# in a static page. Pick a mode, sweep its amplitude:

# %%
GRID_RA = np.repeat(np.arange(0, 360, 30), 7)
GRID_DEC = np.tile(np.linspace(-60, 60, 7), 12)
AMPS = [10, 20, 30, 45]
DEFAULT_I = 2     # start at 30 μas/yr
MODES = [
    ("R_1", "rotation X"), ("R_2", "rotation Y"), ("R_3", "rotation Z"),
    ("D_1", "glide X"), ("D_2", "glide Y"), ("D_3", "glide Z"),
    ("E_22_Re", "quadrupole E22"),
]

# One scale shared by every mode and amplitude. `scale=` is map-degrees per
# unit of *sky angle*, so in μas it needs the 3.6e9 μas-per-degree factor —
# which is exactly why `scale='auto'` exists. But 'auto' renormalizes each
# frame, and then the amplitude slider wouldn't visibly change anything: we
# want one fixed ruler, calibrated so the median arrow spans 6° at the
# default amplitude.
UAS_PER_DEG = 3.6e9
_dlon, _dlat = sph.vsh_field(GRID_RA, GRID_DEC, {"R_3": AMPS[DEFAULT_I]})
ARROW_SCALE = 6.0 * UAS_PER_DEG / np.median(np.hypot(_dlon, _dlat))

ARROW_KW = dict(units="uas", scale=ARROW_SCALE, arrow_size=8, width=2.0,
                color="#88BBEE")


def vector_state(param, amp):
    """Arrow shaft + head coordinates for one (mode, amplitude) pair."""
    def rnd(seq, nd=2):
        return [None if v is None or v != v else round(float(v), nd)
                for v in seq]
    tmp = sphpl.make_figure(projection="AIT", center=180, show_grid=False)
    dlon, dlat = sph.vsh_field(GRID_RA, GRID_DEC, {param: amp})
    sphpl.add_sky_vectors(tmp, GRID_RA, GRID_DEC, dlon, dlat, **ARROW_KW)
    shaft, head = tmp.data[-2], tmp.data[-1]
    ang = [round(float(a), 1) for a in head.marker.angle]
    return {"x": [rnd(shaft.x), rnd(head.x)],
            "y": [rnd(shaft.y), rnd(head.y)],
            "marker.angle": [ang, ang]}


fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark", width=820, height=540,
    title="VSH explorer",
)
dlon, dlat = sph.vsh_field(GRID_RA, GRID_DEC, {"R_3": AMPS[DEFAULT_I]})
sphpl.add_sky_vectors(fig, GRID_RA, GRID_DEC, dlon, dlat, **ARROW_KW)
IDX = [len(fig.data) - 2, len(fig.data) - 1]   # the two arrow traces
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30)
sphpl.add_frame_edge(fig, color=PAL["grid"])


def slider_for(param):
    steps = [dict(method="restyle", label=str(a),
                  args=[vector_state(param, a), IDX]) for a in AMPS]
    return dict(active=DEFAULT_I, steps=steps, x=0.10, len=0.80,
                pad=dict(t=30),
                currentvalue=dict(prefix="amplitude: ", suffix=" μas/yr",
                                  font=dict(size=13)))


buttons = [dict(label=label, method="update",
                args=[vector_state(param, AMPS[DEFAULT_I]),   # new arrows
                      {"sliders": [slider_for(param)]},       # matching slider
                      IDX])
           for param, label in MODES]

fig.update_layout(
    updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                      active=2, x=1.0, xanchor="right", y=1.14)],
    sliders=[slider_for("R_3")],
)
fig.show()

# %% [markdown]
# Try `glide Z` against `rotation Z` — a glide flows *toward* an apex, a
# rotation swirls *around* an axis; on a projected map the two are easy to
# confuse until you see them move. The quadrupole entry shows the
# characteristic ℓ=2 shear that frame-tie analyses fit alongside rotation
# and glide.
#
# > **Note:** choosing a new mode resets the slider to its starting value —
# > every (mode, amplitude) pair is a precomputed state, and the dropdown
# > swaps in the slider belonging to that mode. That's the trade for
# > interactivity with no Python behind it: the controls *replay* prepared
# > states rather than compute new ones. For free-form parameter mixing,
# > reach for `dash` or `ipywidgets` in a live session.

# %% [markdown]
# ## 9. Orbits around the black hole
#
# The VSH explorer's trick — precompute a state for each slider step, let the
# reader drive it — pays off most when the states are a *real system* caught in
# motion. The showpiece: the **S-stars**, a swarm that whips around **Sgr A\***,
# the four-million-solar-mass black hole at the Galactic center, on full
# Keplerian ellipses. Two decades of astrometry (the work behind the 2020 Nobel
# Prize) pinned down their orbits; the bundled `sstar_orbits.csv` carries the
# elements for the 16 best-measured, from Gillessen et al. 2017.
#
# With the elements in hand, each star's position at *any* time is closed-form —
# solve Kepler's equation for the eccentric anomaly, rotate the orbit ellipse
# onto the sky with the Thiele–Innes constants. No integration, no per-epoch
# interpolation. So a `year` slider costs almost nothing: precompute the
# positions (and the live radial velocity, which we get from the same solution)
# at each year, and let the slider replay them.

# %%
orbits = pd.read_csv("../../examples/data/sstar_orbits.csv")
SGRA = (266.41681, -29.00782)                     # Sgr A* (ICRS degrees)
R0_KM = 8.3 * 3.0857e16                           # 8.3 kpc, for velocity scaling
ARCSEC_YR_TO_KMS = R0_KM * (np.pi / 180 / 3600) / 3.1557e7


def kepler_E(mean_anomaly, e, iters=60):
    """Eccentric anomaly from mean anomaly, by Newton's method."""
    M = np.mod(mean_anomaly + np.pi, 2 * np.pi) - np.pi
    E = M + e * np.sin(M)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def star_state(row, t):
    """Sky offsets (arcsec E, N from Sgr A*) and radial velocity (km/s) at year t."""
    a, e = row["a_arcsec"], row["ecc"]
    inc, node, w = np.radians([row["incl_deg"], row["node_deg"],
                               row["periapsis_deg"]])
    E = kepler_E(2 * np.pi * (t - row["t_peri_yr"]) / row["period_yr"], e)
    # Thiele–Innes: orbit-plane (x, y) rotated onto the sky by (node, w, inc).
    A = a * (np.cos(w) * np.cos(node) - np.sin(w) * np.sin(node) * np.cos(inc))
    B = a * (np.cos(w) * np.sin(node) + np.sin(w) * np.cos(node) * np.cos(inc))
    F = a * (-np.sin(w) * np.cos(node) - np.cos(w) * np.sin(node) * np.cos(inc))
    G = a * (-np.sin(w) * np.sin(node) + np.cos(w) * np.cos(node) * np.cos(inc))
    xn, yn = np.cos(E) - e, np.sqrt(1 - e * e) * np.sin(E)
    d_east, d_north = B * xn + G * yn, A * xn + F * yn
    nu = np.arctan2(np.sqrt(1 - e * e) * np.sin(E), np.cos(E) - e)   # true anomaly
    v_r = ((2 * np.pi / row["period_yr"]) * a / np.sqrt(1 - e * e)
           * np.sin(inc) * (np.cos(w + nu) + e * np.cos(w)))
    return d_east, d_north, v_r * ARCSEC_YR_TO_KMS


def offsets_to_xy(d_east, d_north):
    """Arcsec offsets from Sgr A* → canvas coords through the TAN projection."""
    ra = SGRA[0] - (d_east / 3600.0) / np.cos(np.radians(SGRA[1]))   # +East → −RA
    dec = SGRA[1] + d_north / 3600.0
    x, y = sphpl.project(ra, dec, projection="TAN",
                         center=SGRA[0], lat_center=SGRA[1])
    return np.round(x, 7), np.round(y, 7)


names = list(orbits["star"])
kmag = np.where(np.isfinite(orbits["Kmag"]), orbits["Kmag"], 17.0)
sizes = list(np.clip((17.5 - kmag) ** 2 * 0.9, 6, 22))
# Color each star (and its own orbit) by orbital period, in sph's 'mesa' map:
# the tight, fast inner stars at one end, the wide slow ones at the other.
periods = orbits["period_yr"].to_numpy()
pnorm = (periods - periods.min()) / (periods.max() - periods.min())
mesa = sph.get_colormap("sph.mesa")
tint = ["rgb({:.0f},{:.0f},{:.0f})".format(*(c * 255 for c in mesa(p)[:3]))
        for p in pnorm]

# Zoom in on the crowded inner arcsecond, where the action is — the same
# close-in field the Animations tutorial uses. The few widest orbits run off
# the edges, as they do in every real image of this region.
GC_FOV_ARCSEC = 1.34

fig = sphpl.make_figure(
    projection="TAN", center=SGRA[0], lat_center=SGRA[1],
    fov_deg=GC_FOV_ARCSEC / 3600.0, theme="dark",
    width=620, height=650, show_grid=False,
    title="The S-stars orbiting Sgr A* — drag the year slider")

# Each star's full orbit, drawn once as a faint ellipse in its own color.
for j, (_, row) in enumerate(orbits.iterrows()):
    sweep = np.linspace(row["t_peri_yr"], row["t_peri_yr"] + row["period_yr"], 240)
    de, dn, _ = star_state(row, sweep)
    ex, ey = offsets_to_xy(de, dn)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", opacity=0.45,
                             line=dict(color=tint[j], width=1),
                             hoverinfo="skip", showlegend=False))

# Sgr A* itself — fixed at the focus.
bx, by = offsets_to_xy(np.array([0.0]), np.array([0.0]))
fig.add_trace(go.Scatter(x=[float(bx[0])], y=[float(by[0])], mode="markers",
                         marker=dict(symbol="circle", size=16, color="black",
                                     line=dict(color="#FFE08A", width=1.5)),
                         hovertemplate="Sgr A*<extra></extra>", showlegend=False))


def star_frame(t):
    """(x, y, customdata) for all 16 stars at year t — one slider state."""
    de, dn, v_r = np.array([star_state(r, t) for _, r in orbits.iterrows()]).T
    x, y = offsets_to_xy(de, dn)
    sep = np.round(np.hypot(de, dn) * 1000.0, 1)          # mas from Sgr A*
    custom = np.column_stack([names, np.round(kmag, 1), sep, np.round(v_r, 0)])
    return list(x), list(y), custom


YEARS = np.round(np.arange(2000.0, 2034.01, 0.4), 1)
start = int(np.argmin(np.abs(YEARS - 2018.0)))            # S2's recent pericenter
x0, y0, cd0 = star_frame(YEARS[start])

fig.add_trace(go.Scatter(
    x=x0, y=y0, mode="markers", customdata=cd0,
    marker=dict(size=sizes, color=tint, line=dict(width=0)),
    hovertemplate="<b>%{customdata[0]}</b>  (K = %{customdata[1]})<br>"
                  "separation: %{customdata[2]} mas<br>"
                  "radial velocity: %{customdata[3]} km/s<extra></extra>",
    showlegend=False))
STARS = len(fig.data) - 1                                 # the one moving trace

steps = []
for t in YEARS:
    x, y, cd = star_frame(t)
    # Label every step (plotly thins the tick display) so the year readout
    # tracks the slider continuously, not only at whole-year stops.
    steps.append(dict(method="restyle", label=f"{t:g}",
                      args=[{"x": [x], "y": [y], "customdata": [cd]}, [STARS]]))
fig.update_layout(
    xaxis=dict(showticklabels=False, zeroline=False),
    yaxis=dict(showticklabels=False, zeroline=False),
    sliders=[dict(active=start, steps=steps, x=0.08, len=0.86, pad=dict(t=40),
                  currentvalue=dict(prefix="year ", font=dict(size=14)))])
fig.show()

# %% [markdown]
# Drag to 2018 and watch **S2** — the bright one on the tight inner ellipse —
# round Sgr A* at pericenter: hover it and the separation bottoms out near
# 11 mas while the radial velocity swings from +4000 km/s to −1800 km/s in
# under a year, a few percent of light speed. The tight stars blur through
# pericenter and crawl at apocenter (Kepler's second law playing out under
# your cursor), while the black-hole marker never moves.
#
# The whole field is about 1.3 arcseconds across — roughly a fifth of a
# light-year at the Galactic center's 8.3 kpc. This is the same picture the
# UCLA and GRAVITY groups built from a decade of images; here it falls out of
# sixteen rows of orbital elements and a dozen lines of Kepler, and the
# reader's hand is on the clock.
#
# > **Note:** these are *osculating* Keplerian orbits — they ignore the tiny
# > relativistic pericenter precession GRAVITY detected in S2 (far below a
# > pixel at this scale). Radial velocities are relative to Sgr A* and scaled
# > with M = 4.3 × 10⁶ M☉ at R₀ = 8.3 kpc; see `examples/data/README.md` for
# > the data provenance and citation.

# %% [markdown]
# ## 10. The sky in deep time
#
# The S-stars are an extreme close-up; pull all the way back and the *whole*
# sky is in slow motion. Every star drifts across the sky at its own proper
# motion, and over long enough spans the patterns we call constellations
# rearrange completely. Here are the 4,992 naked-eye Hipparcos stars again —
# the same catalog as the [Constellations](constellations.ipynb) star chart —
# with a clock you can wind ±100,000 years.
#
# Two details make it honest and make it pretty. Over that long a baseline a
# fast star sweeps more than 100° of sky, so a straight line in RA/Dec would
# fly off the top of the map; `sph.destination_point` walks each star along
# its **great circle** instead, at its own angular rate. And each star is
# tinted by its *real* perceived color — `sph.bv_to_rgb` turns the catalog's
# B–V index into an RGB, from blue-white hot stars to golden-red cool ones —
# and sized by brightness:

# %%
epoch_stars = pd.read_csv("../../examples/data/hipparcos_bright_pm.csv")
bv = epoch_stars["BV"].fillna(epoch_stars["BV"].median()).to_numpy()
ra0 = epoch_stars["RAICRS"].to_numpy()
dec0 = epoch_stars["DEICRS"].to_numpy()
rate = np.hypot(epoch_stars["pmRA"], epoch_stars["pmDE"]).to_numpy()   # mas/yr
bearing = np.degrees(np.arctan2(epoch_stars["pmRA"], epoch_stars["pmDE"]))


def rgb_str(rgb_rows):
    return ["rgb({:.0f},{:.0f},{:.0f})".format(*(c * 255 for c in row))
            for row in np.atleast_2d(rgb_rows)]


# saturation=1.0 gives the honest perceived colors; the package default of
# 0.55 is a soft-sky default that washes them ~45% toward white. Even at 1.0
# real star colors are genuinely pale — see the Constellations tutorial's §1
# for why (and why a chromaticity diagram oversells the stellar locus).
STAR_SAT = 1.0
star_color = rgb_str(sph.bv_to_rgb(bv, saturation=STAR_SAT))
star_size = list(np.clip((6.6 - epoch_stars["Vmag"].to_numpy()) ** 2 * 0.55,
                         2.0, 16.0))

# Hover data: *why* each star is the color it is. B–V → temperature via
# Ballesteros (2012) is the same first step `bv_to_rgb` takes internally, and
# the class is the bin that B–V falls in. All of it is epoch-independent, so it
# rides on the base trace once — the slider only ever restyles x/y, never this.
teff = 4600.0 * (1.0 / (0.92 * bv + 1.70) + 1.0 / (0.92 * bv + 0.62))
star_class = np.full(bv.shape, "M", dtype="<U3")
for edge, label in reversed([(0.00, "O/B"), (0.30, "A"), (0.58, "F"),
                             (0.81, "G"), (1.40, "K")]):
    star_class[bv < edge] = label
star_custom = np.column_stack([
    epoch_stars["HIP"].to_numpy().astype(int).astype(str),
    star_class,
    np.round(teff).astype(int).astype(str),
    np.round(bv, 2).astype(str),
    np.round(epoch_stars["Vmag"].to_numpy(), 1).astype(str),
])


def epoch_xy(t_yr):
    """Great-circle-propagated canvas coords for every star at year offset t."""
    dist = np.radians(rate * t_yr / 3.6e6)             # signed; +past/−future ok
    lon, lat = sph.destination_point(ra0, dec0, bearing.to_numpy(), dist)
    x, y = sphpl.project(np.asarray(lon), np.asarray(lat),
                         projection="AIT", center=180)
    return list(np.round(x, 1)), list(np.round(y, 1))


YEARS = np.linspace(-100_000, 100_000, 21)
fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark", width=900, height=520,
    title="The naked-eye sky over 200,000 years — wind the clock")
x0, y0 = epoch_xy(0.0)
fig.add_trace(go.Scattergl(                    # WebGL: 5,000 points stay smooth
    x=x0, y=y0, mode="markers", showlegend=False,
    customdata=star_custom,
    hovertemplate=("<b>HIP %{customdata[0]}</b><br>"
                   "class %{customdata[1]} · ≈%{customdata[2]} K<br>"
                   "B–V %{customdata[3]} · V %{customdata[4]}<extra></extra>"),
    marker=dict(size=star_size, color=star_color, line=dict(width=0))))
STARS = len(fig.data) - 1
sphpl.add_frame_edge(fig, color="#3A4258")

steps = [dict(method="restyle", label=f"{t/1000:+.0f} kyr",
              args=[{"x": [epoch_xy(t)[0]], "y": [epoch_xy(t)[1]]}, [STARS]])
         for t in YEARS]

# Two channels, two legend blocks — see the note below.
sun_rgb = rgb_str(sph.bv_to_rgb(0.65, saturation=STAR_SAT))[0]
temperature = {lab: rgb_str(sph.bv_to_rgb(v, saturation=STAR_SAT))[0] for lab, v in
               [("hot (O/B)", -0.25), ("white (A)", 0.1), ("Sun-like (G)", 0.65),
                ("cool (K)", 1.1), ("red (M)", 1.7)]}
sphpl.add_legend(fig, [
    sph.ColorBlock("Star color (B–V)", temperature, swatch="marker"),
    sph.SizeBlock("Brightness (V mag)", values=[6, 4, 2, 0],
                  size_map=(6.6, 0.0, 2.0, 16.0), color=sun_rgb),
])
fig.update_layout(
    xaxis=dict(showticklabels=False, zeroline=False),
    yaxis=dict(showticklabels=False, zeroline=False),
    margin=dict(r=150),                          # room for the key beside the oval
    # itemclick=False: these blocks are a *key*, not trace toggles — without
    # this they dim on click and nothing happens (see the note below).
    legend=dict(x=1.02, xanchor="left", y=0.5, yanchor="middle",
                font=dict(size=10), itemclick=False, itemdoubleclick=False),
    sliders=[dict(active=10, steps=steps, x=0.06, len=0.72, pad=dict(t=40),
                  currentvalue=dict(prefix="epoch: ", font=dict(size=13)))])
fig.show()

# %% [markdown]
# Wind it back and forth. Most stars only plod along slowly, but the nearby, fast ones —
# watch the big bright dots — streak clear across the sky; the whole
# distribution slowly reshuffles. The constellations of 100,000 years ago
# were not our constellations, and ours won't survive the next 100,000.
# **Hover** any star for the numbers behind its color — its B–V, the
# temperature that implies, and the spectral class it lands in. The epoch
# itself stays in the slider's readout rather than the tooltip: it's the same
# for every star at a given step, so repeating it in 5,000 tooltips would mean
# re-sending all 5,000 on *every* step, for something the slider already says.
#
# > **Note:** the star colors are *honest*. `bv_to_rgb` runs here at
# > `saturation=1.0`, which reproduces the canonical blackbody colors — real
# > stars are paler than the vivid palettes many atlases use, so a "red" M star
# > reads as a muted orange rather than fire-engine red. The
# > [Constellations](constellations.ipynb) tutorial's opener explains why a
# > chromaticity diagram makes the stellar locus look far more saturated than
# > any star really is.
# >
# > **Note:** this is constant-velocity proper motion propagated along great
# > circles — the standard deep-time simplification. It ignores the slow
# > *change* in a star's apparent motion as it approaches or recedes (its
# > distance and perspective drift over 100,000 years), which is a small
# > correction for most of these stars.
#
# ### A legend for two channels at once
#
# That figure encodes *two* variables — color for temperature, marker size
# for brightness — and a normal plotly legend can only key one trace at a
# time. `sphpl.add_legend` solves it: you hand it a list of **blocks**, one
# per visual channel, and it renders a compound key. The two above were a
# `ColorBlock` (a labeled swatch per category) and a `SizeBlock` (graduated
# dots, here mapped with the *same* magnitude→size formula as the stars, so
# the key is honest).
#
# > **Note:** a compound key is a *key*, not a control panel. Its swatches are
# > standalone entries — they aren't selection/deselection knobs on the star trace, so a click has
# > nothing to hide and plotly would simply dim the entry. That's why this
# > figure (and the opener's type key) pass `itemclick=False` and
# > `itemdoubleclick=False`, which turns the misleading machinery off. The
# > *clickable* kind of legend is the one in [section 3](#3.-Hover-data) and
# > [section 7](#7.-Constellations), where every entry is a real named trace
# > with data behind it.
#
# The block family is backend-agnostic — the identical block objects render
# on a matplotlib `MultiLegend` too — and covers the channels you actually
# encode with: `ColorBlock`, `SizeBlock`, `GlyphBlock` (marker shapes),
# `FillBlock` (filled vs. open), `LineBlock`, `RegionBlock`, `ColorbarBlock`,
# and more. When a single figure carries color *and* size *and* shape, this
# is how you explain it. The [Catalogs](catalogs.ipynb) tutorial builds a
# richer multi-channel key on a real source catalog.

# %% [markdown]
# ## 11. A drag-rotate globe
#
# The sph projections include orthographic globes
# (`make_figure(projection="SIN", ...)`), and everything above works on
# them. But for one thing there is no substitute: plotly's own *geo* engine
# renders an orthographic globe you can **grab and spin with the mouse**.
# It isn't part of the sph vocabulary — it's plotly's built-in geographic
# machinery, borrowed for the sky — so this section is plain plotly, with
# two idioms worth stealing.
#
# First idiom: geo globes are maps of the *outside* of a sphere, while the
# sky is seen from the *inside* — negate the longitude and east is back on
# the left where astronomers expect it. Second idiom: the globe's
# orientation is pure *layout* (`geo.projection.rotation`), so steering it
# from a slider costs almost nothing — each step is a two-line `relayout`,
# no precomputed data at all:

# %%
def sky2geo(ra):
    """Mirror sky longitudes into geo longitudes (east-left again)."""
    return ((-np.asarray(ra, dtype=float) + 180.0) % 360.0) - 180.0


gal_plane = SkyCoord(l=np.linspace(0, 360, 361) * u.deg, b=0 * u.deg,
                     frame="galactic").icrs

fig = go.Figure()
fig.add_trace(go.Scattergeo(
    lon=sky2geo(gal_plane.ra.deg), lat=gal_plane.dec.deg, mode="lines",
    line=dict(color="rgba(220,220,240,0.8)", width=2),
    hoverinfo="skip"))
fig.add_trace(go.Scattergeo(
    lon=sky2geo(messier["ra_deg"]), lat=messier["dec_deg"], mode="markers",
    marker=dict(size=6, color=C[2], opacity=0.9),
    customdata=np.stack([messier["name"],
                         messier["ra_deg"], messier["dec_deg"]], axis=-1),
    hovertemplate="<b>%{customdata[0]}</b><br>RA %{customdata[1]:.2f}°, "
                  "Dec %{customdata[2]:.2f}°<extra></extra>"))

fig.update_geos(
    projection_type="orthographic",
    projection_rotation=dict(lon=float(sky2geo(270)), lat=-30),
    showcoastlines=False, showland=False, showocean=False,
    showlakes=False, showrivers=False, showcountries=False, showsubunits=False,
    showframe=True, framecolor="#555",
    bgcolor="rgba(0,0,0,0)",
    lataxis=dict(showgrid=True, gridcolor="#2A3140", dtick=30),
    lonaxis=dict(showgrid=True, gridcolor="#2A3140", dtick=30),
)

lon_steps = [dict(method="relayout", label=f"{v}°",
                  args=[{"geo.projection.rotation.lon": float(sky2geo(v))}])
             for v in range(0, 360, 45)]
lat_steps = [dict(method="relayout", label=f"{v}°",
                  args=[{"geo.projection.rotation.lat": float(v)}])
             for v in range(-90, 91, 30)]
fig.update_layout(
    # Same paper color the sph dark theme uses, so this hand-built geo figure
    # sits flush with the rest of the page's figures.
    width=680, height=790, paper_bgcolor="#0a0a14",
    font=dict(color="#C8CCD4"), showlegend=False,
    title="Drag the globe — or steer it with the sliders",
    margin=dict(t=60, b=210),
    sliders=[
        dict(steps=lon_steps, active=6, x=0.12, len=0.76, y=0.145,
             currentvalue=dict(prefix="center RA: ", font=dict(size=12))),
        dict(steps=lat_steps, active=2, x=0.12, len=0.76, y=0.005,
             currentvalue=dict(prefix="center Dec: ", font=dict(size=12))),
    ],
)
fig.show()

# %% [markdown]
# Grab it. The Messier objects and the galactic plane ride along, and hover
# still reports true (un-mirrored) RA/Dec from the `customdata`. The
# rotation also has a `roll` — a third Euler angle — if you need a
# position-angle twist.
#
# > **Note:** the trade-off for the drag magic: this globe speaks
# > *geographic* conventions under the hood (that's why `sky2geo` exists),
# > there's no WCS, and none of the sph overlay helpers apply. For
# > publication-grade globes — far-side masking, planet textures, real tick
# > control — use the matplotlib globe builders in the
# > [Globe & Planet Plotting](globe_plots.ipynb) tutorial. This one is for
# > *spinning*.
#
# ### What a spinnable sphere is really for
#
# Here's a question that flat maps answer badly. Take four corners — two at
# Dec +60°, two at Dec +10°, spanning RA −60° to +60° — and join them. What
# *is* the edge between two corners? A line of constant declination, or the
# great circle (the true shortest path)? The
# [Regions](regions.ipynb) tutorial calls this choice `geodesic=`, and shows
# it on flat projections. But the honest answer lives in three dimensions,
# so let's put both polygons on a sphere you can turn:

# %%
def great_circle_lats(p0, p1, lons):
    """Latitude of the great circle through two points, sampled at `lons`."""
    def unit(lon, lat):
        lo, la = np.radians(lon), np.radians(lat)
        return np.array([np.cos(la) * np.cos(lo),
                         np.cos(la) * np.sin(lo), np.sin(la)])
    pole = np.cross(unit(*p0), unit(*p1))
    pole /= np.linalg.norm(pole)
    lo = np.radians(lons)
    return np.degrees(np.arctan2(
        -(pole[0] * np.cos(lo) + pole[1] * np.sin(lo)), pole[2]))


LO, HI, DEC_LO, DEC_HI = -60.0, 60.0, 10.0, 60.0
GRAT, GEOD = "#E8A33D", "#6FA8FF"
edge = np.linspace(LO, HI, 120)

# Same four corners; the top and bottom edges are drawn two ways.
grat_lon = np.concatenate([edge, edge[::-1]])
grat_lat = np.concatenate([np.full_like(edge, DEC_HI),
                           np.full_like(edge, DEC_LO)])
top_lon, top_lat = sph.great_circle_arc(LO, DEC_HI, HI, DEC_HI, 120)
bot_lon, bot_lat = sph.great_circle_arc(LO, DEC_LO, HI, DEC_LO, 120)
geod_lon = np.concatenate([top_lon, bot_lon[::-1]])
geod_lat = np.concatenate([top_lat, bot_lat[::-1]])

fig = go.Figure()

# Each edge, extended right around the sky: a parallel closes as a small
# circle near the pole; a great circle closes as a full tilted ring.
guides = np.linspace(-179.9, 179.9, 400)
for lat_c in (DEC_HI, DEC_LO):
    fig.add_trace(go.Scattergeo(
        lon=sky2geo(guides), lat=np.full_like(guides, lat_c), mode="lines",
        line=dict(color=GRAT, width=1, dash="dot"),
        hoverinfo="skip", showlegend=False))
for p0, p1 in [((LO, DEC_HI), (HI, DEC_HI)), ((LO, DEC_LO), (HI, DEC_LO))]:
    fig.add_trace(go.Scattergeo(
        lon=sky2geo(guides), lat=great_circle_lats(p0, p1, guides),
        mode="lines", line=dict(color=GEOD, width=1, dash="dot"),
        hoverinfo="skip", showlegend=False))

# Reverse the vertex order: mirroring the longitudes flips the polygon's
# winding, and a filled sphere polygon fills whichever side it walks around.
for lon_v, lat_v, color, fill, label in [
        (grat_lon, grat_lat, GRAT, "rgba(232,163,61,0.30)",
         "geodesic=False — constant Dec"),
        (geod_lon, geod_lat, GEOD, "rgba(111,168,255,0.30)",
         "geodesic=True — great circles")]:
    fig.add_trace(go.Scattergeo(
        lon=sky2geo(lon_v)[::-1], lat=np.asarray(lat_v)[::-1], mode="lines",
        fill="toself", fillcolor=fill, line=dict(color=color, width=2),
        name=label, hoverinfo="name"))

fig.add_trace(go.Scattergeo(
    lon=sky2geo([LO, HI, HI, LO]), lat=[DEC_HI, DEC_HI, DEC_LO, DEC_LO],
    mode="markers", marker=dict(size=8, color="#F2F2F2",
                                line=dict(color="#222", width=1)),
    name="the four corners", hoverinfo="name"))

fig.update_geos(
    projection_type="orthographic",
    projection_rotation=dict(lon=float(sky2geo(0)), lat=35),
    showcoastlines=False, showland=False, showocean=False,
    showlakes=False, showcountries=False,
    showframe=True, framecolor="#555", bgcolor="rgba(0,0,0,0)",
    lataxis=dict(showgrid=True, gridcolor="#2A3140", dtick=30),
    lonaxis=dict(showgrid=True, gridcolor="#2A3140", dtick=30))
fig.update_layout(
    width=660, height=700, paper_bgcolor="#0a0a14",
    font=dict(color="#C8CCD4"),
    title="The same four corners, two kinds of edge — spin it",
    margin=dict(t=60, b=110),
    legend=dict(x=0.5, xanchor="center", y=-0.01, yanchor="top",
                font=dict(size=11)))
fig.show()

# %% [markdown]
# Turn the globe until you are looking straight down on the quadrangle, and
# the two shapes nearly coincide — that's why the distinction never bothers
# you on a small field. Now bring it to the limb, or look down from the
# pole. The great-circle edges bow *poleward* of the parallels: the blue
# polygon claims a crescent of sky above the top edge that the orange one
# doesn't, and gives back a crescent at the bottom. Follow the dotted lines
# to see why. A parallel of constant declination closes into a small circle
# around the pole; a great circle closes into a full ring, tilted through
# the sphere's center. Only one of them is a straight line on a sphere.
#
# This is what `geodesic="auto"` protects you from: on `add_spherical_polygon`
# it switches to great-circle edges once an edge is longer than 10°, right
# about where the two start to visibly disagree.

# %% [markdown]
# ## 12. The FITS viewer
#
# `make_fits_figure()` + `add_fits_image()` put a FITS image on interactive
# sky axes with the full stretch/clip vocabulary from the
# [FITS Images & Quicklook](fits_images.ipynb) tutorial — and hover reads
# back coordinates and pixel values as you sweep the image.
#
# One thing to get right first. An interactive image *carries every pixel
# value with it*: the full 1024×1024 frame would be an ~80 MB figure,
# wonderful in a live session and hopeless in a web page. But almost all of
# that frame is empty sky. Cropping to the 25 mas that actually contain the
# source — with astropy's `Cutout2D`, which carries the WCS along — gives a
# figure a hundred times lighter *and* a better-framed picture:

# %%
# This stacked archival image stores DATE-OBS='MULTIEPOCH'; astropy notes
# the fix, harmlessly, on every load.
warnings.simplefilter("ignore", FITSFixedWarning)

hdu = fits.open("../../examples/data/0316+413.u.stacked.icd.fits")[0]
cutout = Cutout2D(np.squeeze(hdu.data), position=(512, 512), size=256,
                  wcs=WCS(hdu.header).celestial)
data, wcs = cutout.data, cutout.wcs
core_ra, core_dec = float(hdu.header["CRVAL1"]), float(hdu.header["CRVAL2"])
MAS = 1.0 / 3.6e6            # one milliarcsecond, in degrees


def fits_panel(coords, title, left_margin):
    fig = sphpl.make_fits_figure(wcs, theme="dark", width=560, height=560,
                                 title=title)
    sphpl.add_fits_image(fig, data, wcs,
                         coords=coords,           # 'offset' or 'absolute'
                         stretch="sqrt", clip="percentile", plo=5, phi=99.95,
                         colormap="sph.deepsky",  # bundled maps work by name
                         colorbar=True, header=hdu.header,   # beam from header
                         hover="value", display_factor=1e3, bunit="mJy/beam")
    fig.update_layout(margin=dict(l=left_margin, r=10, t=50, b=80))
    return lighten_heatmap(fig)


fits_panel("offset", "coords='offset' — mas from the reference pixel", 70).show()

# %% [markdown]
# Hover anywhere on the jet: you get the offset position *and* the
# brightness in mJy/beam (`display_factor=1e3` rescaled the Jy/beam data,
# `bunit=` named the result). `coords="offset"` relabels the axes as offsets
# from the reference pixel — `offset_units='auto'` picks mas here — which is
# how a VLBI map should read. The restoring beam in the corner came free
# with `header=`.
#
# The alternative is `coords="absolute"`, which labels the true RA and Dec.
# On a 25 mas field, watch what that costs:

# %%
fits_panel("absolute", "coords='absolute' — the same image, true RA / Dec",
           140).show()

# %% [markdown]
# Seven decimal places of degrees, and every tick label nearly identical —
# the axes are technically correct and practically useless. That contrast
# *is* the reason `coords="offset"` exists. On a wide-field image (a survey
# tile, a galaxy) absolute coordinates are exactly what you want; at VLBI
# scales, offsets are.
#
# ### Layering onto the image
#
# Everything else in the toolkit reaches the FITS axes through the WCS.
# `add_fits_scatter()` takes ordinary sky degrees and lands on the right
# pixels — in `mode="lines"` it also draws polylines, which is all a contour
# really is. `make_fits_compound_region()` gives the same set algebra from
# section 5 on image axes, and `add_ruler()` measures true angles across
# the frame:

# %%
# Contours: matplotlib finds the vertices in pixel space; the WCS carries
# them to the sky, and add_fits_scatter draws them back onto the image.
levels = 0.02 * 2.0 ** np.arange(0, 7.0)          # 20 mJy/beam, doubling
cs = plt.contour(np.arange(data.shape[1]), np.arange(data.shape[0]),
                 data, levels=levels)
clon, clat = [], []
for segments in cs.allsegs:
    for seg in segments:                           # NaN separates the pieces
        sky = wcs.pixel_to_world(seg[:, 0], seg[:, 1])
        clon.extend(sky.ra.deg.tolist() + [np.nan])
        clat.extend(sky.dec.deg.tolist() + [np.nan])
plt.close("all")

fig = sphpl.make_fits_figure(
    wcs, theme="dark", width=740, height=700,
    title="3C 84 — contours, an annulus region, jet knots, and a ruler")
sphpl.add_fits_image(fig, data, wcs, coords="offset",
                     stretch="sqrt", clip="percentile", plo=5, phi=99.95,
                     colormap="sph.deepsky", colorbar=True, header=hdu.header,
                     beam_corner="lower right",
                     hover="value", display_factor=1e3, bunit="mJy/beam")

sphpl.add_fits_scatter(fig, clon, clat, mode="lines",
                       line=dict(color="rgba(255,255,255,0.55)", width=0.6),
                       hoverinfo="skip", name="contours")

annulus = (sphpl.make_fits_compound_region(fig)
           .add_circle(core_ra, core_dec, 4 * MAS)
           .subtract_circle(core_ra, core_dec, 1.5 * MAS))
sphpl.add_compound_region(fig, annulus, color="#5FE3D8", width=1.2,
                          fillcolor="rgba(95,227,216,0.10)", opacity=1,
                          name="1.5–4 mas annulus", hover=True)

sphpl.add_fits_scatter(fig, [core_ra] * 3,
                       [core_dec - d * MAS for d in (3.0, 4.5, 6.0)],
                       marker=dict(color="#9DFF8A", size=9, symbol="x",
                                   line=dict(width=1)),
                       name="jet knots")

sphpl.add_ruler(fig, core_ra + 9 * MAS, core_dec + 6 * MAS,
                core_ra + 9 * MAS, core_dec - 6 * MAS,
                geodesic=True, label_unit="mas", tick_interval=4.0,
                tick_side="left", label_side="left", label_rotation=0,
                color="gold", label_fontsize=9)

fig.update_layout(margin=dict(l=70, r=10, t=50, b=70), showlegend=True,
                  legend=dict(x=0.01, y=0.01, xanchor="left",
                              yanchor="bottom", bgcolor="rgba(10,10,20,0.7)",
                              font=dict(size=10)))
lighten_heatmap(fig).show()

# %% [markdown]
# Click "contours" in the legend to strip them off and look at the image
# underneath; hover the annulus to confirm it's a real region object (the
# same one `contains_points()` would answer with). Two more helpers —
# `fits_ticks_for_range()` and `beam_shape_for_range()` — recompute the
# ticks and the beam ellipse for any zoom window; they're the building
# blocks if you ever wire up your own live pan/zoom behavior.
#
# > **Note:** need a static image instead? Any figure will render itself to
# > a PNG with `fig.show(renderer="png")`, which is the honest choice for a
# > full-resolution image you don't want to ship pixel-by-pixel. The
# > `lighten_heatmap()` helper used above is the other half of the story:
# > it rounds the image and hover arrays to a sensible number of decimals,
# > which alone cuts the figure's weight by more than half.
#
# ### The ready-made viewer app
#
# All of that is packaged as a one-call [Dash](https://dash.plotly.com/)
# app — pan, zoom, and inspect a FITS image in the browser with the WCS
# ticks recomputing as you move. It needs a live Python process, so it
# can't run inside this page; on your machine it's three lines
# (`pip install skyplothelper[plotly]` plus the `dash` package):
#
# ```python
# from skyplothelper.plotly.dash_fits import fits_viewer_app
#
# app = fits_viewer_app(data, wcs, coords="offset", stretch="sqrt",
#                       colormap="sph.deepsky", theme="dark",
#                       header=hdu.header)
# app.run(debug=True)        # → http://127.0.0.1:8050
# ```
#
# The underlying callback (`register_fits_relayout`) is public too, so you
# can graft the same live-retick behavior onto a bigger Dash dashboard of
# your own.

# %% [markdown]
# ## 13. A spectral-cube viewer
#
# A spectral cube is a stack of images — one per velocity (or frequency)
# channel. The static way to read it is a grid of channel panels (the
# `channel_map` helper in the [FITS Images](fits_images.ipynb) tutorial); the
# interactive way is to put the channels on a slider and drag through them,
# watching the emission move from one velocity to the next.
#
# `sph.DataCube` does the cube plumbing: it loads and squeezes the FITS,
# splits off the celestial and spectral WCS, labels each channel's velocity,
# and — the part a responsive slider needs — thins the cube on demand. Here
# it downsamples DDO 70's HI cube spatially by 2 and bins every 2 velocity
# channels (which also lifts the per-channel signal-to-noise), then hands each
# channel to `add_fits_image`. One shared `vlimits()` stretch keeps the
# brightness scale fixed across every frame:

# %%
cube = (sph.DataCube.from_fits("../../examples/data/ddo70_hi_subcube.fits")
        .spatial_downsample(2)      # snappier slider; the WCS stays registered
        .spectral_bin(2))           # half the channels, better S/N per channel
vmin, vmax = cube.vlimits(plo=1.0, phi=99.7)      # one stretch for all channels


def channel_display(i):
    """The asinh-stretched display array for channel i (shared vmin/vmax)."""
    tmp = sphpl.make_fits_figure(cube.celestial_wcs)
    sphpl.add_fits_image(tmp, cube.channel(i), cube.celestial_wcs,
                         stretch="asinh", vmin=vmin, vmax=vmax,
                         colormap="sph.dusk", hover=False)
    return np.round(np.asarray(tmp.data[-1].z, dtype=float), 2).tolist()


start = cube.nchan // 2
fig = sphpl.make_fits_figure(
    cube.celestial_wcs, theme="dark", width=560, height=580,
    title="DDO 70 — HI line cube; drag the velocity slider")
sphpl.add_fits_image(fig, cube.channel(start), cube.celestial_wcs,
                     stretch="asinh", vmin=vmin, vmax=vmax,
                     colormap="sph.dusk", colorbar=True, bunit="Jy/beam",
                     hover="value")
HEAT = len(fig.data) - 1
fig.data[HEAT].z = channel_display(start)         # match the stepped (rounded) z

steps = [dict(method="restyle", label=cube.spectral_label(i),
              args=[{"z": [channel_display(i)]}, [HEAT]])
         for i in range(cube.nchan)]
fig.update_layout(
    sliders=[dict(active=start, steps=steps, x=0.06, len=0.9, pad=dict(t=40),
                  currentvalue=dict(prefix="v = ", font=dict(size=13)))])
fig.show()

# %% [markdown]
# Drag from 330 down to 278 km/s and watch the bright gas sweep across the
# galaxy — the disk's rotation, one edge approaching and the other receding.
# Because every channel is stretched with the *same* `vlimits()`, the
# brightness scale holds still while you drag; only the gas moves. The
# static, publication-ready cousin of this figure is the multi-panel
# `channel_map` in the [FITS Images & Quicklook](fits_images.ipynb) tutorial.
#
# > **Note:** a channel slider carries *every channel's pixels* in the figure,
# > so — like the FITS image in section 12 — it pays to shrink the cube first.
# > `DataCube.spatial_downsample()` and `.spectral_bin()` are the two dials;
# > here they take a 43 × 225 × 225 cube down to a ~1.6 MB figure. `DataCube`
# > carries the rest of the cube vocabulary too — moment maps, smoothing,
# > unit-aware spectral labels — shared with the matplotlib `channel_map`.

# %% [markdown]
# ## 14. Sharing and export
#
# Interactive figures are for *exploring* — and the exploration travels: hand
# one to a collaborator and they can pan, zoom, and hover it themselves. The
# quickest export needs no code at all: every plotly figure has a toolbar
# (hover over its top-right corner) whose camera button saves the *current
# view* as a PNG —
# zoom into a detail first and you've made a cropped figure. For anything
# more durable, the workhorse is HTML export — one self-contained file that
# opens in any browser, hover and zoom intact, no Python required at the
# other end:

# %%
out = fig  # the FITS figure from above — any figure works the same way

# Fully self-contained (~4 MB floor: plotly.js rides along in the file):
# out.write_html("skymap.html")

# Much smaller file that loads plotly.js from the CDN when opened online:
# out.write_html("skymap.html", include_plotlyjs="cdn")

# Static raster/vector export (PNG/PDF/SVG) via the kaleido engine:
# out.write_image("skymap.png", scale=2)
print(f"This figure's data: {len(out.to_json()) / 1e6:.2f} MB")
print("HTML with plotly.js inlined:  that, plus a ~4 MB one-time floor")
print("HTML with include_plotlyjs='cdn':  just that")

# %% [markdown]
# Rules of thumb for keeping shared figures light, all of which you've now
# seen in action:
#
# | what | cost | lighter alternative |
# |---|---|---|
# | scatter / lines / labels | tiny (KBs) | — |
# | region + band fills | small | lower `resolution=` |
# | HEALPix tiles | ~0.2–0.6 KB *per tile* | lower `nside`, `add_healpix_sparse`, lower `tile_resolution=` |
# | full IAU constellation boundaries | ~2.2 MB, or 0.9 MB rounded | `round_traces`, or `add_constellation_lines` |
# | FITS images | ~40 bytes *per pixel* | crop first (`Cutout2D`), `lighten_heatmap`, or `max_pixels=` |
# | slider/menu states | one stored data set per state | fewer steps, rounded coordinates |
#
# The two rounding helpers from the setup cell — `lighten_heatmap()` and
# `round_traces()` — are the cheapest wins on that list: a sky map's
# coordinates are meaningless past a couple of decimal places, and an
# image's display values past three. Together they roughly halve a heavy
# figure with no visible change at all.
#
# And for this documentation itself: every live figure on this page is just
# the committed notebook output — `pio.renderers.default =
# "notebook_connected"` embeds each figure's JSON with a CDN loader, so the
# page stays interactive with no kernel behind it. That's the pattern to
# copy for your own rendered notebooks (GitHub, nbviewer, Sphinx, Quarto
# alike).

# %% [markdown]
# ## 15. Putting it together
#
# ### Everything at once
#
# The overlay families compose. Here is most of this tutorial's vocabulary
# on a single map — asterisms for context, two reference planes with their
# parallels, a compound region, a survey polygon, a geodesic circle, a
# reticle, a geodesic ruler, a catalog, coordinate labels, and a frame edge.
# Hover any of it; peel layers off with the legend; zoom wherever you like.
# It all rides in about 0.3 MB:

# %%
fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark", width=980, height=600,
    title="One figure, most of the vocabulary — hover anything, zoom anywhere")

i = len(fig.data)
sphpl.add_constellation_lines(fig, rank_max=1, color="#39415A", width=0.8)
legend_layer(fig, i, "Asterisms")

i = len(fig.data)
sphpl.add_plane_overlay(fig, plane="galactic", color="#FFD37A", width=2.0,
                        parallels=[-30, 30], parallel_opacity=0.45, hover=True)
legend_layer(fig, i, "Galactic plane ±30°", hover=True)

i = len(fig.data)
sphpl.add_plane_overlay(fig, plane="ecliptic", color="#E9807A", width=1.6,
                        parallels=[-23.4, 23.4], parallel_opacity=0.45,
                        hover=True)
legend_layer(fig, i, "Ecliptic + tropics", hover=True)

i = len(fig.data)
zoa = (sphpl.make_compound_region(fig)
       .add_frame_band(-10, 10, frame="galactic")
       .subtract_circle(266.417, -28.936, 12))
sphpl.add_compound_region(fig, zoa, color="#8FB3F0",
                          fillcolor="rgba(140,180,255,0.20)", opacity=1,
                          name="Zone of Avoidance", hover=True)
legend_layer(fig, i, "Zone of Avoidance (GC excised)", hover=True)

i = len(fig.data)
sphpl.add_spherical_polygon(fig, lons=[100, 160, 160, 100],
                            lats=[20, 20, 50, 50], color=C[0],
                            fillcolor="rgba(120,220,180,0.22)", opacity=1,
                            name="Survey footprint", hover=True)
legend_layer(fig, i, "Survey footprint", hover=True)

i = len(fig.data)
sphpl.add_geodesic_circle(fig, 83.82, -5.39, 8.0, fill=True,
                          fillcolor="rgba(255,140,90,0.28)", color="#FF9E6B",
                          width=1.6, opacity=1, name="8° around M42",
                          hover=True)
legend_layer(fig, i, "8° around M42", hover=True)

i = len(fig.data)
sphpl.add_scatter(fig, messier["ra_deg"], messier["dec_deg"],
                  text=messier["name"], name="Messier objects",
                  hovertemplate="<b>%{text}</b><br>RA %{customdata[0]:.2f}°, "
                                "Dec %{customdata[1]:.2f}°<extra></extra>",
                  marker=dict(size=6, color=C[2], opacity=0.95))
fig.data[i].update(legendgroup="Messier objects")

sphpl.add_reticle(fig, 83.82, -5.39, style="circle", size=13,
                  color="#FFE9B0", label="M42")
sphpl.add_ruler(fig, 201.3, -43.0, 148.9, 69.1, geodesic=True, n_ticks=4,
                endcap_style="arrow", labels=False, title="Cen A → M82",
                color=PAL["label"], title_fontsize=10)
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30,
                       lat_exterior=True, color=PAL["label"])
sphpl.add_frame_edge(fig, color=PAL["grid"])
fig.update_layout(showlegend=True,
                  legend=dict(x=0.5, xanchor="center", y=-0.02, yanchor="top",
                              orientation="h", font=dict(size=10),
                              groupclick="togglegroup"))
round_traces(fig)
fig.show()

# %% [markdown]
# Everything you see is seam- and pole-aware, computed on the sphere and
# projected once. The ruler measures a true great-circle angle from
# Centaurus A to M82; the Zone of Avoidance is a real set-algebra region
# with a hole in it; the Messier points know their own names. The elements in the legend are clickable.
#
# ### A coverage report on your own data
#
# The other kind of capstone is the figure you'd actually send to a
# collaborator: *"here's our survey's footprint, and here's every catalog
# object we cover — hover for details."* It marries the compute side (a
# set-algebra footprint + `contains_points` membership) with the interactive
# side (rich hover, context overlays, one-file export):

# %%
fig = sphpl.make_figure(
    projection="AIT", center=180, theme="dark",
    width=860, height=500,
    title="Extragalactic survey coverage — Messier objects, hover for details",
)

# The footprint: a northern cap plus an equatorial strip, avoiding the
# galactic plane, with one bright-star mask punched out.
footprint = (sphpl.make_compound_region(fig)
             .add_latitude_band(30, 75)                          # Dec cap
             .add_lonlat_box(-10, 15, 120, 250, frame="icrs")    # RA strip
             .subtract_frame_band(-15, 15, frame="galactic")
             .subtract_circle(279.2, 38.8, 6))                   # Vega mask
sphpl.add_compound_region(fig, footprint,
                          color=PAL["accent"],
                          fillcolor="rgba(140,180,255,0.22)", opacity=1,
                          name="Survey footprint", hover=True)

# Context: where the Milky Way cuts through.
sphpl.add_plane_overlay(fig, plane="galactic",
                        color=PAL["grid"], width=1.5, hover=True)

# The catalog, split by membership — the same region object does the math.
covered = footprint.contains_points(messier["ra_deg"], messier["dec_deg"])
for sel, color, size, label in [(covered, C[2], 8, "covered"),
                                (~covered, "#5A6070", 5, "not covered")]:
    sphpl.add_scatter(
        fig, messier["ra_deg"][sel], messier["dec_deg"][sel],
        name=label,
        customdata=custom[sel],
        hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                       "V = %{customdata[2]}<br>"
                       "RA %{customdata[3]:.2f}°, Dec %{customdata[4]:.2f}°"
                       "<extra>" + label + "</extra>"),
        marker=dict(size=size, color=color, opacity=0.95),
    )

sphpl.add_frame_edge(fig, color=PAL["grid"])
sphpl.add_coord_labels(fig, lon_spacing=60, lat_spacing=30)
fig.show()

print(f"Footprint covers {footprint.area_frac:.1%} of the sky, "
      f"{covered.sum()} of {len(messier)} Messier objects.")

# One self-contained file to send — hover, zoom, and all:
# fig.write_html("survey_coverage.html", include_plotlyjs="cdn")

# %% [markdown]
# Swap in your own catalog columns and your own footprint verbs, and this
# is a shareable, explorable coverage report in ~30 lines — the compute
# pipeline and the query methods are the same `CompoundRegion` machinery
# you'd use on a matplotlib figure, so nothing about your analysis changes
# when the output medium does.
#
# ## 16. Where to go next
#
# - [Regions & Spherical Polygons](regions.ipynb) — the full compound-region
#   verb families and query methods used in §5 and the capstone.
# - [Vector Fields & Sky Kinematics](vector_fields.ipynb) — what VSH modes
#   mean, and fitting them to real proper motions.
# - [HEALPix Workflows](healpix_workflows.ipynb) — binning, smoothing, and
#   high-resolution HEALPix imaging on the matplotlib side.
# - [FITS Images & Quicklook](fits_images.ipynb) — the stretch/clip
#   vocabulary that `add_fits_image` shares.
# - [Globe & Planet Plotting](globe_plots.ipynb) — publication-grade globes
#   with far-side masking and textures.
# - [Animations](animations.ipynb) — when the third axis is time rather
#   than a mouse.
# - The [plotly user guide page](../guide/plotly.md) — the API map this
#   tutorial toured.
