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
# # Regions & Spherical Polygons
#
# A region is a *piece of sky*: the area a survey covers, the zone you avoid near
# the galactic plane, the cap around a target, the footprint you test a catalog
# against. skyplothelper draws these as **projection-aware** patches — edges follow
# the curved geometry of the sphere, boundaries that cross the map's seam are
# clipped and closed correctly (even when the region wraps a pole), and the *same*
# region can be queried: which of your sources fall inside, and how much sky it
# covers.
#
# This tutorial works through the region system end to end. We start with the
# **three layers** of the system, draw every **simple shape** and band, learn the
# **shared keywords** that steer all of them, then climb to **set algebra** —
# combining shapes into the kind of masks real surveys use — and finally *query*
# those masks for **membership and area**. Throughout, two questions drive every
# section: **"how do I draw this piece of sky?"** and **"how do I adjust it?"**
#
# By the capstone you will build a survey footprint from scratch, test a real
# catalog against it, quote its sky coverage, and check it against a footprint the
# package already ships — the whole workflow in about a dozen lines.
#
# ## Contents
#
# 1. [The three layers](#1.-The-three-layers)
# 2. [Simple regions](#2.-Simple-regions)
# 3. [The shared keywords](#3.-The-shared-keywords)
# 4. [Compound set algebra](#4.-Compound-set-algebra)
# 5. [Membership and area](#5.-Membership-and-area)
# 6. [Planes and Tissot indicatrices](#6.-Planes-and-Tissot-indicatrices)
# 7. [Putting it together](#7.-Putting-it-together)
# 8. [Where to go next](#8.-Where-to-go-next)
#
# > **Note:** The region set algebra runs on [shapely](https://shapely.readthedocs.io/),
# > which ships as a core dependency of skyplothelper — everything in this tutorial
# > works out of the box, no optional extras to install.

# %%
import astropy.units as u
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection used in §2)
import numpy as np
from astropy.coordinates import SkyCoord, get_sun
from astropy.table import Table
from astropy.time import Time
from matplotlib.colors import to_rgb

import skyplothelper as sph

# base='structural' tightens only the frame/tick *geometry* — it leaves colors and
# fonts to the docs light/dark theme, so it composes with the dark-figure pass
# (which sets a theme on top). We avoid base='standard', which would reset the theme.
sph.set_style(base="structural")


# Decoration adapts to the docs light/dark passes: read the active page color off the
# figure background and pick the matching annotation palette (accent/grid/label/...).
def annotation_palette():
    r, g, b = to_rgb(mpl.rcParams["figure.facecolor"])
    return sph.ANNOTATION_PALETTES["dark" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5
                                  else "publication"]


# Region fills are pulled from one of sph's built-in *cycle* palettes. `uranometria`
# (mode 'dual') was designed for exactly this job — its muted tones read on both
# light and dark pages and hold up under the transparency region fills want. We also
# drive matplotlib's color cycle with it so any auto-colored series match.
sph.set_palette("uranometria")
_CYC = sph.CYCLE_PALETTES["uranometria"]["colors"]
RC = {  # descriptive handles into the cycle (all dual-mode, transparency-safe)
    "blue":  _CYC[0],   # #46618A
    "tan":   _CYC[1],   # #B97C52
    "gold":  _CYC[2],   # #C29B3C
    "gray":  _CYC[3],   # #9DA3AB
    "green": _CYC[4],   # #5E8C7E
    "rust":  _CYC[5],   # #8A4540
    "mauve": _CYC[6],   # #716A8E
}

# Theme-adaptive accent/label colors for catalog points and annotations.
PAL = annotation_palette()


# An orthographic (SIN) globe with a *visible* graticule — the lon/lat lines are what
# give a globe its 3-D read, so we lift them above the very faint default.
def globe(subplot, clon=0, clat=10):
    return sph.make_globe_frame(subplot, center_LONdeg=clon, center_LATdeg=clat,
                                gridcolor=RC["gray"], gridalpha=0.55)


def great_circle_lats(p0, p1, lons):
    """Latitudes of the great circle through two (lon, lat) points, sampled at lons."""
    def _v(lon, lat):
        lo, la = np.radians(lon), np.radians(lat)
        return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    pole = np.cross(_v(*p0), _v(*p1))
    pole /= np.linalg.norm(pole)
    lo = np.radians(lons)
    return np.degrees(np.arctan2(-(pole[0] * np.cos(lo) + pole[1] * np.sin(lo)), pole[2]))


def draw_guide(ax, lons, lats, color, view_center=None, lw=1.0):
    """Dashed guide curve in world coords. With a view_center, the hemisphere facing
    *away* (the 'back of the sphere') is drawn fainter and dotted, the near side solid-
    dashed — so on a flat map you can still tell which arc would be hidden on a globe."""
    tr = ax.get_transform("world")
    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    if view_center is None:
        ax.plot(lons, lats, transform=tr, color=color, ls="--", lw=lw, alpha=0.9, zorder=2)
        return
    clon, clat = np.radians(view_center)
    la, lo = np.radians(lats), np.radians(lons)
    front = (np.sin(clat) * np.sin(la)
             + np.cos(clat) * np.cos(la) * np.cos(lo - clon)) >= 0
    ax.plot(np.where(front, lons, np.nan), lats, transform=tr,
            color=color, ls="--", lw=lw, alpha=0.9, zorder=2)
    ax.plot(np.where(~front, lons, np.nan), lats, transform=tr,
            color=color, ls=":", lw=lw * 0.9, alpha=0.30, zorder=2)

# %% [markdown]
# Two real catalogs ride along for the membership examples later: the 110 **Messier**
# objects (a northern-biased deep-sky list) and the 303 **ICRF3 defining** radio
# sources (an all-sky astrometric grid). We load them once.

# %%
messier = Table.read("../../examples/data/messier.csv", format="ascii.csv")
icrf3 = Table.read("../../examples/data/icrf3_defining.csv", format="ascii.csv")
print(f"Messier: {len(messier)} objects   ICRF3: {len(icrf3)} sources")

# %% [markdown]
# Two things make sky regions trickier than drawing shapes on a flat plot — and the
# opener previews both. **On the left, edges curve.** The same four corners near the
# pole bound very different areas depending on whether the edges follow lines of
# constant declination (a *graticule* box) or the true shortest path on the sphere
# (a *great circle*) — near the pole the gap is enormous. **On the right, real
# footprints are set algebra:** a survey box, minus the galactic plane, minus a hole,
# carrying a real catalog. By the end of this tutorial you can build and query that
# mask in a dozen lines. Everything below unpacks these two ideas.

# %%
# fig-slug: overview
fig = plt.figure(figsize=(12, 5.4))

# (a) Near-pole quadrangle — graticule edges vs great-circle edges, seen pole-on.
ax = globe(121, clon=0, clat=90)
qlon, qlat = [-70, 70, 70, -70], [45, 45, 78, 78]
sph.add_spherical_polygon(ax, lons=qlon, lats=qlat, geodesic=False,
                          facecolor="none", edgecolor=RC["rust"], lw=2.4,
                          label="constant-Dec edges")
sph.add_spherical_polygon(ax, lons=qlon, lats=qlat, geodesic=True,
                          facecolor=RC["blue"], alpha=0.40, edgecolor=RC["blue"],
                          lw=1.6, label="great-circle edges")
# Extend both top/bottom edges as thin dashed guides: constant-Dec parallels (rust)
# and great circles (blue). The globe hides whatever wraps to the back hemisphere.
_full = np.linspace(0, 360, 240)
for lat_edge in (45, 78):
    draw_guide(ax, _full, np.full_like(_full, lat_edge), RC["rust"], lw=0.7)
_span = np.linspace(-179, 179, 240)
for p0, p1 in [((-70, 45), (70, 45)), ((-70, 78), (70, 78))]:
    draw_guide(ax, _span, great_circle_lats(p0, p1, _span), RC["blue"], lw=0.7)
# A few unobtrusive in-frame labels: lon round a near-edge parallel, lat up a meridian
# well away from the quadrangle.
sph.add_curved_lon_ticks(ax, tick_lat=18, lon_spacing=60, fontsize=7, color=RC["gray"])
sph.add_overlay_ticks(ax, lon_vals=[], lat_vals=[30, 60], lat_at="lon=120",
                      label_kwargs=dict(fontsize=7, color=RC["gray"]))
ax.legend(loc="lower center", fontsize=8, framealpha=0.85)
ax.set_title("Edges curve on the sphere", fontsize=11)

# (b) An obvious set-algebra region carrying a catalog — where we're headed.
ax = sph.make_wcs_frame(122, "AIT", center=180, fig=fig)
teaser = (sph.CompoundRegion(ax)
          .add_circle(150, 12, radius_deg=34)         # cap A  ┐ symmetric
          .xor_circle(210, 12, radius_deg=34))        # cap B  ┘ difference
teaser.render(facecolor=RC["blue"], alpha=0.30)
teaser.render_boundary(color=RC["blue"], linewidth=1.3)
mra0, mdec0 = np.asarray(messier["ra_deg"]), np.asarray(messier["dec_deg"])
hit = teaser.contains_points(mra0, mdec0)
ax.scatter(mra0[hit], mdec0[hit], transform=ax.get_transform("world"),
           s=16, c=PAL["accent"], zorder=6)
ax.set_title("Footprints are set algebra", fontsize=11)

fig.suptitle("Regions & spherical polygons", y=1.0, fontsize=13)
fig.tight_layout()

# %% [markdown]
# ## 1. The three layers
#
# The region system is three layers stacked from raw geometry up to set algebra.
# You can interact at whichever level matches what you need:
#
# 1. **Vertex constructors** (`geodesic_circle`, `rectangle`, `ellipse`) compute the
#    boundary `(lon, lat)` and hand it back — *no drawing*. Reach for these when you
#    want the raw outline for your own machinery (write it to a file, feed another
#    library, compute something).
# 2. **Renderers** (the `add_*` family) draw a *single* region onto an axes in one
#    call — projection, seam handling, and fill all taken care of.
# 3. **`CompoundRegion`** combines shapes with set algebra — union, difference,
#    intersection, symmetric difference — and renders the result as one patch, holes
#    included. This is the layer for things like "inside survey A but outside the galactic plane."
#
# The same circle at all three layers:

# %%
fig = plt.figure(figsize=(13, 4.2))

# --- Layer 1: vertex constructor — coordinates only, you do the drawing ---
ax1 = sph.make_wcs_frame(131, "AIT", center=180, fig=fig)
lons, lats = sph.geodesic_circle(180, 0, radius_deg=30)        # -> two arrays
ax1.plot(lons, lats, transform=ax1.get_transform("world"),
         color=RC["rust"], lw=2)
ax1.set_title("Layer 1 — vertices\ngeodesic_circle() → (lon, lat)", fontsize=9)

# --- Layer 2: one-call renderer — a filled, seam-aware patch ---
ax2 = sph.make_wcs_frame(132, "AIT", center=180, fig=fig)
sph.add_geodesic_circle(ax2, 180, 0, radius_deg=30,
                        facecolor=RC["green"], edgecolor=RC["blue"], alpha=0.5)
ax2.set_title("Layer 2 — renderer\nadd_geodesic_circle()", fontsize=9)

# --- Layer 3: set algebra — a cap with a hole punched out ---
ax3 = sph.make_wcs_frame(133, "AIT", center=180, fig=fig)
region = (sph.CompoundRegion(ax3)
          .add_circle(180, 0, radius_deg=30)
          .subtract_circle(180, 0, radius_deg=12))
region.render(facecolor=RC["gold"], alpha=0.6)
region.render_boundary(color=RC["tan"], linewidth=1.5)
ax3.set_title("Layer 3 — set algebra\nCompoundRegion: cap − hole", fontsize=9)

fig.suptitle("One circle, three layers", y=1.02)
fig.tight_layout()

# %% [markdown]
# Layer 1 gives you arrays and nothing else — note we drew them ourselves with
# `ax.plot(..., transform=ax.get_transform("world"))`. Layer 2 is the standard
# entry point to simple regions: one call, a filled patch, the seam handled. Layer 3 adds the verbs
# that turn shapes into *masks*. The rest of this tutorial climbs that stack.
#
# > **Note:** Every renderer takes world coordinates in the frame's native system —
# > RA/Dec on an equatorial map, galactic longitude and latitude on a galactic one.
# > Hand it a `SkyCoord` in *any* frame and it converts automatically (a galactic
# > `SkyCoord` lands correctly on an equatorial map), and angular sizes accept
# > astropy `Quantity` values (`15 * u.deg`, `900 * u.arcmin`) as well as plain
# > degrees. The trajectory example in §5 uses both.

# %% [markdown]
# ## 2. Simple regions
#
# The `add_*` family covers the common shapes. Each one takes a center
# (or a set of vertices), angular sizes **on the sky** — not RA intervals, so the
# `cos(Dec)` stretch near the poles is handled for you — and any matplotlib patch
# keyword (`facecolor`, `edgecolor`, `alpha`, `hatch`, `lw`, `zorder`) can be passed.
#
# ### Centered shapes
#
# Six shape helpers, shown here on orthographic globes (where curvature is easiest
# to read). Sizes are angular distances; `angle=` is a position angle measured from
# North through East, the standard astronomical convention.

# %%
fig = plt.figure(figsize=(12, 7.5))
specs = [  # (title, draw-fn)
    ("geodesic circle\nadd_geodesic_circle(r=25°)",
     lambda ax: sph.add_geodesic_circle(ax, 0, 0, radius_deg=25,
                                        facecolor=RC["green"], alpha=0.55)),
    ("rectangle + square\nadd_rectangle / add_square",
     lambda ax: (sph.add_rectangle(ax, 0, 0, width=50, height=28, angle=20,
                                   facecolor=RC["rust"], alpha=0.5),
                 sph.add_square(ax, 0, 0, size=16, facecolor=RC["tan"], alpha=0.7))),
    ("ellipse (PA = 35°)\nadd_ellipse",
     lambda ax: sph.add_ellipse(ax, 0, 0, semi_major=30, semi_minor=14, angle=35,
                                facecolor=RC["gold"], alpha=0.6)),
    ("annulus\nadd_annulus",
     lambda ax: sph.add_annulus(ax, 0, 0, inner_radius=14, outer_radius=28,
                                facecolor=RC["blue"], alpha=0.6)),
    ("spherical polygon\nadd_spherical_polygon",
     lambda ax: sph.add_spherical_polygon(
         ax, lons=[-30, 25, 35, -5, -35], lats=[-25, -28, 12, 30, 8],
         facecolor=RC["tan"], edgecolor=RC["rust"], alpha=0.6)),
    ("lon/lat box\nadd_lonlat_box",
     lambda ax: sph.add_lonlat_box(ax, lat_min=-20, lat_max=20,
                                   lon_min=-30, lon_max=30, frame="icrs",
                                   facecolor=RC["mauve"], alpha=0.5)),
]
for i, (title, draw) in enumerate(specs, start=1):
    ax = globe(231 + (i - 1), clon=0, clat=10)
    draw(ax)
    ax.set_title(title, fontsize=9)
fig.suptitle("The centered-shape helpers", y=1.0)
fig.tight_layout()

# %% [markdown]
# `add_square` is just `add_rectangle` with one size, and `add_lonlat_box` is the
# special case whose edges follow constant lon/lat **graticule lines** rather than
# great circles — useful if a footprint is defined in constant RA/Dec limits instead of spherical geometry arcs
# (more on that distinction below).
#
# > **Note:** unlike the rest of the family, `add_lonlat_box` — and the matching
# > `*_lonlat_box` methods in §4 — defaults to `frame='galactic'`, since boxes in
# > galactic coordinates are its most common job. Pass `frame='icrs'` (as above)
# > when your box is defined in RA/Dec.
#
# ### Bands
#
# A *band* sweeps the sky between two limits. Three standalone helpers cover the
# common cases, and a fourth draws a band defined in *another* coordinate frame:

# %%
fig = plt.figure(figsize=(12, 7))

ax = sph.make_wcs_frame(221, "AIT", center=180, fig=fig)
sph.add_latitude_band(ax, lat_min=-10, lat_max=10, facecolor=RC["green"], alpha=0.5)
sph.add_latitude_band(ax, lat_min=30, lat_max=50, facecolor=RC["rust"], alpha=0.5)
ax.set_title("add_latitude_band — Dec strips", fontsize=9)

ax = sph.make_wcs_frame(222, "AIT", center=180, fig=fig)
sph.add_longitude_band(ax, lon_min=120, lon_max=200, facecolor=RC["gold"], alpha=0.55)
ax.set_title("add_longitude_band — an RA strip", fontsize=9)

ax = sph.make_wcs_frame(223, "AIT", center=180, fig=fig)
sph.add_great_circle_band(ax, ra_pole=40, dec_pole=15, half_width=10,
                          facecolor=RC["blue"], alpha=0.55)
ax.set_title("add_great_circle_band — any orientation\n(a scan strip / orbital plane)",
             fontsize=9)

ax = sph.make_wcs_frame(224, "AIT", center=180, fig=fig)
sph.add_frame_band(ax, lat_min=-10, lat_max=10, frame="galactic",
                   facecolor=RC["tan"], alpha=0.4)
ax.set_title("add_frame_band(frame='galactic')\n— another frame's band, here",
             fontsize=9)

fig.suptitle("Bands: lat / lon / great-circle / cross-frame", y=1.0)
fig.tight_layout()

# %% [markdown]
# `add_latitude_band` and `add_longitude_band` work in the map's **native** frame
# (on a galactic map, the "latitude" band is galactic). `add_great_circle_band`
# generalizes them to *any* orientation — give it the **pole** of the great circle
# and a half-width, and you can draw an orbital plane, a satellite scan strip, or a
# custom avoidance zone tilted however you like. `add_frame_band` goes one step
# further: it draws a band defined in a *different named frame* onto your map,
# projecting the curved boundary correctly — the one-call way to shade the galactic
# plane on an equatorial chart. (Behind the scenes the galactic plane *is* the
# great circle about the North Galactic Pole, so `add_great_circle_band` could draw
# it too; `add_frame_band` just lets you name the frame instead of its pole.)
#
# ### Geodesic vs. linear edges
#
# On a sphere, "the straight edge between two corners" is ambiguous: a **great
# circle** (the true shortest path), or a line of constant lon/lat (a **graticule**
# edge)? For small shapes the two are indistinguishable; for survey-scale polygons
# they diverge dramatically. `add_spherical_polygon` exposes the choice as
# `geodesic=`:

# %%
qlon = [-60, 60, 60, -60]                             # a wide quadrangle, corners at
qlat = [10, 10, 60, 60]                                # constant Dec — where it matters
glons = np.linspace(-179, 179, 240)
view = (0, 35)                                         # ~quadrangle center, for front/back


def _edge_guides(mode):
    """The top/bottom edges extended across the sky under the chosen interpretation."""
    if mode is False:
        return [np.full_like(glons, 60.0), np.full_like(glons, 10.0)]
    return [great_circle_lats((-60, 60), (60, 60), glons),
            great_circle_lats((-60, 10), (60, 10), glons)]


modes = [
    ("geodesic=False\n(graticule edges)", False, RC["rust"]),
    ("geodesic=True\n(great-circle edges)", True, RC["blue"]),
    ("geodesic='auto'\n(per-edge by length)", "auto", RC["gold"]),
]
fig = plt.figure(figsize=(13, 4.2))
for i, (title, mode, col) in enumerate(modes, start=1):
    ax = sph.make_wcs_frame(131 + (i - 1), "AIT", center=0, fig=fig)
    for g in _edge_guides(mode):                       # dashed near side, dotted far side
        draw_guide(ax, glons, g, RC["gray"], view_center=view, lw=1.0)
    sph.add_spherical_polygon(ax, lons=qlon, lats=qlat, geodesic=mode,
                              facecolor=col, edgecolor=PAL["text"], lw=1.2, alpha=0.6)
    ax.set_title(title, fontsize=9)
fig.suptitle("The same four corners, three edge interpretations "
             "(dashed = edge extended; dotted = the arc on the sphere's far side)", y=1.02)
fig.tight_layout()

# %% [markdown]
# The same three interpretations on a **globe** make the front/back split literal: the
# extended edges simply disappear where they wrap behind the sphere, so a geodesic edge
# reads unmistakably as a great circle slicing across, while the constant-Dec edge rides
# its little circle of latitude.

# %%
fig = plt.figure(figsize=(13, 4.6))
for i, (title, mode, col) in enumerate(modes, start=1):
    ax = globe(131 + (i - 1), clon=0, clat=35)
    for g in _edge_guides(mode):                       # the globe hides the back for us
        draw_guide(ax, glons, g, RC["gray"], lw=1.0)
    sph.add_spherical_polygon(ax, lons=qlon, lats=qlat, geodesic=mode,
                              facecolor=col, edgecolor=PAL["text"], lw=1.2, alpha=0.7)
    ax.set_title(title, fontsize=9)
fig.suptitle("…and the same three on a sphere — the far-side arcs are simply hidden",
             y=1.0)
fig.tight_layout()

# %% [markdown]
# The top edge sits at Dec = +60°. With `geodesic=False` it hugs that parallel
# (flat in the figure); with `geodesic=True` it bows toward the pole along the
# great circle — the geometrically correct shortest path. `'auto'` (the default)
# decides per edge by length, so small fields stay fast-and-linear while large
# footprints get the correct curvature.
#
# > **Rule of thumb:** survey footprints defined by constant-RA/Dec boundaries want
# > `geodesic=False` (edges should follow the catalog's graticule lines); physical
# > regions on the sky want `geodesic=True` or `'auto'`.
#
# A flat map can still *under-sell* the difference, so it is worth seeing the same
# four corners on a real sphere. Spun to three viewing angles below, the linear edges
# (rust) ride the latitude circles while the geodesic edges (blue) cut straight
# across as great circles — the gap that the projection above only hinted at.

# %%
def _xyz(lon, lat, r=1.0):
    lo, la = np.radians(lon), np.radians(lat)
    return np.array([r * np.cos(la) * np.cos(lo),
                     r * np.cos(la) * np.sin(lo),
                     r * np.sin(la)])


def _edge(p0, p1, geodesic, n=48):
    """Sample an edge between two (lon, lat) corners, linearly or along a great circle."""
    if not geodesic:
        return np.linspace(p0[0], p1[0], n), np.linspace(p0[1], p1[1], n)
    a, b = _xyz(*p0), _xyz(*p1)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    omega = np.arccos(np.clip(a @ b, -1.0, 1.0))
    t = np.linspace(0, 1, n)[:, None]
    pts = (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)
    return (np.degrees(np.arctan2(pts[:, 1], pts[:, 0])),
            np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1))))


def _ring(corners, geodesic):
    lons, lats = [], []
    for k in range(len(corners)):
        lo, la = _edge(corners[k], corners[(k + 1) % len(corners)], geodesic)
        lons.append(lo)
        lats.append(la)
    return np.concatenate(lons), np.concatenate(lats)


corners = list(zip(qlon, qlat))                       # the same quadrangle as above
gridc = PAL["grid"]
fig = plt.figure(figsize=(13, 5.2))
for panel, (elev, azim, title) in enumerate([
        (22, -35, "front"), (18, -90, "side"), (68, -35, "pole-on")]):
    ax = fig.add_subplot(1, 3, panel + 1, projection="3d")
    # faint wireframe + lon/lat graticule (az/pol are the sphere-mesh parameters)
    az, pol = np.linspace(0, 2 * np.pi, 48), np.linspace(0, np.pi, 24)
    ax.plot_wireframe(np.outer(np.cos(az), np.sin(pol)), np.outer(np.sin(az), np.sin(pol)),
                      np.outer(np.ones_like(az), np.cos(pol)),
                      color=gridc, linewidth=0.3, alpha=0.4)
    for g in [-60, -30, 0, 30, 60]:
        xyz = _xyz(np.linspace(0, 360, 120), np.full(120, g))
        ax.plot(*xyz, color=gridc, lw=0.5, alpha=0.6)
    for g in np.arange(0, 360, 30):
        xyz = _xyz(np.full(120, g), np.linspace(-90, 90, 120))
        ax.plot(*xyz, color=gridc, lw=0.5, alpha=0.6)
    # the two edge interpretations + the geodesic fill
    for geo, col, lab in [(False, RC["rust"], "linear (const-Dec)"),
                          (True, RC["blue"], "geodesic")]:
        lo, la = _ring(corners, geo)
        ax.plot(*_xyz(lo, la, r=1.01), color=col, lw=2.6, zorder=5, label=lab)
    glo, gla = _ring(corners, True)
    poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection(
        [np.column_stack(_xyz(glo, gla, r=1.002))], alpha=0.18,
        facecolor=RC["blue"], edgecolor="none")
    ax.add_collection3d(poly)
    for lo, la in corners:                            # vertices
        ax.scatter(*_xyz(lo, la, r=1.02), color=PAL["text"], s=22, zorder=10)
    ax.set_box_aspect((1, 1, 1), zoom=1.5)            # fill the panel; shrink the gaps
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    if panel == 0:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
fig.suptitle("The same quadrangle on the sphere — linear edges hug parallels, "
             "geodesics cut across", y=0.97)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.0, wspace=0.0)

# %% [markdown]
# > **Note:** these sphere views are static stills, but the same geodesic machinery
# > runs on the **interactive** plotly backend, where a globe like this is a live
# > figure — hover any curve, zoom in, and pick the facing hemisphere with
# > `center=`/`lat_center=` (only that hemisphere is drawn, so curves run off the
# > limb and stop on their own). See
# > [Interactive Plotting](interactive_plotly.ipynb).

# %% [markdown]
# ## 3. The shared keywords
#
# Every closed-region renderer — circles, boxes, bands, polygons — shares the same
# four keywords, so one mental model covers all of them. You will rarely need the
# first two; the defaults are chosen well.
#
# | Keyword | What it does | When to reach for it |
# |---|---|---|
# | `clip=` | how the projection seam (antimeridian) and frame edge are handled | leave on `'auto'` (→ `'d3'`, full spherical clipping); drop to `'simple'` only to trade robustness for speed on thousands of tiny shapes, or to debug a seam |
# | `backend=` | what kind of artist is produced | `'patch'` (a filled path) is the default and usually the only option; `add_frame_band` also takes `'contour'`, and the HEALPix renderers take `'pcolormesh'`/`'imshow'` |
# | `resolution=` | number of points sampled along the boundary | raise it if a large region's edge looks faceted in a strongly curved projection |
# | `complement=` | fill everything *except* the region | the quick way to draw an exclusion zone / avoidance mask |
#
# ### `clip=` — handling the seam
#
# A region that straddles the map's antimeridian is the hard case. Picture a cap
# centered just past the seam of a `center=0` map — most of it on one side, a sliver
# wrapping onto the other. **That is what you want it to look like** (left). The
# default `'d3'` clipping delivers exactly that: it splits the boundary at the seam
# and closes each piece correctly. `'simple'` (right) skips the seam handling and
# projects raw vertices, smearing the cap across the whole map where its longitudes
# jump from +180° to −180°.

# %%
fig = plt.figure(figsize=(9.5, 4))
for i, (mode, title) in enumerate([("d3", "clip='d3' (the 'auto' default) — correct"),
                                   ("simple", "clip='simple' — raw projection")], start=1):
    ax = sph.make_wcs_frame(120 + i, "AIT", center=0, fig=fig)
    sph.add_geodesic_circle(ax, 192, 18, radius_deg=36, clip=mode,
                            facecolor=RC["rust"], edgecolor=RC["tan"], alpha=0.6)
    ax.set_title(title, fontsize=9)
fig.suptitle("clip= closes a seam-crossing region", y=1.0)
fig.tight_layout()

# %% [markdown]
# The left panel is the region as it truly sits on the sky: the bulk near 12–13ʰ and
# a sliver reappearing, closed, at the far edge. The right panel collapses into a band
# of nonsense across the middle. You get the left result for free — `'auto'` picks it.
#
# ### `resolution=` and `complement=`
#
# `resolution=` controls how finely the boundary is sampled. The default is fine for
# most shapes; a large region in a strongly curved projection can look faceted until
# you raise it. `complement=True` flips any shape inside-out — instead of filling the
# region, it fills everything else, which can be useful as an avoidance mask.

# %%
fig = plt.figure(figsize=(13, 4.2))

ax = sph.make_wcs_frame(131, "AIT", center=180, fig=fig)
sph.add_geodesic_circle(ax, 180, 0, radius_deg=55, resolution=6,
                        facecolor=RC["gold"], edgecolor=RC["tan"], alpha=0.6)
ax.set_title("resolution=6 — faceted", fontsize=9)

ax = sph.make_wcs_frame(132, "AIT", center=180, fig=fig)
sph.add_geodesic_circle(ax, 180, 0, radius_deg=55, resolution=200,
                        facecolor=RC["gold"], edgecolor=RC["tan"], alpha=0.6)
ax.set_title("resolution=200 (default) — smooth", fontsize=9)

ax = sph.make_wcs_frame(133, "AIT", center=180, fig=fig)
sph.add_geodesic_circle(ax, 180, 0, radius_deg=30, complement=True,
                        facecolor=RC["blue"], alpha=0.4)
ax.set_title("complement=True — avoid the cap", fontsize=9)

fig.suptitle("resolution= (boundary sampling) and complement= (invert)", y=1.0)
fig.tight_layout()

# %% [markdown]
# ### Complex regions are handled too
#
# The seam machinery is not just for a single round cap. Bands with their complements,
# separated pieces joined by a bridge, many independent strips, deep concavities, and
# cavities *inside* a footprint all clip and close correctly — at any frame rotation.
# Every panel below uses the default `clip='auto'`; the only thing that changes is the
# map center, which slides each region on or off the seam.

# %%
# A purpose-built "G"/backwards-C outline with a deep notch — a concavity stress case.
def _g_shape():
    lons, lats = [], []
    for lon in np.linspace(210, 150, 20):
        lons.append(lon)
        lats.append(25)
    for lat in np.linspace(25, -25, 20):
        lons.append(150 - 4 * np.sin(np.pi * (lat + 25) / 50))
        lats.append(lat)
    for lon in np.linspace(150, 210, 20):
        lons.append(lon)
        lats.append(-25)
    for lo, la in [(210, -3), (185, -3), (185, -15), (165, -15)]:
        lons.append(lo)
        lats.append(la)
    for lat in np.linspace(-15, 15, 10):
        lons.append(165)
        lats.append(lat)
    for lo, la in [(185, 15), (185, 3), (210, 3)]:
        lons.append(lo)
        lats.append(la)
    return lons, lats


def _strips_bridge(ax):
    r = sph.CompoundRegion(ax)
    for dec_c in (-18, 0, 18):
        r.add_polygon([120, 240, 240, 120],
                      [dec_c - 4, dec_c - 4, dec_c + 4, dec_c + 4], geodesic=False)
    r.add_circle(180, 0, radius_deg=25)
    return r


def _holey(ax):
    main_lons = [130, 145, 160, 175, 190, 205, 220, 230, 230, 225,
                 220, 210, 200, 190, 180, 170, 160, 150, 140, 130]
    main_lats = [5, 8, 10, 10, 10, 8, 5, 0, -45, -48,
                 -50, -50, -50, -48, -45, -42, -40, -35, -25, -10]
    return (sph.CompoundRegion(ax)
            .add_polygon(main_lons, main_lats, geodesic=False)
            .subtract_circle(175, -20, radius_deg=8)
            .subtract_polygon([170, 180, 190, 180], [-35, -30, -35, -40], geodesic=False)
            .subtract_circle(135, -25, radius_deg=10))


_STRIPS5 = [([150, 210, 210, 150], [-22, -22, -12, -12]),
            ([155, 215, 215, 155], [-5, -5, 5, 5]),
            ([160, 220, 220, 160], [10, 10, 20, 20]),
            ([165, 225, 225, 165], [24, 24, 34, 34]),
            ([148, 212, 212, 148], [-37, -37, -27, -27])]


def _draw_strips_bridge(ax):
    r = _strips_bridge(ax)
    r.render(facecolor=RC["green"], alpha=0.5)
    r.render_boundary(color=RC["green"], linewidth=1)


def _draw_gshape(ax):
    gl, ga = _g_shape()
    sph.add_spherical_polygon(ax, gl, ga, geodesic=False,
                              facecolor=RC["mauve"], edgecolor=RC["blue"], lw=1, alpha=0.5)


def _draw_holey(ax):
    h = _holey(ax)
    h.render(facecolor=RC["rust"], alpha=0.45)
    h.render_boundary(color=RC["rust"], linewidth=1)


def _draw_galband(ax):
    sph.add_frame_band(ax, -30, 30, frame="galactic", facecolor=RC["gold"], alpha=0.4)


def _draw_strips5(ax):
    for (lo, la), ck in zip(_STRIPS5, ["blue", "green", "gold", "rust", "mauve"]):
        sph.add_spherical_polygon(ax, lo, la, geodesic=False,
                                  facecolor=RC[ck], edgecolor="white", lw=0.5, alpha=0.6)


# Each row = one region, shown twice: centered (legible), then on a rotated frame so
# the *same* region must clip and re-close at the seam. The galactic band always wraps
# the sky, so for it "centered" means the galactic center is mid-map, "on the seam"
# means the galactic center sits on the antimeridian.
stress_rows = [
    ("separated strips + a bridge", _draw_strips_bridge, (180, 0)),
    ("a concave 'C' with a deep notch", _draw_gshape, (180, 0)),
    ("a footprint with interior cavities", _draw_holey, (180, 0)),
    ("a wide galactic band", _draw_galband, (266, 86)),
    ("five independent strips", _draw_strips5, (180, 0)),
]
fig = plt.figure(figsize=(11, 15.5))
gs = fig.add_gridspec(len(stress_rows), 2)
for row, (label, draw, (c_mid, c_seam)) in enumerate(stress_rows):
    for col, (center, tag) in enumerate([(c_mid, "centered"),
                                         (c_seam, "rotated → on the seam")]):
        ax = sph.make_wcs_frame(gs[row, col], "AIT", center=center, fig=fig)
        draw(ax)
        ax.set_title(f"{label} — {tag}", fontsize=8)
fig.suptitle("One clipper, every awkward case — left centered, right rotated onto the "
             "seam (all clip='auto')", y=0.995)
fig.tight_layout()

# %% [markdown]
# > **Note:** the `clip=` knob governs the *vector* region and band families. Raster
# > data (a HEALPix `pcolormesh`) and open curves (`add_great_circle`) are cleaned a
# > different way — see [Core concepts](../guide/concepts.md) §"Projection, clipping
# > & rendering" for the shared pipeline these modes plug into.

# %% [markdown]
# ## 4. Compound set algebra
#
# <video controls autoplay loop muted playsinline poster="../_static/manim/regions__set-algebra.poster.jpg" width="100%" style="max-width:760px;display:block;margin:0.5em auto;" aria-label="Set algebra as an animated Venn diagram: two overlapping sets A and B; union, intersection, and difference each fill in and gather into a corner, then land as real skyplothelper regions across one globe, stretched near its edges by the projection; finally a survey footprint is built up piece by piece on an all-sky map"><source src="../_static/manim/regions__set-algebra.mp4" type="video/mp4"></video>
#
# *Set algebra is the same union / intersection / difference you know from a Venn
# diagram — only here each operation is a real region on the sky. The three gather
# onto one globe (watch the projection stretch them toward the edges), then combine
# into a real survey footprint, built up one operation at a time — exactly what this
# section does.*
#
# A single shape is rarely a real footprint. Real footprints are *combinations*:
# "this declination band, minus the galactic plane, minus a few bright-star holes."
# `CompoundRegion` is the layer that builds them. You construct it against an axes
# (it needs the frame's projection to do its planar geometry), then chain four verb
# families over the shape vocabulary — `add_*` (union), `subtract_*` (difference),
# `intersect_*`, and `xor_*` — and render the accumulated result as a single patch,
# holes and all. Each operation returns the region itself, so the calls chain into
# a readable recipe.
#
# ### The four verbs
#
# The clearest way to visualize the difference is two overlapping circles, combined four
# ways. The thin outlines are the two inputs; the fill is the result.

# %%
fig = plt.figure(figsize=(11, 9))
A = dict(lon=-20, lat=8, radius_deg=26)
B = dict(lon=20, lat=-6, radius_deg=26)
ops = [
    ("add — union (A ∪ B)", lambda r: r.add_circle(**A).add_circle(**B), "green"),
    ("subtract — difference (A − B)", lambda r: r.add_circle(**A).subtract_circle(**B), "rust"),
    ("intersect — overlap (A ∩ B)", lambda r: r.add_circle(**A).intersect_circle(**B), "gold"),
    ("xor — symmetric difference", lambda r: r.add_circle(**A).xor_circle(**B), "blue"),
]
for i, (title, build, colkey) in enumerate(ops, start=1):
    ax = globe(221 + (i - 1), clon=0, clat=0)
    # the two inputs, as faint reference outlines
    sph.add_geodesic_circle(ax, A["lon"], A["lat"], A["radius_deg"],
                            facecolor="none", edgecolor=RC["gray"], lw=0.8)
    sph.add_geodesic_circle(ax, B["lon"], B["lat"], B["radius_deg"],
                            facecolor="none", edgecolor=RC["gray"], lw=0.8)
    region = build(sph.CompoundRegion(ax))
    region.render(facecolor=RC[colkey], alpha=0.65)
    region.render_boundary(color=RC[colkey], linewidth=1.4)
    ax.set_title(title, fontsize=10)
fig.suptitle("Two circles, four set operations", y=0.98)
fig.tight_layout()

# %% [markdown]
# `add_`/`subtract_` cover all eleven shapes; `intersect_`/`xor_` cover most of them
# (a few bands are union/difference-only — the
# [API reference](../api/geometry.md) has the exact list). The vocabulary mirrors the
# `add_*` renderers from §2: `add_circle`, `add_ellipse`, `add_annulus`,
# `add_rectangle`, `add_square`, `add_lonlat_box`, the band methods, and
# `add_polygon` — so an arbitrary outline you would draw with `add_spherical_polygon`
# can equally be unioned, subtracted, or intersected here as
# `add_polygon(lons, lats)` / `subtract_polygon(...)`.
#
# ### Complement, expand, and contract
#
# Three transformations act on the *accumulated* region rather than adding a shape:
# `complement()` inverts it (region ↔ everything else), and `expand(deg)` /
# `contract(deg)` buffer the boundary outward or inward by an angular margin — the
# way to grow a guard zone around a footprint or erode a safe interior.

# %%
fig = plt.figure(figsize=(14, 3.6))
base_shape = dict(lon=180, lat=15)


def fresh_region(ax):
    return (sph.CompoundRegion(ax)
            .add_ellipse(base_shape["lon"], base_shape["lat"],
                         semi_major=34, semi_minor=20, angle=25))


for i, (title, xform, colkey) in enumerate([
        ("base region", lambda r: r, "green"),
        (".complement() — everything else", lambda r: r.complement(), "mauve"),
        (".expand(10) — +10° margin", lambda r: r.expand(10), "rust"),
        (".contract(10) — −10° erosion", lambda r: r.contract(10), "gold")], start=1):
    ax = sph.make_wcs_frame(141 + (i - 1), "AIT", center=180, fig=fig)
    # show the base outline for reference on the transformed panels (white stroke so
    # the gray reads against the colored fills)
    fresh_region(ax).render_boundary(color=RC["gray"], linewidth=1.0,
                                     path_effects=[pe.withStroke(linewidth=2.6,
                                                                 foreground="white")])
    region = xform(fresh_region(ax))
    region.render(facecolor=RC[colkey], alpha=0.55)
    region.render_boundary(color=RC[colkey], linewidth=1.3)
    ax.set_title(title, fontsize=9)
fig.suptitle("Transforming the accumulated region (gray = original boundary)", y=1.02)
fig.tight_layout()

# %% [markdown]
# Two things to notice. The contracted ellipse (right) develops a faint **cusp** at
# its pointed end — buffering inward by a fixed angular margin pinches hardest where
# the boundary curves most, an expected quirk of the offsetting geometry. And because
# these act on the *whole accumulated region*, they apply just as well to a compound
# mask. Below, the same two operations buffer a great-circle band along the galactic
# plane that has the galactic center punched out and a rectangular field bolted on —
# band, hole, and field all grow and shrink together.

# %%
fig = plt.figure(figsize=(13, 4.2))
GAL_CENTER = (266.4, -28.9)


def fresh_compound(ax):
    return (sph.CompoundRegion(ax)
            .add_great_circle_band(ra_pole=192.85, dec_pole=27.13, half_width=11)
            .subtract_circle(GAL_CENTER[0], GAL_CENTER[1], radius_deg=17)
            .add_rectangle(45, 48, width=34, height=22))


for i, (title, xform, colkey) in enumerate([
        ("base compound region", lambda r: r, "green"),
        (".expand(6) — guard zone", lambda r: r.expand(6), "rust"),
        (".contract(6) — safe interior", lambda r: r.contract(6), "gold")], start=1):
    ax = sph.make_wcs_frame(131 + (i - 1), "AIT", center=266, fig=fig)
    fresh_compound(ax).render_boundary(color=RC["gray"], linewidth=1.0,
                                       path_effects=[pe.withStroke(linewidth=2.6,
                                                                   foreground="white")])
    region = xform(fresh_compound(ax))
    region.render(facecolor=RC[colkey], alpha=0.5)
    region.render_boundary(color=RC[colkey], linewidth=1.1)
    ax.set_title(title, fontsize=9)
fig.suptitle("…and on a compound region: the band, the hole, and the field buffer together",
             y=1.0)
fig.tight_layout()

# %% [markdown]
# ### Start from a real footprint
#
# You do not always start from scratch. skyplothelper ships a catalog of real survey
# footprints; discover them with `survey_keys()` / `list_surveys()` and drop one onto
# a map with `add_survey_footprint`. They render through this same region machinery, so
# they clip and close correctly. (Left to its own devices `add_survey_footprint` colors
# them from the built-in `REGION_PALETTE` — or `REGION_PALETTE_NAMED` for by-name
# access; here we tint them from our cycle palette instead.)

# %%
print(f"{len(sph.survey_keys())} built-in survey footprints, e.g.:",
      ", ".join(sph.survey_keys()[:12]), "...")

fig = plt.figure(figsize=(11, 5.5))
ax = sph.make_wcs_frame(111, "AIT", center=180, fig=fig)
for sv, colkey in [("des", "rust"), ("sdss", "blue"),
                   ("lsst", "green"), ("euclid", "gold")]:
    sph.add_survey_footprint(ax, sv, color=RC[colkey], alpha=0.3, label=sv.upper())
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Four real survey footprints, drawn from the built-in catalog")
fig.tight_layout()

# %% [markdown]
# ### Build your own
#
# Here is a custom footprint assembled the way a real one is — a wide RA/Dec box,
# minus the galactic plane, minus a bright-source exclusion hole — built up one
# operation at a time so you can watch each verb act. The box deliberately runs from
# RA 200° round past 0° to 20°, so it *wraps the seam*; the galactic-plane subtraction
# then takes a big diagonal bite out of the middle. The `add_lonlat_box` base uses
# graticule edges (constant RA/Dec), one possible choice for a survey boundary (say, a telescope elevation pointing limit). To make
# the subtractions obvious, each cut-out zone is drawn **solid while it is still part of
# the picture**, then switches to a **dashed outline once it has been cut** — so you
# can watch the rust galactic band and the gold exclusion circle drop away step by step.

# %%
fig = plt.figure(figsize=(13, 4.2))


def _box(r):
    return r.add_lonlat_box(lat_min=-20, lat_max=78, lon_min=200, lon_max=380,
                            frame="icrs")


def _cut_zones(ax, band, circle):
    """Solid = zone still present; dashed outline = zone already subtracted."""
    if band == "solid":
        sph.add_frame_band(ax, -20, 20, frame="galactic",
                           facecolor=RC["rust"], alpha=0.30, zorder=4)
    else:
        sph.add_frame_band(ax, -20, 20, frame="galactic", facecolor="none",
                           edgecolor=RC["rust"], lw=1.2, ls="--", zorder=4)
    if circle == "solid":
        sph.add_geodesic_circle(ax, 240, 58, radius_deg=9, facecolor=RC["gold"],
                                edgecolor=RC["gold"], alpha=0.5, zorder=4)
    else:
        sph.add_geodesic_circle(ax, 240, 58, radius_deg=9, facecolor="none",
                                edgecolor=RC["gold"], lw=1.2, ls="--", zorder=4)


steps = [
    ("1. add_lonlat_box (wraps the seam)", _box, "solid", "solid"),
    ("2. − frame_band (avoid the plane)",
     lambda r: _box(r).subtract_frame_band(-20, 20, frame="galactic"), "dashed", "solid"),
    ("3. − circle (punch a hole)",
     lambda r: _box(r).subtract_frame_band(-20, 20, frame="galactic")
              .subtract_circle(240, 58, radius_deg=9), "dashed", "dashed"),
]
for i, (title, build, bmode, cmode) in enumerate(steps, start=1):
    ax = sph.make_wcs_frame(131 + (i - 1), "AIT", center=180, fig=fig)
    region = build(sph.CompoundRegion(ax))
    region.render(facecolor=RC["blue"], alpha=0.5)
    region.render_boundary(color=RC["blue"], linewidth=1.2)
    _cut_zones(ax, bmode, cmode)
    ax.set_title(title, fontsize=9)
fig.suptitle("Building a footprint with set algebra "
             "(solid = still present, dashed = cut)", y=1.0)
fig.tight_layout()

# %% [markdown]
# Subtraction is only one verb. Three more recipes show the range — an observability
# mask built by *avoidance*, a footprint that mixes union, intersection, *and*
# difference, and a galactic-plane survey with its crowded center carved out:

# %%
fig = plt.figure(figsize=(13, 4.4))

# (1) Survey creation: everything a northern telescope can reach, minus the plane and
#     a couple of bright-source exclusion zones.
ax = sph.make_wcs_frame(131, "AIT", center=180, fig=fig)
survey = (sph.CompoundRegion(ax)
          .add_latitude_band(-20, 90)                          # Dec > −20° is reachable
          .subtract_frame_band(-12, 12, frame="galactic")      # avoid the plane
          .subtract_circle(83.6, 22.0, radius_deg=10)          # Taurus A / Crab
          .subtract_circle(187.7, 12.4, radius_deg=9))         # Virgo A / M87
survey.render(facecolor=RC["green"], alpha=0.45)
survey.render_boundary(color=RC["green"], linewidth=1.1)
ax.set_title("survey creation:\nDec > −20° − plane − bright sources", fontsize=8)

# (2) Mixed operations: two fields unioned, intersected with a Dec slice, minus plane.
ax = sph.make_wcs_frame(132, "AIT", center=180, fig=fig)
mixed = (sph.CompoundRegion(ax)
         .add_circle(150, 22, radius_deg=32)                   # field A  ┐ union
         .add_circle(215, 26, radius_deg=32)                   # field B  ┘
         .intersect_latitude_band(0, 48)                       # ∩ a Dec slice
         .subtract_frame_band(-13, 13, frame="galactic"))      # − the plane
mixed.render(facecolor=RC["gold"], alpha=0.5)
mixed.render_boundary(color=RC["gold"], linewidth=1.1)
ax.set_title("mixed ops:\ntwo fields ∪, ∩ a Dec slice, − plane", fontsize=8)

# (3) A galactic-plane survey with the crowded galactic center excised.
ax = sph.make_wcs_frame(133, "AIT", center=180, fig=fig)
plane = (sph.CompoundRegion(ax)
         .add_frame_band(-15, 15, frame="galactic")            # the plane itself
         .subtract_lonlat_box(lat_min=-12, lat_max=12,
                              lon_min=-25, lon_max=25, frame="galactic"))  # excise GC
plane.render(facecolor=RC["rust"], alpha=0.5)
plane.render_boundary(color=RC["rust"], linewidth=1.1)
ax.set_title("galactic-plane survey,\ncrowded center excised", fontsize=8)

fig.suptitle("More recipes: avoidance masks, mixed set operations, frame-defined cuts",
             y=1.02)
fig.tight_layout()

# %% [markdown]
# Each of these is a single queryable mask. In the next section we put one to work:
# asking which sources fall inside it and how much sky it covers.

# %% [markdown]
# ## 5. Membership and area
#
# A region is not only something you *draw* — it is a queryable object. Two common
# questions for sky regions: **which of my sources fall inside?** and **how much sky
# area does it cover?** `CompoundRegion` answers both directly.
#
# - `contains_point(ra, dec)` / `contains_points(ra, dec)` — membership tests (the
#   plural one is vectorized and fast, and both accept a `SkyCoord`).
# - `area_frac` — the fraction of the map the region covers; `solid_angle` — the same
#   thing as a physical area (`sq_deg` and `sr`).
# - `is_empty` — a sanity check after aggressive intersections.
#
# We work up to it in three steps: an intuitive dense-catalog test, a moving-target
# avoidance check, then a real catalog against a survey footprint.
#
# ### A dense catalog, an obvious region
#
# The clearest way to *see* `contains_points` is a uniform catalog blanketing the
# whole sky against a compound region. The region is the two-crescent `xor` from the
# opening figure — the symmetric difference of two overlapping caps — so which points
# fall inside is unmistakable.

# %%
rng = np.random.default_rng(7)
n = 1200
u_ra = rng.uniform(0, 360, n)
u_dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))    # area-uniform on the sphere

fig = plt.figure(figsize=(11, 5.6))
ax = sph.make_wcs_frame(111, "AIT", center=180, fig=fig)
region = (sph.CompoundRegion(ax)
          .add_circle(150, 12, radius_deg=34)
          .xor_circle(210, 12, radius_deg=34))
region.render(facecolor=RC["green"], alpha=0.30)
region.render_boundary(color=RC["blue"], linewidth=1.2)
inside = region.contains_points(u_ra, u_dec)
ax.scatter(u_ra[~inside], u_dec[~inside], transform=ax.get_transform("world"),
           s=5, c=RC["gray"], label=f"outside ({(~inside).sum()})")
ax.scatter(u_ra[inside], u_dec[inside], transform=ax.get_transform("world"),
           s=11, c=PAL["accent"], label=f"inside ({inside.sum()})", zorder=6)
ax.legend(loc="lower left", fontsize=8)
ax.set_title(f"contains_points on {n} uniform sources — "
             f"two caps xor'd ({region.area_frac:.1%} of sky)")
fig.tight_layout()

# %% [markdown]
# ### A trajectory through an avoidance zone
#
# Membership is not only for static catalogs. Suppose a target tracks across the sky
# — a satellite, a survey scan, a moving body — and you must keep clear of the Sun.
# Here the Sun sits at its real position for 2026 August 1 (astropy's `get_sun`), the
# avoidance zone is a 40° circle around it, and the trajectory's samples are tested
# against that zone: the ones inside are the moments you cannot observe. Note the
# inputs — the circle's center is a `SkyCoord` and its radius an astropy `Quantity`,
# both accepted directly.

# %%
sun_app = get_sun(Time("2026-08-01"))                  # the Sun's position that day
sun = SkyCoord(sun_app.ra, sun_app.dec)                # keep just the sky direction
AVOID_RADIUS = 40 * u.deg
t = np.linspace(0, 1, 90)
traj_ra = 60 + 180 * t                                 # sweeps RA 60° → 240°
traj_dec = 35 * np.sin(2 * np.pi * t) + 5              # a wavy track

fig = plt.figure(figsize=(11, 5.6))
ax = sph.make_wcs_frame(111, "AIT", center=180, fig=fig)
avoid = sph.CompoundRegion(ax).add_circle(sun, AVOID_RADIUS)
avoid.render(facecolor=RC["gold"], alpha=0.22)
avoid.render_boundary(color=RC["gold"], linewidth=1.3)

blocked = avoid.contains_points(traj_ra, traj_dec)
ax.plot(traj_ra, traj_dec, transform=ax.get_transform("world"),
        color=RC["gray"], lw=1, alpha=0.7, zorder=4)
ax.scatter(traj_ra[~blocked], traj_dec[~blocked], transform=ax.get_transform("world"),
           s=14, c=RC["green"], label="observable", zorder=5)
ax.scatter(traj_ra[blocked], traj_dec[blocked], transform=ax.get_transform("world"),
           s=20, c=RC["rust"], label="too near the Sun", zorder=5)
# Place the Sun icon at its sky position (world → pixel, since imscatter wants data coords).
spx, spy = ax.wcs.world_to_pixel(sun)
sph.imscatter(spx, spy, "../../examples/data/icons/sun1_120pix.png",
              ax=ax, zoom=0.32, zorder=6)
ax.legend(loc="lower left", fontsize=8)
ax.set_title(f"A trajectory through a {AVOID_RADIUS.value:.0f}° solar-avoidance zone — "
             f"{blocked.sum()} of {len(t)} samples blocked")
fig.tight_layout()

# %% [markdown]
# ### A real catalog and a survey footprint
#
# A real catalog case: the 110 Messier objects against a survey-style footprint
# (a box, the galactic plane, a hole — the §4 recipe).

# %%
mra, mdec = np.asarray(messier["ra_deg"]), np.asarray(messier["dec_deg"])

fig = plt.figure(figsize=(11, 5.6))
ax = sph.make_wcs_frame(111, "AIT", center=180, fig=fig)
footprint = (sph.CompoundRegion(ax)
             .add_lonlat_box(lat_min=-12, lat_max=70, lon_min=110, lon_max=260,
                             frame="icrs")
             .subtract_frame_band(-25, 25, frame="galactic")
             .subtract_circle(180, 35, radius_deg=8))
footprint.render(facecolor=RC["green"], alpha=0.30)
footprint.render_boundary(color=RC["blue"], linewidth=1.2)

# contains_points accepts a SkyCoord array (in any frame — converted automatically)
m_coords = SkyCoord(mra, mdec, unit="deg")
inside = footprint.contains_points(m_coords)

ax.scatter(mra[~inside], mdec[~inside], transform=ax.get_transform("world"),
           s=12, c=RC["gray"], label=f"outside ({(~inside).sum()})")
ax.scatter(mra[inside], mdec[inside], transform=ax.get_transform("world"),
           s=26, c=PAL["accent"], edgecolor="white", linewidth=0.4,
           label=f"inside ({inside.sum()})", zorder=6)
ax.legend(loc="lower left", fontsize=8)

sa = footprint.solid_angle
ax.set_title(f"Messier objects in a custom footprint — "
             f"{footprint.area_frac:.1%} of sky ≈ {sa['sq_deg']:.0f} deg²")
fig.tight_layout()

# %% [markdown]
# The membership test reads the holes correctly: a Messier object that lands in the
# carved-out galactic strip or the punched hole counts as *outside*. The `solid_angle`
# property turns the same geometry into a number you can quote in a proposal.

# %%
# Scalar test for a single target, and an emptiness sanity check.
print("M51 (RA 202.5, Dec +47.2) inside footprint?",
      footprint.contains_point(202.5, 47.2))
print(f"footprint covers {sa['sq_deg']:.0f} deg² = {sa['sr']:.3f} sr")

nothing = (sph.CompoundRegion(ax)
           .add_circle(180, 20, radius_deg=15)
           .subtract_circle(180, 20, radius_deg=15))   # a cap minus itself
print("a region minus itself is_empty:", nothing.is_empty)

# %% [markdown]
# > **Tip:** `contains_points` is how you turn a footprint into a catalog filter —
# > `catalog[footprint.contains_points(ra, dec)]` keeps just the sources on your
# > survey. The companion [Catalogs](catalogs.ipynb) tutorial builds on this for coordinate-frame
# > handling and large catalogs. And for anything these queries don't cover, the
# > region's underlying shapely geometry is available as `region.geometry`.

# %% [markdown]
# ## 6. Planes and Tissot indicatrices
#
# Two region-flavored views round out the toolkit.
#
# ### A plane as a line, or as a band
#
# A reference *plane* (galactic, ecliptic, …) on the sky is a great circle. You can mark it as a
# thin line with `add_plane_overlay` — the lightweight "where is the galactic
# equator?" annotation — or shade the zone around it as a *band* (`add_frame_band` /
# `add_great_circle_band`) when the *width* matters, e.g. an avoidance region.

# %%
fig = plt.figure(figsize=(12, 5))

ax = sph.make_wcs_frame(121, "AIT", center=180, fig=fig)
sph.add_plane_overlay(ax, plane="galactic", color=RC["tan"], lw=2, label="galactic")
sph.add_plane_overlay(ax, plane="ecliptic", color=RC["blue"], lw=2, ls="--",
                      label="ecliptic")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Planes as lines — add_plane_overlay", fontsize=10)

ax = sph.make_wcs_frame(122, "AIT", center=180, fig=fig)
sph.add_frame_band(ax, -10, 10, frame="galactic", facecolor=RC["tan"], alpha=0.4)
sph.add_plane_overlay(ax, plane="galactic", color=RC["tan"], lw=1)
ax.set_title("…or as a shaded avoidance band — add_frame_band", fontsize=10)

fig.suptitle("The galactic plane: a line marks it, a band gives it width", y=1.0)
fig.tight_layout()

# %% [markdown]
# ### Tissot indicatrices
#
# `tissot` drops a lattice of *equal-radius geodesic circles* across the frame
# (a sensible default grid — pass `lons=`/`lats=` arrays to place your own) —
# each one an `add_geodesic_circle` region under the hood. Where they render as true
# circles the projection is locally faithful; where they stretch into ellipses you
# read the distortion directly. (The [Projections](projections.ipynb) tutorial introduces Tissot as a
# distortion tool; here it is the same idea, framed as a grid of geodesic-circle
# regions.) Because every indicatrix routes through the same clip pipeline, the same
# one line works on *any* frame — a globe, the rectangular Plate Carrée, the HEALPix
# grid, and the wilder polyconic / Bonne / conic projections alike:

# %%
tissot_projs = [("AIT", False), ("MOL", False), ("SIN", True),
                ("CAR", False), ("PAR", False), ("HPX", False),
                ("PCO", False), ("BON", False), ("COE", False)]
# cycle the (saturated) uranometria tones so each projection reads as its own
tissot_cycle = [RC["blue"], RC["gold"], RC["green"], RC["rust"], RC["mauve"], RC["tan"]]
fig = plt.figure(figsize=(13, 11))
for i, (proj, is_globe) in enumerate(tissot_projs, start=1):
    if is_globe:
        ax = sph.make_globe_frame(330 + i, center_LONdeg=0, center_LATdeg=25,
                                  gridcolor=RC["gray"], gridalpha=0.5)
    else:
        ax = sph.make_wcs_frame(330 + i, proj, center=0, fig=fig)
    col = tissot_cycle[(i - 1) % len(tissot_cycle)]
    sph.tissot(ax, rad_deg=8, facecolor=col, edgecolor=col, alpha=0.42)
    ax.set_title(proj + (" (globe)" if is_globe else ""), fontsize=10)
fig.suptitle("Tissot indicatrices across projection families — true circles where "
             "faithful, ellipses where distorted", y=1.0)
fig.tight_layout()

# %% [markdown]
# ## 7. Putting it together
#
# Everything in one figure on the kind of data you actually have: a custom footprint
# built from set algebra, a real catalog tested against it, the sky coverage quoted,
# the galactic plane drawn in for context — and, as an honest check, the *real* SDSS
# footprint the package already ships, overlaid as a dashed outline. Your hand-built
# mask lands right where the survey actually is.

# %%
fig = plt.figure(figsize=(12, 6.2))
ax = sph.make_wcs_frame(111, "AIT", center=180, fig=fig)

# Context: the galactic plane (why we carve it) and the real SDSS footprint (the check).
sph.add_plane_overlay(ax, plane="galactic", color=RC["tan"], lw=1.3, ls=":",
                      alpha=0.9, label="galactic plane")
sph.add_survey_footprint(ax, "sdss", fill=False, edgecolor=RC["gray"], lw=1.3,
                         alpha=0.8, ls="--", label="SDSS (built-in)")

# Our custom footprint: box − galactic plane − a bright-star hole.
footprint = (sph.CompoundRegion(ax)
             .add_lonlat_box(lat_min=-12, lat_max=70, lon_min=110, lon_max=260,
                             frame="icrs")
             .subtract_frame_band(-25, 25, frame="galactic")
             .subtract_circle(180, 35, radius_deg=8))
footprint.render(facecolor=RC["green"], alpha=0.28)
footprint.render_boundary(color=RC["blue"], linewidth=1.6)

# The catalog, split by membership.
mra, mdec = np.asarray(messier["ra_deg"]), np.asarray(messier["dec_deg"])
inside = footprint.contains_points(mra, mdec)
ax.scatter(mra[~inside], mdec[~inside], transform=ax.get_transform("world"),
           s=12, c=RC["gray"], label=f"Messier outside ({(~inside).sum()})")
ax.scatter(mra[inside], mdec[inside], transform=ax.get_transform("world"),
           s=28, c=PAL["accent"], edgecolor="white", linewidth=0.4, zorder=6,
           label=f"Messier inside ({inside.sum()})")

sa = footprint.solid_angle
ax.text(0.015, 0.04,
        f"footprint area: {footprint.area_frac:.1%} of sky\n"
        f"≈ {sa['sq_deg']:.0f} deg²  ({sa['sr']:.2f} sr)\n"
        f"Messier inside: {inside.sum()} / {len(inside)}",
        transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
        bbox=dict(boxstyle="round", fc=PAL["ax_bg"], ec=PAL["frame"], alpha=0.9))
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.set_title("A custom footprint, a real catalog, and the real survey for comparison")
fig.tight_layout()

# %% [markdown]
# That is the whole arc: simple shapes (§2) combine through set algebra (§4) into a
# mask you can both *draw* and *query* (§5) — and it agrees with the footprint a real
# survey publishes. Swap in your own RA/Dec limits, your own catalog, and you have a
# membership filter and a coverage number for a proposal in a handful of lines.

# %% [markdown]
# ## 8. Where to go next
#
# - **[Core concepts](../guide/concepts.md)** §"Projection, clipping & rendering" —
#   the shared antimeridian-clip → project → frame-clip pipeline that the `clip=`
#   modes plug into, and why the same region renders identically on matplotlib and
#   plotly.
# - **[Regions guide](../guide/regions.md)** — the reference companion to this tour,
#   with the full method list and pitfalls.
# - **[HEALPix workflows](healpix_workflows.ipynb)** — region → pixel membership and
#   binning catalogs into sky maps.
# - **[Catalogs](catalogs.ipynb)** — `contains_points` at scale, coordinate-frame
#   handling, and testing large source lists against survey footprints.
# - **[Interactive plotting](interactive_plotly.ipynb)** — the same `CompoundRegion`
#   drives the plotly backend: build it with `sphpl.make_compound_region(fig)` and
#   render with `sphpl.add_compound_region(fig, region)`; holes and seams render
#   there too, and the globes are live figures you can hover and zoom.
