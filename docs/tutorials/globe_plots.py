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
# # Globe and Planet Plotting
#
# An orthographic *globe* — the sphere drawn as a disk, seen from outside — is one
# of the most evocative ways to show spherical data, and skyplothelper draws two
# kinds from the same machinery: a **celestial** globe (a hemisphere of sky) and a
# **planet** globe (Earth, the Moon, a world at any orientation, including a
# physically tilted, day-and-night Earth). Under the hood both are SIN-projection
# WCS frames, so everything else in the package — overlays, regions, HEALPix maps —
# draws straight onto them; what this tutorial adds is the *orientation* machinery,
# *hemisphere-aware* plotting, *surface* textures, and globe-specific *decorations*.
#
# We work through it the way you'd actually use it: build and aim a globe, plot
# points/lines/fields on it without tracks bleeding through the far side, drape
# real planet surfaces, tilt the Earth with Euler angles, add a day–night
# terminator, hang decorations and geodesics on it, draw terrestrial features with
# and without cartopy — and finish by stacking the whole toolkit into one figure.
# Each section answers the two questions a globe raises: **"how do I show my data
# on a globe?"** and **"how do I adjust it?"**
#
# > **A note on the data.** The NASA raster maps (Blue Marble, Black Marble, the
# > planet textures) are large and ship outside the pip package, so they live in
# > `examples/data/` locally; the committed notebook outputs show every figure, and
# > the code is exactly what you'd run with the files in place.
# >
# > **Vector Earth features are fetched once, on demand.** The coastlines, land /
# > lakes / rivers, tectonic plates, and time zones used in §7 are *not* bundled with
# > the package — they download and cache locally the first time you use them. Run
# > `sph.prepare_earth_data()` once per environment before using them; it needs the
# > optional `cartopy` extra (for the Natural Earth layers) and a network connection.
# >
# > **Scope, stated honestly.** skyplothelper's Earth maps are an earnest effort to
# > make whole-globe views and simple planetary plots look good with little setup —
# > enough to orient a figure, drape a texture, or sketch station geography. For
# > heavier terrestrial cartography — fine-resolution features, national borders,
# > filled land/ocean at scale, or GIS-style feature queries — reach for
# > [cartopy](https://scitools.org.uk/cartopy/); §8 shows the built-in bridge.
#
# ## Contents
#
# 1. [Building a globe](#1.-Building-a-globe)
# 2. [Plotting on a globe](#2.-Plotting-on-a-globe)
# 3. [Earth and planet surfaces](#3.-Earth-and-planet-surfaces)
# 4. [Tilting the view and the TiltedEarthFrame](#4.-Tilting-the-view-and-the-TiltedEarthFrame)
# 5. [Day and night with nightshade](#5.-Day-and-night-with-nightshade)
# 6. [Globe decorations and geodesics](#6.-Globe-decorations-and-geodesics)
# 7. [Earth features without cartopy](#7.-Earth-features-without-cartopy)
# 8. [The cartopy backend](#8.-The-cartopy-backend)
# 9. [Putting it together](#9.-Putting-it-together)
# 10. [Where to go next](#10.-Where-to-go-next)

# %%
import datetime as dt

import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import ITRS
from matplotlib.colors import to_rgb
from scipy.spatial.transform import Rotation

import skyplothelper as sph

# A globe is an orthographic SIN-projection WCS frame, so the whole package draws
# onto it. We set the **structural** base style for a clean, in-theme feel
# throughout; set_style applies each layer (base / theme / palette) independently,
# so this is safe with the docs' light and dark passes alike.
sph.set_style(base="structural")


# Pick an in-theme annotation palette by reading the active page background, so a
# single code path adapts to the docs' light and dark renders.
def annotation_palette():
    r, g, b = to_rgb(mpl.rcParams["figure.facecolor"])
    dark = (0.299 * r + 0.587 * g + 0.114 * b) < 0.5
    return sph.ANNOTATION_PALETTES["dark" if dark else "publication"]


PAL = annotation_palette()

# Cycle palette for multi-series data (dual-mode, transparency-safe).
URANOMETRIA = sph.CYCLE_PALETTES["uranometria"]["colors"]

# %% [markdown]
# Throughout we lean on a few **data anchors** so the focus stays on the globe
# machinery, not on wrangling data:
#
# - a **real catalog** — the 110 Messier objects — for celestial scatter;
# - a **smooth synthetic all-sky field** (a notional foreground model) for the
#   gridded-field plotters;
# - NASA **raster maps** (Black Marble night lights, Blue Marble day, and the
#   2k planet textures) for Earth and planet surfaces;
# - the bundled **`planet_radii` / `obliquities` / `rot_periods`** tables, which
#   supply physically motivated radii and tilts for any solar-system body.
#
# Where each of these datasets comes from — with download links for the large
# raster maps — is listed in `examples/data/README.md`.

# %%
DATA = "../../examples/data"

# Real catalog: the Messier objects (name, RA, Dec, type, V magnitude).
messier = np.genfromtxt(f"{DATA}/messier.csv", delimiter=",", names=True,
                        dtype=None, encoding="utf-8")
m_ra, m_dec, m_vmag = messier["ra_deg"], messier["dec_deg"], messier["vmag"]

# Smooth synthetic all-sky field — a notional, slowly varying foreground: a broad
# galactic-plane-like band plus a couple of soft hot spots. Defined on a regular
# (lon, lat) grid, which is exactly what the globe field plotters consume.
_lon1 = np.linspace(-180, 180, 145)
_lat1 = np.linspace(-90, 90, 73)
LONG, LATG = np.meshgrid(_lon1, _lat1)


def _blob(lon0, lat0, amp, width):
    d = np.hypot((LONG - lon0) * np.cos(np.radians(LATG)), LATG - lat0)
    return amp * np.exp(-(d / width) ** 2)


FIELD = (np.exp(-(LATG / 22.0) ** 2)              # a smooth equatorial band
         + _blob(40, 35, 0.8, 28)
         + _blob(-110, -20, 0.6, 30))

# NASA raster textures (equirectangular). Day = Blue Marble, night = Black Marble.
EARTH_DAY = f"{DATA}/world.topo.bathy.200412.3x5400x2700.jpg"
EARTH_NIGHT = f"{DATA}/BlackMarble_2016_01deg.jpg"


def drape(ax, hdu, zorder=-10):
    """Resample an equirectangular raster HDU onto a globe frame's pixel grid.

    Globe frames carry a synthetic WCS, so the standard reprojection machinery
    drapes any raster onto them; we size the output to the frame's pixel extent
    and place it below the graticule.
    """
    out_hdr = ax.wcs.to_header()
    nx = round(ax.get_xlim()[1] - ax.get_xlim()[0])
    ny = round(ax.get_ylim()[1] - ax.get_ylim()[0])
    out_hdr["NAXIS1"], out_hdr["NAXIS2"] = nx, ny
    bg = sph.reproject_rgb_map(hdu, out_hdr, shape_out=(ny, nx))
    return ax.imshow(np.nan_to_num(bg), zorder=zorder), out_hdr


def wire_grid(ax, color="steelblue", **kw):
    """Graticule for a see-through (wireframe) globe.

    Both hemispheres share one color; the far side is drawn fainter (more
    transparent and thinner) rather than in a different color, so it reads as
    'the back of the same sphere' instead of a separate, inverted grid.
    """
    sph.plot_ortho_grid(ax, front_color=color, back_color=color, front_lw=0.8,
                        back_lw=0.55, back_alpha=0.3, **kw)


def surface_grid(ax, color="0.85", **kw):
    """Graticule for a globe with an opaque raster surface.

    The surface hides the far side, so we draw only the front hemisphere
    (`show_back=False`) — back gridlines would imply we can see through the body.
    """
    sph.plot_ortho_grid(ax, front_color=color, front_lw=0.4, show_back=False, **kw)


def globe_labels(ax, fontsize=8, color="white"):
    """In-frame coordinate labels for a raster globe.

    Matches the in-frame look the frame builders use by default (labels along the
    central parallel and meridian, `lon_at='axis'` / `lat_at='axis'` — not the
    flat 'native' edge labels), but recolored with a fine dark stroke so they stay
    legible over any surface — dark ocean or the bright face of the Sun alike.
    `add_overlay_ticks` takes the stroke directly via `stroke_lw` / `stroke_color`.
    """
    sph.add_overlay_ticks(ax, lon_at="axis", lat_at="axis", suppress_default="both",
                          show_ticks=False, stroke_lw=1.1, stroke_color="0.1",
                          label_kwargs={"color": color, "fontsize": fontsize})


# %% [markdown]
# ## Two kinds of globe
#
# Everything here is an **orthographic** view — a sphere drawn as a disk, seen
# from far away — but it comes in two flavors, and the difference is the single
# thing most likely to trip you up. A **celestial** globe is a hemisphere of
# *sky*, where longitude (RA) runs **east-left** by astronomy convention. A
# **planet** globe is a solid body seen from outside, where longitude runs
# **east-right** like a map. Draw a world with the sky's east-left convention and
# the continents come out **mirrored** (center panel below — same frame, same center,
# only `direction=` changed); `make_planet_frame()`'s geographic default is what
# un-mirrors them (right panel). Reach for `make_globe_frame()` for the sky and
# `make_planet_frame()` for a world.
#
# > **A second, separate gotcha: match the frames.** Longitude *direction* is one
# > thing; the coordinate *frame* is another. Draping a geographic raster onto a
# > celestial frame makes the reprojection apply a real ITRS↔ICRS rotation, which
# > slides the map by tens of degrees of longitude — and because that rotation
# > tracks Earth's spin, the result depends on the epoch. skyplothelper warns when
# > you do it. The fix is to match them: keep an Earth map on a planet frame.

# %%
fig = plt.figure(figsize=(13, 4.5))

# A celestial globe: a hemisphere of sky with the Messier catalog. Centered at the
# same latitude (+20) as the Earth panels for an apples-to-apples comparison.
ax = sph.make_globe_frame(131, center_LONdeg=275, center_LATdeg=20, grid=False,
                          Naxispix=400)
wire_grid(ax)
sph.plot_scatter_globe(ax, m_ra, m_dec, s=22, c=PAL["accent"],
                       edgecolors=PAL["frame"], linewidths=0.4, zorder=5)
ax.set_title("Celestial globe (sky)\nmake_globe_frame", fontsize=10)

# The same Earth, same frame, but with the sky's east-left longitude: mirrored.
# Using direction= here isolates the convention as the single cause; the
# panels differ in nothing else, so the flip is unmistakable.
earth_hdu0 = sph.pseudofits_from_image(EARTH_DAY, geo=True)
ax = sph.make_planet_frame(132, body="earth", center_LONdeg=-90, center_LATdeg=20,
                           direction="sky", grid=False, Naxispix=450)
drape(ax, earth_hdu0)
globe_labels(ax)
surface_grid(ax)
ax.set_title("East-left (sky) convention\nmirrored — wrong", fontsize=10,
             color=PAL["accent"])

# The same Earth on a planet frame: un-mirrored.
ax = sph.make_planet_frame(133, body="earth", center_LONdeg=-90, center_LATdeg=20,
                           grid=False, Naxispix=450)
drape(ax, earth_hdu0)
globe_labels(ax)
surface_grid(ax)
ax.set_title("Earth on a planet frame\nmake_planet_frame — right", fontsize=10)

fig.suptitle("Two kinds of globe: a hemisphere of sky, and a world", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# %% [markdown]
# ## 1. Building a globe
#
# The globe builders render an **orthographic** view — the sphere shown as a disk,
# as if seen from far away. `make_globe_frame()` builds a **celestial** globe (a
# hemisphere of sky); `make_planet_frame()` builds a **solid-body** globe (Earth,
# Moon, a planet). The split matters because of one convention: sky longitude runs
# **east-left** (`direction='sky'`, the astronomy default), while a planet's
# surface runs **east-right** like a geographic map. `make_planet_frame()` selects
# the geographic convention for you, so continents come out un-mirrored — more on
# that in [Earth and planet surfaces](#3.-Earth-and-planet-surfaces).
#
# You aim a celestial globe with `center_LONdeg` / `center_LATdeg`: center it on a
# target for a hemisphere of context, on a pole for a polar-cap view, or on your
# zenith for a "what's up tonight." `plot_ortho_grid()` draws the graticule with
# independent front/back-hemisphere styling — so we build the frame with
# `grid=False` and let `plot_ortho_grid()` own the graticule (otherwise the
# builder's own grid shows through as a faint double).

# %%
fig = plt.figure(figsize=(13, 4.4))
centers = [(0, 0, "On the equator (RA 0ʰ)"),
           (0, 90, "On the north pole (a polar cap)"),
           (266, -29, "On a target (Galactic center)")]
for i, (clon, clat, title) in enumerate(centers):
    ax = sph.make_globe_frame(131 + i, center_LONdeg=clon, center_LATdeg=clat,
                              grid=False)
    wire_grid(ax)
    ax.set_title(title, fontsize=10)
fig.suptitle("Celestial globes: aim with center_LONdeg / center_LATdeg",
             fontsize=12, y=1.0)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Resolution.** The globe builder's `Naxispix` sets the synthetic-WCS pixel grid
# (default 360). Vectors — graticules, points, lines — are drawn analytically and
# don't care, but a *raster* draped on the globe is resampled onto that grid, so
# bump `Naxispix` to ~600–1000 for a crisp surface (§3).
#
# ### Aiming the globe: rotation, obliquity, perspective
#
# A globe doesn't have to sit upright. The intuitive way to orient one is with
# three **physical** angles, and `euler_to_fits_ortho()` converts them into the
# `(center_lon, center_lat, lonpole)` the frame builders take:
#
# - **rotation** — spin about the pole (turn the globe like a top);
# - **obliquity** — tilt the pole toward or away from the viewer;
# - **perspective** — *precession*: swing the tilted pole around, changing the
#   direction the tilt leans (only meaningful once there *is* a tilt).
#
# <video controls autoplay loop muted playsinline poster="../_static/manim/globe_plots__euler-trio.poster.jpg" width="100%" style="max-width:640px;display:block;margin:0.5em auto;" aria-label="The three Euler angles posing a globe in 3-D: rotation spins it about the pole, obliquity tilts the pole toward the viewer, and perspective precesses the tilted pole; the wireframe then dissolves into the real skyplothelper Blue-Marble globe posed at exactly those angles."><source src="../_static/manim/globe_plots__euler-trio.mp4" type="video/mp4"></video>
#
# *The three angles in 3-D: spin about the pole, tilt the pole toward you, then
# precess the tilt — and the wireframe dissolves into the real skyplothelper globe
# posed at exactly those angles.*
#
# Sweeping each angle in turn, with the other two fixed, shows what each does:
#
# > **Gimbal lock.** `rotation` and `perspective` are both rotations about the
# > polar axis, so at zero obliquity they collapse to one degree of freedom — only
# > their *sum* matters, and a perspective sweep just looks like extra spin. The
# > perspective row below therefore uses a real 35° tilt, and cancels the
# > bookkeeping spin (`rotation = -perspective`) so the precession reads on its own
# > rather than riding on a longitude drift.

# %%
# A 3x4 grid of globes needs more than the 9 a 3-digit subplot number allows, so
# we build a gridspec and hand each cell's SubplotSpec to make_globe_frame. We
# color the equator (rust) and prime meridian (green) so the eye can follow the
# orientation as the wireframe turns.
EQ_C, PM_C = URANOMETRIA[3], URANOMETRIA[2]
fig = plt.figure(figsize=(13, 9.6))
gs = fig.add_gridspec(3, 4)
# The perspective row keeps a fixed 35° tilt and sets rotation = -perspective, so
# the longitude drift cancels and only the precession remains — the tilt leaning a
# new way each panel rather than the globe spinning past in longitude.
rows = [("rotation", "Spin\n(about the pole)", [0, 90, 180, 270], dict(obliquity=0, perspective=0)),
        ("obliquity", "Obliquity\n(tilt the pole)", [0, 23.44, 45, 90], dict(rotation=0, perspective=0)),
        ("perspective", "Perspective\n(precession, 35° tilt)", [0, 45, 90, 135], dict(rotation=0, obliquity=35))]
for r, (key, label, values, fixed) in enumerate(rows):
    for c, val in enumerate(values):
        kw = dict(fixed)
        kw[key] = val
        if key == "perspective":
            kw["rotation"] = -val   # cancel the longitude drift; show precession alone
        clon, clat, pole = sph.euler_to_fits_ortho(**kw)
        ax = sph.make_globe_frame(gs[r, c], center_LONdeg=clon,
                                  center_LATdeg=clat, lonpole=pole, grid=False)
        wire_grid(ax, equator_color=EQ_C, equator_lw=1.4,
                  prime_meridian_color=PM_C, prime_meridian_lw=1.4)
        ax.set_title(f"{key}={val:g}°", fontsize=9)
        if c == 0:
            ax.text(-0.30, 0.5, label, transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=11, fontweight="bold")
fig.suptitle("Aiming the globe: rotation, obliquity, perspective", fontsize=13, y=0.99)
fig.tight_layout(rect=[0.03, 0, 1, 0.97])
plt.show()

# %% [markdown]
# **The same three angles are three *motions*.** Let them change continuously and
# you have the axial motions of a spinning body: a steady `rotation` is its
# **spin**, a nod of `obliquity` is **nutation**, and a sweep of `perspective` is
# **precession** — the tilted pole tracing a cone. The
# [Animations](animations.ipynb) tutorial (§4) drives this as a matplotlib
# look-down Earth; here it is in 3-D, dissolving into the real skyplothelper globe
# and its spin axis. *(The nutation amplitude and precession rate in the clip are
# greatly exaggerated so the motions read — real ones are far slower and smaller.)*
#
# <video controls autoplay loop muted playsinline poster="../_static/manim/globe_plots__spin-nutation-precession.poster.jpg" width="100%" style="max-width:640px;display:block;margin:0.5em auto;" aria-label="A spinning globe's three axial motions in 3-D: spin turns the body about its axis, nutation nods the axis, and precession sweeps the tilted axis around a cone with the pole tracing a circle; then it dissolves into the real skyplothelper Black-Marble look-down globe with its spin axis."><source src="../_static/manim/globe_plots__spin-nutation-precession.mp4" type="video/mp4"></video>

# %% [markdown]
# **Generating the motion series: `make_globe_angles()`.** To turn those motions
# into an animation you need the orientation at every frame, and
# `make_globe_angles()` builds that series for you. Give it a starting
# `[rotation, obliquity, perspective]` and either per-frame rates (`spin_rate`,
# `prec_rate`, `nut_rate` / `nut_amp`) or — often handier — the **whole-animation
# totals**: `spin_total=360` for one seamless looping rotation, `prec_total=` for a
# full precession, `nut_cycles=` for a set number of nutation wobbles. It returns
# the `(center_lon, center_lat, lonpole)` for every frame, ready to hand straight to
# `make_planet_frame()`. (Precession here drives the `perspective` Euler angle; as
# the frame re-aims each step, that reads as the pole precessing.) The
# [Animations](animations.ipynb) tutorial builds the moving loop — a strip of its
# frames:

# %%
# One clean rotation loop, sampled as five stills: spin_total=360 gives evenly
# spaced longitudes (0, 72, ... 288) that would join seamlessly end to end.
lons, lats, poles = sph.make_globe_angles([0, 23.44, 0], n_steps=5, spin_total=360)
fig = plt.figure(figsize=(13, 2.9))
earth_hduM = sph.pseudofits_from_image(EARTH_DAY, geo=True)
for i, (m_clon, m_clat, m_pole) in enumerate(zip(lons, lats, poles)):
    ax = sph.make_planet_frame(151 + i, body="earth", center_LONdeg=m_clon,
                               center_LATdeg=m_clat, lonpole=m_pole, Naxispix=360,
                               grid=False)
    drape(ax, earth_hduM)
    surface_grid(ax)
    ax.set_title(f"frame {i}", fontsize=9)
fig.suptitle("make_globe_angles: five frames of a spin_total=360° loop", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# %% [markdown]
# ### Euler angles and quaternions
#
# The three angles of §1 are **Euler angles** — an orientation written as three
# rotations applied in sequence. skyplothelper uses the *proper* **z-x'-z″**
# convention (`'ZXZ'`): spin about the pole, tilt about the new x-axis, spin about
# the new pole — the `[rotation, obliquity, perspective]` triple
# `euler_to_fits_ortho()` takes. (Other orders exist — Tait-Bryan **z-y'-x″**
# yaw/pitch/roll is the aviation one — so any library that speaks Euler angles will
# ask *which* order it means; aiming a globe from one of those is a quaternion away,
# below.) They are wonderfully intuitive, but two things bite:
# at zero tilt the first and third rotations line up and collapse into one
# (**gimbal lock**, §1), and they don't interpolate smoothly, so an animation
# blended between two Euler poses can wobble.
#
# **Quaternions** sidestep both. A rotation quaternion encodes the whole
# orientation as four numbers with no preferred axis — no gimbal lock — and two of
# them *slerp* (spherically interpolate) along the shortest smooth arc, which is
# why attitude- and pointing-tracking systems and animation pipelines speak
# quaternions natively. `quaternion_to_fits_ortho()` is the drop-in counterpart of
# `euler_to_fits_ortho()`: same `(center_lon, center_lat, lonpole)` out. Pass
# `scalar_first=False` for the SciPy / ROS `(x, y, z, w)` order.
#
# The two agree for the same orientation — here the quaternion equivalent of the
# Euler triple `(60°, 23.44°, 30°)` lands on the identical globe:

# %%
# The same orientation as euler_to_fits_ortho(rotation, obliquity, perspective),
# expressed as a quaternion. SciPy's Rotation uses *active* rotations with the
# scalar-LAST (x, y, z, w) order, so we negate the angles per the convention note
# in the docstring — a quaternion straight from a tracker needs no such juggling.
rot, obl, persp = 60, 23.44, 30
q = Rotation.from_euler("ZXZ", [-rot, -obl, -persp], degrees=True).as_quat()
clon_q, clat_q, pole_q = sph.quaternion_to_fits_ortho(q, scalar_first=False)
clon_e, clat_e, pole_e = sph.euler_to_fits_ortho(rot, obl, persp)
print(f"euler ({rot}, {obl}, {persp}) is the quaternion (x, y, z, w) = "
      f"[{q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f}]")
print(f"quaternion_to_fits_ortho -> ({clon_q:7.2f}, {clat_q:6.2f}, {pole_q:6.2f})")
print(f"euler_to_fits_ortho      -> ({clon_e:7.2f}, {clat_e:6.2f}, {pole_e:6.2f})"
      f"  # identical")

fig = plt.figure(figsize=(5.2, 5.2))
ax = sph.make_planet_frame(111, body="earth", center_LONdeg=clon_q,
                           center_LATdeg=clat_q, lonpole=pole_q, Naxispix=500,
                           grid=False)
drape(ax, sph.pseudofits_from_image(EARTH_DAY, geo=True))
globe_labels(ax)
surface_grid(ax)
ax.set_title("Globe aimed by a quaternion\n(identical to the Euler result)", fontsize=10)
plt.show()

# %% [markdown]
# That quaternion is also the bridge to **any other Euler order**. To aim a globe
# from aviation-style Tait-Bryan yaw/pitch/roll (**z-y'-x″**) you need no new sph
# function — build the rotation in that order and pass its quaternion in, exactly as
# above but with a `"ZYX"` order string:
#
# ```python
# yaw, pitch, roll = 60, 23.44, 30
# q = Rotation.from_euler("ZYX", [-yaw, -pitch, -roll], degrees=True).as_quat()
# clon, clat, pole = sph.quaternion_to_fits_ortho(q, scalar_first=False)
# ```
#
# (the leading minus signs carry SciPy's active-rotation convention across, just as
# in the z-x'-z″ example). So `euler_to_fits_ortho()`'s fixed z-x'-z″ triple is a
# convenience for the physical spin/tilt/precession picture, not a limit on the
# angles you can start from — `quaternion_to_fits_ortho()` accepts an orientation
# built in whatever convention your source speaks.

# %% [markdown]
# In short — reach for whichever fits the job:
#
# | | Euler angles | Quaternion |
# |---|---|---|
# | **Form** | three angles `[rotation, obliquity, perspective]` (z-x'-z″) | four numbers `(w, x, y, z)` |
# | **Intuition** | high — each is a physical turn | low — read off a device, not by hand |
# | **Gimbal lock** | yes (at zero tilt) | none |
# | **Interpolation** | can wobble between poses | smooth (slerp) |
# | **Typical source** | posing a view by hand (§1) | attitude / pointing systems, animation |
# | **sph entry point** | `euler_to_fits_ortho()` | `quaternion_to_fits_ortho()` |

# %% [markdown]
# ## 2. Plotting on a globe
#
# The far side of a globe is something to be considered: a raw `ax.plot(...)`
# happily draws a track *through* the sphere, out the back, and onto the front
# again. The globe plotters are **hemisphere-aware** — they mask (or restyle)
# whatever falls on the far side. The family mirrors matplotlib's:
#
# - `plot_scatter_globe()` — points;
# - `plot_line_globe()` — polylines (with optional densification so a straight
#   segment in data follows the sphere's curve);
# - `plot_pcolormesh_globe()` / `plot_contour_globe()` — gridded fields.
#
# Here they are over a celestial globe centered on the rich Sagittarius region: the
# 110 Messier objects for the points, great-circle arcs between a few of them for
# the lines, and the synthetic foreground field for the gridded plotters. The
# fields use the package's bundled `sph.deepsky` colormap by its registered name —
# any of the bundled maps work anywhere matplotlib takes a `cmap`
# (see [Themes, Palettes & Fonts](styling.ipynb) for the full set).

# %%
CC = (275.0, -15.0)        # celestial-globe center for this section (RA, Dec)

# Marker size from V magnitude (brighter -> bigger); a couple of objects have no
# listed magnitude, so fill those with the catalog median before scaling.
vmag = np.where(np.isnan(m_vmag), np.nanmedian(m_vmag), m_vmag)
m_size = np.clip((11.0 - vmag) * 7, 8, 90)

# A few well-spread Messier objects to string great-circle "routes" between, so
# the arcs read as long curves bending across the sphere.
route_names = ["M13", "M27", "M2", "M30"]
idx = {n: int(np.where(messier["name"] == n)[0][0]) for n in route_names}

fig = plt.figure(figsize=(13, 8.6))

# (a) Scatter — the Messier catalog, sized by brightness. plot_scatter_globe culls
# the far-side objects itself, carrying the per-point size array along with them.
ax = sph.make_globe_frame(231, center_LONdeg=CC[0], center_LATdeg=CC[1], grid=False)
wire_grid(ax)
sph.plot_scatter_globe(ax, m_ra, m_dec, s=m_size, c=PAL["accent"],
                       edgecolors=PAL["frame"], linewidths=0.4, zorder=5)
ax.set_title("(a) Scatter — Messier catalog", fontsize=10)

# (b) Lines — great-circle arcs between bright objects, far side hidden.
ax = sph.make_globe_frame(232, center_LONdeg=CC[0], center_LATdeg=CC[1], grid=False)
wire_grid(ax)
for k in range(len(route_names) - 1):
    i, j = idx[route_names[k]], idx[route_names[k + 1]]
    lo, la = sph.great_circle_arc(m_ra[i], m_dec[i], m_ra[j], m_dec[j], n_pts=80)
    sph.plot_line_globe(ax, lo, la, color=URANOMETRIA[k], lw=2.0, densify=False)
sph.plot_scatter_globe(ax, [m_ra[idx[n]] for n in route_names],
                       [m_dec[idx[n]] for n in route_names],
                       s=40, c=PAL["frame"], zorder=6)
ax.set_title("(b) Great-circle routes", fontsize=10)

# (c) Pcolormesh — the synthetic foreground field, with a colorbar. The plotters
# return the matplotlib mappable, so add_colorbar attaches to it directly.
ax = sph.make_globe_frame(233, center_LONdeg=CC[0], center_LATdeg=CC[1], grid=False)
mesh = sph.plot_pcolormesh_globe(ax, LONG, LATG, FIELD, cmap="sph.deepsky",
                                 shading="auto")
wire_grid(ax)
sph.add_colorbar(mesh, ax=ax, label="intensity (arb.)")
ax.set_title("(c) Pcolormesh — synthetic field", fontsize=10)

# (d) Line contours.
ax = sph.make_globe_frame(234, center_LONdeg=CC[0], center_LATdeg=CC[1], grid=False)
wire_grid(ax)
sph.plot_contour_globe(ax, LONG, LATG, FIELD, levels=8, cmap="sph.deepsky",
                       linewidths=1.0)
ax.set_title("(d) Contours", fontsize=10)

# (e) Filled contours.
ax = sph.make_globe_frame(235, center_LONdeg=CC[0], center_LATdeg=CC[1], grid=False)
sph.plot_contour_globe(ax, LONG, LATG, FIELD, levels=10, cmap="sph.deepsky",
                       filled=True)
wire_grid(ax)
ax.set_title("(e) Filled contours", fontsize=10)

# (f) Combined — field as a muted backdrop, catalog on top. A cool backdrop
# (sph.lagoon) keeps the warm catalog markers clearly separate, and a thin white
# stroke lifts them off the field where it is busiest.
ax = sph.make_globe_frame(236, center_LONdeg=CC[0], center_LATdeg=CC[1], grid=False)
sph.plot_contour_globe(ax, LONG, LATG, FIELD, levels=10, cmap="sph.lagoon",
                       filled=True, alpha=0.5)
wire_grid(ax)
sph.plot_scatter_globe(ax, m_ra, m_dec, s=m_size, c=PAL["accent"],
                       edgecolors="white", linewidths=0.6, zorder=5)
ax.set_title("(f) Field + catalog", fontsize=10)

fig.suptitle("The globe plotters: points, lines, and gridded fields", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %% [markdown]
# **Front, back, and the far side.** A great circle wraps all the way around the
# sphere, so half of it is always behind. `highlight_great_circle()` — a
# *decoration* — traces a **complete** great circle, given by its pole, two points,
# or (here) an `inclination` and `node`, at one matched color with the only
# front/back difference being the linestyle. Draw it all solid (left) and you can't
# tell which arcs sit behind; dash the far side (middle) and the depth reads
# instantly.
#
# Your **data** plotters take the opposite, safer tack: `plot_line_globe()` and
# `plot_scatter_globe()` **mask** the far side outright (right) — the far half of a
# track, and any far-side points, simply drop out, so nothing plows through the
# sphere. That's the right default for data you're reading quantitatively (a point
# you can see is unambiguously on the near face). The sources here sit on the near
# side throughout for exactly that reason; the great-circle *rings*, being
# decoration, are what show the far side. (To keep a far-side **network** visible
# but faded rather than dropped, `plot_baselines()` has `back_hemisphere_markers` /
# `_linestyle` / `_alpha` — see §6.) *(The meridian-only
# `highlight_meridian_tracer()` is a thin wrapper of `highlight_great_circle()`.)*

# %%
VC = (275, 35)                          # view center, tilted down so the far side shows
rings = [(55, 250, URANOMETRIA[3]), (70, 330, URANOMETRIA[4])]   # (inclination, node, color)
# A handful of catalog sources spread around the sphere, so some fall on the near
# side of this view and some on the far side.
msub = np.arange(0, len(m_ra), 9)
dm_lon, dm_lat = m_ra[msub], m_dec[msub]
MK = dict(s=26, c=PAL["accent"], edgecolors=PAL["frame"], linewidths=0.5, zorder=6)


def great_circle_lonlat(inclination, node, n_pts=361):
    """(lon, lat) of the full great circle at this inclination and ascending node.

    Same convention `highlight_great_circle()` uses, so we can hand the very same
    ring to the data plotters in panel (c).
    """
    t = np.radians(np.linspace(0, 360, n_pts))
    i = np.radians(inclination)
    lat = np.degrees(np.arcsin(np.sin(i) * np.sin(t)))
    lon = node + np.degrees(np.arctan2(np.cos(i) * np.sin(t), np.cos(t)))
    return lon, lat


fig = plt.figure(figsize=(13, 4.4))

# (a) Rings all solid — ambiguous: which arcs sit behind the globe? (Sources sit on
# the near side; plot_scatter_globe drops any that fall on the far side.)
ax = sph.make_globe_frame(131, center_LONdeg=VC[0], center_LATdeg=VC[1], grid=False)
wire_grid(ax)
for inc, node, col in rings:
    sph.highlight_great_circle(ax, inclination=inc, node=node, color=col, lw=2.2, back_ls="-")
sph.plot_scatter_globe(ax, dm_lon, dm_lat, **MK)
ax.set_title("(a) All solid — front/back ambiguous", fontsize=10)

# (b) Far side dashed — the same rings, depth now obvious (decoration draws both
# halves; the data points are still near-side only).
ax = sph.make_globe_frame(132, center_LONdeg=VC[0], center_LATdeg=VC[1], grid=False)
wire_grid(ax)
for inc, node, col in rings:
    sph.highlight_great_circle(ax, inclination=inc, node=node, color=col, lw=2.2, back_ls="--")
sph.plot_scatter_globe(ax, dm_lon, dm_lat, **MK)
ax.set_title("(b) Far side dashed — front + back", fontsize=10)

# (c) Your data: the plot_*_globe family masks the far side. The same rings, handed
# to plot_line_globe, keep only their front arcs — no dashed back — and the far-side
# points drop out. We hide the back graticule too (show_back=False).
ax = sph.make_globe_frame(133, center_LONdeg=VC[0], center_LATdeg=VC[1], grid=False)
wire_grid(ax, show_back=False)
for inc, node, col in rings:
    gl, ga = great_circle_lonlat(inc, node)
    sph.plot_line_globe(ax, gl, ga, color=col, lw=2.2, densify=False)
sph.plot_scatter_globe(ax, dm_lon, dm_lat, **MK)
ax.set_title("(c) plot_*_globe — far side masked", fontsize=10)

fig.suptitle("Hemisphere awareness: the far side is handled for you", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# > **Note:** the gridded plotters and the visibility test build on a small
# > toolkit of functions you can call directly: `orthographic_forward()` /
# > `orthographic_inverse()` for the projection math, and
# > `orthographic_visibility()` for the "is this point on the visible side?"
# > test — handy when you want to mask your own artists. The arcs themselves come
# > from a general-purpose spherical-geometry toolkit you can use anywhere:
# > `great_circle_arc()` and `great_circle_distance()` (the path and length between
# > two points), `midpoint()` and `destination_point()` (where you end up going a
# > bearing and distance), `initial_bearing()`, and `small_circle()` (a ring of
# > constant angular radius — a field of view, a horizon). Constellation boundaries
# > also ride on a globe (`plot_boundaries_globe()`); the full constellation
# > treatment — lines, labels, fills — is its own topic (see
# > [Constellations & Asterisms](constellations.ipynb)).
#
# The typical use of `small_circle()` is a **field of view**: a ring of constant
# angular radius around a pointing. Feed its output to `plot_line_globe()` and the
# ring bends with the sphere and masks on the far side like everything else. Here,
# two instruments aimed at the Small Sagittarius Star Cloud (M24) — the ~7° field
# of 7×50 binoculars, and the ~25° field of a wide-angle camera lens:

# %%
FOV_CENTER = (274.2, -18.5)          # M24, the Small Sagittarius Star Cloud
fig = plt.figure(figsize=(5.8, 5.8))
ax = sph.make_globe_frame(111, center_LONdeg=CC[0], center_LATdeg=CC[1], grid=False)
wire_grid(ax)
sph.plot_scatter_globe(ax, m_ra, m_dec, s=m_size * 0.6, c=PAL["accent2"],
                       edgecolors=PAL["frame"], linewidths=0.4, zorder=5)
for radius, name, col in [(7, "7×50 binoculars (~7°)", URANOMETRIA[3]),
                          (25, "wide-angle lens (~25°)", URANOMETRIA[4])]:
    fov_lon, fov_lat = sph.small_circle(*FOV_CENTER, radius_deg=radius)
    sph.plot_line_globe(ax, fov_lon, fov_lat, color=col, lw=2.2, densify=False,
                        label=name, zorder=6)
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Fields of view: small_circle rings around M24", fontsize=11)
plt.show()

# %% [markdown]
# ## 3. Earth and planet surfaces
#
# A globe doesn't have to represent the sky — it can be a solid body. `make_planet_frame()`
# is the entry point: it sets the **geographic** longitude convention and the
# body-fixed coordinate system in one call, so continents (and Martian
# volcanoes, and lunar maria) come out the right way round rather than mirrored.
# This is the one place the package's astro east-left default would trip you up;
# `make_planet_frame()` handles it.
#
# Any equirectangular texture becomes a surface in three steps:
# `pseudofits_from_image(path, geo=True)` wraps the image in a synthetic WCS, and
# our `drape()` helper resamples it onto the frame's pixel grid below the
# graticule. The bundled `obliquities` table gives each body's real axial tilt, so
# the gallery below shows every world at its true obliquity — from Earth's 23.4°
# to Uranus's 98° (nearly pole-on):

# %%
gallery = [
    ("earth", EARTH_DAY, 70),
    ("mars", f"{DATA}/planet_maps/2k_mars.jpg", 250),
    ("moon", f"{DATA}/planet_maps/2k_moon.jpg", 200),
    ("sun", f"{DATA}/planet_maps/2k_sun.jpg", 0),
    ("jupiter", f"{DATA}/planet_maps/2k_jupiter.jpg", 160),   # 160 brings the Great Red Spot into view
    ("uranus", f"{DATA}/planet_maps/2k_uranus.jpg", 0),
]
fig = plt.figure(figsize=(13, 8.4))
for i, (body, path, spin) in enumerate(gallery):
    obl = sph.obliquities[body]
    clon, clat, pole = sph.euler_to_fits_ortho(rotation=spin, obliquity=obl, perspective=0)
    ax = sph.make_planet_frame(231 + i, body=body, center_LONdeg=clon,
                               center_LATdeg=clat, lonpole=pole, Naxispix=600,
                               grid=False)
    hdu = sph.pseudofits_from_image(path, geo=True)
    drape(ax, hdu)
    globe_labels(ax)
    surface_grid(ax)
    ax.set_title(f"{body.capitalize()} — obliquity {obl:g}°", fontsize=10)
fig.suptitle("Solar-system bodies at their true axial tilts", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %% [markdown]
# The physical constants come from three bundled dictionaries, keyed by body
# name, that you can use anywhere — they feed the tilt above and the distance
# scale bars in §6:
#
# | Table | What it holds | Example |
# |-------|---------------|---------|
# | `planet_radii` | equatorial radius (km) | `earth → 6371`, `jupiter → 69911` |
# | `obliquities` | axial tilt (°) | `earth → 23.44`, `uranus → 97.86` |
# | `rot_periods` | sidereal spin period | `earth → 23.93 h`, `venus → 5832 h` |
#
# They cover the planets, the Sun and Moon, and major moons and dwarf planets —
# so a Mars map, a Ganymede map, and a Pluto map each get the right radius and
# tilt without you looking anything up.

# %%
# A quick peek at the tables:
for body in ["earth", "mars", "jupiter", "uranus"]:
    print(f"{body:8s}  R = {sph.planet_radii[body]:6.0f} km   "
          f"obliquity = {sph.obliquities[body]:6.2f}°   "
          f"spin = {sph.rot_periods[body]:.2f}")

# %% [markdown]
# **Surface resolution.** Because the texture is resampled onto the frame's
# pixel grid, the globe builder's `Naxispix` controls how crisp the surface
# looks. The default (360) is fine for thumbnails; bump it up to keep a larger
# image crisp. Points and graticules are drawn analytically and stay sharp
# regardless — only the raster cares. One companion knob: `Naxispix` sets the
# pixels *available*, but what you *see* also depends on the figure — a
# high-resolution globe rendered into a small `figsize` (or saved at low
# `dpi`) will still look soft, so size the figure for its final save state too.

# %%
fig = plt.figure(figsize=(13, 3.6))
earth_hdu = sph.pseudofits_from_image(EARTH_DAY, geo=True)
clon, clat, pole = sph.euler_to_fits_ortho(rotation=70, obliquity=0, perspective=0)
for i, npix in enumerate([90, 180, 360, 720]):
    ax = sph.make_planet_frame(141 + i, body="earth", center_LONdeg=clon,
                               center_LATdeg=clat, lonpole=pole, Naxispix=npix,
                               grid=False)
    drape(ax, earth_hdu)
    globe_labels(ax)
    ax.set_title(f"Naxispix={npix}", fontsize=10)
fig.suptitle("Naxispix sets the draped-surface resolution", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# %% [markdown]
# ### Where the surface maps come from
#
# Any **equirectangular** (plate-carrée) image works as a surface — wrap it with
# `pseudofits_from_image(path, geo=True)` and drape it. The maps in this notebook are
# all free to use:
#
# - **Earth day** — NASA *Blue Marble Next Generation* (topography + bathymetry);
# - **Earth night** — NASA *Black Marble* 2016 night lights (Suomi NPP VIIRS);
# - **Planets, Moon, Sun** — the *Solar System Scope* texture set, built on NASA
#   imagery and elevation data.
#
# The NASA maps are public domain; the Solar System Scope textures are CC BY 4.0, so
# credit them if you reuse the figures. Per-file sources and licenses live in
# `examples/data/README.md`.
#
# A few other places to find free plate-carrée maps you can drop straight in:
#
# | Source | What | Terms |
# |--------|------|-------|
# | [NASA Visible Earth](https://visibleearth.nasa.gov/) | Earth — day, night, clouds, topography | Public domain |
# | [NASA SVS — Black Marble](https://svs.gsfc.nasa.gov/30876/) | Earth night lights | Public domain |
# | [Solar System Scope](https://www.solarsystemscope.com/textures/) | Planets, moons, Sun, starfield | CC BY 4.0 |
# | [USGS Astrogeology](https://astrogeology.usgs.gov/) | Planetary / lunar mosaics (Mars, Io, …) | Mostly public domain |
# | [Natural Earth](https://www.naturalearthdata.com/) | Earth raster + vector (coastlines, borders, …) | Public domain |

# %% [markdown]
# ## 4. Tilting the view and the TiltedEarthFrame
#
# §1 aimed the camera at a tilted globe — and that is the *main* way you'll use it:
# the tilt, spin, and perspective are baked straight into the view, so a draped
# planet map renders at any physical orientation. The four panels below are the
# **same Earth as the main axis**, sweeping obliquity and then adding a
# perspective (precession) swing:
#
# > **Why this matters.** A globe plotting function with both an axial
# > **tilt** *and* a viewing **perspective** is surprisingly hard to come by:
# > cartopy's orthographic view only aims at a (longitude, latitude) center,
# > and coaxing a raw WCS into an oblique view means hand-crafting the header.
# > `euler_to_fits_ortho()` (and its companion coordinate frame
# > `TiltedEarthFrame`) turn it into three intuitive angles.

# %%
orients = [(70, 0, 0, "Upright\n(0° tilt)"),
           (70, 23.44, 0, "Earth's tilt\n(23.4°)"),
           (70, 45, 0, "Exaggerated\n(45°)"),
           (70, 23.44, 45, "+ precession\n(23.4° tilt, persp 45°)")]
fig = plt.figure(figsize=(13, 3.8))
earth_hduA = sph.pseudofits_from_image(EARTH_DAY, geo=True)
for i, (rot, obl, per, title) in enumerate(orients):
    clon, clat, pole = sph.euler_to_fits_ortho(rotation=rot, obliquity=obl, perspective=per)
    ax = sph.make_planet_frame(141 + i, body="earth", center_LONdeg=clon,
                               center_LATdeg=clat, lonpole=pole, Naxispix=500, grid=False)
    drape(ax, earth_hduA)
    surface_grid(ax)        # a light graticule makes the changing tilt legible
    globe_labels(ax)
    ax.set_title(title, fontsize=9)
fig.suptitle("Orienting the view itself — the tilt is the main axis",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()

# %% [markdown]
# Data goes straight onto the tilted view. Here a physically tilted Earth (23.4°
# obliquity, spun to face the Americas) with a handful of world cities;
# `plot_scatter_globe()` keeps the far-side ones from showing through:

# %%
cities = {
    "New York": (-74.0, 40.7), "London": (-0.1, 51.5), "Cairo": (31.2, 30.0),
    "Tokyo": (139.7, 35.7), "Sydney": (151.2, -33.9), "Rio": (-43.2, -22.9),
    "Cape Town": (18.4, -33.9), "Delhi": (77.2, 28.6), "Nairobi": (36.8, -1.3),
}
clon, clat, pole = sph.euler_to_fits_ortho(rotation=70, obliquity=23.44, perspective=15)
fig = plt.figure(figsize=(6.6, 6.6))
ax = sph.make_planet_frame(111, body="earth", center_LONdeg=clon, center_LATdeg=clat,
                           lonpole=pole, Naxispix=700, grid=False)
drape(ax, sph.pseudofits_from_image(EARTH_DAY, geo=True))
globe_labels(ax)
surface_grid(ax)
c_lon = np.array([v[0] for v in cities.values()])
c_lat = np.array([v[1] for v in cities.values()])
sph.plot_scatter_globe(ax, c_lon, c_lat, s=45, c=PAL["accent"],
                       edgecolors="white", linewidths=0.8, zorder=6)
ax.set_title("A tilted Earth with a draped surface and city scatter", fontsize=11)
plt.show()

# %% [markdown]
# ### The `TiltedEarthFrame` coordinate frame
#
# `TiltedEarthFrame` is skyplothelper's own astropy coordinate frame — a subclass
# of astropy's **ITRS** (the Earth-fixed frame) carrying three Euler-angle
# attributes (`rotation`, `obliquity`, `perspective`). Because it is a real
# registered frame, it does double duty:
#
# - **as a coordinate overlay** — drop its rotated graticule onto *another* globe
#   with `ax.get_coords_overlay(te)`, so you can read tilted-frame coordinates off
#   an upright body grid;
# - **as a transform** — astropy registers the frame transforms, so you can convert
#   positions between standard ITRS and the tilted/rotated view directly.
#
# The two panels below contrast a plain **ITRS** overlay (north-up, red) with the
# **tilted** frame (accent) on the same upright body grid. Each overlay's labels
# are drawn in its own grid color; the tilted frame's pole is marked with an ×:

# %%
overlays = [
    (ITRS(), URANOMETRIA[3], "Standard ITRS\n(north-up reference)"),
    (sph.TiltedEarthFrame(rotation=0 * u.deg, obliquity=35 * u.deg,
                          perspective=20 * u.deg),
     PAL["accent"], "TiltedEarthFrame\n(35° tilt, 20° precession)"),
]
fig = plt.figure(figsize=(10.4, 5.4))
for i, (frame, col, title) in enumerate(overlays):
    ax = sph.make_globe_frame(121 + i, center_LONdeg=0, center_LATdeg=0,
                              lon_deg_spacing=30, lat_deg_spacing=30,
                              gridcolor="0.75", gridalpha=0.5)
    # Hide the host body grid's own labels; each colored overlay supplies its own.
    sph.add_overlay_ticks(ax, lon_at="axis", lat_at="axis",
                          show_labels=False, show_ticks=False)
    overlay = ax.get_coords_overlay(frame)
    for ci in (0, 1):
        overlay[ci].set_ticks(spacing=30 * u.deg, color=col)
        overlay[ci].set_ticks_visible(False)
        overlay[ci].set_format_unit(u.deg)
        overlay[ci].set_ticklabel(color=col, size=7)
    overlay.grid(color=col, alpha=0.9, lw=1.1)
    ax.plot(0, 90, marker="x", ms=11, mew=2.6, color=col,
            transform=ax.get_transform(frame), zorder=6)
    ax.set_title(title, fontsize=10)
fig.suptitle("A coordinate overlay: north-up ITRS vs the same frame tilted",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# %% [markdown]
# ## 5. Day and night with nightshade
#
# A satellite view of Earth has a day side and a night side, divided by the
# **terminator** — the twilight line where the Sun sits on the horizon.
# `make_nightshade_blend()` computes that line for a given date and time and
# returns an RGBA overlay: transparent on the day side, opaque on the night side,
# with a soft twilight falloff in between. Lay it over a day map and the night
# side darkens (or, with a night-lights image, lights up).
#
# The progression below builds it up one step at a time — first the flat maps,
# then the blend, then the blend on a globe.
#
# **Step 1 — the day and night maps, and the blend, in plate carrée.** The raw
# equirectangular rasters are just images; the blended panel overlays the
# nightshade RGBA (built from the night-lights map) on the day map for a chosen
# moment.

# %%
day_img = plt.imread(EARTH_DAY)
night_img = plt.imread(EARTH_NIGHT)
night_f = night_img.astype(float) / 255.0
when = dt.datetime(2024, 6, 21, 21, 0)        # solstice, 21:00 UTC — night over Europe/Africa
EXTENT = [-180, 180, -90, 90]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.0))
axes[0].imshow(day_img, extent=EXTENT, origin="upper")
axes[0].set_title("Day map (Blue Marble)", fontsize=10)
axes[1].imshow(night_img, extent=EXTENT, origin="upper")
axes[1].set_title("Night lights (Black Marble)", fontsize=10)
axes[2].imshow(day_img, extent=EXTENT, origin="upper")
night_rgba = sph.make_nightshade_blend(night_f, when, blend_sigma=60)
axes[2].imshow(night_rgba, extent=EXTENT, origin="upper")
# A finished map wants a scale bar; on a cylindrical (plate carrée) map the scale
# varies with latitude, so add_scale_bar_cylindrical (what add_scale_bar hands off
# to here) draws it exactly along a chosen parallel.
sph.add_scale_bar_cylindrical(axes[2], lat=45, body="earth", length_km=2000,
                              color="white", stroke_color="0.1")
axes[2].set_title("Blended at 21:00 UTC (+ scale bar at 45°)", fontsize=10)
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Nightshade, step by step: flat maps then the blend", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# %% [markdown]
# **Two terminator models.** The default `blend='elevation'` computes the *actual*
# solar elevation across the surface, so the terminator is physical and twilight
# fades over the real civil/nautical/astronomical range — note how the band
# narrows toward the poles near the solstice Sun. `blend='gaussian'` instead
# smooths a hard day/night mask by a fixed `blend_sigma` pixels: a stylized,
# uniform fade for when you want a softer look rather than physical twilight.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 3.4))
for ax, blend, subtitle in [(axes[0], "elevation", "physical twilight (default)"),
                            (axes[1], "gaussian", "stylized uniform fade")]:
    ax.imshow(day_img, extent=EXTENT, origin="upper")
    rgba = sph.make_nightshade_blend(night_f, when, blend=blend, blend_sigma=60)
    ax.imshow(rgba, extent=EXTENT, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"blend='{blend}' — {subtitle}", fontsize=10)
fig.suptitle("The two terminator models at the same moment", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# %% [markdown]
# **Step 2 — the blend on a globe.** Drape the day map on a tilted Earth, then
# reproject the same nightshade RGBA onto the frame and lay it on top. The
# terminator now curves across the sphere exactly as it does from orbit.

# %%
# Center on Africa so the night side (over Asia/the Indian Ocean at 21:00 UTC) is
# squarely in view, not hiding around the eastern limb.
clon, clat, pole = sph.euler_to_fits_ortho(rotation=-25, obliquity=23.44, perspective=12)
fig = plt.figure(figsize=(6.6, 6.6))
ax = sph.make_planet_frame(111, body="earth", center_LONdeg=clon, center_LATdeg=clat,
                           lonpole=pole, Naxispix=700, grid=False)
day_hdu = sph.pseudofits_from_image(EARTH_DAY, geo=True)
night_hdu = sph.pseudofits_from_image(EARTH_NIGHT, geo=True)
_, out_hdr = drape(ax, day_hdu, zorder=1)
globe_labels(ax)
# Reproject the nightshade overlay onto the same frame and lay it on top.
night_rgba = sph.make_nightshade_blend(night_f, when, blend_sigma=80)
night_tmp = pyfits.ImageHDU(night_rgba, night_hdu.header)
ax.imshow(np.nan_to_num(sph.reproject_rgb_map(
    night_tmp, out_hdr, shape_out=(out_hdr["NAXIS2"], out_hdr["NAXIS1"]))), zorder=2)
surface_grid(ax)
ax.set_title("Nightshade on a tilted globe (21:00 UTC, summer solstice)", fontsize=11)
plt.show()

# %% [markdown]
# **Step 3 — the terminator through the day.** Holding the globe still and
# stepping the clock through the day sweeps the terminator across the face — the
# Americas go from dawn to dusk. (Equivalently, you could spin the globe under a
# fixed Sun; `make_globe_angles()` generates the orientation sequences for an
# animation — see [Animations](animations.ipynb).)

# %%
clon, clat, pole = sph.euler_to_fits_ortho(rotation=75, obliquity=23.44, perspective=0)
times = [dt.datetime(2024, 6, 21, h, 0) for h in (6, 12, 18, 23)]
fig = plt.figure(figsize=(13, 3.6))
for i, t in enumerate(times):
    ax = sph.make_planet_frame(141 + i, body="earth", center_LONdeg=clon,
                               center_LATdeg=clat, lonpole=pole, Naxispix=400,
                               grid=False)
    _, out_hdr = drape(ax, day_hdu, zorder=1)
    globe_labels(ax)
    rgba = sph.make_nightshade_blend(night_f, t, blend_sigma=80)
    tmp = pyfits.ImageHDU(rgba, night_hdu.header)
    ax.imshow(np.nan_to_num(sph.reproject_rgb_map(
        tmp, out_hdr, shape_out=(out_hdr["NAXIS2"], out_hdr["NAXIS1"]))), zorder=2)
    ax.set_title(t.strftime("%H:%M UTC"), fontsize=10)
fig.suptitle("The terminator sweeping across the day", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# %% [markdown]
# ## 6. Globe decorations and geodesics
#
# The simplest map furniture comes first, on a flat plate-carrée map:
#
# - `add_checkered_border()` — the surveyor-style alternating border, in any pair
#   of `colors`. On a rectangular map it frames the edges; on a **circular** polar
#   plot it rings the disk (its natural home — think a pole-centered chart with
#   Antarctica in the middle).
# - `add_compass_rose()` — the N/E/S/W rose, in a `'simple'` star or `'arrow'` style.
# - `add_scale_bar()` — a real-distance bar; on a cylindrical map it dispatches to
#   `add_scale_bar_cylindrical()`, exact along a chosen parallel.
#
# Decorations over imagery want a color that pops *and* a thin stroke, so they read
# on both the bright land and the dark ocean. Here a custom navy/cream border, with
# a warm-orange compass and scale bar; and a south-polar view where the circular
# border belongs:

# %%
DECO = "#E0552E"        # warm orange — reads over land and ocean alike
fig = plt.figure(figsize=(13, 4.6))

# (left) Plate carrée — a custom-colored checkered border, a compass, a scale bar.
ax = fig.add_subplot(121)
ax.imshow(day_img, extent=[-180, 180, -90, 90], origin="upper")
ax.set_xticks([])
ax.set_yticks([])
sph.add_checkered_border(ax, segment_spacing_deg=30, colors=("#1F3A5F", "#E8DDB5"))
# A full N/E/S/W rose shows the longitude (east-right) direction, not just north.
sph.add_compass_rose(ax, x=0.1, y=0.8, size=26, style="simple", color=DECO,
                     label_color=DECO, stroke_color="white", stroke_lw=1.4)
sph.add_scale_bar_cylindrical(ax, lat=0, body="earth", length_km=3000,
                              color=DECO, stroke_color="white", stroke_lw=1.4)
ax.set_title("Plate carrée — custom-colored border", fontsize=10)

# (right) South-polar globe — the circular border ringing the disk. Longitude
# labels follow an outer parallel (radial/tangent), instead of the default labels
# that pile up near the pole at the center.
ax = sph.make_planet_frame(122, body="earth", center_LONdeg=0, center_LATdeg=-90,
                           Naxispix=600, grid=False)
drape(ax, sph.pseudofits_from_image(EARTH_DAY, geo=True))
surface_grid(ax)
# Longitude labels follow an outer parallel (lat=-30, radial along the meridians)
# and latitude labels run up the central meridian — one add_overlay_ticks call
# restyles both, replacing the frame's default labels.
sph.add_overlay_ticks(ax, lon_at="lat=-30", lat_at="axis",
                      lon_vals=np.arange(0, 360, 45), show_ticks=False,
                      stroke_lw=1.1, stroke_color="0.1",
                      label_kwargs={"color": "white", "fontsize": 8})
sph.add_checkered_border(ax, n_segments=24, colors=("0.1", "white"))
ax.set_title("South-polar view — circular border", fontsize=10)

fig.suptitle("Checkered borders: rectangular on a flat map, circular on a pole",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# %% [markdown]
# On a globe, the orientation furniture is the **pole rod**, a **compass**, and a
# distance scale bar. Because these sit on a raster surface, we color them for
# contrast (light strokes on the dark ocean) rather than from the theme palette.
# `add_pole_rod()` skewers the poles — occluded correctly where it passes behind
# the sphere — so the spin axis reads at a glance. For the compass there are two
# kinds: `add_compass_rose()` pins a fixed rose to a **corner** (in axes fraction),
# while `add_surface_compass()` plants one **on the surface** at a chosen lon/lat,
# warping with the projection to show the local N/E/S/W *there*.

# %%
clon, clat, pole = sph.euler_to_fits_ortho(rotation=60, obliquity=23.44, perspective=10)
fig = plt.figure(figsize=(6.8, 6.8))
ax = sph.make_planet_frame(111, body="earth", center_LONdeg=clon, center_LATdeg=clat,
                           lonpole=pole, Naxispix=700, grid=False)
drape(ax, sph.pseudofits_from_image(EARTH_DAY, geo=True))
globe_labels(ax)
surface_grid(ax)
sph.add_pole_rod(ax, color="white", stroke_color="0.1")
# Two-tone roses read over light land and dark ocean alike: give the rose a real
# color and the white half comes from color_alt / the hollow points (an all-white
# rose would blend its two halves together). Corner rose (axes fraction) +
# on-surface compass; add_surface_compass's size_deg is the full tip-to-tip span,
# so size_deg=10 is a ~10° rose.
COMPASS_C = "#C0392B"   # warm red — distinct against the blue ocean
sph.add_compass_rose(ax, x=0.12, y=0.85, size=34, color=COMPASS_C, label_color="white",
                     style="simple", stroke_color="0.1", stroke_lw=1.2)
sph.add_surface_compass(ax, -30, 45, size_deg=10, style="star", color=COMPASS_C,
                        color_alt="white", label_color="white", stroke_color="0.1")
sph.add_scale_bar(ax, lon_0=clon, lat_0=clat, body="earth", length_km=4000,
                  color="white", stroke_color="0.1")
ax.set_title("Globe furniture: pole rod, compasses, scale bar", fontsize=11)
plt.show()

# %% [markdown]
# **A km is not the same number of degrees on every body.** The scale bar
# reads the body radius from `planet_radii`, so the *same* angular span is a
# very different distance on Earth versus the Moon versus Jupiter. Pass
# `body=` and the length comes out right; the `add_scale_bar_curved_parallel`
# variant also takes a `style='checkered'` for a segmented ruler.

# %%
# Each body draped with its own surface; Jupiter rotated to bring the Great Red
# Spot into view. The scale bar reads each body's radius automatically.
bars = [
    ("earth", EARTH_DAY, 2000, "plain", (20, 5)),
    ("mars", f"{DATA}/planet_maps/2k_mars.jpg", 1000, "plain", (290, 5)),
    ("moon", f"{DATA}/planet_maps/2k_moon.jpg", 500, "checkered", (0, 5)),
    ("jupiter", f"{DATA}/planet_maps/2k_jupiter.jpg", 20000, "checkered", (200, 5)),
]
fig = plt.figure(figsize=(13, 4.0))
for i, (body, path, length, style, (clon, clat)) in enumerate(bars):
    ax = sph.make_planet_frame(141 + i, body=body, center_LONdeg=clon,
                               center_LATdeg=clat, Naxispix=500, grid=False)
    drape(ax, sph.pseudofits_from_image(path, geo=True))
    # These rasters look identical in light and dark mode, so the graticule and
    # labels use a fixed high-contrast pale gray (0.9) with a thin dark stroke —
    # mid-gray would vanish against the bright lunar surface.
    surface_grid(ax, color="0.9")
    globe_labels(ax, color="0.9")
    sph.add_scale_bar_curved_parallel(ax, lon_0=clon, lat_0=clat, body=body,
                                      length_km=length, style=style, n_segments=4,
                                      color="white", colors=("white", "0.2"),
                                      stroke_color="0.1")
    ax.set_title(f"{body.capitalize()} — {length:,} km bar\n(R = {sph.planet_radii[body]:,.0f} km)",
                 fontsize=9)
fig.suptitle("Scale bars read the body radius from planet_radii", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# %% [markdown]
# **Geodesics on a globe.** The shortest path between two points on a sphere is a
# great circle, and the globe plotters draw it correctly — bending across the
# face, hidden on the far side. `plot_baselines()` draws a whole **network** of
# them between named ground stations (the interferometric baselines of a VLBI
# array), while `plot_line_globe()` with a `great_circle_arc()` traces a single
# **flight path**. Station co-visibility and the deeper baseline-network analysis
# live with the other kinematics tools (see
# [Vector Fields & Sky Kinematics](vector_fields.ipynb)).

# %%
# A global VLBI array — antennas on four continents. plot_baselines draws every
# pairwise baseline; sites on the far side of the globe (and their labels) are
# culled automatically, so the network reads cleanly from any viewpoint.
vlbi = {
    "MK": (-155.46, 19.80), "GBT": (-79.84, 38.43), "Yebes": (-3.09, 40.52),
    "Effelsberg": (6.88, 50.52), "Onsala": (11.93, 57.40), "Tianma": (121.20, 31.10),
    "HartRAO": (27.69, -25.89), "Hobart": (147.44, -42.80),
}
fp_lon, fp_lat = sph.great_circle_arc(-74.0, 40.7, -0.1, 51.5, n_pts=120)   # NY -> London
fig = plt.figure(figsize=(13, 6.2))

# (left) On a globe — baselines bend across the face, far side culled.
ax = sph.make_planet_frame(121, body="earth", center_LONdeg=-35, center_LATdeg=35,
                           Naxispix=600, grid=False)
drape(ax, sph.pseudofits_from_image(EARTH_DAY, geo=True))
globe_labels(ax)
surface_grid(ax)
sph.plot_baselines(ax, vlbi, color=PAL["accent"], linewidth=1.0,
                   marker_color="white", marker_edgecolor=PAL["frame"],
                   site_label_color="white", site_label_fontsize=7)
sph.plot_line_globe(ax, fp_lon, fp_lat, color=PAL["accent2"], lw=2.6, ls="--",
                    densify=False)
ax.set_title("On a globe", fontsize=11)

# (right) The same network on a flat plate-carrée map, for comparison.
ax = fig.add_subplot(122)
ax.imshow(day_img, extent=[-180, 180, -90, 90], origin="upper")
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xticks([])
ax.set_yticks([])
sph.plot_baselines(ax, vlbi, color=PAL["accent"], linewidth=1.0,
                   marker_color="white", marker_edgecolor=PAL["frame"],
                   site_label_color="white", site_label_fontsize=7)
ax.plot(fp_lon, fp_lat, color=PAL["accent2"], lw=2.0, ls="--")
ax.set_title("On plate carrée", fontsize=11)

fig.suptitle("A global VLBI network and a great-circle flight path", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# ## 7. Earth features without cartopy
#
# You don't need cartopy to draw Earth's coastlines, land, rivers, plate
# boundaries, or time zones — skyplothelper ships lightweight vector versions that
# go straight onto a planet frame:
#
# - `plot_coastlines()` — continental outlines;
# - `plot_land()` — filled land (optionally `lakes=True` to punch lake holes);
# - `plot_lakes()` / `plot_rivers()` — inland water;
# - `plot_tectonic_plates()` — plate boundaries, or filled plates
#   (`fill=True`, with a categorical / single / `values=`-choropleth color);
# - `plot_time_zones()` — the UTC-offset meridians (or filled bands);
# - `clip_to_land()` / `clip_to_ocean()` — mask any artist to the coastline.
#
# These draw from the small vector data files fetched once by
# `sph.prepare_earth_data()` (see the note at the top — Natural Earth for the
# coastline/land/water/time-zone layers, Bird 2003 via fraxen for the plate
# polygons), or `sph.fetch_boundary_data()` if you point it at a mirror. Under the
# hood `plot_boundaries_globe()` / `plot_boundaries_ortho()` draw any boundary
# dataset, and `split_segments()` breaks polylines at the visibility horizon so
# nothing trails across the back of the globe. The fills route through the same
# region machinery as `add_spherical_polygon` (§6 of the *Regions* tutorial), so
# they work on the flat all-sky projections and the custom Robinson/Eckert frames
# too, not just the globe.

# %%
# One-time setup: fetch the vector Earth data (coastlines / land / lakes / rivers
# / tectonic plates / time zones) into the local cache. Run this once per
# environment before the feature helpers below will find anything to draw. Needs
# the optional `cartopy` extra (Natural Earth layers) and a network connection.
# sph.prepare_earth_data()

clon, clat, pole = sph.euler_to_fits_ortho(rotation=50, obliquity=23.44, perspective=8)
fig = plt.figure(figsize=(13, 6.4))

# (a) Vector features on a wireframe globe — no raster, no cartopy.
ax = sph.make_planet_frame(121, body="earth", center_LONdeg=clon, center_LATdeg=clat,
                           lonpole=pole, Naxispix=500, grid=True,
                           gridcolor="0.85", gridalpha=0.7)
sph.plot_coastlines(ax, color=PAL["label"], lw=0.6)
sph.plot_tectonic_plates(ax, color=PAL["accent"], lw=1.0)
sph.plot_time_zones(ax, color="0.7", lw=0.4)
ax.set_title("(a) Coastlines + plate boundaries + time zones", fontsize=10)

# (b) The same vector features as analysis over the Blue Marble photo.
ax = sph.make_planet_frame(122, body="earth", center_LONdeg=clon, center_LATdeg=clat,
                           lonpole=pole, Naxispix=600, grid=False)
drape(ax, sph.pseudofits_from_image(EARTH_DAY, geo=True))
globe_labels(ax)
sph.plot_tectonic_plates(ax, color="#FF5A3C", lw=1.4)
sph.plot_time_zones(ax, color="white", lw=0.4, alpha=0.5)
ax.set_title("(b) Plate boundaries + time zones over Blue Marble", fontsize=10)

fig.suptitle("Earth features drawn directly on a planet frame — no cartopy", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# ### Filled features
#
# The features above are drawn as *lines* (matplotlib ``Line2D`` objects, via
# ``ax.plot``); the same Earth data can also be drawn as **filled regions**. The
# fills route through skyplothelper's region machinery (the engine behind
# `add_spherical_polygon`), so they honor the projection seam on whatever frame
# you give them:
#
# - **`plot_land(lakes=True)`** — filled continents with the lakes punched out as
#   holes; `plot_lakes()` / `plot_rivers()` add inland water;
# - **`plot_tectonic_plates(fill=True)`** — filled plates: a single color, a
#   categorical map (panel b), or a `values=`-driven **choropleth**;
# - **`clip_to_land()` / `clip_to_ocean()`** — mask *any* artist (an image drape,
#   a data field) to the coastline, so an analysis shows only where it means
#   something (panel c).
#
# Here we draw them on a flat, whole-world **Mollweide** map (`projection='MOL'`)
# so every plate and coastline is visible at once — the same calls fill on the
# SIN globe above, but a flat all-sky frame shows the whole surface in one view.

# %%
fig = plt.figure(figsize=(15, 5))

# (a) A filled physical map: land (with lake holes), rivers, coastlines.
ax = sph.make_planet_frame(131, body="earth", projection="MOL", center_LONdeg=0,
                           grid=True, gridcolor="0.6", gridalpha=0.35)
sph.plot_land(ax, lakes=True, facecolor="#cbb994")
sph.plot_rivers(ax, color="#3a7bd5", lw=0.5)
sph.plot_coastlines(ax, color=PAL["label"], lw=0.4)
ax.set_title("(a) Filled land, lakes, rivers", fontsize=10)

# (b) Tectonic plates as a filled categorical choropleth.
ax = sph.make_planet_frame(132, body="earth", projection="MOL", center_LONdeg=0,
                           grid=False)
sph.plot_tectonic_plates(ax, fill=True, cmap="tab20", alpha=0.85, edgecolor="0.3")
sph.plot_coastlines(ax, color=PAL["label"], lw=0.4)
ax.set_title("(b) Tectonic plates, filled", fontsize=10)

# (c) Mask a data field to the ocean with clip_to_ocean.
ax = sph.make_planet_frame(133, body="earth", projection="MOL", center_LONdeg=0,
                           grid=True, gridcolor="0.6", gridalpha=0.35)
mesh = ax.pcolormesh(LONG, LATG, FIELD, transform=ax.get_transform("world"),
                     cmap="magma", shading="auto", zorder=1)
sph.clip_to_ocean(ax, mesh)
sph.plot_coastlines(ax, color=PAL["label"], lw=0.4)
ax.set_title("(c) A data field, clipped to ocean", fontsize=10)

fig.suptitle("Filled Earth features — land/lakes/rivers, plate choropleth, "
             "clip-to-coastline", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# ## 8. The cartopy backend
#
# When you *do* want cartopy's full feature stack — land/ocean fills, national
# borders, its whole projection library — build a cartopy `GeoAxes` instead of a
# WCS globe. `make_cartopy_frame()` (and the one-call `cartopy_figure()`) wrap it,
# defaulting to the geographic east-right convention like `make_planet_frame()`.
# `list_cartopy_projections()` enumerates the options:

# %%
sph.list_cartopy_projections()

# %% [markdown]
# The two backends complement each other. Cartopy brings the richer terrestrial
# feature set; the skyplothelper WCS globe brings a true **obliquity + perspective
# tilt** — the physically tilted, spinning planet of §4 — which cartopy's standard
# orthographic view doesn't offer. Side by side, with the same view:

# %%
fig = plt.figure(figsize=(12, 6))

# Left: the sph WCS globe, tilted to Earth's real obliquity.
clon, clat, pole = sph.euler_to_fits_ortho(rotation=60, obliquity=23.44, perspective=12)
ax1 = sph.make_planet_frame(121, body="earth", center_LONdeg=clon, center_LATdeg=clat,
                            lonpole=pole, Naxispix=600, grid=False)
drape(ax1, sph.pseudofits_from_image(EARTH_DAY, geo=True))
globe_labels(ax1)
surface_grid(ax1)
ax1.set_title("skyplothelper WCS globe — tilted 23.4°", fontsize=10)

# Right: a cartopy orthographic GeoAxes with the feature stack.
ax2 = sph.make_cartopy_frame(122, projection="orthographic", center=(-60, 30),
                             frame="ITRS", coastlines=True, land=True, ocean=True,
                             land_color="0.75", ocean_color="#AFC7DA", fig=fig)
ax2.set_title("cartopy GeoAxes — land/ocean/coastlines", fontsize=10)

fig.suptitle("Two backends: WCS-globe tilt vs cartopy's feature stack", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# > **Note:** the cartopy path needs the optional `cartopy` extra. Reach for it
# > when you want borders, filled land/ocean, or a cartopy-only projection; reach
# > for the WCS globe when you want an obliquely tilted planet, a celestial
# > hemisphere, or to drive the rest of the skyplothelper toolkit (overlays,
# > regions, HEALPix maps) onto the same frame.

# %% [markdown]
# ## 9. Putting it together
#
# One figure that stacks the whole toolkit: a physically tilted Earth with a
# draped Blue Marble surface and a nightshade terminator, the VLBA baseline network
# and a great-circle flight path on top, and the orientation furniture — pole rod,
# an on-surface compass, and a real-distance scale bar. Swap in your own raster,
# your own station list, and your own date, and the same code is your finished
# figure.

# %%
# The ten VLBA antennas fill this Americas-facing view densely (the global array
# of §6 would mostly fall on the far side here).
vlba = {
    "MK": (-155.46, 19.80), "BR": (-119.68, 48.13), "OV": (-118.28, 37.23),
    "KP": (-111.61, 31.96), "PT": (-108.12, 34.30), "LA": (-106.25, 35.78),
    "FD": (-103.94, 30.64), "NL": (-91.57, 41.77), "HN": (-71.99, 42.93),
    "SC": (-64.58, 17.76),
}
cap_lon, cap_lat, cap_pole = sph.euler_to_fits_ortho(rotation=86, obliquity=23.44,
                                                     perspective=8)
when_cap = dt.datetime(2024, 6, 21, 2, 0)        # evening over the Americas

fig = plt.figure(figsize=(7.6, 7.6))
ax = sph.make_planet_frame(111, body="earth", center_LONdeg=cap_lon,
                           center_LATdeg=cap_lat, lonpole=cap_pole, Naxispix=800,
                           grid=False)
# Day surface + nightshade terminator.
_, cap_hdr = drape(ax, sph.pseudofits_from_image(EARTH_DAY, geo=True), zorder=1)
globe_labels(ax)
cap_night = sph.make_nightshade_blend(night_f, when_cap, blend_sigma=80)
cap_tmp = pyfits.ImageHDU(cap_night, sph.pseudofits_from_image(EARTH_NIGHT, geo=True).header)
ax.imshow(np.nan_to_num(sph.reproject_rgb_map(
    cap_tmp, cap_hdr, shape_out=(cap_hdr["NAXIS2"], cap_hdr["NAXIS1"]))), zorder=2)
surface_grid(ax)
# VLBA network + flight path.
sph.plot_baselines(ax, vlba, color=PAL["accent"], linewidth=1.0, marker_color="white",
                   marker_edgecolor=PAL["frame"], site_label_color="white",
                   site_label_fontsize=6)
fp_lon, fp_lat = sph.great_circle_arc(-74.0, 40.7, -157.86, 21.3, n_pts=120)
sph.plot_line_globe(ax, fp_lon, fp_lat, color=PAL["accent2"], lw=2.4, ls="--",
                    densify=False)
# Orientation furniture. On a globe an on-surface compass reads more naturally
# than a corner rose, so the capstone plants one out over the South Pacific — in
# uranometria green and the wheat used on the §6 checkered border.
WHEAT = "#E8DDB5"
sph.add_pole_rod(ax, color=WHEAT, stroke_color="0.1")
sph.add_surface_compass(ax, 240, -15, size_deg=14, style="star", color=URANOMETRIA[4],
                        color_alt=WHEAT, label_color=WHEAT, stroke_color="0.1")
sph.add_scale_bar(ax, lon_0=cap_lon, lat_0=cap_lat, body="earth", length_km=4000,
                  color=WHEAT, stroke_color="0.1")
ax.set_title("A tilted Earth at dusk: surface, terminator, network, and furniture",
             fontsize=11)
plt.show()

# %% [markdown]
# ## 10. Where to go next
#
# - **[Markers: Rotatable & Image](markers.ipynb)** — scatter *image stamps*
#   (icons, planet thumbnails) at sky or surface positions with the `imscatter`
#   family, including the hemisphere-aware `imscatter_globe()`.
# - **[Vector Fields & Sky Kinematics](vector_fields.ipynb)** — station
#   co-visibility regions and the deeper VLBI baseline-network analysis that
#   builds on `plot_baselines()`.
# - **[Constellations & Asterisms](constellations.ipynb)** — the full
#   constellation-boundary treatment (lines, labels, fills), including the globe
#   case that §2 only previewed.
# - **[Animations](animations.ipynb)** — rotate and nutate a planet by feeding
#   `make_globe_angles()` sequences through `euler_to_fits_ortho()`, one frame at
#   a time.
# - [A Tour of Projections](projections.ipynb) — where the orthographic globe sits
#   in the wider projection landscape.
# - Guide pages: [Globe and planet plots](../guide/globe.md),
#   [Core concepts](../guide/concepts.md) (the astro/geo convention in depth).
