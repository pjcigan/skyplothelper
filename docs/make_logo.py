"""Build skyplothelper's logo / branding marks — committed, NOT run at build time.

Five logo directions, each rendered with skyplothelper itself (dogfooded, so the
marks track the styling system), written to ``docs/_static/logo/``. They serve
different purposes — docs header, favicon, README, and a provenance watermark
for the animation / manim clips people may lift into talks (BSD-licensed, but
worth marking as ours).

Products (each ``-light`` and ``-dark``; the line marks also a transparent
``-mark`` for stamping on video):

* ``logo_1_split_tiles`` / ``logo_1_split_smooth`` — the AIT sky-ellipse, its
  own central meridian bisecting it: bare graticule left, data right (HEALPix
  diamond tiles, or a smooth density field). The "raw sky <-> your data" mark.
* ``logo_2_rosette``   — a gradient-shaded HEALPix diamond cluster inside a
  gridded globe frame; compact monogram / favicon candidate.
* ``logo_3_eye``       — the bare ellipse + graticule (transparent interior).
  The pure line mark; its ``-mark`` variant is the corner watermark.
* ``logo_4_globe``     — an orthographic globe, half wireframe / half tiled
  data; the 3D cousin of #1.
* ``logo_5_reticle``   — a reticle over a faint star field (the "helper" read).
* ``logo_6_wordmark``  — the #3 eye glyph + ``skyplothelper`` set in DejaVu Sans.

Ticks are stripped on every mark (icon, not plot); the graticule alone carries
the "sky frame" read and, on the filled marks, is drawn over the tiles. Colors
are all in-house: ``sph.deepsky`` / ``sph.dusk`` colormaps and the
``publication`` / ``dark`` annotation palettes.

Run it (outputs are committed, so this only needs re-running when you change a
mark)::

    python docs/make_logo.py            # all marks
    python docs/make_logo.py 1 3 6      # just those concept ids (6 = wordmark)
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Ensure the in-repo package wins over any same-named shim on PYTHONPATH.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
OUT = os.path.join(_REPO, "docs/_static/logo")
os.makedirs(OUT, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import healpy as hp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyplothelper as sph  # noqa: E402

DARK_BG = "#010409"   # matches the dark_sky figure.facecolor


# --- theme handling --------------------------------------------------------
# Each mark is drawn once per THEME. ``line_ink`` is the theme-appropriate
# foreground for line marks / text: dark on light, white on dark — used for the
# solid AND the transparent ``-mark`` variants (a watermark stamps in the ink of
# the surface it lands on, not always white).
THEMES = {
    "light": dict(annot="publication",
                  set=dict(base="publication")),
    "dark": dict(annot="dark",
                 set=dict(base="structural", theme="dark_sky", palette="nightcap")),
}


def apply_theme(name):
    t = THEMES[name]
    sph.set_style(**t["set"])
    # full-canvas saves so the marks aren't content-cropped to varying sizes
    plt.rcParams.update({"savefig.bbox": "standard", "savefig.pad_inches": 0.0})
    pal = sph.ANNOTATION_PALETTES[t["annot"]]
    bg = DARK_BG if name == "dark" else "white"
    line_ink = "white" if name == "dark" else pal["frame"]
    return pal, bg, line_ink


def clean_frame(ax):
    """Strip all coordinate ticks + text — an icon, not a plot; the grid alone
    carries the "this is a sky frame" read.

    allsky_figure / make_globe_frame draw their in-frame tick MARKS and LABELS
    not as native ticklabels but as tagged overlay artists (``_sph_overlay_tick``
    / ``_sph_overlay_ticklabel``) plus a few plain Text labels — so hiding the
    coords doesn't remove them. Delete the tagged artists (and any stray Text)
    after a draw; the removal survives the savefig redraw. The caller draws once
    first so the artists exist to remove.
    """
    for c in (0, 1):
        ax.coords[c].set_ticks(size=0)
        ax.coords[c].set_ticklabel_visible(False)
        ax.coords[c].set_axislabel("")
    for art in list(ax.lines) + list(ax.texts) + list(ax.collections):
        if (getattr(art, "_sph_overlay_tick", False)
                or getattr(art, "_sph_overlay_ticklabel", False)):
            art.remove()
    for t in list(ax.texts):
        t.remove()


def grid_over(ax, color, alpha=0.9, lw=0.7):
    """Restyle the graticule so it stays visible over the HEALPix tiles. WCSAxes
    ignores a ``zorder`` on gridlines, so the caller instead pushes the tiles to
    a negative zorder (below the grid's fixed draw stage); here we just set a
    solid, legible line."""
    for c in (0, 1):
        ax.coords[c].grid(draw_grid=True, color=color, alpha=alpha,
                          linewidth=lw, linestyle="-")


def savefig(fig, stem, bg, transparent=False, bbox_inches=None, pad_inches=None):
    path = os.path.join(OUT, stem + ".png")
    # bbox_inches='tight' crops to the drawn content with a uniform pad on every
    # side (used by the wordmark so it centers instead of floating in a fixed
    # canvas); the marks otherwise save full-canvas so they aren't content-cropped
    # to varying sizes.
    extra = {}
    if bbox_inches is not None:
        extra["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        extra["pad_inches"] = pad_inches
    fig.savefig(path, dpi=200, facecolor="none" if transparent else bg,
                transparent=transparent, **extra)
    plt.close(fig)
    print(f"  {stem + '.png':32s} {os.path.getsize(path) // 1024:4d} KB")
    return path


# --- a reusable synthetic "sky density" on HEALPix pixel centers -----------
def density_on_pixels(lon, lat):
    """A few smooth galactic-ish overdensities as a function of (lon, lat)."""
    blobs = [(60, 8, 34, 26, 1.0), (110, -18, 24, 20, 0.8),
             (150, 22, 20, 16, 0.7), (95, 40, 22, 14, 0.6),
             (40, -35, 26, 18, 0.55)]
    f = np.zeros_like(lon, dtype=float)
    for l0, b0, sl, sb, amp in blobs:
        dl = ((lon - l0 + 180) % 360) - 180
        f += amp * np.exp(-((dl / sl) ** 2 + ((lat - b0) / sb) ** 2))
    return f + 0.04


# ===========================================================================
# 1 — split ellipse (AIT): bare graticule left, data right
# ===========================================================================
def logo_split_ellipse(theme):
    for variant, nside, tiles in [("tiles", 8, True), ("smooth", 32, False)]:
        pal, bg, _ = apply_theme(theme)
        fig, ax = sph.allsky_figure(projection="AIT", center=180,
                                    figsize=(6.4, 3.4), grid=True)
        ax.set_position([0.02, 0.04, 0.96, 0.92])
        fig.canvas.draw()
        # RIGHT half (0 < lon < 180 on east-left center=180) gets the data,
        # drawn at low zorder so the graticule can sit on top of it.
        npix = hp.nside2npix(nside)
        lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        field = density_on_pixels(lon, lat)
        right = (lon > 2) & (lon < 178)
        sph.plot_healpix_sparse(
            np.arange(npix)[right], field[right], nside=nside, ax=ax,
            backend="patch", cmap="sph.deepsky", vmin=0.0,
            vmax=float(field.max()), show_boundaries=tiles,
            boundary_color=pal["grid"], boundary_lw=0.25, set_extent=False,
            zorder=-10)
        # graticule over the tiles (visible on the blank left AND the data)
        grid_over(ax, color=pal["grid"], lw=0.7, alpha=0.9)
        ax.coords.frame.set_color(pal["frame"])
        ax.coords.frame.set_linewidth(1.6)
        clean_frame(ax)
        fig.canvas.draw()
        savefig(fig, f"logo_1_split_{variant}-{theme}", bg)


# ===========================================================================
# 2 — HEALPix rosette monogram
# ===========================================================================
def logo_rosette(theme):
    pal, bg, _ = apply_theme(theme)
    nside = 4
    center = (45.0, 20.0)
    cvec = np.asarray(hp.ang2vec(center[0], center[1], lonlat=True))
    idx = hp.query_disc(nside, cvec, np.radians(42), inclusive=True)
    lon, lat = hp.pix2ang(nside, idx, lonlat=True)
    # gradient shade by angular distance from center -> luminous core
    vecs = hp.ang2vec(lon, lat, lonlat=True)          # (N, 3)
    d = np.arccos(np.clip(vecs @ cvec, -1.0, 1.0))
    val = 1.0 - (d / (d.max() + 1e-9))
    # a fine 15deg grid drawn OVER the tiles reads as an obvious framed sky-plot
    ax = sph.make_globe_frame(center_LONdeg=center[0], center_LATdeg=center[1],
                              projection="SIN", grid=True, gridcolor=pal["grid"],
                              gridalpha=0.6, lon_deg_spacing=15,
                              lat_deg_spacing=15, Naxispix=360)
    fig = ax.figure
    fig.set_size_inches(3.6, 3.6)
    ax.set_position([0.06, 0.06, 0.88, 0.88])
    sph.plot_healpix_sparse(idx, val, nside=nside, ax=ax, backend="patch",
                            cmap="sph.dusk", vmin=0.0, vmax=1.0,
                            show_boundaries=True, boundary_color=bg,
                            boundary_lw=1.4, set_extent=False, zorder=-10)
    grid_over(ax, color=pal["grid"], lw=0.6, alpha=0.85)
    ax.set_facecolor("none")
    fig.patch.set_facecolor(bg)
    fig.canvas.draw()
    clean_frame(ax)
    fig.canvas.draw()
    savefig(fig, f"logo_2_rosette-{theme}", bg)


# ===========================================================================
# 3 — graticule "eye" line mark + transparent watermark
# ===========================================================================
def _eye_figure(ink, grid_lw, grid_alpha, frame_lw=2.0):
    """Build the bare AIT "eye": transparent sky, a styled graticule + oval rim,
    all ticks and text stripped. Shared by the #3 line mark and the #6 wordmark
    glyph. The grid weight/opacity is a parameter because the wordmark shows the
    eye much smaller (e.g. a corner mark in the animations), where the #3 mark's
    thin, semi-transparent graticule washes out against the rim — so it renders
    a heavier, opaque grid."""
    fig, ax = sph.allsky_figure(projection="AIT", center=180,
                                figsize=(6.4, 3.4), grid=True)
    ax.set_position([0.02, 0.04, 0.96, 0.92])
    ax.set_facecolor("none")
    fig.canvas.draw()
    sph.style_grid(ax, color=ink, lw=grid_lw, alpha=grid_alpha, ls="-")
    ax.coords.frame.set_color(ink)
    ax.coords.frame.set_linewidth(frame_lw)
    clean_frame(ax)
    fig.canvas.draw()
    return fig


def logo_eye(theme):
    pal, bg, ink = apply_theme(theme)
    for transparent in (False, True):
        # pure LINE mark: transparent sky so only the graticule + rim show
        fig = _eye_figure(ink, grid_lw=0.9, grid_alpha=0.55 if transparent else 0.85)
        savefig(fig, f"logo_3_eye-{theme}" + ("_mark" if transparent else ""),
                bg, transparent=transparent)


# ===========================================================================
# 4 — orthographic globe: half wireframe, half tiled data
# ===========================================================================
def logo_globe(theme):
    pal, bg, _ = apply_theme(theme)
    center = (180.0, 15.0)
    # doubled grid density (15deg) so the graticule reads clearly as a globe;
    # drawn over the tiles, no ticks.
    ax = sph.make_globe_frame(center_LONdeg=center[0], center_LATdeg=center[1],
                              projection="SIN", grid=True, gridcolor=pal["grid"],
                              gridalpha=0.6, lon_deg_spacing=15,
                              lat_deg_spacing=15, Naxispix=360)
    fig = ax.figure
    fig.set_size_inches(3.8, 3.8)
    ax.set_position([0.05, 0.05, 0.90, 0.90])
    nside = 16
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    field = density_on_pixels(lon, lat)
    # Right half of the visible disk, split by POSITION on the sphere rather
    # than a longitude wedge — near the pole the visible hemisphere reaches
    # |dlon| > 90, so a wedge leaves an unfilled sliver up to the pole. Front
    # hemisphere: v.c > 0; right of the central meridian: v.east < 0, where
    # east = north x center points image-LEFT on an east-left globe (small
    # margin keeps a thin seam at the meridian).
    cvec = np.asarray(hp.ang2vec(center[0], center[1], lonlat=True))
    east = np.cross([0.0, 0.0, 1.0], cvec)
    east /= np.linalg.norm(east)
    vecs = hp.ang2vec(lon, lat, lonlat=True)          # (npix, 3)
    right = (vecs @ cvec > 0.0) & (vecs @ east < -0.02)
    sph.plot_healpix_sparse(
        np.arange(npix)[right], field[right], nside=nside, ax=ax,
        backend="patch", cmap="sph.deepsky", vmin=0.0, vmax=float(field.max()),
        show_boundaries=False, set_extent=False, zorder=-10)
    grid_over(ax, color=pal["grid"], lw=0.6, alpha=0.85)
    ax.set_facecolor("none")
    fig.patch.set_facecolor(bg)
    fig.canvas.draw()
    clean_frame(ax)
    fig.canvas.draw()
    savefig(fig, f"logo_4_globe-{theme}", bg)


# ===========================================================================
# 5 — reticle on a faint star field
# ===========================================================================
def logo_reticle(theme):
    pal, bg, _ = apply_theme(theme)
    rng = np.random.default_rng(7)
    fig, ax = sph.allsky_figure(projection="AIT", center=180,
                                figsize=(6.4, 3.4), grid=True)
    ax.set_position([0.02, 0.04, 0.96, 0.92])
    fig.canvas.draw()
    sph.style_grid(ax, color=pal["grid"], lw=0.6, alpha=0.5, ls="-")
    ra = rng.uniform(0, 360, 260)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 260)))
    sz = rng.uniform(1, 5, 260) ** 2 * 0.5
    ax.scatter(ra, dec, s=sz, c=pal.get("stars", pal["label"]), lw=0,
               alpha=0.85, zorder=3, transform=ax.get_transform("world"))
    sph.add_reticle(ax, (180, 6), style="circle", size=22,
                    color=pal["accent"], lw=2.0)
    ax.coords.frame.set_color(pal["frame"])
    ax.coords.frame.set_linewidth(1.6)
    clean_frame(ax)
    fig.canvas.draw()
    savefig(fig, f"logo_5_reticle-{theme}", bg)


# ===========================================================================
# 6 — wordmark lockup: the #3 eye glyph + "skyplothelper"
# ===========================================================================
def logo_wordmark(theme):
    """Eye glyph + "skyplothelper", theme-inked so the two match on both the
    solid and the transparent ``-mark`` variant. Renders its OWN eye with a
    heavier, opaque grid (rather than reusing the #3 watermark eye) so the sky
    graticule stays legible at the small sizes this lockup is used at."""
    pal, bg, ink = apply_theme(theme)
    # Render the eye once and grab it as a transparent RGBA raster to composite.
    # 200 dpi so the buffer is crisp when the lockup is saved at 200 dpi too.
    eye_fig = _eye_figure(ink, grid_lw=1.4, grid_alpha=0.9)
    eye_fig.set_dpi(200)
    eye_fig.patch.set_alpha(0.0)               # transparent behind the glyph
    eye_fig.canvas.draw()
    eye = np.asarray(eye_fig.canvas.buffer_rgba()).copy()
    plt.close(eye_fig)
    for transparent in (False, True):
        # Generous fixed canvas, then bbox_inches='tight' crops it back to the
        # eye + text with equal padding all round — so the lockup centers instead
        # of leaving a wide gap to the right of the (shorter-than-canvas) text.
        fig = plt.figure(figsize=(7.0, 1.7))
        axg = fig.add_axes([0.01, 0.02, 0.28, 0.96])
        axg.axis("off")
        axg.imshow(eye)
        fig.text(0.30, 0.5, "skyplothelper", ha="left", va="center",
                 fontsize=34, color=ink, family="DejaVu Sans")
        savefig(fig, f"logo_6_wordmark-{theme}" + ("_mark" if transparent else ""),
                bg, transparent=transparent, bbox_inches="tight", pad_inches=0.08)


# ===========================================================================
# Template assets — tightly-cropped derivatives wired into the Sphinx theme
# (navbar glyph + favicon). These are the "sized versions" of #3 and #2:
# no page banners, just the chrome marks. See conf.py's html_theme_options
# ["logo"] and html_favicon.
# ===========================================================================
def logo_navbar(theme):
    """Navbar glyph — the #3 eye, cropped tight (no padding) and drawn a touch
    bolder so it holds up beside the title text at ~30px. Transparent so the
    navbar's own light/dark background shows through (image_light = dark ink,
    image_dark = white ink)."""
    pal, bg, ink = apply_theme(theme)
    fig, ax = sph.allsky_figure(projection="AIT", center=180,
                                figsize=(2.2, 1.15), grid=True)
    ax.set_position([0.03, 0.03, 0.94, 0.94])   # ellipse ~fills the canvas
    ax.set_facecolor("none")
    fig.canvas.draw()
    sph.style_grid(ax, color=ink, lw=1.0, alpha=0.8, ls="-")
    ax.coords.frame.set_color(ink)
    ax.coords.frame.set_linewidth(2.4)
    clean_frame(ax)
    fig.canvas.draw()
    savefig(fig, f"logo_navbar-{theme}", bg, transparent=True)


def logo_favicon():
    """Favicon — the #2 rosette, filled out to a near-full tessellated disk and
    cropped tight so the diamonds (not framing) survive at 16-32px. One
    theme-agnostic file on a transparent background: the colored tiles read on
    any tab color; no grid or frame ring (they vanish at that size)."""
    apply_theme("dark")   # only the sph.dusk colormap is used; theme irrelevant
    nside = 4
    center = (45.0, 20.0)
    cvec = np.asarray(hp.ang2vec(center[0], center[1], lonlat=True))
    idx = hp.query_disc(nside, cvec, np.radians(68), inclusive=True)
    lon, lat = hp.pix2ang(nside, idx, lonlat=True)
    vecs = hp.ang2vec(lon, lat, lonlat=True)
    d = np.arccos(np.clip(vecs @ cvec, -1.0, 1.0))
    val = 1.0 - (d / (d.max() + 1e-9))
    ax = sph.make_globe_frame(center_LONdeg=center[0], center_LATdeg=center[1],
                              projection="SIN", grid=False, Naxispix=360)
    fig = ax.figure
    fig.set_size_inches(2.4, 2.4)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    sph.plot_healpix_sparse(idx, val, nside=nside, ax=ax, backend="patch",
                            cmap="sph.dusk", vmin=0.0, vmax=1.0,
                            show_boundaries=True, boundary_color="#0e1117",
                            boundary_lw=1.6, set_extent=False)
    ax.set_facecolor("none")
    try:
        ax.coords.frame.set_visible(False)   # no disk ring for the favicon
    except Exception:
        pass
    clean_frame(ax)
    fig.canvas.draw()
    savefig(fig, "logo_favicon", "white", transparent=True)


CONCEPTS = {
    "1": logo_split_ellipse,
    "2": logo_rosette,
    "3": logo_eye,
    "4": logo_globe,
    "5": logo_reticle,
    "6": logo_wordmark,
    "7": logo_navbar,
}

if __name__ == "__main__":
    # "fav" = the single-file favicon; everything else is per-theme.
    which = sys.argv[1:] or list(CONCEPTS) + ["fav"]
    for key in sorted(which, key=lambda k: (k == "fav", k)):
        if key == "fav":
            print("[fav] logo_favicon")
            logo_favicon()
            continue
        fn = CONCEPTS.get(key)
        if fn is None:
            print(f"unknown concept {key!r}")
            continue
        print(f"[{key}] {fn.__name__}")
        for theme in ("light", "dark"):
            fn(theme)
    print("done ->", OUT)
