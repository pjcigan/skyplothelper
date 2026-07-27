"""Build the README's visual assets — committed, NOT run at build time.

Two products, written to ``docs/_static/readme/``:

* ``hero.jpg``    — the hero banner: several identically-projected AIT panels,
                    each reconstructing a dark-mode tutorial figure, diagonally
                    spliced into one coherent all-sky map (same frame +
                    graticule throughout; only the content changes across the
                    wedges). Left -> right: capstone-with-insets, regions,
                    constellations, HEALPix (its colorbar lands on the right).
* ``gallery-{light,dark}.jpg`` — a justified "photo-wall" mosaic of the Feature
                    Gallery (globes, planets, cones, HEALPix, regions, …) for the
                    README body, one per color scheme for the README's
                    ``<picture>`` switcher. Recurate via ``GALLERY_ORDER``.

Run it (outputs are committed, so this only needs re-running when you change
the panels or the gallery)::

    python docs/make_readme_assets.py            # both
    python docs/make_readme_assets.py hero       # just the hero
    python docs/make_readme_assets.py gallery    # just the mosaic

Customizing the hero: edit the ``panel_*`` functions in ``build_hero`` (and the
``HERO_PANELS`` list) and the knobs ``HERO_ANGLE`` / ``HERO_FEATHER`` /
``HERO_SEAM``. Keep everything in the "IDENTICAL GEOMETRY" block the same across
panels — that (``ax.set_position(SKY_RECT)``) is what makes the ovals register
when spliced. Only the rightmost panel's colorbar shows in the composite; the
others reserve no colorbar. The look matches the notebooks' dark pass
(``set_style(theme='dark_sky', palette='nightcap')``, ``uranometria`` element
colors, ``sph.deepsky`` / ``sph.dusk`` colormaps).

Non-bundled data (background images, catalogs) is read through the ``DATA`` dict
near the top via ``optional_path`` — keep the committed copy's paths as
PLACEHOLDERS (your real local paths stay in a private copy, e.g. under
``hidden/``); missing files are skipped so panels fall back to synthetic data.
"""
import base64
import io
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

# Ensure the in-repo package wins over any same-named shim on PYTHONPATH
# (only skyplothelper, imported lazily inside build_hero, needs this).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

OUT = os.path.join(_REPO, "docs/_static/readme")
PLOT_TYPES = os.path.join(_REPO, "docs/_static/features")
os.makedirs(OUT, exist_ok=True)

# --- Optional external data for the hero panels ----------------------------
# Panels may want background images / catalogs that are NOT bundled with the
# package. Keep the committed copy's entries as PLACEHOLDERS so no
# machine-specific paths ever ship — put your real local paths in a private
# copy of this file (e.g. under hidden/). Missing files are simply skipped
# (panels fall back to synthetic data), so this committed script still runs
# anywhere and produces a valid hero.
DATA = {
    # Equirectangular Milky Way panorama for the globe inset (panel 1).
    #"allsky_background": "PATH/TO/your_allsky_panorama.jpg",
    "allsky_background": os.path.join(_REPO, "examples/data/Allsky_noirlab2430b_1280x640.jpg"),
}

# Bundled (git-tracked) catalogs — safe to reference directly.
HIP_CSV = os.path.join(_REPO, "examples/data/hipparcos_bright_pm.csv")


def optional_path(key):
    """Return an existing, expanded path for DATA[key], or None (with a note)."""
    p = os.path.expanduser(DATA.get(key, "") or "")
    if p and os.path.exists(p):
        return p
    if p:
        print(f"  (note: DATA[{key!r}] not found — panel falls back to synthetic data)")
    return None


# ---------------------------------------------------------------------------
# 1. Diagonal "sliced-panel" compositor
# ---------------------------------------------------------------------------
def diagonal_composite(paths, angle_deg=18.0, feather_px=10.0, seam=True,
                       seam_color=(70, 70, 70), seam_width=2, white=248):
    """Stitch N same-size images, split by tilted slashes at uniform x, with
    feathered seams. A thin seam line (optional) is masked to inked pixels so
    it never streaks across white margins. Returns a PIL ``Image``."""
    imgs = [np.asarray(Image.open(p).convert("RGB"), dtype=float) for p in paths]
    H, W, _ = imgs[0].shape
    for p, im in zip(paths, imgs):
        if im.shape != imgs[0].shape:
            raise SystemExit(f"size mismatch: {p} is {im.shape[1]}x{im.shape[0]}, "
                             f"expected {W}x{H} — all panels must share dimensions.")
    N = len(imgs)
    t = np.tan(np.deg2rad(angle_deg))
    xd = np.arange(W)[None, :] + (np.arange(H)[:, None] - H / 2.0) * t  # sheared x
    band = W / N

    def smoothstep(u):
        u = np.clip(u, 0.0, 1.0)
        return u * u * (3 - 2 * u)

    f = max(feather_px, 1e-6)
    weights = []
    for i in range(N):
        left, right = i * band, (i + 1) * band
        wl = np.ones((H, W)) if i == 0 else smoothstep((xd - (left - f)) / (2 * f))
        wr = np.ones((H, W)) if i == N - 1 else 1.0 - smoothstep((xd - (right - f)) / (2 * f))
        weights.append(wl * wr)
    wsum = np.sum(weights, axis=0) + 1e-9
    out = sum(imgs[i] * weights[i][..., None] for i in range(N)) / wsum[..., None]

    if seam:
        layer = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(layer)
        for k in range(1, N):
            xk = k * band
            d.line([(xk + (H / 2.0) * t, 0), (xk - (H / 2.0) * t, H)], fill=255, width=seam_width)
        m = (np.asarray(layer, float) / 255.0 * (out.min(axis=2) < white))[..., None]
        out = out * (1 - m) + np.array(seam_color, float) * m

    return Image.fromarray(out.clip(0, 255).astype("uint8"))


# ---------------------------------------------------------------------------
# 2. Hero panels — identical geometry, dark, spliced into hero.jpg
# ---------------------------------------------------------------------------
HERO_ANGLE, HERO_FEATHER, HERO_SEAM = 18.0, 13.0, False   # feathered, no line (dark)

# IDENTICAL GEOMETRY — keep the same in every panel.
FIGSIZE, DPI = (8, 4.2), 150
SKY_RECT = [0.02, 0.05, 0.83, 0.90]      # WCSAxes position -> identical pixels
CBAR_RECT = [0.885, 0.15, 0.022, 0.70]   # colorbar axes position (only panel 4)
DARK_BG = "#010409"                      # dark_sky figure.facecolor
KEEP_PANELS = False                      # set True (e.g. from a runner) to keep
#                                          the un-sliced per-panel PNGs for inspection


def build_hero():
    import shutil
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap

    import skyplothelper as sph

    # Dark look, matching the notebooks' dark pass.
    sph.set_style(base="structural", theme="dark_sky", palette="nightcap")
    # The 'structural' preset enables savefig.bbox='tight', which content-crops
    # each panel to a different size — force full-canvas saves so every panel
    # shares identical pixel dimensions (required for the diagonal splice).
    plt.rcParams.update({"savefig.bbox": "standard", "savefig.pad_inches": 0.0})
    PAL = sph.ANNOTATION_PALETTES["dark"]
    URAN = sph.CYCLE_PALETTES["uranometria"]["colors"]

    if KEEP_PANELS:
        panel_dir = os.path.join(OUT, "_hero_panels")
        os.makedirs(panel_dir, exist_ok=True)
    else:
        panel_dir = tempfile.mkdtemp(prefix="sph_hero_")
    rng = np.random.default_rng(11)

    # ---- identical-geometry helpers ----
    def new_frame(colorbar=False):
        fig, ax = sph.allsky_figure(projection="AIT", center=180, figsize=FIGSIZE)
        ax.set_position(SKY_RECT)
        fig.canvas.draw()
        cax = fig.add_axes(CBAR_RECT) if colorbar else None
        return fig, ax, cax

    def world(ax):
        return ax.get_transform("world")

    def save(fig, name):
        path = os.path.join(panel_dir, name)
        fig.savefig(path, dpi=DPI, facecolor=DARK_BG)
        plt.close(fig)
        return path

    def arrow_cmap(name, lo=0.12, hi=0.72):
        base = sph.get_colormap(name)
        return LinearSegmentedColormap.from_list(f"{name}_hi", base(np.linspace(lo, hi, 256)))

    def star_sizes(vmag, scale=0.5, mlim=6.5):
        return np.clip(mlim - np.asarray(vmag), 0.1, None) ** 2 * scale

    def sphere_grid(n_lat=9, lat_max=70.0, dlon=20.0):
        lon, lat = [], []
        for b in np.linspace(-lat_max, lat_max, n_lat):
            n = max(4, round(360 * np.cos(np.radians(b)) / dlon))
            ll = np.linspace(0, 360, n, endpoint=False)
            lon += list(ll)
            lat += [b] * len(ll)
        return np.array(lon), np.array(lat)

    def rand_sky(n):
        return rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))

    # ================= EDIT: one function per panel =================
    # Each returns a saved PNG path. Only panel_healpix draws a colorbar.

    # -- Panel 1: capstone frame with two LEFT insets (globe = MW raster +
    #    galactic-aberration vectors; zoom = TAN patch). --
    def panel_insets():
        from astropy.visualization.wcsaxes.frame import EllipticalFrame
        fig, ax, _ = new_frame()
        # ROI on the LEFT of the map (RA>180 is left on center=180 east-left), so
        # the marked box + connector lines fall inside this leftmost slice.
        # Kept low (dec<0) so it clears the top-left globe inset; nudged further
        # left (higher RA) so both inset connector lines are visible.
        ROI = (300, -8)
        gl, gb = rand_sky(600)
        gz = rng.uniform(0.02, 0.16, gl.size)
        ax.scatter(gl, gb, c=gz, s=6, cmap="sph.dusk", vmin=0.01, vmax=0.16,
                   alpha=0.7, lw=0, transform=world(ax), zorder=3)
        try:  # denser second-coordinate (galactic) grid on the main axis
            sph.add_coord_overlay(ax, frame="galactic", color=URAN[5], alpha=0.30, lw=0.7,
                                  lon_vals=np.arange(0, 360, 15),
                                  lat_vals=np.arange(-75, 76, 15))
        except Exception as e:
            print(f"    (insets: overlay grid skipped: {e})")
        # smooth background contours of a made-up scalar field, drawn BEHIND the
        # data — and repeated on the zoom inset below, so the same overlay renders
        # on the inset, driving home that it is its own independent WCSAxes.
        def sky_field(lon, lat):   # a few compact "image features" near the ROI
            return (np.exp(-(((lon - 300) / 13) ** 2 + ((lat + 3) / 11) ** 2))
                    + 0.85 * np.exp(-(((lon - 285) / 9) ** 2 + ((lat - 9) / 8) ** 2))
                    + 0.6 * np.exp(-(((lon - 313) / 11) ** 2 + ((lat - 12) / 7) ** 2))
                    + 0.5 * np.exp(-(((lon - 296) / 8) ** 2 + ((lat + 16) / 6) ** 2)))
        try:   # cmapped contours tracing local features around the ROI
            fx, fy = np.meshgrid(np.linspace(ROI[0] - 34, ROI[0] + 34, 90),
                                 np.linspace(ROI[1] - 28, ROI[1] + 28, 80))
            sph.add_contour_overlay(ax, fx, fy, sky_field(fx, fy), levels=8,
                                    cmap="sph.thicket", linewidths=0.9,
                                    alpha=0.85, zorder=2)
        except Exception as e:
            print(f"    (insets: background contours skipped: {e})")
        # globe inset (upper-left, smaller): MW raster + galactic aberration,
        # a subtle grid, and small geodesic ticklabels along the top + left.
        try:
            globe = sph.reproject_inset_axes(
                ax, rect=[0.02, 0.62, 0.135, 0.27], transform="figure",
                projection="SIN", center=(60, 12), size=180, frame_class=EllipticalFrame)
            bg = optional_path("allsky_background")
            if bg is not None:
                img, hdr = sph.load_sky_image(bg, frame="galactic", center=0)
                globe.imshow(sph.reproject_background(img, hdr, globe), origin="lower", zorder=0)
            glon, glat = sphere_grid(9, 70, 22)
            dlon, dlat = sph.vsh_field(glon, glat, {"D_1": 5.05})
            sph.plot_sky_vectors(globe, glon, glat, dlon, dlat, units="uas", scale="auto",
                                 auto_target_deg=13.0, color_by_magnitude=True,
                                 cmap=arrow_cmap("dusk"), width=0.015)
            try:
                globe.coords.grid(True, color=PAL["grid"], alpha=0.55, lw=0.5)
                globe.coords[0].set_ticklabel(size=4.5, color=PAL["label"])
                globe.coords[1].set_ticklabel(size=4.5, color=PAL["label"])
                globe.coords[0].set_ticklabel_position("t")   # longitude on top
                globe.coords[1].set_ticklabel_position("l")   # latitude on left
            except Exception as e:
                print(f"    (insets: globe grid/ticks skipped: {e})")
        except Exception as e:
            print(f"    (insets: globe inset skipped: {e})")
        # zoom inset (lower-left, smaller): TAN patch on the ROI
        try:
            zoom = sph.reproject_inset_axes(
                ax, rect=[0.05, 0.11, 0.145, 0.185], transform="figure",
                projection="TAN", center=ROI, size=(44, 32), bg_color=DARK_BG)
            try:   # the SAME background contours as the main axis (independent WCSAxes)
                zfx, zfy = np.meshgrid(np.linspace(ROI[0] - 28, ROI[0] + 28, 60),
                                       np.linspace(ROI[1] - 22, ROI[1] + 22, 60))
                sph.add_contour_overlay(zoom, zfx, zfy, sky_field(zfx, zfy), levels=8,
                                        cmap="sph.thicket", linewidths=0.9,
                                        alpha=0.95, zorder=1)
            except Exception as e:
                print(f"    (insets: zoom contours skipped: {e})")
            m = (np.abs(((gl - ROI[0] + 180) % 360) - 180) < 28) & (np.abs(gb - ROI[1]) < 20)
            zoom.scatter(gl[m], gb[m], c=gz[m], s=26, cmap="sph.dusk", vmin=0.01, vmax=0.16,
                         alpha=0.85, lw=0, transform=zoom.get_transform("world"), zorder=3)
            try:   # relabel the zoom as OFFSET coords to demo that feature
                sph.apply_offset_ticks(zoom, unit="deg", show_unit=False)
            except Exception as e:
                print(f"    (insets: zoom offset ticks skipped: {e})")
            zoom.coords[0].set_axislabel(r"$\Delta\alpha\,\cos\delta$", fontsize=5.5)
            zoom.coords[1].set_axislabel(r"$\Delta\delta$", fontsize=5.5)
            for c in (0, 1):
                zoom.coords[c].set_ticklabel(size=5)
            sph.mark_inset_axes(ax, zoom, edgecolor=PAL["accent2"], linewidth=1.5)
            sph.connect_inset_axes(ax, zoom, color=PAL["accent2"], linewidth=1.2)
        except Exception as e:
            print(f"    (insets: zoom inset skipped: {e})")
        return save(fig, "1_insets.png")

    # -- Panel 2: a trajectory through a solar-avoidance zone, galactic band,
    #    plus a compound "complex region" crescent. --
    def panel_regions():
        fig, ax, _ = new_frame()
        SUN_POS = (215, 10)   # stand-in "Sun" placed in this slice (RA~215 = left-center)
        SUN_PNG = os.path.join(_REPO, "examples/data/icons/sun1_120pix.png")
        # galactic-plane band
        try:
            sph.add_frame_band(ax, -10, 10, frame="galactic", facecolor=URAN[2], alpha=0.20)
        except Exception as e:
            print(f"    (regions: band skipped: {e})")
        # solar-avoidance zone (swap SUN_POS for get_sun(Time(...)) if you like)
        avoid = sph.CompoundRegion(ax).add_circle(*SUN_POS, radius_deg=38)
        avoid.render(facecolor=URAN[4], alpha=0.20)
        avoid.render_boundary(color=URAN[4], linewidth=1.3)
        # dotted trajectory, split observable / too-near-Sun
        t = np.linspace(0, 1, 110)
        tra, tdec = 130 + 150 * t, 40 * np.sin(2 * np.pi * t) + 6
        blocked = np.asarray(avoid.contains_points(tra, tdec), dtype=bool)
        ax.plot(tra, tdec, transform=world(ax), color=PAL["grid"], lw=1.0,
                ls=(0, (2, 3)), zorder=4)
        ax.scatter(tra[~blocked], tdec[~blocked], transform=world(ax), s=15,
                   color=URAN[2], zorder=5, label="observable")
        # blocked points a vivid red, drawn on top so the avoidance-region fill
        # can't desaturate them into a washed-out grey.
        ax.scatter(tra[blocked], tdec[blocked], transform=world(ax), s=18,
                   color="#e0556b", edgecolor=DARK_BG, lw=0.4, zorder=7,
                   label="too near Sun")
        # small Sun icon on the track, inside the avoidance zone
        try:
            spx, spy = ax.wcs.world_to_pixel_values(*SUN_POS)
            sph.imscatter(spx, spy, SUN_PNG, ax=ax, zoom=0.16, zorder=6)
        except Exception as e:
            print(f"    (regions: sun icon skipped: {e})")
        # fill the top: compass + ruler
        try:
            sph.add_compass(ax, (0.55, 0.88), color=PAL["accent"], length=0.06,
                            fontsize=8, stroke_color=DARK_BG)
        except Exception as e:
            print(f"    (regions: compass skipped: {e})")
        # a boxed second-coordinate (galactic) overlay up top — demos overlay grids
        try:
            from matplotlib.patches import Polygon as MplPolygon
            gbox = np.array([[200, 32], [252, 32], [252, 59], [200, 59], [200, 32]], float)
            ax.plot(gbox[:, 0], gbox[:, 1], transform=world(ax), color=URAN[0], lw=1.3, zorder=6)
            clip = MplPolygon(gbox, closed=True, facecolor="none", edgecolor="none",
                              transform=world(ax))
            ax.add_patch(clip)
            inner = sph.add_coord_overlay(ax, frame="galactic", color=URAN[5],
                                          alpha=0.9, lw=0.7, ls="-",
                                          lon_vals=np.arange(0, 360, 10),
                                          lat_vals=np.arange(-80, 81, 10))
            for arts in inner.lon_artists + inner.lat_artists:
                for ln in (arts if isinstance(arts, (list, tuple)) else [arts]):
                    ln.set_clip_path(clip)
            # in-frame tick labels: along the galactic gridlines through the box
            # (a central parallel / meridian), not the box edges; small + red
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            gc = SkyCoord(gbox[:-1, 0] * u.deg, gbox[:-1, 1] * u.deg).galactic
            gls, gbs = gc.l.deg, gc.b.deg
            lon_vals = np.arange(np.floor(gls.min() / 10) * 10, gls.max() + 5, 10)
            lat_vals = np.arange(np.floor(gbs.min() / 5) * 5, gbs.max() + 2, 5)
            sph.add_overlay_ticks(
                ax, frame="galactic", lon_vals=lon_vals, lat_vals=lat_vals,
                lon_at=f"lat={np.mean(gbs):.0f}", lat_at=f"lon={np.mean(gls):.0f}",
                tick_kwargs=dict(color=URAN[5], length=2, lw=0.6),
                label_kwargs=dict(color=URAN[5], fontsize=5))
        except Exception as e:
            print(f"    (regions: overlay box skipped: {e})")
        # a compound "complex region" crescent lower
        try:
            sph.CompoundRegion(ax).add_circle(215, -28, radius_deg=26) \
               .xor_circle(245, -28, radius_deg=26) \
               .render(facecolor=URAN[0], alpha=0.16, edgecolor=URAN[0], lw=1.0)
        except Exception as e:
            print(f"    (regions: crescent skipped: {e})")
        return save(fig, "2_regions.png")

    # -- Panel 3: a star chart at a glance (dark) — Hipparcos stars,
    #    constellation lines/labels, galactic band, reticle + ruler on M51. --
    def panel_constellations():
        fig, ax, _ = new_frame()
        # colored constellation regions (in this slice: RA ~90-180 on center=180),
        # like the notebook's add_constellation_polygon fills.
        for name, ci in [("Leo", 0), ("Cnc", 2), ("Gem", 4), ("Hya", 5),
                         ("LMi", 3), ("Vir", 1)]:
            try:
                sph.add_constellation_polygon(ax, name, facecolor=URAN[ci],
                                              edgecolor="none", alpha=0.22, zorder=1)
            except Exception as e:
                print(f"    (constellations: polygon {name} skipped: {e})")
        try:
            stars = pd.read_csv(HIP_CSV)
            ax.scatter(stars.RAICRS, stars.DEICRS, s=star_sizes(stars.Vmag, 0.5),
                       c=PAL["stars"], lw=0, alpha=0.85, zorder=3, transform=world(ax))
        except Exception as e:
            print(f"    (constellations: star catalog skipped: {e})")
        for fn in (lambda: sph.add_constellation_boundaries(ax),
                   lambda: sph.add_constellation_lines(ax, rank_max=1, color=PAL["accent"],
                                                       lw=0.9, alpha=0.85),
                   lambda: sph.add_constellation_labels(ax, fontsize=6),
                   lambda: sph.add_frame_band(ax, -10, 10, frame="galactic",
                                              facecolor=URAN[2], alpha=0.14)):
            try:
                fn()
            except Exception as e:
                print(f"    (constellations: overlay skipped: {e})")
        # reticle on M44 (Beehive Cluster, in Cancer) — within this slice.
        # (The Ruler is a field/image measurement tool and does not render on an
        #  all-sky frame, so it is not used in this hero.)
        m44 = (130.05, 19.62)
        try:
            sph.add_reticle(ax, m44, style="circle", size=15, color=PAL["accent2"],
                            lw=1.6, label="M44", label_color=PAL["accent2"],
                            label_side="E")
        except Exception as e:
            print(f"    (constellations: reticle skipped: {e})")
        return save(fig, "3_constellations.png")

    # -- Panel 4 (rightmost): HEALPix density as discrete TILES + a catalog
    #    scatter (size = uncertainty, color = redshift via a 2nd colorbar, shape
    #    = flag), with a MultiLegend (size + shape) in the bottom-right — the
    #    flagship multi-channel legend. Two stacked colorbars (density + z).
    #    Bright tiles concentrated on the RIGHT (RA~40). --
    def panel_healpix():
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        fig, ax, _ = new_frame()
        cax_top = fig.add_axes([0.885, 0.545, 0.022, 0.355])   # HEALPix density
        cax_bot = fig.add_axes([0.885, 0.115, 0.022, 0.355])   # redshift z
        # HEALPix tile map with a bright clump on the right
        n = 9000
        ra = np.concatenate([rng.uniform(0, 360, n // 2),
                             rng.normal(40, 16, n // 2) % 360])
        dec = np.concatenate([np.degrees(np.arcsin(rng.uniform(-1, 1, n // 2))),
                              rng.normal(10, 12, n // 2)])
        m8 = sph.healpix_smooth(
            sph.bin_data_as_healpix(ra, dec, np.ones(ra.size), nside=8,
                                    statistic="count", blank_value=0)[0], sigma_deg=10.0)
        vlo, vhi = float(np.nanmin(m8)), float(np.nanmax(m8))
        coll = sph.plot_healpix_sparse(np.arange(len(m8)), m8, nside=8, ax=ax,
                                       backend="patch", cmap="sph.deepsky",
                                       vmin=vlo, vmax=vhi, show_boundaries=True,
                                       boundary_color="0.35", boundary_lw=0.3,
                                       set_extent=False)
        try:
            sph.add_colorbar(coll, cax=cax_top, label="HEALPix density")
        except Exception as e:
            print(f"    (healpix: tile colorbar failed ({e}); building one)")
            sm = ScalarMappable(norm=Normalize(vlo, vhi), cmap=sph.get_colormap("deepsky"))
            plt.colorbar(sm, cax=cax_top, label="HEALPix density")
        # catalog scatter: size=uncertainty, color=redshift (2nd colorbar),
        # shape=flag (2 shapes, one plot_catalog call each, shared color scale)
        try:
            ns = 150
            sra = rng.uniform(0, 95, ns)
            sdec = rng.uniform(-50, 45, ns)
            unc = rng.uniform(1.0, 50.0, ns)          # "uncertainty" -> marker size
            zz = rng.uniform(0.02, 0.16, ns)          # redshift -> color
            flag = rng.random(ns) < 0.3               # 2 shapes
            cp_ref = None
            for fl, mk, z in [(True, "^", 8), (False, "o", 7)]:
                mm = flag == fl
                if not mm.any():
                    continue
                cp = sph.plot_catalog(
                    ax, {"ra": sra[mm], "dec": sdec[mm], "unc": unc[mm], "z": zz[mm]},
                    sizeby="unc", size_vlim=(1.0, 50.0), size_scale="sqrt",
                    smin=4, smax=70, colorby="z", cmap="sph.mariner",
                    vmin=0.01, vmax=0.16, marker=mk, alpha=0.7, edgecolor=DARK_BG,
                    linewidths=0.4, frame="icrs", zorder=z)
                cp_ref = cp_ref or cp
            mappable = getattr(cp_ref, "scatter", cp_ref)
            sph.add_colorbar(mappable, cax=cax_bot, label="redshift  z")
            # compact 2-column (horizontal) legend tucked into the lower-right gap
            (sph.MultiLegend(ax, loc="lower right", orientation="horizontal",
                             palette="dark", facecolor=PAL["ax_bg"], framealpha=0.9,
                             frameon=True, fontsize=6, borderpad=0.4, block_sep=10,
                             stroke_color=DARK_BG, stroke_lw=1.5)
                .add_size_from(cp_ref, values=[5, 20, 50], title="unc.",
                               fmt=".0f", ncol=1)
                .add_shape("flag", {"flagged": "^", "clean": "o"}, size=7)
                .draw())
        except Exception as e:
            print(f"    (healpix: scatter/MultiLegend skipped: {e})")
        return save(fig, "4_healpix.png")

    # ---- optional extra panels (not in the default HERO_PANELS) ----
    def panel_overlaygrids():   # optional extra panel (add to HERO_PANELS to use)
        fig, ax, _ = new_frame()
        try:
            sph.style_grid(ax, color="0.4", lw=0.6, ls=":", alpha=1.0)
            sph.add_coord_overlay(ax, frame="galactic", color=URAN[5], alpha=0.85, lw=1.0)
            box = np.array([[110, -40], [250, -40], [250, 40], [110, 40], [110, -40]], float)
            ax.plot(box[:, 0], box[:, 1], transform=world(ax), color=URAN[0], lw=1.8)
            sph.add_overlay_ticks(ax, frame="galactic", boundary=box,
                                  lon_at="boundary", lat_at="boundary")
        except Exception as e:
            print(f"    (overlaygrids skipped: {e})")
        return save(fig, "5_overlaygrids.png")

    def panel_vectors():        # optional extra panel (add to HERO_PANELS to use)
        fig, ax = sph.allsky_figure(projection="AIT", center=0, frame="galactic", figsize=FIGSIZE)
        ax.set_position(SKY_RECT)
        fig.canvas.draw()
        glon, glat = sphere_grid(11, 75, 15)
        dlon, dlat = sph.vsh_field(glon, glat, {"D_1": 5.05})
        sph.plot_sky_vectors(ax, glon, glat, dlon, dlat, units="uas", scale="auto",
                             auto_target_deg=9.0, color_by_magnitude=True,
                             cmap=arrow_cmap("dusk"), width=0.005)
        return save(fig, "6_vectors.png")

    # left -> right slices: insets (left), regions, constellations, healpix (right)
    HERO_PANELS = [panel_insets, panel_regions, panel_constellations, panel_healpix]

    def safe(fn):
        try:
            return fn()
        except Exception as e:
            import traceback
            print(f"  PANEL {fn.__name__} FAILED: {e}")
            traceback.print_exc()
            fig, ax, _ = new_frame()
            return save(fig, fn.__name__ + "_blank.png")

    paths = [safe(fn) for fn in HERO_PANELS]
    img = diagonal_composite(paths, angle_deg=HERO_ANGLE, feather_px=HERO_FEATHER, seam=HERO_SEAM)
    dest = os.path.join(OUT, "hero.jpg")
    img.save(dest, quality=90, optimize=True, progressive=True)
    if not KEEP_PANELS:
        shutil.rmtree(panel_dir, ignore_errors=True)
    else:
        print(f"  (kept per-panel PNGs in {panel_dir})")
    print(f"hero.jpg    {os.path.getsize(dest)//1024:4d} KB  {img.size}  ({len(paths)} panels)")


# ---------------------------------------------------------------------------
# 3. Body gallery — justified "photo-wall" mosaic, light + dark
# ---------------------------------------------------------------------------
# Built in BOTH modes (gallery-light.jpg / gallery-dark.jpg) for the README's
# <picture> switcher. Tiles come from three reproducible sources, all read from
# committed artifacts (nothing hand-clipped):
#   ("nb", stem)      a tutorial capstone. The DARK tile is the transparent
#                     docs/_static/nb_dark/<stem>.png (flattened); the LIGHT tile
#                     is that figure's own inline output in the executed
#                     docs/tutorials/<notebook>.ipynb. The two are paired via the
#                     <notebook>.dark.js manifest (slugs in figure-output order,
#                     written by make_tutorial_dark_figs.py) — NOT by filename
#                     sort, which does not match cell order. A fail-loud aspect
#                     check catches any future drift.
#   ("feature", stem) a Feature Gallery entry's {stem}-{mode}.png pair.
#   ("render", key)   a figure re-rendered here in both modes (RENDER_TILES) —
#                     for showcases that add an sph feature on top of a notebook
#                     figure (e.g. a Ruler on the mcf SN 1987A composite).
# Gutters/background match each mode's figure facecolor exactly (white for the
# light figures, #1D1C1A for the 'denim' dark figures) so tiles blend seamlessly.
GALLERY_W, GALLERY_GUT, GALLERY_H = 1200, 8, 250
GALLERY_BG = {"light": (255, 255, 255), "dark": (0x1D, 0x1C, 0x1A)}
NB_DARK = os.path.join(_REPO, "docs/_static/nb_dark")
TUTORIALS = os.path.join(_REPO, "docs/tutorials")

# Curated for variety + wow; edit freely to recurate. Ordered to interleave wide
# all-sky ovals with square/portrait tiles so the justified rows pack evenly.
GALLERY_TILES = [
    ("nb", "globe_plots__9-putting-it-together"),          # tilted Earth at dusk
    ("nb", "projections__9-putting-it-together-2"),        # ICRF3 sources all-sky
    ("nb", "annotations__3-putting-it-together"),          # SN 1987A ALMA, furnished
    ("nb", "decorating_frames__highlighting-families-with-a-colormap"),
    ("nb", "fits_images__4-quicklook-in-one-call"),        # 3C 84 jet
    ("nb", "constellations__galactic-frame-chart"),        # constellations, galactic
    ("nb", "catalogs__capstone-virgo"),                    # Virgo cluster, labeled
    ("nb", "vector_fields__1-the-moving-sky"),             # galactic-aberration field
    ("nb", "insets_and_zoom__postage-stamps"),             # all-sky + zoom insets
    ("nb", "vector_fields__8-planning-a-vlbi-session"),    # region contains sources
    ("nb", "cone_bowtie__bowtie-horizontal"),              # redshift bowtie
    ("nb", "catalogs__legend-capstone"),                   # USNO VLBI, 3 channels
    ("nb", "healpix_workflows__7-putting-it-together"),    # HEALPix density + mask
    ("nb", "overlay_grids__6-putting-it-together"),        # Messier over Milky Way
    # Optical SN 1987A: the mcf four-band HST composite with an sph Ruler on top.
    # Off by default (the ALMA SN 1987A tile above already shows beam + scale);
    # uncomment to swap in the optical view, and drop a tile to keep the count.
    # ("render", "sn1987a_ruler"),
]


# Render-tile builders (mode, bg) -> opaque PIL image, populated below.
RENDER_TILES: dict = {}


def _sn1987a_ruler_tile(mode, bg):
    """The mcf SN 1987A four-band HST composite (YBPR) with an sph Ruler scale
    bar drawn on top — a showcase of an sph decoration over an mcf color image,
    doing more than a plain mcf display. Rendered fresh in both modes so it can
    drop straight into GALLERY_TILES. It is commented out there by default: the
    ALMA SN 1987A tile already carries the beam + projected-size info, so this is
    the optical alternative to swap in, not a second SN 1987A on by default.

    Recipe mirrors the fits_images tutorial's worked example (to_grey_rgb ->
    colorize_image -> combine_multicolor); the Ruler is the added sph element."""
    import warnings

    import matplotlib.pyplot as plt
    import multicolorfits as mcf
    from astropy.io import fits
    from astropy.wcs import WCS

    import skyplothelper as sph

    # On-image ticks read against the black composite, so they stay light in both
    # modes (like the other deep-sky tiles); only the figure margin follows the
    # mode. margin ink = dark on the light page, warm off-white on the dark page.
    margin_ink = "#1A1A1A" if mode == "light" else "#D9D5C5"

    def load(name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = fits.open(os.path.join(_REPO, "examples/data", name))[0]
            return np.squeeze(h.data).astype(float), h.header, WCS(h.header).celestial

    bands = ["sn1987a_hst_F625W.fits", "sn1987a_hst_F656N.fits",
             "sn1987a_hst_F658N.fits", "sn1987a_hst_F502N.fits"]
    colors = ["#FBFCCF", "#729FCF", "#75507B", "#EF2929"]
    labels = ["F625W", "Hα (F656N)", "[N II] (F658N)", "[O III] (F502N)"]
    grays, wcs, hdr = [], None, None
    for b in bands:
        d, h, w = load(b)
        wcs = wcs or w
        hdr = hdr if hdr is not None else h
        grays.append(mcf.to_grey_rgb(d, rescalefn="log", scaletype="perc",
                                     min_max=[40, 99.9]))
    rgb = mcf.combine_multicolor(
        [mcf.colorize_image(g, c, colorintype="hex", gammacorr_color=2.2)
         for g, c in zip(grays, colors)], gamma=2.2)

    with plt.style.context("default"):
        fig = plt.figure(figsize=(5.6, 5.6), facecolor=_hex(bg))
        ax = fig.add_subplot(111, projection=wcs)
        ax.imshow(rgb, origin="lower")
        sph.format_ticklabels(ax, color="0.85")          # light ticks on the image
        for k in ("ra", "dec"):
            ax.coords[k].set_axislabel(
                {"ra": "Right Ascension (J2000)", "dec": "Declination (J2000)"}[k],
                color=margin_ink, fontsize=9)
        ax.coords["ra"].set_ticklabel(color="0.85", fontsize=7)
        ax.coords["dec"].set_ticklabel(color="0.85", fontsize=7)
        sph.add_bandlabels(ax, labels, colors, fontsize=9)
        # The added sph feature over the plain mcf image: an angular scale bar.
        # (A Ruler suits a long measured span; at SN 1987A's ~1" ring scale the
        # purpose-built sizebar reads far cleaner than a tick-subdivided Ruler.)
        sph.add_sizebar_asec(ax, hdr, 1.0, '1"', color="0.9",
                             stroke_color="black", stroke_lw=2.2)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=118, facecolor=fig.get_facecolor())
        plt.close(fig)
    buf.seek(0)
    return _flatten(Image.open(buf), bg)


RENDER_TILES["sn1987a_ruler"] = _sn1987a_ruler_tile


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


def _manifest_slugs(notebook):
    """Figure slugs in figure-output (= inline-output) order, from the
    ``<notebook>.dark.js`` manifest make_tutorial_dark_figs.py writes."""
    js = open(os.path.join(NB_DARK, f"{notebook}.dark.js")).read()
    return json.loads(re.search(r"=\s*(\[.*\]);", js, re.S).group(1))


def _flatten(im, bg):
    im = im.convert("RGBA")
    out = Image.new("RGB", im.size, bg)
    out.paste(im, mask=im.split()[3])
    return out


def _nb_tile(stem, mode, bg):
    """(image) for a tutorial capstone in *mode*. Dark = the nb_dark transparent
    PNG; light = the same figure's inline output in the executed notebook, paired
    through the dark.js manifest. Raises if the pair's aspect ratios disagree
    (manifest/notebook drift), rather than silently mosaicking mismatched tiles."""
    notebook, slug = stem.split("__", 1)
    idx = _manifest_slugs(notebook).index(slug)
    nb = json.load(open(os.path.join(TUTORIALS, f"{notebook}.ipynb")))
    inline = [o["data"]["image/png"]
              for c in nb["cells"] if c["cell_type"] == "code"
              for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    light = Image.open(io.BytesIO(base64.b64decode(inline[idx])))
    dark = Image.open(os.path.join(NB_DARK, f"{stem}.png"))
    if abs(light.width / light.height - dark.width / dark.height) > 0.06:
        raise SystemExit(f"{stem}: light/dark aspect mismatch "
                         f"({light.size} vs {dark.size}) — dark.js/notebook drift?")
    return _flatten(light if mode == "light" else dark, bg)


def _feature_tile(stem, mode, bg):
    return _flatten(Image.open(os.path.join(PLOT_TYPES, f"{stem}-{mode}.png")), bg)


def _tile(kind, ident, mode, bg):
    if kind == "nb":
        return _nb_tile(ident, mode, bg)
    if kind == "feature":
        return _feature_tile(ident, mode, bg)
    if kind == "render":
        return RENDER_TILES[ident](mode, bg)
    raise SystemExit(f"unknown tile kind {kind!r}")


def build_gallery():
    for mode in ("light", "dark"):
        _build_gallery_mode(mode, GALLERY_BG[mode])


def _build_gallery_mode(mode, bg):
    imgs = [_tile(kind, ident, mode, bg) for kind, ident in GALLERY_TILES]
    rows, cur = [], []
    for im in imgs:
        cur.append(im)
        w = sum(round(i.width * GALLERY_H / i.height) for i in cur) + GALLERY_GUT * (len(cur) - 1)
        if w >= GALLERY_W:
            rows.append(cur)
            cur = []
    if cur:
        rows.append(cur)

    rendered = []
    for ri, row in enumerate(rows):
        avail = GALLERY_W - GALLERY_GUT * (len(row) - 1)
        h = avail / sum(i.width / i.height for i in row)
        if ri == len(rows) - 1 and h > GALLERY_H * 1.35:
            h = GALLERY_H
        h = int(round(h))
        rendered.append([i.resize((max(1, round(i.width * h / i.height)), h), Image.LANCZOS) for i in row])

    total_h = sum(r[0].height for r in rendered) + GALLERY_GUT * (len(rendered) - 1)
    canvas = Image.new("RGB", (GALLERY_W, total_h), bg)
    y = 0
    for row in rendered:
        row_w = sum(im.width for im in row) + GALLERY_GUT * (len(row) - 1)
        x = max(0, (GALLERY_W - row_w) // 2)
        for im in row:
            canvas.paste(im, (x, y))
            x += im.width + GALLERY_GUT
        y += row[0].height + GALLERY_GUT

    dest = os.path.join(OUT, f"gallery-{mode}.jpg")
    canvas.save(dest, quality=86, optimize=True, progressive=True)
    print(f"gallery-{mode}.jpg {os.path.getsize(dest)//1024:4d} KB  {canvas.size}  ({len(rendered)} rows)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "hero"):
        build_hero()
    if which in ("all", "gallery"):
        build_gallery()
