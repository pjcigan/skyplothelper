"""Render HEALPix numerical-correctness demonstrations for visual eyeballing.

Produces:
  - hp_01_allsky_smooth.png    — full-sky smoothed Gaussian field
  - hp_02_sparse.png           — plot_healpix_sparse on a few pixels
  - hp_03_smooth_kernel.png    — delta function before/after Gaussian smoothing
  - hp_04_upgrade_downgrade.png — same map at three resolutions
  - hp_05_sources_to_healpix.png — bin sources by count + mean statistic
  - hp_06_query_results.png    — circle_query and polygon_query overlays
  - hp_07_sparse_clusters.png  — three connected clusters (mid-lat, polar, antimeridian)
  - hp_08_highlight_tile.png   — highlight a single HEALPix tile across
                                 frames via ``pc.patch_pixel_index`` (also
                                 covers globe / tilted-globe rendering)
  - hp_09_backends.png         — 4 rendering backends side-by-side
                                 (canvas+imshow, canvas+pcolormesh,
                                 lonlat+pcolormesh legacy, patches)
"""

import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

import skyplothelper as sph
from skyplothelper.healpix import (
    bin_data_as_healpix,
    healpix_allsky_figure,
    healpix_circle_query,
    healpix_downgrade,
    healpix_polygon_query,
    healpix_smooth,
    healpix_to_celestial,
    healpix_upgrade,
    plot_healpix_sparse,
)
from skyplothelper.wcs_frame import make_wcs_frame

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


@_panel("hp_01_allsky_smooth")
def render_allsky_smooth():
    """Smoothed Gaussian field on the full sky."""
    rng = np.random.default_rng(0)
    nside = 32
    npix = hp.nside2npix(nside)
    m = rng.normal(0, 1, npix)
    m_smooth = healpix_smooth(m, sigma_deg=5.0)
    result = healpix_allsky_figure(
        m_smooth, projection="MOL", center=0, cmap="RdBu_r",
        title="HEALPix random field, smoothed with σ=5° "
              "(healpix_allsky_figure)",
        figsize=(11, 5.5),
    )
    return result.fig


@_panel("hp_02_sparse")
def render_sparse():
    """plot_healpix_sparse — sparse ring rendered on six different WCS
    frame types. Verifies the geometry-pipeline integration handles
    antimeridian crossings (the pixel at lon≈0) and aspect-ratio
    preservation consistently across:

      * elliptical all-sky (MOL, AIT)
      * rectangular all-sky (CAR — what the standalone development
        gallery used; renders correctly)
      * sinusoidal (SFL — non-trivial frame shape)
      * tangential field projection (TAN, centered on the ring)
      * orthographic globe (SIN, viewed from above the ring)
    """
    nside = 32
    # Select 30 pixels evenly along one declination (lat=20°)
    lons = np.linspace(0, 360, 30, endpoint=False)
    lats = np.full(30, 20.0)
    pixels = hp.ang2pix(nside, lons, lats, lonlat=True)
    values = np.linspace(0, 1, 30)

    panels = [
        ("MOL", 180, None,    "MOL (elliptical, all-sky)"),
        ("AIT", 180, None,    "AIT (elliptical, all-sky)"),
        ("CAR", 180, None,    "CAR (rectangular, all-sky)"),
        ("SFL", 180, None,    "SFL (sinusoidal, all-sky)"),
        ("TAN", 180, 20,      "TAN (tangent plane, on ring)"),
        ("SIN", 180, 60,      "SIN (orthographic globe, lat₀=60°)"),
    ]
    fig = plt.figure(figsize=(15, 9))
    for idx, (proj, c_lon, c_lat, label) in enumerate(panels, start=1):
        center = (c_lon, c_lat) if c_lat is not None else c_lon
        ax = make_wcs_frame((2, 3, idx), projection=proj, center=center,
                            fig=fig)
        fig.canvas.draw()
        plot_healpix_sparse(pixels, values, nside=nside, ax=ax,
                            cmap="plasma", show_boundaries=True,
                            boundary_color="0.4", boundary_lw=0.5)
        ax.set_title(label, fontsize=10)
    fig.suptitle("plot_healpix_sparse — 30 pixels along lat=20° on six "
                 "WCS frame types", fontsize=12)
    fig.subplots_adjust(top=0.92, hspace=0.40, wspace=0.30)
    return fig


@_panel("hp_07_sparse_clusters")
def render_sparse_clusters():
    """plot_healpix_sparse — connected clusters of tiles. Verifies that
    adjacent HEALPix pixels render seam-to-seam (no gaps from the
    seam-to-seam with no inter-tile gaps) and
    that clusters spanning the antimeridian, the poles, and a generic
    mid-latitude region all behave correctly.

    Three clusters on a single all-sky AIT panel:
      * a ~20°×20° box around (170°, 30°) — generic mid-latitude
      * a north polar cap (lat ≥ 70°) — pole containment
      * a ~20°×20° box straddling the antimeridian — wrap behavior

    Plus a zoomed TAN view that should show one cluster filling
    most of the field with no inter-tile gaps.
    """
    nside = 32
    rng = np.random.default_rng(0)

    # Cluster A: 20°×20° box at (170°, 30°)
    pix_a = hp.query_polygon(
        nside,
        hp.ang2vec([160, 180, 180, 160], [20, 20, 40, 40], lonlat=True),
    )
    vals_a = rng.uniform(0.2, 0.4, len(pix_a))

    # Cluster B: north polar cap, lat ≥ 70°
    pix_b = hp.query_disc(
        nside, hp.ang2vec(0, 90, lonlat=True),
        np.radians(20),
    )
    vals_b = rng.uniform(0.5, 0.7, len(pix_b))

    # Cluster C: antimeridian-straddling 20°×20° box at (350°→10°, -30°→-10°)
    # (built as two adjacent boxes; query_polygon doesn't natively split
    # so use circle_query around the centroid)
    pix_c = hp.query_disc(
        nside, hp.ang2vec(0, -20, lonlat=True),
        np.radians(12),
    )
    vals_c = rng.uniform(0.8, 1.0, len(pix_c))

    all_pix = np.concatenate([pix_a, pix_b, pix_c])
    all_vals = np.concatenate([vals_a, vals_b, vals_c])

    # Panel 1: AIT all-sky with all three clusters
    fig = plt.figure(figsize=(14, 6))
    ax = make_wcs_frame((1, 2, 1), projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    plot_healpix_sparse(all_pix, all_vals, nside=nside, ax=ax,
                        cmap="viridis", show_boundaries=True,
                        boundary_color="0.3", boundary_lw=0.3)
    ax.set_title("AIT — three connected clusters: mid-lat box, polar cap, "
                 "antimeridian crosser", fontsize=10)

    # Panel 2: TAN zoomed to cluster A — should show the cluster filling
    # the field with zero gaps between adjacent tiles.
    ax2 = make_wcs_frame((1, 2, 2), projection="TAN", center=(170, 30),
                         fig=fig)
    fig.canvas.draw()
    plot_healpix_sparse(pix_a, vals_a, nside=nside, ax=ax2,
                        cmap="viridis", show_boundaries=True,
                        boundary_color="0.3", boundary_lw=0.5)
    ax2.set_title("TAN @ (170°, 30°) zoomed — adjacent tiles should "
                  "touch edge-to-edge", fontsize=10)
    fig.subplots_adjust(top=0.9, wspace=0.25)
    return fig


@_panel("hp_09_backends")
def render_backends():
    """Side-by-side comparison of the four ``plot_healpix_allsky``
    backends on the same smooth scalar field.

    Backend-comparison doc panel. Headlines:

    * ``imshow`` (default): canvas-pixel sampling, fastest, lowest
      memory, returns ``AxesImage``.
    * ``pcolormesh + canvas``: same sampling, returns ``QuadMesh``
      (per-cell event picking).
    * ``pcolormesh + lonlat``: legacy lon/lat-grid path, slower,
      kept as a fallback / debug option.
    * ``patch``: per-tile Polygons through the geometry pipeline —
      true tile boundaries, slowest by ~20×.

    At sub-pixel sampling density, all four are visually
    indistinguishable; the choice is about performance and
    return type.
    """
    nside = 32
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    m = np.cos(np.radians(2 * lon)) * np.cos(np.radians(2 * lat))

    fig = plt.figure(figsize=(15, 9))
    panels = [
        ("imshow",     None,      "imshow + canvas (default)"),
        ("pcolormesh", "canvas",  "pcolormesh + canvas"),
        ("pcolormesh", "lonlat",  "pcolormesh + lonlat (legacy)"),
        ("patch",      None,      "patches"),
    ]
    for idx, (backend, sampling, label) in enumerate(panels, start=1):
        ax = make_wcs_frame((2, 2, idx), projection="AIT", center=180,
                            fig=fig)
        fig.canvas.draw()
        # Replicate plot_healpix_allsky's dispatch on the existing
        # axes (the public helper builds its own figure, so we
        # call the lower-level paths directly to get one panel per
        # subplot).
        if backend == "imshow":
            arr, ext = sph.healpix_to_canvas(m, ax)
            ax.imshow(arr, extent=ext, origin="lower", cmap="viridis",
                       vmin=-1, vmax=1, interpolation="nearest",
                       aspect="auto")
        elif backend == "pcolormesh" and sampling == "canvas":
            arr, ext = sph.healpix_to_canvas(m, ax)
            xmin, xmax, ymin, ymax = ext
            ny, nx = arr.shape
            xe = np.linspace(xmin, xmax, nx + 1)
            ye = np.linspace(ymin, ymax, ny + 1)
            ax.pcolormesh(xe, ye, arr, cmap="viridis", vmin=-1, vmax=1)
        elif backend == "pcolormesh":  # lonlat
            plonc, platc, pvals = healpix_to_celestial(
                m, "allsky", 180, (1500, 750))
            ax.pcolormesh(plonc, platc, pvals,
                          transform=ax.get_transform("world"),
                          cmap="viridis", vmin=-1, vmax=1)
        else:  # patch
            plot_healpix_sparse(np.arange(npix), m, nside=nside, ax=ax,
                                 cmap="viridis", vmin=-1, vmax=1,
                                 set_extent=False, backend="patch")
        ax.set_title(label, fontsize=10)
    fig.suptitle("plot_healpix_allsky backends — same data, "
                 "cos(2λ)cos(2β), nside=32",
                 fontsize=12)
    fig.subplots_adjust(top=0.92, hspace=0.30, wspace=0.20)
    return fig


@_panel("hp_08_highlight_tile")
def render_highlight_tile():
    """Highlight a single HEALPix tile across four frame types using the
    ``patch_pixel_index`` attribute exposed by the patches backend
    . Doubles as the visual coverage for HEALPix
    rendering on the orthographic / globe frame.

    Same data on every panel: a small cluster around (180°, 20°) at
    nside=8, with one tile (the cluster's centroid pixel) highlighted
    in red. The four panels exercise:

      * AIT all-sky — context view, target tile is one of many
      * SIN globe at center=(180°, 0°) — equatorial globe; tiles must
        clip cleanly at the limb (no spillover into back hemisphere)
      * SIN globe at center=(180°, 45°) — *tilted* globe; verifies
        that non-zero lat0 works the same way
      * TAN at center=(180°, 20°) — zoomed view, target tile filling
        most of the field with neighbors visible

    The highlight is drawn by masking ``pc.patch_pixel_index`` against
    the target pixel id and tracing the matching path(s) on top of the
    rendered collection. This is the canonical "tutorial recipe" for
    per-tile annotation.
    """
    nside = 8
    # A small cluster around (180, 20) in front of all globe centers
    # we'll use, so every tile is visible on every panel.
    pix = healpix_circle_query(180.0, 20.0, 18.0, nside)
    rng = np.random.default_rng(7)
    vals = rng.uniform(0.0, 1.0, len(pix))

    # Choose the target tile: the pixel whose center is closest to
    # the cluster center. (Stable choice that doesn't depend on
    # query ordering.)
    plon, plat = hp.pix2ang(nside, pix, lonlat=True)
    dlon = (plon - 180.0 + 180.0) % 360.0 - 180.0
    target_idx = int(np.argmin(np.hypot(dlon, plat - 20.0)))
    target_pix = int(pix[target_idx])

    # Panels: keep full-frame context on all-sky / globe frames
    # (set_extent=False) so the cluster's location on the celestial
    # sphere is visible; let TAN auto-zoom for the close-up.
    panels = [
        ("AIT", 180,           "AIT (all-sky)",                False),
        ("SIN", (180, 0),      "SIN-globe @ lat₀=0°",         False),
        ("SIN", (180, 45),     "SIN-globe @ lat₀=45° (tilted)", False),
        ("TAN", (180, 20),     "TAN @ (180°, 20°) zoomed",    True),
    ]
    fig = plt.figure(figsize=(13, 9))
    for idx, (proj, center, label, set_ext) in enumerate(panels, start=1):
        ax = make_wcs_frame((2, 2, idx), projection=proj, center=center,
                            fig=fig)
        fig.canvas.draw()
        pc = plot_healpix_sparse(pix, vals, nside=nside, ax=ax,
                                 cmap="viridis",
                                 show_boundaries=True,
                                 boundary_color="0.4", boundary_lw=0.4,
                                 set_extent=set_ext)
        # Highlight via patch_pixel_index.
        mask = pc.patch_pixel_index == target_pix
        for path in np.asarray(pc.get_paths(), dtype=object)[mask]:
            verts = path.vertices
            ax.plot(verts[:, 0], verts[:, 1], color="red", lw=2.5,
                    solid_joinstyle="round", zorder=5)
        # Annotate each tile with its HEALPix pixel id on the TAN
        # zoom panel (only — on the all-sky / globe panels the tiles
        # are too small for legible labels).
        if proj == "TAN":
            for path, pix_id in zip(pc.get_paths(), pc.patch_pixel_index):
                cx, cy = np.mean(path.vertices, axis=0)
                color = "red" if pix_id == target_pix else "0.15"
                weight = "bold" if pix_id == target_pix else "normal"
                ax.text(cx, cy, str(int(pix_id)),
                        ha="center", va="center",
                        fontsize=8, color=color, fontweight=weight,
                        zorder=6)
        ax.set_title(label, fontsize=10)
    fig.suptitle(
        "plot_healpix_sparse — highlight one tile via "
        "pc.patch_pixel_index (target = HEALPix pix "
        f"{target_pix} @ nside={nside})",
        fontsize=12,
    )
    fig.subplots_adjust(top=0.90, hspace=0.40, wspace=0.30)
    return fig


@_panel("hp_03_smooth_kernel")
def render_smooth_kernel():
    """Delta-function smoothing — visualize a δ before and after Gaussian
    smoothing with σ=3°."""
    nside = 64
    npix = hp.nside2npix(nside)
    m = np.zeros(npix)
    pix_center = hp.ang2pix(nside, 0.0, 0.0, lonlat=True)
    m[pix_center] = 1.0

    sigma_deg = 3.0
    m_smooth = healpix_smooth(m, sigma_deg=sigma_deg)

    fig = plt.figure(figsize=(15, 5))
    for col, (data, label) in enumerate([
        (m, "δ-function (1 pixel ON, all others 0)"),
        (m_smooth, f"smoothed with σ={sigma_deg}° "
                   f"(FWHM ≈ {2.355 * sigma_deg:.1f}°)"),
    ], start=1):
        ax = make_wcs_frame((1, 2, col), projection="AIT", center=0,
                            fig=fig)
        plonc, platc, pvals = healpix_to_celestial(data,
                                                       lonlatlims="allsky",
                                                       center_deg=0)
        ax.pcolormesh(plonc, platc, pvals,
                      transform=ax.get_transform("world"),
                      cmap="viridis")
        ax.set_title(label, fontsize=10)
    fig.suptitle("healpix_smooth — Gaussian beam convolution on the sphere",
                 fontsize=12)
    fig.subplots_adjust(top=0.85, wspace=0.2)
    return fig


@_panel("hp_04_upgrade_downgrade")
def render_upgrade_downgrade():
    """Same map at three resolutions: original, upgrade, downgrade.

    Uses the ``'patches'`` backend so the actual HEALPix tile structure
    is visible at each nside — the resolution differences are immediately
    obvious from tile size rather than blurred away by a meshgrid
    interpolation.
    """
    rng = np.random.default_rng(1)
    nside_in = 8
    npix = hp.nside2npix(nside_in)
    m = healpix_smooth(rng.normal(0, 1, npix), sigma_deg=15)

    m_up = healpix_upgrade(m, nside_out=32)
    m_down = healpix_downgrade(m, nside_out=2)

    vmin = float(np.nanmin(m))
    vmax = float(np.nanmax(m))

    fig = plt.figure(figsize=(18, 6))
    panels = [
        (m_down, f"downgraded to nside=2 (npix={hp.nside2npix(2)})", 2),
        (m,      f"original nside={nside_in} (npix={npix})", nside_in),
        (m_up,   f"upgraded to nside=32 (npix={hp.nside2npix(32)})", 32),
    ]
    for col, (data, label, _n) in enumerate(panels, start=1):
        ax = make_wcs_frame((1, 3, col), projection="MOL", center=0, fig=fig)
        fig.canvas.draw()
        n_data = hp.npix2nside(len(data))
        pix_idx = np.arange(len(data))
        # show_boundaries makes tile structure visible at low/medium
        # nside, but at the highest nside in this demo the tiles are
        # already small enough that the boundary lines visually
        # compress the fill colors at the panel size used here. Drop
        # boundaries on the rightmost panel and keep them on the
        # other two.
        show_b = n_data <= 8
        plot_healpix_sparse(pix_idx, data, nside=n_data, ax=ax,
                            cmap="RdBu_r", vmin=vmin, vmax=vmax,
                            show_boundaries=show_b, boundary_color="0.3",
                            boundary_lw=0.4,
                            set_extent=False)
        ax.set_title(label, fontsize=10)
    fig.suptitle("healpix_upgrade / healpix_downgrade — same map at "
                 "three resolutions (patches backend; tile structure "
                 "visible)", fontsize=12)
    fig.subplots_adjust(top=0.88, wspace=0.15)
    return fig


@_panel("hp_05_sources_to_healpix")
def render_sources_to_healpix():
    """bin_data_as_healpix — count + mean statistics of synthetic sources."""
    rng = np.random.default_rng(2)
    n = 30000
    # Cluster sources around (180°, 30°) with sigma 20°
    lons = (180 + rng.normal(0, 20, n)) % 360
    lats = np.clip(30 + rng.normal(0, 15, n), -89.9, 89.9)
    data = rng.uniform(0, 10, n)

    nside = 16
    count_map, _, _, _ = bin_data_as_healpix(
        lons, lats, data, nside, statistic="count",
    )
    mean_map, _, _, _ = bin_data_as_healpix(
        lons, lats, data, nside, statistic="mean",
    )
    # Replace NaNs with 0 for visualization
    count_map = np.where(np.isfinite(count_map), count_map, 0)
    mean_map = np.where(np.isfinite(mean_map), mean_map, np.nan)

    fig = plt.figure(figsize=(15, 5))
    for col, (data, label, cmap) in enumerate([
        (count_map, "statistic='count'", "magma"),
        (mean_map,  "statistic='mean'",  "viridis"),
    ], start=1):
        ax = make_wcs_frame((1, 2, col), projection="MOL", center=180,
                            fig=fig)
        plonc, platc, pvals = healpix_to_celestial(data,
                                                       lonlatlims="allsky",
                                                       center_deg=180)
        ax.pcolormesh(plonc, platc, pvals,
                      transform=ax.get_transform("world"),
                      cmap=cmap)
        ax.set_title(label, fontsize=10)
    fig.suptitle("bin_data_as_healpix — 30k sources clustered around "
                 "(180°, 30°)", fontsize=12)
    fig.subplots_adjust(top=0.85, wspace=0.2)
    return fig


@_panel("hp_06_query_results")
def render_query_results():
    """circle_query and polygon_query — several regions scattered
    across an all-sky AIT frame. Demonstrates each query at varied
    lat/lon positions including high-lat caps and an antimeridian-
    straddling polygon, so the queries' behavior at different
    sky-coverage extremes is visible at a glance.

    Rendered via ``healpix_to_celestial`` + ``pcolormesh`` with
    NaN masking on un-selected pixels (so the value-0 cmap color
    doesn't flood the background). The patches backend
    ``plot_healpix_sparse`` would normally be cleaner per-pixel,
    but currently leaves sub-pixel diagonal gaps between adjacent
    tiles at high nside.
    """
    nside = 64
    npix = hp.nside2npix(nside)
    m = np.full(npix, np.nan)

    # Three circle queries (value 1, distinct in cmap) — north,
    # equatorial, south.
    for (lon, lat, r) in [(120.0, 60.0, 12.0),
                          (210.0, 0.0, 10.0),
                          (40.0, -50.0, 15.0)]:
        m[healpix_circle_query(lon, lat, r, nside)] = 1.0

    # Two polygon queries (value 2) — a mid-lat box and an
    # antimeridian-straddling box centered at (0°, 30°).
    m[healpix_polygon_query(
        [(255, -45), (285, -45), (285, -15), (255, -15)],
        nside=nside)] = 2.0
    m[healpix_polygon_query(
        [(345, 15), (15, 15), (15, 45), (345, 45)],
        nside=nside)] = 2.0

    fig = plt.figure(figsize=(11, 5.5))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    plonc, platc, pvals = healpix_to_celestial(m,
                                                   lonlatlims="allsky",
                                                   center_deg=180)
    ax.pcolormesh(plonc, platc, pvals,
                  transform=ax.get_transform("world"),
                  cmap="Set1", vmin=0, vmax=3)
    ax.set_title("healpix_circle_query (3× across the sky) + "
                 "healpix_polygon_query (2× including an "
                 "antimeridian-straddling box)",
                 fontsize=10)
    return fig


def main():
    banner("HEALPix numerical correctness — gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
