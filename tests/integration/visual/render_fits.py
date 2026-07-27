"""Render the plotly FITS-image viewer for visual eyeballing.

Two side-by-side panels of the bundled MOJAVE 3C 84 (0316+413) VLBA image —
the same data shown in **offset** (mas, native round + zoom-adaptive ticks,
east left) and **absolute** (pixel axes with RA/Dec labels) coordinates — each
with the restoring beam pinned to a corner and a couple of sph overlays (a
CompoundRegion mask + a marker) projected onto the image via the WCS.

Run:  python render_fits.py   (writes to output/, gitignored)
"""

import os
import sys
import warnings

from _common import OUTPUT_DIR, banner

import skyplothelper.plotly as sphpl

_HERE = os.path.dirname(os.path.abspath(__file__))
_FITS = os.path.join(_HERE, "..", "..", "..", "examples", "data",
                     "0316+413.u.stacked.icd.fits")


def _build_panel(data, wcs, hdr, ra0, dec0, *, coords, title):
    fig = sphpl.make_fits_figure(wcs, theme="dark", width=620, height=640,
                                 title=title)
    sphpl.add_fits_image(
        fig, data, wcs, coords=coords, stretch="asinh", colormap="inferno",
        colorbar=True, header=hdr, field_size=0.04,   # 40 mas, in arcsec
        display_factor=1e3, bunit="mJy/beam",         # Jy/beam → mJy/beam
        # Downsample the display grid so the standalone HTML stays light
        # (overlays + WCS hover still use the full-resolution WCS).
        max_pixels=250_000)

    # A CompoundRegion mask (3 mas core circle) + a marker at the phase
    # center, both projected onto the image through the figure's projector.
    region = sphpl.make_fits_compound_region(fig).add_circle(
        ra0, dec0, 3.0 / 3600.0 / 1000.0)
    sphpl.add_compound_region(fig, region, color="cyan",
                              fillcolor="rgba(0,255,255,0.12)", name="3 mas core")
    sphpl.add_fits_scatter(fig, [ra0], [dec0],
                           marker=dict(color="lime", size=10, symbol="x"),
                           name="phase center")
    return fig


def _build_kitchensink(data, wcs, hdr, ra0, dec0):
    """Single offset-coords panel exercising the sph features that work on
    FITS WCS axes: stretched image + beam, a boolean CompoundRegion, catalog
    markers, contours (plt.contour vertices drawn as a polyline), and the
    standalone overlay helpers now routed through the image WCS — a
    geodesic-angle ruler and an angular scale bar (both via add_ruler).
    (Sparse healpix also reprojects through the WCS, but a healpix tile dwarfs
    this ~40 mas VLBI field at any sensible nside, so it isn't shown here; its
    FITS routing is covered in tests/test_plotly_fits.py.)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go

    from skyplothelper.plotly.projector import _offset_linear_map
    from skyplothelper.ticks import OffsetFormatter

    data2d = np.squeeze(np.asarray(data, dtype=float))
    ny, nx = data2d.shape
    fig = sphpl.make_fits_figure(
        wcs, theme="dark", width=820, height=800,
        title="MOJAVE 3C 84 — sph features on FITS WCS axes (offset coords)")
    # (rescale_image's stretch set has no 'symlog'; for this image 'asinh'
    # gives a similar pleasing log-like look.) Use the full data range
    # (vmin = image min) to recover the fainter / negative structure.
    sphpl.add_fits_image(
        fig, data, wcs, coords="offset", stretch="sqrt",
        vmin=float(np.nanmin(data2d)), vmax=float(np.nanmax(data2d)),
        colormap="inferno", colorbar=True, header=hdr, field_size=0.04,
        display_factor=1e3, bunit="mJy/beam", hover="value", max_pixels=200_000)

    # Offset (mas) pixel->display map matching the image, for the contours.
    factor = OffsetFormatter._UNIT_LABELS["mas"][1]
    sx, sy, cx, cy = _offset_linear_map(wcs.celestial, (ra0, dec0), factor)

    # Contours at the recommended 3C 84 geometric levels (base 2.9 mJy/bm).
    start = 2.9e-3
    nlev = max(int(np.log2(np.nanmax(data2d) / start)), 1)
    levels = start * 2.0 ** np.arange(0, nlev, 1.0)
    cs = plt.contour(np.arange(nx), np.arange(ny), data2d, levels=levels)
    cxs, cys = [], []
    for segs in cs.allsegs:
        for seg in segs:
            cxs.extend((sx * (seg[:, 0] - cx)).tolist() + [np.nan])
            cys.extend((sy * (seg[:, 1] - cy)).tolist() + [np.nan])
    plt.close("all")
    fig.add_trace(go.Scatter(x=cxs, y=cys, mode="lines",
                             line=dict(color="white", width=0.5), opacity=0.7,
                             hoverinfo="skip", name="contours"))

    # CompoundRegion set-algebra (6 mas circle minus a 2 mas core = annulus).
    region = (sphpl.make_fits_compound_region(fig)
              .add_circle(ra0, dec0, 6.0 / 3600 / 1000)
              .subtract_circle(ra0, dec0, 2.0 / 3600 / 1000))
    sphpl.add_compound_region(fig, region, color="cyan",
                              fillcolor="rgba(0,255,255,0.2)",
                              name="annulus region")

    # Catalog markers up the jet (a few mas north of the core).
    import numpy as _np
    jet_dec = dec0 + _np.array([2, 4, 6]) / 3600 / 1000
    sphpl.add_fits_scatter(fig, [ra0] * 3, list(jet_dec),
                           marker=dict(color="lime", size=7, symbol="x"),
                           name="jet markers")

    # Standalone overlay helpers, now routed through the image WCS projector
    # (offset/mas coords) — they land on the FITS axes like the markers.
    mas = 1.0 / 3600.0 / 1000.0
    cosd = _np.cos(_np.radians(dec0))
    # Geodesic-angle ruler: core -> 15 mas north, ticks every 5 mas.
    sphpl.add_ruler(fig, ra0, dec0, ra0, dec0 + 15 * mas, geodesic=True,
                    label_unit="mas", tick_interval=5.0, tick_side="right",
                    color="gold", title="offset from core", label_fontsize=9)
    # 10 mas angular scale bar, lower-right, ticks at 0 / 5 / 10 mas.
    bar_dec = dec0 - 16 * mas
    bar_ra2 = ra0 - 13 * mas / cosd            # west end (image right)
    bar_ra1 = bar_ra2 + 10 * mas / cosd        # east end (image left)
    sphpl.add_ruler(fig, bar_ra1, bar_dec, bar_ra2, bar_dec,
                    label_unit="mas", tick_interval=5.0, tick_side="left",
                    color="white", label_fontsize=9)

    # A small sparse-healpix cluster tucked into the upper-left corner, to
    # verify tile reprojection onto the FITS axes. A healpix tile is huge at
    # any normal resolution, so this needs a high nside to get ~5 mas tiles
    # that fit the ~40 mas field; light edges make the boundaries easy to see.
    import healpy as hp
    hp_nside = 2 ** 27                          # ~2.7 mas tiles
    hp_dec = dec0 + 14 * mas
    hp_ra = ra0 + 14 * mas / cosd             # east (image left)
    ic = hp.ang2pix(hp_nside, hp_ra, hp_dec, lonlat=True)
    ipix = [ic] + [p for p in hp.get_all_neighbours(hp_nside, ic) if p >= 0]
    sphpl.add_healpix_sparse(
        fig, ipix, list(range(len(ipix))), hp_nside,
        colorscale="Blues", opacity=0.3, line_width=1.2,
        line_color="lightcyan")
    return fig


def render():
    banner("plotly FITS viewer — MOJAVE 3C 84 (0316+413): offset vs absolute")
    if not os.path.exists(_FITS):
        print(f"  example FITS not found: {_FITS} — skipping.")
        return
    from astropy.io import fits
    from astropy.wcs import WCS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdr = fits.getheader(_FITS)
        data = fits.getdata(_FITS)
        wcs = WCS(hdr)
        ra0, dec0 = float(hdr["CRVAL1"]), float(hdr["CRVAL2"])

        fig_off = _build_panel(
            data, wcs, hdr, ra0, dec0, coords="offset",
            title="offset coords (mas) — native round ticks, east left")
        fig_abs = _build_panel(
            data, wcs, hdr, ra0, dec0, coords="absolute",
            title="absolute coords — pixel axes with RA/Dec labels")
        fig_ks = _build_kitchensink(data, wcs, hdr, ra0, dec0)

    # Second page: the single-panel feature check.
    ks_html = fig_ks.to_html(full_html=True, include_plotlyjs="cdn")
    ks_out = os.path.join(OUTPUT_DIR, "fits_02_mojave_kitchensink.html")
    with open(ks_out, "w", encoding="utf-8") as fh:
        fh.write(ks_html)
    print(f"  saved: {ks_out} ({os.path.getsize(ks_out) / 1024:.0f} kB, "
          f"{len(fig_ks.data)} traces, {len(fig_ks.layout.shapes)} shapes)")

    html_off = fig_off.to_html(full_html=False, include_plotlyjs="cdn")
    html_abs = fig_abs.to_html(full_html=False, include_plotlyjs=False)
    combined = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>skyplothelper.plotly FITS viewer — MOJAVE 3C 84</title>"
        "<style>body{margin:0;padding:8px;background:#06060a;color:#dcdcdc;"
        "font-family:sans-serif}h2{font-weight:400}"
        ".row{display:flex;flex-wrap:wrap;gap:8px}</style></head><body>"
        "<h2>MOJAVE 3C 84 (0316+413) — VLBA U-band — offset vs absolute</h2>"
        f"<div class='row'>{html_off}{html_abs}</div></body></html>"
    )
    out = os.path.join(OUTPUT_DIR, "fits_01_mojave.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(combined)
    size_kb = os.path.getsize(out) / 1024
    print(f"  saved: {out} ({size_kb:.0f} kB)")


if __name__ == "__main__":
    render()
    print("\nDone.")
    sys.exit(0)
