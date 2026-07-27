"""Render the overlay/annotation helpers for visual eyeballing.

Covers: ``add_compass``, ``add_axis_inlay``, the :class:`Beam` /
:class:`BeamStack` classes (anchored + free, every style, the PSF
inset + fit factory), ``add_sizebar`` / ``add_sizebar_asec``,
``add_bandlabels``, ``add_colorbar``, ``add_contour_overlay``,
``style_ax_colors``.

Usage
-----
    python render_annotations.py            # save PNGs to output/
    python render_annotations.py --show     # display interactively
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show
from astropy.io import fits

from skyplothelper.overlays.annotations import (
    add_axis_inlay,
    add_bandlabels,
    add_colorbar,
    add_compass,
    add_contour_overlay,
    add_sizebar,
    add_sizebar_asec,
    style_ax_colors,
)
from skyplothelper.overlays.beam import Beam, BeamStack
from skyplothelper.overlays.ruler import Ruler
from skyplothelper.wcs_frame import make_wcs_frame

# Builder registry — name → no-arg function that returns a Figure.
PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _tan_axes(fov_arcsec=200.0, fig=None):
    """A modest TAN field for annotation tests, ~3 arcmin wide."""
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    cdelt = (fov_arcsec / 100.0) / 3600.0  # 100 px → fov_arcsec
    ax = make_wcs_frame(
        111, projection="TAN", center=(180.0, 0.0),
        cdelt=cdelt, npix=(100, 100), fig=fig,
    )
    fig.canvas.draw()
    return fig, ax


def _tan_header(fov_arcsec=200.0):
    """Matching FITS header (CDELT in degrees) plus BMAJ/BMIN/BPA."""
    cdelt_deg = (fov_arcsec / 100.0) / 3600.0
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 100
    hdr["NAXIS2"] = 100
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = 50.5
    hdr["CRPIX2"] = 50.5
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 0.0
    hdr["CDELT1"] = -cdelt_deg
    hdr["CDELT2"] = cdelt_deg
    hdr["BMAJ"] = 12.0 / 3600.0   # 12 asec FWHM major
    hdr["BMIN"] = 7.0 / 3600.0    # 7 asec FWHM minor
    hdr["BPA"] = 35.0
    return hdr


@_panel("annot_01_colorbar_contour")
def render_colorbar_and_contour():
    """add_colorbar + add_contour_overlay on a synthetic gradient field."""
    fig, ax = _tan_axes(fov_arcsec=200.0)

    # Background image — a 2D Gaussian "source" near the center
    nx, ny = 100, 100
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = np.exp(-((xx - 50) ** 2 + (yy - 45) ** 2) / (2 * 12 ** 2))
    im = ax.imshow(img, origin="lower", cmap="gist_yarg",
                   transform=ax.get_transform("pixel"))
    add_colorbar(im, ax=ax, label="brightness [arb.]")

    # Contour overlay on the same field (sky-coord grid)
    npts = 80
    lon = np.linspace(179.97, 180.03, npts)
    lat = np.linspace(-0.03, 0.03, npts)
    LON, LAT = np.meshgrid(lon, lat)
    vals = np.exp(-((LON - 180.0) ** 2 + (LAT) ** 2) / (2 * 0.005 ** 2))
    add_contour_overlay(ax, LON, LAT, vals,
                        levels=[0.2, 0.5, 0.8],
                        colors=["#FF6060", "#FF3030", "#A00000"],
                        linewidths=1.2)
    ax.set_title("add_colorbar + add_contour_overlay")
    return fig


@_panel("annot_02_compass")
def render_compass():
    """add_compass — default (N=up) vs a rotated WCS so the compass
    has to handle a tilted north (e.g. HST images that aren't
    registered to north-up). Light grid makes the WCS rotation easy
    to read at a glance."""
    from astropy.wcs import WCS

    fig = plt.figure(figsize=(11, 5.5))
    # Panel 1: standard RA-inverted TAN, N=up
    ax1 = make_wcs_frame((1, 2, 1), projection="TAN", center=(180.0, 0.0),
                         cdelt=200 / 100.0 / 3600.0, npix=(100, 100),
                         fig=fig)
    fig.canvas.draw()
    ax1.grid(True, color="0.85", lw=0.5, ls="-")
    add_compass(ax1)
    ax1.set_title("add_compass — standard WCS (N=up, E=left)")

    # Panel 2: same TAN but with a 30° rotation applied to the CD
    # matrix, built manually since make_wcs_frame doesn't accept PC
    # directly. Start from the unrotated standard astronomical CD =
    # diag(-cdelt, +cdelt) (RA-inverted, north-up), then left-multiply
    # by the rotation matrix R(30°).
    cdelt_deg = 200 / 100.0 / 3600.0
    rot = np.radians(30)
    cos_t, sin_t = np.cos(rot), np.sin(rot)
    cd_unrot = np.array([[-cdelt_deg, 0], [0, cdelt_deg]])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    wcs2 = WCS(naxis=2)
    wcs2.wcs.crpix = [50.5, 50.5]
    wcs2.wcs.crval = [180.0, 0.0]
    wcs2.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs2.wcs.cd = R @ cd_unrot
    ax2 = fig.add_subplot(1, 2, 2, projection=wcs2)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    fig.canvas.draw()
    ax2.grid(True, color="0.85", lw=0.5, ls="-")
    add_compass(ax2)
    ax2.set_title("add_compass — rotated WCS (N tilted 30°)")
    fig.subplots_adjust(top=0.9, wspace=0.25)
    return fig


@_panel("annot_03_axis_inlay")
def render_axis_inlay():
    """add_axis_inlay — three inlays on a single plot demonstrating
    both lon-direction conventions and three frame label styles:

      * upper-left: galactic (ℓ→left, b↑) on a galactic-style WCS
      * upper-right: Earth-style cartographic (lon→right, lat↑)
      * lower-right: equatorial RA/Dec (RA→left, Dec↑) — the
        astronomical default with RA-inverted axes
    """
    fig, ax = _tan_axes()
    # Auto-detected RA/Dec — CDELT1 < 0 → lon arrow inverts (←)
    add_axis_inlay(ax, lon_label="RA", lat_label="Dec",
                   loc="lower right")
    # Forced Earth-style — lon arrow points right
    add_axis_inlay(ax, lon_label="lon", lat_label="lat",
                   loc="upper right", lon_invert=False)
    # Galactic labels with the inverted convention
    add_axis_inlay(ax, lon_label="ℓ", lat_label="b",
                   loc="upper left", lon_invert=True)
    ax.set_title("add_axis_inlay — three styles\n"
                 "RA/Dec (auto), Earth lon/lat, galactic ℓ/b")
    return fig


def _elliptical_gaussian_2d(xx, yy, x0, y0, fwhm_major, fwhm_minor, pa_deg):
    """Synthetic elliptical 2D Gaussian source (FWHM/PA in pixel coords).

    ``pa_deg`` follows matplotlib's ``Ellipse(angle=...)`` convention
    — CCW from the +x axis in degrees — which matches what Beam
    receives when constructed with ``pa_convention='plot'``.
    """
    sm = fwhm_major / 2.3548  # FWHM → sigma
    sn = fwhm_minor / 2.3548
    theta = np.radians(pa_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xr = (xx - x0) * cos_t + (yy - y0) * sin_t
    yr = -(xx - x0) * sin_t + (yy - y0) * cos_t
    return np.exp(-0.5 * ((xr / sm) ** 2 + (yr / sn) ** 2))


def _scatter_sources(nx, ny, fwhm_major, fwhm_minor, pa_deg, n=4):
    """Build an image with a handful of identical-shape Gaussian sources
    at scattered positions; deterministic positions for stable baselines."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = np.zeros((ny, nx))
    centers = [(25, 25), (75, 30), (30, 70), (70, 75)][:n]
    for cx, cy in centers:
        img += _elliptical_gaussian_2d(xx, yy, cx, cy,
                                        fwhm_major, fwhm_minor, pa_deg)
    return np.clip(img, 0, 1)


@_panel("annot_05_beam_anchored_explicit")
def render_beam_anchored_explicit():
    """``Beam.add_anchored`` — explicit pixel-size form, with 4
    synthetic 2D Gaussian sources of matching FWHM and PA. The
    corner-anchored beam should visually match the source shape and
    orientation. ``pa_convention='plot'`` keeps the historical
    ``add_beamsize`` interpretation of ``angle`` (CCW from +x)."""
    fig, ax = _tan_axes()
    img = _scatter_sources(100, 100, fwhm_major=20.0, fwhm_minor=12.0,
                           pa_deg=35.0)
    ax.imshow(img, origin="lower", cmap="gray_r",
              transform=ax.get_transform("pixel"),
              vmin=0, vmax=1)
    beam = Beam((0, 0), bmaj_pix=20.0, bmin_pix=12.0, bpa_deg=35.0,
                pa_convention='plot',
                fc='white', ec='k')
    beam.add_anchored(ax, loc='lower left')
    ax.set_title("Beam.add_anchored (explicit: 20 × 12 px, PA=35°)\n"
                 "beam should match source shape")
    return fig


@_panel("annot_06_beam_anchored_from_header")
def render_beam_anchored_from_header():
    """``Beam.from_header(...).add_anchored(...)`` — header-driven
    form. Synthetic sources are sized from the header's BMAJ /
    BMIN / BPA after converting arcsec → pixels via CDELT, so the
    anchored beam and the source ellipses should be indistinguishable
    in shape."""
    fig, ax = _tan_axes()
    hdr = _tan_header(fov_arcsec=200.0)
    asec_per_pix = abs(hdr["CDELT2"]) * 3600.0
    fwhm_maj_px = (hdr["BMAJ"] * 3600.0) / asec_per_pix
    fwhm_min_px = (hdr["BMIN"] * 3600.0) / asec_per_pix
    img = _scatter_sources(100, 100,
                            fwhm_major=fwhm_maj_px,
                            fwhm_minor=fwhm_min_px,
                            pa_deg=hdr["BPA"])
    ax.imshow(img, origin="lower", cmap="gray_r",
              transform=ax.get_transform("pixel"),
              vmin=0, vmax=1)
    # Header BPA in our synthetic test is interpreted in mpl-angle
    # convention by _scatter_sources, so build the Beam with
    # pa_convention='plot' to match.
    beam = Beam.from_arcsec(
        bmaj_asec=hdr["BMAJ"] * 3600.0,
        bmin_asec=hdr["BMIN"] * 3600.0,
        bpa_deg=hdr["BPA"],
        pa_convention='plot',
        pixscale_asec=asec_per_pix,
        xy=(0, 0),
        fc='white', ec='C0',
    )
    beam.add_anchored(ax, loc='lower left')
    ax.set_title("Beam.from_arcsec + add_anchored "
                 "(12″ × 7″, PA=35°)\n"
                 "sources sized from BMAJ/BMIN")
    return fig


@_panel("annot_06b_beam_styles")
def render_beam_styles():
    """``Beam`` class — render every style on a 2×3 grid of plain
    axes. Each panel uses identical geometry (FWHM major/minor 40 px /
    20 px, PA=30°) so the only difference is the display style. The
    crosshair child lines should track the major + minor axes of the
    ellipse for the two ``crosshair*`` styles; the hatch fill patterns
    should appear on ``hatch`` and the two ``*grid`` variants; the
    ``filled`` and ``filledgrid`` styles default ``fc=ec`` so the
    beam reads as solid (instead of an empty outline)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    styles = ['ellipse', 'crosshair', 'crosshairgrid',
              'hatch', 'filled', 'filledgrid']
    for ax, style in zip(axes.flat, styles):
        ax.set_aspect('equal')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])
        beam = Beam((50, 50), bmaj_pix=40, bmin_pix=20, bpa_deg=30,
                    style=style, ec='C0', lw=1.2)
        beam.add_to(ax)
        ax.set_title(f"style={style!r}", fontsize=10)
    fig.suptitle("Beam class — six display styles (40 × 20 px, PA=30°)",
                 fontsize=12)
    fig.subplots_adjust(top=0.90, hspace=0.3, wspace=0.2)
    return fig


@_panel("annot_06c_beam_components")
def render_beam_components():
    """``Beam`` class — demonstrate independent manual control over
    every visual component, plus the alternate factories.

    Six panels:

    1. **Defaults** — bare ``Beam`` with ``style='crosshairgrid'``.
       Reference look against which the manual overrides read.
    2. **Patch edge** — heavy navy outline + dashed linestyle. The
       inherited ``Ellipse.set_edgecolor/lw/ls`` work directly.
    3. **Face + alpha** — solid wash color with semi-transparent
       alpha (the publication "shaded beam" look).
    4. **Crosshair recolored** — patch and crosshair lines styled
       independently. Crimson crosshairs on a navy beam outline; the
       crosshair is also thicker and dashed.
    5. **Grid customized** — marker switched from ``'+'`` to ``'x'``,
       density reduced to 4. Demonstrates that hatch pattern is
       constructible from ``grid_marker × grid_density`` instead of
       hard-coding the hatch string.
    6. **Alternative PA convention** — same beam constructed via
       ``pa_convention='plot'`` (matplotlib's CCW-from-+x angle) —
       confirms the bpa-plot setter inverts to the FITS BPA stored
       internally, and that ``from_arcsec`` is also exercised here.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax in axes.flat:
        ax.set_aspect('equal')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])

    # 1) Defaults
    ax = axes[0, 0]
    Beam((50, 50), bmaj_pix=40, bmin_pix=20, bpa_deg=30,
         style='crosshairgrid', ec='C0', lw=1.2).add_to(ax)
    ax.set_title("(1) Defaults", fontsize=10)

    # 2) Patch edge — color/lw/ls customized
    ax = axes[0, 1]
    Beam((50, 50), bmaj_pix=40, bmin_pix=20, bpa_deg=30,
         style='ellipse',
         ec='navy', lw=3.0, ls='--').add_to(ax)
    ax.set_title("(2) Patch edge: navy, lw=3, ls='--'", fontsize=10)

    # 3) Face + alpha
    ax = axes[0, 2]
    Beam((50, 50), bmaj_pix=40, bmin_pix=20, bpa_deg=30,
         style='ellipse',
         ec='C0', fc='lightsteelblue', alpha=0.55, lw=1.2).add_to(ax)
    ax.set_title("(3) Face + alpha", fontsize=10)

    # 4) Crosshair independently styled
    ax = axes[1, 0]
    Beam((50, 50), bmaj_pix=40, bmin_pix=20, bpa_deg=30,
         style='crosshair',
         ec='navy', lw=1.2,
         crosshair_color='crimson', crosshair_lw=1.8,
         crosshair_ls='--').add_to(ax)
    ax.set_title("(4) Crosshair: crimson, lw=1.8, '--'", fontsize=10)

    # 5) Grid: custom marker + density
    ax = axes[1, 1]
    Beam((50, 50), bmaj_pix=40, bmin_pix=20, bpa_deg=30,
         style='crosshairgrid',
         ec='C2', lw=1.2,
         grid_marker='x', grid_density=4).add_to(ax)
    ax.set_title("(5) Grid: marker='x', density=4", fontsize=10)

    # 6) Alternate factory + PA convention
    ax = axes[1, 2]
    # 'plot' convention: angle=30 means CCW from +x by 30° — should
    # match panel (1) shape since panel (1)'s FITS BPA=30 (E of N,
    # i.e. 30° CCW from +y = 120° CCW from +x). To MATCH panel (1)
    # in this panel, pass bpa_deg=120 in plot convention.
    beam = Beam.from_arcsec(
        bmaj_asec=20.0, bmin_asec=10.0, bpa_deg=120.0,
        pa_convention='plot',
        pixscale_asec=0.5, xy=(50, 50),
        style='crosshair', ec='C0', lw=1.2)
    beam.add_to(ax)
    ax.set_title("(6) from_arcsec + pa_convention='plot'",
                 fontsize=10)

    fig.suptitle("Beam class — component access "
                 "(all six panels: 40 × 20 px, PA=30° east of north)",
                 fontsize=12)
    fig.subplots_adjust(top=0.90, hspace=0.3, wspace=0.2)
    return fig


@_panel("annot_06d_beam_stack")
def render_beam_stack():
    """``BeamStack`` — co-located beams stacked at one canvas position.

    The publication idiom for showing a synthesis observation built
    from multiple array configurations. Each member beam is a full
    :class:`Beam` (independent size / style / color) but
    :class:`BeamStack` adds them as a unit and lets matplotlib
    discover the per-beam ``label=`` for a single legend call.

    The four panels show:

    1. **ALMA 12 m + 7 m + combined** — concentric beams ordered
       largest-outermost, smallest-innermost-and-filled. The typical
       ALMA Cycle paper inset.
    2. **VLA A/B/C/D configurations** — four progressively larger
       beams, all outlined, mapping the resolution achieved at each
       configuration. The smallest (A-config) is filled for
       emphasis.
    3. **Multi-frequency** — same antenna at three frequencies; the
       beam shrinks with frequency. All same style, distinguished by
       color.
    4. **BeamStack.from_specs** — same shape as panel (1) but built
       from a list of kwarg dicts (a single ``style=`` and ``lw=``
       shared, sizes / colors / labels per spec) to illustrate the
       compact builder.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])

    # (1) ALMA 12 m + 7 m + combined
    ax = axes[0, 0]
    BeamStack([
        Beam((50, 50), bmaj_pix=42, bmin_pix=32, bpa_deg=20,
             style='ellipse', ec='C3', lw=1.5, label='7 m'),
        Beam((50, 50), bmaj_pix=24, bmin_pix=18, bpa_deg=20,
             style='ellipse', ec='C0', lw=1.5, label='12 m'),
        Beam((50, 50), bmaj_pix=10, bmin_pix=7, bpa_deg=20,
             style='filled', ec='C2', label='12 m + 7 m'),
    ]).add_to(ax)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title("(1) ALMA combo: 7 m + 12 m + combined", fontsize=10)

    # (2) VLA A/B/C/D configurations
    ax = axes[0, 1]
    BeamStack([
        Beam((50, 50), bmaj_pix=60, bmin_pix=50, bpa_deg=-15,
             style='ellipse', ec='C5', lw=1.2, label='D config'),
        Beam((50, 50), bmaj_pix=35, bmin_pix=29, bpa_deg=-15,
             style='ellipse', ec='C4', lw=1.2, label='C config'),
        Beam((50, 50), bmaj_pix=18, bmin_pix=14, bpa_deg=-15,
             style='ellipse', ec='C1', lw=1.2, label='B config'),
        Beam((50, 50), bmaj_pix=8, bmin_pix=6, bpa_deg=-15,
             style='filled', ec='C0', label='A config'),
    ]).add_to(ax)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title("(2) VLA A/B/C/D configurations", fontsize=10)

    # (3) Multi-frequency: beam shrinks ∝ 1/ν
    ax = axes[1, 0]
    BeamStack([
        Beam((50, 50), bmaj_pix=45, bmin_pix=35, bpa_deg=10,
             style='ellipse', ec='C3', lw=1.3, label='1.4 GHz'),
        Beam((50, 50), bmaj_pix=28, bmin_pix=22, bpa_deg=10,
             style='ellipse', ec='C0', lw=1.3, label='5 GHz'),
        Beam((50, 50), bmaj_pix=15, bmin_pix=12, bpa_deg=10,
             style='ellipse', ec='C2', lw=1.3, label='15 GHz'),
    ]).add_to(ax)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title("(3) Multi-frequency (1.4 / 5 / 15 GHz)", fontsize=10)

    # (4) Same shape as (1), built via from_specs
    ax = axes[1, 1]
    BeamStack.from_specs(
        [dict(bmaj_pix=42, bmin_pix=32, bpa_deg=20, ec='C3',
              label='7 m'),
         dict(bmaj_pix=24, bmin_pix=18, bpa_deg=20, ec='C0',
              label='12 m'),
         dict(bmaj_pix=10, bmin_pix=7, bpa_deg=20, ec='C2',
              style='filled', label='12 m + 7 m')],
        xy=(50, 50),
        style='ellipse', lw=1.5,
    ).add_to(ax)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title("(4) Same as (1) via BeamStack.from_specs",
                 fontsize=10)

    fig.suptitle("BeamStack — co-located beams for combined-array "
                 "publications", fontsize=12)
    fig.subplots_adjust(top=0.93, hspace=0.18, wspace=0.12)
    return fig


@_panel("annot_06e_beam_psf_inset")
def render_beam_psf_inset():
    """``Beam.from_psf_fit`` + ``Beam.add_psf_inset`` — fit a Beam to a
    synthetic elliptical-Gaussian PSF (with a faint sidelobe ring),
    then render the parent image, the fitted FWHM ellipse, and a
    thumbnail PSF inset in one figure.

    The inset's white FWHM overlay (the same Beam re-drawn on the
    inset axes) should hug the bright core of the PSF — a direct
    visual sanity check that ``from_psf_fit`` recovered the correct
    shape and orientation."""
    import numpy as np

    rng = np.random.default_rng(0)
    nx = ny = 81
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

    # Elliptical Gaussian PSF: bmaj_fwhm = 14 px, bmin_fwhm = 7 px
    # tilted 25° (mpl convention).
    sig_minor = 7.0 / 2.3548
    sig_major = 14.0 / 2.3548
    th = np.radians(25.0)
    c, s = np.cos(th), np.sin(th)
    xr = (xx - cx) * c + (yy - cy) * s
    yr = -(xx - cx) * s + (yy - cy) * c
    core = np.exp(-0.5 * ((xr / sig_minor) ** 2 + (yr / sig_major) ** 2))

    # A faint Airy-like ring at radius ~22 px.
    r = np.sqrt(xr ** 2 + yr ** 2)
    ring = -0.08 * np.exp(-0.5 * ((r - 22.0) / 4.0) ** 2)

    psf = core + ring + 0.004 * rng.standard_normal((ny, nx))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(psf, cmap='gray_r', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Beam.from_psf_fit + Beam.add_psf_inset")

    # Fit and render the FWHM ellipse on the parent image.
    beam = Beam.from_psf_fit(psf, xy=(cx, cy),
                              style='crosshair',
                              ec='C0', lw=1.2,
                              crosshair_color='C0',
                              crosshair_lw=0.9)
    beam.add_to(ax)

    # PSF thumbnail inset (asinh stretch reveals the negative ring).
    beam.add_psf_inset(
        ax, psf,
        size='30%', loc='upper right',
        stretch='asinh', cmap='viridis',
        show_beam=True,
        beam_kwargs={'style': 'ellipse', 'ec': 'white', 'lw': 1.2},
        border_color='C0', border_lw=1.0,
        title='PSF inset (asinh)',
    )
    return fig


@_panel("annot_06f_ruler")
def render_ruler():
    """``Ruler`` — two-point distance annotation companion to the
    corner-anchored scale bars. Four panels:

    1. **Plain axes, no scale** — falls back to pixel-unit labels and
       auto-picks a 1/2/5×10^n tick interval that gives ~4 ticks across
       the line. Centered ticks on both sides.
    2. **WCS axes, arcsec auto-units** — pixel scale read from the
       axes' WCS so labels render in arcsec (auto-promoted to arcmin /
       degrees by magnitude). One-sided ticks (left of the line).
    3. **Explicit interval + custom styling** — ``tick_interval=`` pins
       the spacing in the active unit; the line, ticks, and labels are
       styled independently to demonstrate component access.
    4. **Great-circle on a wide field** — ``geodesic=True`` traces the
       on-sky great-circle path between two points; for the small TAN
       field used here, the visual difference vs a straight line is
       subtle, but the underlying line is a polyline (visible as a
       gentle curve at high zoom). The bottom-right panel uses
       ``Ruler.from_world`` with SkyCoord endpoints to demonstrate
       the world-coord factory.
    """
    from astropy.coordinates import SkyCoord

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))

    # (1) Plain axes, pixel-unit labels, auto-nice interval
    ax = axes[0, 0]
    ax.set_aspect("equal")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    Ruler((15, 20), (85, 80),
          color="C0", lw=1.4,
          tick_side="both").add_to(ax)
    ax.set_title("(1) Plain axes — pixel units, auto interval",
                  fontsize=10)

    # (2) WCS axes with arcsec labels, one-sided ticks
    ax = axes[0, 1]
    cdelt = (200.0 / 100.0) / 3600.0  # 2 arcsec/pix → 200" field
    # Build inline so we share the figure
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 2, 2), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100),
                        fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler((20, 30), (85, 75), ax=ax,
          color="C3", lw=1.4,
          tick_side="left",
          label_fontsize=8).add_to(ax)
    ax.set_title("(2) WCS axes — arcsec auto-units, ticks on left",
                  fontsize=10)

    # (3) Explicit interval + custom styling
    ax = axes[1, 0]
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 2, 3), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100),
                        fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler((10, 50), (90, 50), ax=ax,
          tick_interval=20.0,    # 20 arcsec spacing
          tick_length=8.0,
          color="navy", lw=1.5, ls="-",
          tick_color="crimson", tick_lw=1.2,
          label_color="navy", label_fontsize=9,
          label_offset=4.0).add_to(ax)
    ax.set_title("(3) Explicit interval (20\") + component styling",
                  fontsize=10)

    # (4) Geodesic via from_world with SkyCoord endpoints. Use a much
    # wider field so the great-circle curvature is visible.
    ax = axes[1, 1]
    fig.delaxes(ax)
    cdelt_wide = 0.05  # 0.05 deg/pix → 5° field
    ax = make_wcs_frame((2, 2, 4), projection="AIT",
                        center=(0.0, 0.0),
                        cdelt=cdelt_wide, npix=(100, 100),
                        fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    c1 = SkyCoord(2.0, -2.0, unit="deg")
    c2 = SkyCoord(-2.0, +2.0, unit="deg")
    Ruler.from_world(c1, c2, ax=ax,
                     geodesic=True, n_geodesic_pts=64,
                     color="C2", lw=1.4,
                     tick_side="both",
                     label_fontsize=8).add_to(ax)
    ax.set_title("(4) from_world geodesic (4°×4° SkyCoord pair)",
                  fontsize=10)

    fig.suptitle("Ruler — two-point distance annotation",
                  fontsize=12)
    fig.subplots_adjust(top=0.93, hspace=0.22, wspace=0.18)
    return fig


@_panel("annot_06g_ruler_convert")
def render_ruler_convert():
    """``Ruler`` with ``convert=`` — physical-unit tick labels.

    Four panels showcasing every supported form of the conversion
    stretch feature:

    1. **Callable** — a bare ``lambda asec: asec * factor`` paired with
       ``convert_unit='kpc'``. The simplest form; useful when the user
       already has a scale factor in hand.
    2. **Redshift z = 0.5** — astropy Planck18 cosmology drives the
       conversion via ``cosmo.kpc_proper_per_arcmin(z)``. Labels render
       in projected kpc, the publication standard for high-z imaging.
    3. **Redshift comparison** — same image, three rulers at three
       redshifts (z=0.1 / 1.0 / 3.0). Shows how the projected
       physical scale of a fixed angular separation shrinks then
       recovers slightly as cosmological angular diameter distance
       evolves with z.
    4. **Distance form** — small-angle ``size ≈ distance × θ`` for a
       nearby resolved source (1 kpc distance, labels in AU). The
       canonical use case is a Galactic stellar cluster / nebula
       where projected sizes are best read in physical units.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))

    # Shared TAN field — 2 arcsec/pix, 100 px wide ≈ 200" field.
    cdelt = (200.0 / 100.0) / 3600.0

    # (1) Callable form
    ax = axes[0, 0]
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 2, 1), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler((10, 50), (90, 50), ax=ax,
          tick_interval=40.0,
          convert=lambda asec: asec * 0.05,
          convert_unit="arb. unit",
          color="C0", lw=1.4,
          label_fontsize=9).add_to(ax)
    ax.set_title("(1) Callable convert= (lambda asec: asec * 0.05)",
                  fontsize=10)

    # (2) Redshift z=0.5
    ax = axes[0, 1]
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 2, 2), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler((10, 50), (90, 50), ax=ax,
          tick_interval=40.0,
          convert=dict(redshift=0.5, cosmo="Planck18", unit="kpc"),
          color="C3", lw=1.4,
          label_fontsize=9).add_to(ax)
    ax.set_title("(2) Redshift z=0.5 (Planck18) — projected kpc",
                  fontsize=10)

    # (3) Redshift comparison: three rulers, same angular length,
    # different z. Stack them vertically at different y offsets.
    ax = axes[1, 0]
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 2, 3), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    redshifts = [(0.1, "C0", 70), (1.0, "C2", 50), (3.0, "C1", 30)]
    for z, color, y in redshifts:
        Ruler((15, y), (85, y), ax=ax,
              tick_interval=40.0,
              convert=dict(redshift=z, unit="kpc"),
              color=color, lw=1.4,
              label_fontsize=8).add_to(ax)
        ax.text(90, y, f"z={z}", color=color, fontsize=9,
                 va="center", ha="left",
                 transform=ax.transData)
    ax.set_title("(3) Same 140″ ruler at z = 0.1 / 1.0 / 3.0",
                  fontsize=10)

    # (4) Distance form — 1 kpc distance, labels in AU
    ax = axes[1, 1]
    fig.delaxes(ax)
    # Tighter field for AU-scale labels — 0.05"/px so 100 px = 5"
    cdelt_tight = (5.0 / 100.0) / 3600.0
    ax = make_wcs_frame((2, 2, 4), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt_tight, npix=(100, 100),
                        fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler((10, 50), (90, 50), ax=ax,
          tick_interval=1.0,
          convert=dict(distance=1.0, distance_unit="kpc", unit="au"),
          color="C5", lw=1.4,
          label_fontsize=9).add_to(ax)
    ax.set_title("(4) Distance form (d=1 kpc, labels in AU)",
                  fontsize=10)

    fig.suptitle("Ruler — convert= stretch feature "
                  "(callable / redshift / distance)",
                  fontsize=12)
    fig.subplots_adjust(top=0.93, hspace=0.22, wspace=0.18)
    return fig


@_panel("annot_06h_ruler_styling")
def render_ruler_styling():
    """``Ruler`` v2 refinements — six panels covering title, fmt,
    label_side, label_rotation, tick_positions, and the from_polar
    factory. The endpoint collision fix from this same commit makes
    panel 4 of ``annot_06f`` (the from_world geodesic) cleaner too.

    1. **Title + auto rotation** — ``label_rotation='auto'`` (default
       in v2) makes labels track the local tangent; ``title='...'``
       caption sits opposite the labels.
    2. **fmt + compact title** — ``fmt='%.1f'`` pins the numeric
       format; ``title='Size in arcsec'`` lets the per-tick labels
       drop their unit suffix via ``label_unit='pix'`` (numeric only).
    3. **label_side flip** — both-sided ticks with labels forced to
       the right side via ``label_side='right'`` (Kapteyn's
       ``fliplabelside``).
    4. **Perpendicular labels + custom positions** — ``label_rotation=
       'perpendicular'`` rotates each label 90° from the line;
       ``tick_positions=`` supplies an explicit non-uniform list.
    5. **from_polar polar factory** — anchor + length + PA bar
       (radio-astronomy style "10″ at PA=45°").
    6. **Title + redshift convert** — title "Projected distance"
       lets per-tick labels drop the 'kpc' suffix via a label_fmt
       callable that strips the unit.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    cdelt = (200.0 / 100.0) / 3600.0   # 2"/pix → 200" field

    # (1) Title + auto-rotation
    ax = axes[0, 0]
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 3, 1), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler((15, 20), (85, 80), ax=ax,
          color="C0", lw=1.4,
          tick_interval=40.0,
          title="Source A → B",
          title_fontsize=10,
          label_fontsize=8).add_to(ax)
    ax.set_title("(1) Title + auto-rotation", fontsize=9)

    # (2) fmt + title
    ax = axes[0, 1]
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 3, 2), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler((10, 50), (90, 50), ax=ax,
          color="C3", lw=1.4,
          tick_interval=40.0,
          fmt="%.0f",
          label_unit="arcsec",         # pin to arcsec; title carries unit
          title="Size in arcsec",
          title_fontsize=10,
          label_fontsize=9,
          label_fmt=lambda v, _u: f"{v:.0f}").add_to(ax)
    ax.set_title("(2) fmt + title carries unit", fontsize=9)

    # (3) label_side flip
    ax = axes[0, 2]
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 3, 3), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler((10, 50), (90, 50), ax=ax,
          color="C2", lw=1.4,
          tick_side="both",
          label_side="right",
          tick_interval=40.0,
          label_fontsize=9,
          title="ticks both, labels below",
          title_fontsize=9).add_to(ax)
    ax.set_title("(3) label_side='right' (Kapteyn fliplabelside)",
                  fontsize=9)

    # (4) Perpendicular rotation + custom tick positions
    ax = axes[1, 0]
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 3, 4), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler((10, 50), (90, 50), ax=ax,
          color="C4", lw=1.4,
          tick_positions=[0, 25, 50, 100, 160],
          label_rotation="perpendicular",
          label_offset=6.0,
          label_fontsize=8).add_to(ax)
    ax.set_title("(4) perpendicular rotation + custom tick_positions",
                  fontsize=9)

    # (5) from_polar
    ax = axes[1, 1]
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 3, 5), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler.from_polar((50, 50), length=60.0, angle=45.0,
                      ax=ax, length_unit="arcsec",
                      angle_convention="fits",
                      color="C5", lw=1.4,
                      tick_interval=20.0,
                      title="60″ @ PA=45°",
                      title_fontsize=9,
                      label_fontsize=8).add_to(ax)
    ax.set_title("(5) from_polar (60″ at PA=45°)", fontsize=9)

    # (6) Title compactifies a convert= ruler
    ax = axes[1, 2]
    fig.delaxes(ax)
    ax = make_wcs_frame((2, 3, 6), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    ax.grid(True, color="0.85", lw=0.5, ls="-")
    Ruler((10, 50), (90, 50), ax=ax,
          color="C1", lw=1.4,
          tick_interval=40.0,
          convert=dict(redshift=0.5, unit="kpc"),
          fmt="%.0f",
          label_fmt=lambda v, _u: f"{v * 0.5 * 12.583:.0f}",  # ~6.29 kpc/asec
          title="Projected distance (kpc, z=0.5)",
          title_fontsize=9,
          label_fontsize=9).add_to(ax)
    ax.set_title("(6) Title + convert= (unit in title, not labels)",
                  fontsize=9)

    fig.suptitle("Ruler v2 refinements — title / fmt / side / rotation "
                  "/ positions / from_polar", fontsize=12)
    fig.subplots_adjust(top=0.91, hspace=0.30, wspace=0.18)
    return fig


@_panel("annot_06i_ruler_endcaps")
def render_ruler_endcaps():
    """``Ruler`` endcap styles — visually distinguish endpoints
    from regular ticks.

    All panels use a 54″ ruler with tick_interval=10″ — the
    endpoint at 54″ sits 4″ past the last regular tick (40 %
    of an interval), which is *below* the 50 % collision
    threshold, so the default ``endcap_style='none'`` drops it.
    Endcaps re-introduce a visually-distinct endpoint marker
    (and, with ``endcap_label='auto'``, re-introduce the
    endpoint label too).

    Six panels:

    1. **Default ('none')** — endpoint dropped by the collision
       rule. The ruler line extends visibly to 54″ but only the
       0/10/20/30/40/50 labels appear.
    2. **endcap_style='tick'** — endpoints render as longer
       ticks (1.8× length here). Endpoint label re-included.
    3. **endcap_style='arrow' (both)** — outward arrowheads at
       both endpoints; the regular endpoint ticks are replaced.
    4. **endcap_style='arrow', endcaps='end'** — arrow only at
       xy2; xy1 keeps a regular tick. Matches the from_polar
       use case ("54″ in this direction starting from here").
    5. **endcap_label=False** — endcap drawn, endpoint label
       suppressed (the caller wants the cap as a pure
       direction / extent marker without distance markup).
    6. **from_polar default** — auto-picks ``endcap_style='arrow'``
       and ``endcaps='end'``; the bar reads as "120″ at PA=30°
       from this point".
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    cdelt = (1.0) / 3600.0    # 1″/pix → 100" field

    def _setup(ax_slot, idx):
        fig.delaxes(ax_slot)
        ax = make_wcs_frame((2, 3, idx), projection="TAN",
                            center=(180.0, 0.0),
                            cdelt=cdelt, npix=(100, 100), fig=fig)
        fig.canvas.draw()
        ax.grid(True, color="0.85", lw=0.5, ls="-")
        return ax

    # (1) Default — collision-suppressed endpoint
    ax = _setup(axes[0, 0], 1)
    Ruler((20, 50), (74, 50), ax=ax,
          tick_interval=10.0,
          color="C0", lw=1.4,
          label_fontsize=9).add_to(ax)
    ax.set_title("(1) endcap_style='none' (default)\n"
                 "endpoint dropped by collision rule",
                 fontsize=9)

    # (2) tick endcaps
    ax = _setup(axes[0, 1], 2)
    Ruler((20, 50), (74, 50), ax=ax,
          tick_interval=10.0,
          color="C3", lw=1.4,
          endcap_style="tick",
          endcap_length_scale=1.8,
          label_fontsize=9).add_to(ax)
    ax.set_title("(2) endcap_style='tick' (1.8× length)\n"
                 "endpoint auto-labeled, endcap disambiguates",
                 fontsize=9)

    # (3) arrow endcaps (both)
    ax = _setup(axes[0, 2], 3)
    Ruler((20, 50), (74, 50), ax=ax,
          tick_interval=10.0,
          color="C2", lw=1.6,
          endcap_style="arrow",
          label_fontsize=9).add_to(ax)
    ax.set_title("(3) endcap_style='arrow' (both)\n"
                 "outward arrowheads replace endpoint ticks",
                 fontsize=9)

    # (4) arrow at end only
    ax = _setup(axes[1, 0], 4)
    Ruler((20, 50), (74, 50), ax=ax,
          tick_interval=10.0,
          color="C4", lw=1.6,
          endcap_style="arrow", endcaps="end",
          label_fontsize=9).add_to(ax)
    ax.set_title("(4) endcaps='end' only\n"
                 "start keeps regular tick",
                 fontsize=9)

    # (5) arrow + endcap_label=False
    ax = _setup(axes[1, 1], 5)
    Ruler((20, 50), (74, 50), ax=ax,
          tick_interval=10.0,
          color="C1", lw=1.6,
          endcap_style="arrow", endcaps="end",
          endcap_label=False,
          label_fontsize=9).add_to(ax)
    ax.set_title("(5) endcap_label=False\n"
                 "arrow drawn, endpoint label suppressed",
                 fontsize=9)

    # (6) from_polar default
    ax = _setup(axes[1, 2], 6)
    # PA=30° east of north on N-up E-left → tangent points up-and-left
    # (mpl angle 120°), so anchor at lower-right keeps the bar inside
    # the 100x100 px frame: end = (75-30, 25+52) = (45, 77).
    Ruler.from_polar((75, 25), length=60.0, angle=30.0,
                      ax=ax, length_unit="arcsec",
                      angle_convention="fits",
                      color="C5", lw=1.6,
                      tick_interval=15.0,
                      title="60″ @ PA=30°",
                      title_fontsize=10,
                      label_fontsize=9).add_to(ax)
    ax.set_title("(6) Ruler.from_polar default\n"
                 "auto endcap_style='arrow', endcaps='end'",
                 fontsize=9)

    fig.suptitle("Ruler endcap styles — none / tick / arrow + "
                  "endcap_label + from_polar default", fontsize=12)
    fig.subplots_adjust(top=0.91, hspace=0.35, wspace=0.20)
    return fig


@_panel("annot_06j_ruler_from_zero")
def render_ruler_from_zero():
    """``Ruler.from_zero`` factory + ``lambda0=`` low-level kwarg —
    anchor the value-0 tick at a specific coordinate, with the
    ruler extending in both directions.

    Four panels:

    1. **lambda0=0 (default)** — current scale-bar semantic: ticks
       run 0 → total from xy1 to xy2.
    2. **lambda0=0.5** — same endpoints, but the value-0 tick lands
       at the midpoint; labels are signed ±values.
    3. **from_zero symmetric** — coordinate-based factory: zero
       at xy_anchor, extends ``extent`` in both directions.
       Default endcaps are bidirectional arrows.
    4. **from_zero asymmetric** — different ``extent`` /
       ``extent_back``: zero remains at the anchor coordinate but
       the ruler is longer on one side than the other.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    cdelt = (1.0) / 3600.0    # 1″/pix → 100" field

    def _setup(ax_slot, idx):
        fig.delaxes(ax_slot)
        ax = make_wcs_frame((2, 2, idx), projection="TAN",
                            center=(180.0, 0.0),
                            cdelt=cdelt, npix=(100, 100), fig=fig)
        fig.canvas.draw()
        ax.grid(True, color="0.85", lw=0.5, ls="-")
        return ax

    # (1) lambda0=0 (default) — v1 baseline behavior
    ax = _setup(axes[0, 0], 1)
    Ruler((20, 50), (80, 50), ax=ax,
          tick_interval=10.0,
          color="C0", lw=1.4,
          label_fontsize=9).add_to(ax)
    ax.set_title("(1) lambda0=0 (default)\n"
                 "zero at xy1, labels 0 → total",
                 fontsize=9)

    # (2) lambda0=0.5 — symmetric around midpoint
    ax = _setup(axes[0, 1], 2)
    Ruler((20, 50), (80, 50), ax=ax,
          tick_interval=10.0,
          color="C3", lw=1.4,
          lambda0=0.5,
          label_fontsize=9).add_to(ax)
    ax.set_title("(2) lambda0=0.5\n"
                 "zero at midpoint, signed ±labels",
                 fontsize=9)

    # (3) from_zero symmetric — coordinate-based factory
    ax = _setup(axes[1, 0], 3)
    Ruler.from_zero((50, 50), extent=25.0, angle=0.0,
                     ax=ax, length_unit="arcsec",
                     angle_convention="plot",
                     tick_interval=10.0,
                     color="C2", lw=1.6,
                     label_fontsize=9).add_to(ax)
    ax.set_title("(3) from_zero(xy_anchor, extent=25″)\n"
                 "symmetric, arrows on both ends (default)",
                 fontsize=9)

    # (4) from_zero asymmetric
    ax = _setup(axes[1, 1], 4)
    Ruler.from_zero((40, 50), extent=40.0, extent_back=15.0,
                     angle=0.0,
                     ax=ax, length_unit="arcsec",
                     angle_convention="plot",
                     tick_interval=10.0,
                     color="C5", lw=1.6,
                     label_fontsize=9).add_to(ax)
    ax.set_title("(4) from_zero asymmetric\n"
                 "extent=40″, extent_back=15″",
                 fontsize=9)

    fig.suptitle("Ruler.from_zero + lambda0= — zero-anchored rulers "
                  "with signed labels", fontsize=12)
    fig.subplots_adjust(top=0.91, hspace=0.32, wspace=0.20)
    return fig


@_panel("annot_06k_ruler_from_axes_fraction")
def render_ruler_from_axes_fraction():
    """``Ruler.from_axes_fraction`` — axes-fraction (0–1) endpoint
    coordinates that stay dynamically pinned during pan / zoom /
    resize. Natural for pseudo twin-axis spines just outside the
    plot frame.

    The 2-panel demo places the rulers on the **right Y axis** to
    avoid overlap with the host axes' own x-tick labels along the
    bottom. The ruler kwargs are configured to mimic a real
    matplotlib twin y-axis: one-sided ticks pointing outward,
    horizontal label rotation, and a title sitting on the same
    side as the labels (further out so it clears them).

    1. **Single twin Y-axis spine** — one Ruler at axes-fraction
       x=1.10, spanning the full y-extent, ticks every 40″.
    2. **Stacked twin Y-axes** — two parallel Rulers, the first
       showing offset arcsec, the second showing the same span
       converted to projected kpc at z=0.5. Stack as many as
       desired by progressively increasing ``x_frac``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    cdelt = (2.0) / 3600.0     # 2"/pix → 200" field

    # Shared twin-axis styling kwargs. ``title_beyond_labels=True``
    # auto-computes the title offset so the title sits past the
    # rendered label bboxes (the matplotlib twin-axis look —
    # title further out than the labels on the same side).
    # ``lambda0=0.5`` places the value-0 tick at the *center* of
    # the spine — the natural choice for an offset-coordinate
    # twin axis (labels read ±values from the host axes' center).
    twin_kwargs = dict(
        tick_side='right',           # ticks point outward only
        label_side='right',          # labels on the outward (+x) side
        label_rotation='horizontal',   # labels read horizontally
        title_side='right',           # title on same side as labels
        title_beyond_labels=True,     # auto-clear the label bbox
        title_rotation='auto',        # title rotated along the spine
        lambda0=0.5,                   # zero at center of spine
    )

    # (1) Single twin spine on the right Y axis
    ax = axes[0]
    fig.delaxes(ax)
    ax = make_wcs_frame((1, 2, 1), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    Ruler.from_axes_fraction(
        (1.10, 0.0), (1.10, 1.0), ax=ax,
        tick_interval=40.0,
        color="C0", lw=1.4,
        title="Offset (arcsec)",
        title_fontsize=10,
        label_fontsize=9,
        **twin_kwargs,
    ).add_to(ax)
    ax.set_title("(1) Twin Y-spine at axes-fraction x = 1.10\n"
                 "lambda0=0.5 → zero at center, ±offsets",
                  fontsize=10)

    # (2) Stacked twin Y-spines on the right
    ax = axes[1]
    fig.delaxes(ax)
    ax = make_wcs_frame((1, 2, 2), projection="TAN",
                        center=(180.0, 0.0),
                        cdelt=cdelt, npix=(100, 100), fig=fig)
    fig.canvas.draw()
    # First twin: arcsec offset from center, closer to the host axes
    Ruler.from_axes_fraction(
        (1.10, 0.0), (1.10, 1.0), ax=ax,
        tick_interval=40.0,
        color="C3", lw=1.4,
        title="Offset (arcsec)",
        title_fontsize=9,
        label_fontsize=8,
        **twin_kwargs,
    ).add_to(ax)
    # Second twin: same span converted to kpc at z=0.5 (Planck18),
    # placed further to the right.
    Ruler.from_axes_fraction(
        (1.35, 0.0), (1.35, 1.0), ax=ax,
        tick_interval=40.0,
        convert=dict(redshift=0.5, unit="kpc"),
        fmt="%.0f",
        color="C2", lw=1.4,
        title="Projected distance (kpc, z=0.5)",
        title_fontsize=9,
        label_fontsize=8,
        **twin_kwargs,
    ).add_to(ax)
    ax.set_title("(2) Stacked twins (arcsec + kpc), centered at zero",
                  fontsize=10)

    fig.suptitle("Ruler.from_axes_fraction — pinned twin-axis spines",
                  fontsize=12)
    # Leave generous right margin so the stacked twins + titles fit.
    fig.subplots_adjust(left=0.06, right=0.78, top=0.86, wspace=0.55)
    return fig


@_panel("annot_07_sizebar_pixels")
def render_sizebar_pixels():
    """add_sizebar — pixel-units form."""
    fig, ax = _tan_axes(fov_arcsec=200.0)
    add_sizebar(ax, length_pixels=25, label="25 px",
                color="black")
    ax.set_title("add_sizebar (length_pixels=25)")
    return fig


@_panel("annot_08_sizebar_asec")
def render_sizebar_asec():
    """add_sizebar_asec — arcsec-units form."""
    fig, ax = _tan_axes(fov_arcsec=200.0)
    hdr = _tan_header(fov_arcsec=200.0)
    add_sizebar_asec(ax, hdr, length_asec=30.0, label="30″",
                     color="black")
    ax.set_title("add_sizebar_asec (length_asec=30 → ~15 px)")
    return fig


@_panel("annot_09_bandlabels")
def render_bandlabels():
    """add_bandlabels on a multi-color (RGB) composite.

    Builds a synthetic 3-channel image where each filter contributes a
    Gaussian source at a different position, then labels each band in
    its own color at the top-left corner — exactly the standard
    "multicolor-composite" labeling pattern the function is designed
    for.
    """
    import matplotlib.patheffects as PathEffects

    fig, ax = _tan_axes()

    # Build a 3-channel RGB composite: each band = one Gaussian source
    nx, ny = 100, 100
    yy, xx = np.mgrid[0:ny, 0:nx]
    sigma = 11.0

    def gauss(cx, cy, amp=1.0):
        return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    R = 0.20 + 0.80 * gauss(28, 35)   # U-band — at lower-left
    G = 0.20 + 0.80 * gauss(50, 50)   # B-band — at center
    B = 0.20 + 0.80 * gauss(72, 65)   # V-band — at upper-right
    rgb = np.dstack([R, G, B])
    rgb = np.clip(rgb, 0, 1)

    ax.imshow(rgb, origin="lower",
              transform=ax.get_transform("pixel"))

    # Add the band labels with a stroke effect for legibility against
    # any background color.
    pe = [PathEffects.withStroke(linewidth=2, foreground="black")]
    band_colors = ["#FF6060", "#60D060", "#6090FF"]  # match RGB channels
    add_bandlabels(ax,
                   labels=["U", "B", "V"],
                   labcolors=band_colors,
                   fontsize=22,
                   xy=(0.05, 0.95),
                   textpad=0.5)
    # Stroke wasn't a kwarg of add_bandlabels — apply manually via ax.texts
    for txt in ax.texts[-3:]:
        txt.set_path_effects(pe)

    ax.set_title("add_bandlabels — U / B / V on RGB composite")
    return fig


@_panel("annot_10_style_ax_colors")
def render_style_ax_colors():
    """style_ax_colors on a plain matplotlib axes — applies a single
    color to ticks, labels, title, and spines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5))
    ax1.plot(np.linspace(0, 10), np.sin(np.linspace(0, 10)))
    ax1.set_xlabel("x")
    ax1.set_ylabel("sin(x)")
    ax1.set_title("default colors (black on white)")

    ax2.plot(np.linspace(0, 10), np.cos(np.linspace(0, 10)))
    ax2.set_xlabel("x")
    ax2.set_ylabel("cos(x)")
    ax2.set_title("after style_ax_colors('darkred')")
    ax2.set_facecolor("#fff8f0")
    style_ax_colors(ax2, color="darkred")

    fig.suptitle("style_ax_colors — recolor ticks/labels/spines/title",
                 y=1.02)
    return fig


def main():
    banner("overlays.annotations — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
