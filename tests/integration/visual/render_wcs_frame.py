"""Render every WCS projection registered in skyplothelper.

Produces:
  - wcsframe_01_allsky_grid.png — 5×4 grid of all 20 all-sky projections
  - wcsframe_02_field_grid.png  — 4×3 grid of all 12 zenithal/conic field projections
  - wcsframe_03_center_shift.png — Plate Carrée centered at 0° vs 180° (RA wrap demo)
  - wcsframe_04_frame_change.png — AIT in ICRS / Galactic / Ecliptic frames
  - wcsframe_05_non_fits_focus.png — Robinson, Kavrayskiy, Eckert IV, Winkel Tripel,
                                       McBryde-Thomas side-by-side (the 5 non-FITS)

Usage
-----
    python render_wcs_frame.py            # save PNGs to output/
    python render_wcs_frame.py --show     # display interactively
"""

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper.coord_overlay import add_coord_overlay, add_overlay_ticks
from skyplothelper.overlays.planes import add_plane_overlay
from skyplothelper.projections._boundaries import (
    bonne_boundary,
    conic_boundary,
    healpix_boundary,
    polyconic_boundary,
)
from skyplothelper.projections.registry import _PROJECTION_REGISTRY
from skyplothelper.wcs_frame import (
    _get_wcs_frame_name,
    make_wcs_frame,
)

# Conic projections (COD/COE/COO/COP) get the PV2_1=45 default
# supplied by ``make_wcs_frame``, so they construct directly through
# the public API and appear in the field-view gallery.
#
# "Zoomed-rather-than-allsky" projections — these are
# registered with allsky=True (the projection is mathematically
# all-sky) but their natural visual presentation is one zoomed
# face / one polar region, not the full cube-unfold or butterfly.
# They appear in wcsframe_02 with an explicit fov_deg, which
# triggers the field-view CDELT branch in make_wcs_frame.
_ZOOMED_FIELD_KEYS = ('csc', 'tsc', 'qsc', 'xph')

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _safe_make(fig, subspec, projection, **kw):
    """Build axes for one projection panel; return None on failure.

    For projections where ``_decorate_panel`` will draw its own
    custom-styled overlay ticks (circular zenithals + the
    ``_CUSTOM_BOUNDARY_HELPERS`` family — BON/PCO/HPX/conics), pass
    ``tick_style='native'`` so the ``make_wcs_frame`` auto-trigger
    doesn't draw a competing set on top (which otherwise
    reads as doubled labels). All other projections — SFL / PAR /
    AIT / MOL / TAN / etc. — keep the default ``tick_style='auto'``,
    which lets the auto-trigger handle pseudo-cylindrical apex-label
    glitches (the ``'$'`` truncated-mathtext at the SFL/PAR south
    apex that astropy's default labeler produces).
    """
    info = _PROJECTION_REGISTRY.get(projection.lower())
    needs_native = info is not None and (
        info.frame_shape == 'circular'
        or (info.fits_code is not None
            and info.fits_code in _CUSTOM_BOUNDARY_HELPERS))
    if needs_native:
        kw.setdefault('tick_style', 'native')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return make_wcs_frame(subspec, projection=projection, fig=fig, **kw)
        except Exception as e:
            print(f"  WARNING: {projection!r} failed to construct: "
                  f"{type(e).__name__}: {e}")
            return None


# Projections where the astropy-default WCSAxes ticks look noticeably
# worse than axis-curve overlay placement. ``make_wcs_frame``
# exposes this via ``tick_style='in_frame'`` (the
# auto-default for circular frame_shapes). This gallery overrides to
# ``tick_style='native'`` (see ``_safe_make``) so ``_decorate_panel``
# can apply its own custom-styled overlay ticks per panel.
#
# Currently limited to circular-frame zenithals (AIR / ARC / AZP /
# SIN / STG / SZP / ZEA): default lon labels stack horizontally along
# the bbox top/bottom, producing the orthographic-style smooshed look.
# Axis-curve mode puts lon labels along the center-lat parallel
# (bowing with the spine) and lat labels along the center-lon meridian
# — all confined to the visible circle via :func:`add_overlay_ticks`'s
# boundary-polygon clip.
#
# Not auto-applied for:
# (XPH HEALPix butterfly in wcsframe_02 already looks fine with the
# astropy default — same situation as the spherical cubes.)

# FITS code → (boundary helper, render kwargs) for projections whose
# visible region requires a custom boundary curve (their astropy frame
# spine is rectangular/elliptical but the visible region is something
# else). Each helper returns a closed ``(lon, lat)`` polyline that
# ``add_overlay_ticks(... boundary=...)`` uses for both clipping and
# (when ``lon_at='axis'``/``lat_at='axis'`` mode is active) staying
# inside the visible region.
_CUSTOM_BOUNDARY_HELPERS = {
    'BON': bonne_boundary,
    'COD': conic_boundary,
    'COE': conic_boundary,
    'COO': conic_boundary,
    'COP': conic_boundary,
    'PCO': polyconic_boundary,
    'HPX': healpix_boundary,
}

# Per-family opt-ins for the three additional visual polishes that
# sit on top of the axis-curve overlay ticks. Each is independent.

# Draw the projection's natural boundary as a black outline. BON's
# cardioid is already drawn by ``make_wcs_frame`` via
# ``_draw_allsky_lon_boundary``. HPX draws via the stepped-diamond
# boundary helper on its rectangular frame.
#
# Conics (COD/COE/COO/COP) are deliberately excluded: their
# visible-region geometries differ wildly per conic flavor (COD
# overshoots the bbox, COE fits, COO extends FAR past the bbox in
# pixel space, COP returns NaN at the seam-epsilon meridian) — no
# single boundary helper traces a clean outline for all four, and
# an explicit wedge outline reads as "a dark curve cutting through
# the projection" rather than bounding it; bbox spines + gridlines
# convey the visible region more cleanly.
_DRAW_BOUNDARY_CODES = frozenset({'PCO', 'HPX'})

# Add a same-frame ``add_coord_overlay`` pass to backfill gridline
# segments astropy's default rendering misses on these projections.
# BON / PCO / HPX are full-sky non-rectangular envelopes where the
# astropy gridline densifier truncates or skips wrap-side segments.
# Conics (COD/COE/COO/COP) are included — the
# backfill + boundary clip (passed via ``_backfill_gridlines``'s
# ``boundary_lonlat=`` kwarg) cleanly confines the wedge gridlines
# to the visible region instead of letting astropy's defaults
# extend across the full rectangular bbox.
_BACKFILL_GRID_CODES = frozenset({'BON', 'PCO', 'HPX',
                                  'COD', 'COE', 'COO', 'COP'})

# Tighten ``ax.set_xlim`` / ``ax.set_ylim`` to the boundary's actual
# pixel extent so the projection fills the panel without empty space.
# BON specifically had a noticeable gap on the sides under the
# default make_wcs_frame extents.
_TIGHTEN_AXES_CODES = frozenset({'BON'})


def _draw_projection_boundary(ax, lonlat, *, color='black', lw=0.7,
                              zorder=1.5, margin=20.0):
    """Plot a ``(lon, lat)`` polyline as the projection boundary
    outline.

    Sanitizes the polyline in pixel space — NaN samples (from
    projection singularities like the COP cone apex) and samples
    outside the axes bbox plus ``margin`` pixels get NaN-marked so
    matplotlib's line plotting doesn't draw triangular connecting
    segments across them. The tight margin is essential for conics
    where the bottom-arc samples at lat=-89.99 extrapolate to
    valid-but-divergent pixels well outside the visible wedge.
    """
    pix = ax.get_transform('world').transform(lonlat)
    bb = ax.bbox
    finite = np.isfinite(pix).all(axis=1)
    inside = ((pix[:, 0] >= bb.x0 - margin) & (pix[:, 0] <= bb.x1 + margin)
              & (pix[:, 1] >= bb.y0 - margin) & (pix[:, 1] <= bb.y1 + margin))
    keep = finite & inside
    # Replace dropped samples with NaN to break the polyline at gaps.
    pix_plot = pix.copy()
    pix_plot[~keep] = np.nan
    # Convert to data coords (NaN survives) and plot via transData.
    data = ax.transData.inverted().transform(pix_plot)
    ax.plot(data[:, 0], data[:, 1],
            color=color, lw=lw, zorder=zorder)


def _backfill_gridlines(ax, boundary_lonlat=None,
                        *, color='gray', alpha=0.4, lw=0.5, ls=':'):
    """Add an axes-frame ``add_coord_overlay`` pass so projection
    gridlines astropy's default rendering missed get drawn (the
    overlay machinery samples gridlines densely and handles wrap-
    side / multi-face segments cleanly).

    Also suppresses astropy's default gridline rendering so the
    incomplete default lines don't bleed through underneath the
    complete overlay set.

    Parameters
    ----------
    ax : WCSAxes
    boundary_lonlat : (N, 2) array_like, optional
        Closed ``(lon, lat)`` polyline of the projection's visible
        region. When provided, every overlay gridline artist is
        clipped to this polygon so wrap-side / extrapolated segments
        that wander outside the visible region (e.g. conic wedge
        gridlines spilling into the bbox corners) get cleanly trimmed.
    """
    try:
        ax.coords[0].grid(draw_grid=False)
        ax.coords[1].grid(draw_grid=False)
        overlay = add_coord_overlay(
            ax, frame=_get_wcs_frame_name(ax),
            color=color, alpha=alpha, lw=lw, ls=ls)
        if boundary_lonlat is not None:
            _clip_overlay_to_boundary(ax, overlay, boundary_lonlat)
    except Exception as e:
        print(f"  WARNING: gridline backfill failed: "
              f"{type(e).__name__}: {e}")


def _clip_overlay_to_boundary(ax, overlay, lonlat, margin=20.0):
    """Clip every gridline artist in *overlay* to the polygon traced by
    the ``(lon, lat)`` boundary polyline.

    Polyline is sanitized in pixel space — NaN samples (from projection
    singularities like the COP cone apex) and samples outside
    ``ax.bbox`` plus ``margin`` pixels get dropped before the polygon
    is formed. Same recipe ``_draw_projection_boundary`` uses for line
    drawing; here it keeps the clip polygon from collapsing or
    inverting when a boundary side extrapolates to infinity.
    """
    from matplotlib.path import Path

    pix = ax.get_transform('world').transform(lonlat)
    bb = ax.bbox
    finite = np.isfinite(pix).all(axis=1)
    inside = ((pix[:, 0] >= bb.x0 - margin) & (pix[:, 0] <= bb.x1 + margin)
              & (pix[:, 1] >= bb.y0 - margin) & (pix[:, 1] <= bb.y1 + margin))
    keep = finite & inside
    if keep.sum() < 3:
        return
    data = ax.transData.inverted().transform(pix[keep])
    path = Path(data)
    for artists in overlay.lon_artists + overlay.lat_artists:
        for ln in artists:
            ln.set_clip_path(path, transform=ax.transData)


def _tighten_axes_to_boundary(ax, lonlat, pad=0.05):
    """Set ``ax.set_xlim`` / ``ax.set_ylim`` to the boundary's actual
    data-coord extent (plus ``pad`` fractional margin)."""
    pix = ax.get_transform('world').transform(lonlat)
    pix = pix[np.isfinite(pix).all(axis=1)]
    if len(pix) < 4:
        return
    data = ax.transData.inverted().transform(pix)
    x_pad = (data[:, 0].max() - data[:, 0].min()) * pad
    y_pad = (data[:, 1].max() - data[:, 1].min()) * pad
    ax.set_xlim(data[:, 0].min() - x_pad, data[:, 0].max() + x_pad)
    ax.set_ylim(data[:, 1].min() - y_pad, data[:, 1].max() + y_pad)


def _apply_overlay_ticks_if_useful(ax, info):
    """Per-projection-family policy for the wcsframe gallery: apply
    axis-curve overlay ticks where they read more naturally than
    astropy's defaults.
    """
    if ax is None:
        return
    frame_shape = getattr(info, "frame_shape", None)
    fits_code = (getattr(info, "fits_code", None) or "").upper()

    # Path 1: native-circular-spine projections (zenithals with
    # fov-limited views). The astropy frame already gives us the
    # right spine — no custom boundary needed.
    if frame_shape == "circular":
        try:
            add_overlay_ticks(
                ax, lon_at="axis", lat_at="axis",
                tick_kwargs={"length": 4, "color": "0.2", "lw": 0.8},
                label_kwargs={"color": "0.2"})
        except Exception as e:
            print(f"  WARNING: add_overlay_ticks failed on "
                  f"{fits_code!r}: {type(e).__name__}: {e}")
        return

    # Path 2: projections with a custom-computed boundary (BON
    # cardioid, PCO egg, HPX stepped diamond, conic wedges).
    boundary_fn = _CUSTOM_BOUNDARY_HELPERS.get(fits_code)
    if boundary_fn is not None:
        try:
            lonlat = boundary_fn(ax)
            if fits_code in _TIGHTEN_AXES_CODES:
                _tighten_axes_to_boundary(ax, lonlat)
            if fits_code in _DRAW_BOUNDARY_CODES:
                _draw_projection_boundary(ax, lonlat)
            if fits_code in _BACKFILL_GRID_CODES:
                # Conics: skip the overlay clip. The world_rect polygon
                # doesn't correctly bound conic visible regions in
                # pixel space — its meridians at lon=CRVAL±180 sit on
                # the multi-valued antimeridian seam and collapse to a
                # narrow wedge that excludes most of the visible
                # teardrop. The natural NaN-projection at invalid
                # points already keeps overlay gridlines within bounds,
                # and skipping the clip lets the gridlines fill the
                # full visible region (matching what the scatter probe
                # shows is actually projectable).
                is_conic = fits_code in {'COD', 'COE', 'COO', 'COP'}
                _backfill_gridlines(
                    ax,
                    boundary_lonlat=(None if is_conic else lonlat),
                    color='0.35', alpha=0.65, lw=0.6)
            add_overlay_ticks(
                ax, lon_at="axis", lat_at="axis",
                boundary=lonlat,
                tick_kwargs={"length": 4, "color": "0.2", "lw": 0.8},
                label_kwargs={"color": "0.2"})
        except Exception as e:
            print(f"  WARNING: add_overlay_ticks failed on "
                  f"{fits_code!r}: {type(e).__name__}: {e}")
        return


def _decorate_panel(ax, title, info):
    """Common per-panel styling: title, light galactic-plane overlay,
    and (for select projections) axis-curve overlay ticks that read
    better than the astropy defaults."""
    if ax is None:
        return
    try:
        add_plane_overlay(ax, plane="galactic", color="C3", lw=0.7, alpha=0.7)
    except Exception:
        pass
    _apply_overlay_ticks_if_useful(ax, info)
    ax.set_title(title, fontsize=8)


@_panel("wcsframe_01_allsky_grid")
def render_allsky_grid():
    """Grid of all-sky projections, EXCLUDING zoomed-only members
    (cubes + XPH — see _ZOOMED_FIELD_KEYS) which now live in
    wcsframe_02 as field-view zooms."""
    allsky = [(k, v) for k, v in _PROJECTION_REGISTRY.items()
              if v.allsky and k not in _ZOOMED_FIELD_KEYS]
    allsky.sort(key=lambda kv: (kv[1].fits_code or "z" + kv[0]))
    nrows, ncols = 4, 4
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"All-sky projections — full-sky bounded rendering "
        f"({len(allsky)} of {len(_PROJECTION_REGISTRY)})",
        fontsize=14, y=0.995,
    )
    for idx, (key, info) in enumerate(allsky, start=1):
        ax = _safe_make(fig, (nrows, ncols, idx), key, center=180)
        code = info.fits_code if info.fits_code else f"({key})"
        _decorate_panel(ax, f"{code} — {info.description}", info)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.04,
                        hspace=0.5, wspace=0.4)
    return fig


@_panel("wcsframe_02_field_grid")
def render_field_grid():
    """Every non-allsky projection PLUS the zoomed-only allsky
    projections (cubes + XPH) — field-of-view zooms."""
    field = [(k, v) for k, v in _PROJECTION_REGISTRY.items()
             if not v.allsky]
    field.sort(key=lambda kv: (kv[1].fits_code or "z" + kv[0]))
    # Append the zoomed-only allsky entries (cubes + XPH) in
    # registry order
    zoomed = [(k, _PROJECTION_REGISTRY[k]) for k in _ZOOMED_FIELD_KEYS
              if k in _PROJECTION_REGISTRY]
    panels = field + zoomed
    n = len(panels)
    # Re-tile depending on count; was 4×3=12, now 4×4=16 with the
    # zoomed additions
    nrows, ncols = 4, 4
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle(
        f"Field-of-view projections — non-allsky + zoomed-only "
        f"allsky ({n} usable; conics use PV2_1=45 default; cubes "
        f"/ XPH use fov_deg=70)",
        fontsize=13, y=0.995,
    )
    for idx, (key, info) in enumerate(panels, start=1):
        is_zoomed = key in _ZOOMED_FIELD_KEYS
        # ``fov_deg=70`` for cubes / XPH keeps the view safely inside
        # one cube face (or the XPH "central diamond") and avoids
        # gridline-compression strip artifacts at the face boundaries
        # (which occur at lat≈±37-45° depending on cube formula —
        # TSC and QSC have face boundaries below CSC's 45° due to
        # their tangential/quadrilateralized formulations).
        kwargs = {'center': (0.0, 0.0), 'fov_deg': 70.0} if is_zoomed \
            else {'center': (180.0, 30.0)}
        ax = _safe_make(fig, (nrows, ncols, idx), key, **kwargs)
        code = info.fits_code if info.fits_code else f"({key})"
        _decorate_panel(ax, f"{code} — {info.description}", info)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.05,
                        hspace=0.5, wspace=0.4)
    return fig


@_panel("wcsframe_03_center_shift")
def render_center_shift():
    """Same projection (CAR), two different centers — RA-wrap demo."""
    fig = plt.figure(figsize=(14, 5))
    for idx, lon0 in enumerate((0, 180), start=1):
        ax = _safe_make(fig, 120 + idx, "CAR", center=lon0)
        try:
            add_plane_overlay(ax, plane="galactic", color="C3", lw=1.0)
        except Exception:
            pass
        ax.set_title(f"Plate Carrée — center=({lon0}, 0)")
    fig.suptitle("center-shift demo: RA wrap point moves with `center=` parameter",
                 fontsize=12)
    fig.subplots_adjust(top=0.85, wspace=0.25)
    return fig


@_panel("wcsframe_04_frame_change")
def render_frame_change():
    """Same AIT projection, three different frames — galactic plane in each."""
    fig = plt.figure(figsize=(15, 5))
    frames = [
        ("ICRS", "RA / Dec axes"),
        ("Galactic", "GLON / GLAT axes"),
        ("geocentrictrueecliptic", "Ecliptic lon/lat axes"),
    ]
    for idx, (frame, subtitle) in enumerate(frames, start=1):
        ax = _safe_make(fig, 130 + idx, "AIT", center=0, frame=frame)
        try:
            add_plane_overlay(ax, plane="galactic", color="C3", lw=1.0,
                              label="galactic plane")
            add_plane_overlay(ax, plane="ecliptic", color="C2", lw=1.0,
                              label="ecliptic")
        except Exception:
            pass
        ax.set_title(f"{frame} — {subtitle}")
    fig.suptitle("frame= parameter — same AIT axes in three coord frames",
                 fontsize=12)
    fig.subplots_adjust(top=0.86, wspace=0.3)
    return fig


@_panel("wcsframe_05_non_fits_focus")
def render_non_fits_focus():
    """The 5 non-FITS pseudocylindrical projections in a 2-column,
    3-row layout (one slot empty) so each frame keeps a usable
    aspect ratio rather than getting squished by a 1×5 strip."""
    keys = ["robinson", "kavrayskiy", "eckert_iv", "winkel_tripel", "mcbryde"]
    nrows, ncols = 3, 2
    fig = plt.figure(figsize=(13, 11))
    for idx, key in enumerate(keys, start=1):
        ax = _safe_make(fig, (nrows, ncols, idx), key, center=180)
        info = _PROJECTION_REGISTRY[key]
        try:
            add_plane_overlay(ax, plane="galactic", color="C3", lw=0.9)
            add_plane_overlay(ax, plane="ecliptic", color="C2", lw=0.9)
        except Exception:
            pass
        ax.set_title(info.description.replace(" [non-FITS]", ""), fontsize=10)
    fig.suptitle(
        "Non-FITS projections — implemented via CurvedTransform "
        "(no FITS code in the registry)", fontsize=12,
    )
    fig.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.04,
                        hspace=0.4, wspace=0.25)
    return fig


def main():
    banner("wcs_frame — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
