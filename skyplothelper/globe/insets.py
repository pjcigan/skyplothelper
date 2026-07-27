"""Reproject inset axes and connector helpers.

``reproject_inset_axes`` builds a child axes whose WCS is a reprojection
of a region of the parent axes' WCS; ``mark_inset_axes`` and
``connect_inset_axes`` draw the parent-side ROI marker and the
connector lines respectively. The API is inspired by ligo.skymap's
inset helpers (https://lscsoft.docs.ligo.org/ligo.skymap/plot/allsky.html#insets).

Typical usage::

    fig = plt.figure(figsize=(10, 6))
    main_ax = ...  # an all-sky WCSAxes (AIT, MOL, ...)
    inset = reproject_inset_axes(
        main_ax, rect=[0.6, 0.1, 0.35, 0.4],
        projection='TAN', center=(45, 30), size=5)
    mark_inset_axes(main_ax, inset, style='rectangle', edgecolor='red')
    connect_inset_axes(main_ax, inset)

The inset inherits the parent's coordinate frame (RA/Dec, galactic,
ecliptic, ...) by default, swapping just the projection code (e.g.
``RA---AIT`` → ``RA---TAN`` for a TAN zoom on an AIT parent).

Known limitations:

* ROI outlines crossing the parent's antimeridian are split at the
  wrap (see ``mark_inset_axes``'s ``wrap_fix=``); this cleans up the
  outline, but *filled* ROIs may still show artifacts near the wrap.
* Connector endpoints track figure resizes (they are evaluated at
  draw time), but the *choice* of which corners to connect is made at
  call time — re-call ``connect_inset_axes`` after drastically
  repositioning the inset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from astropy.wcs import WCS
from matplotlib.patches import ConnectionPatch  # noqa: F401
from matplotlib.path import Path as MplPath  # noqa: F401

from ..wcs_frame import (  # noqa: F401  (re-exported; used here + by tests)
    _east_increases_right,
    _is_wcsaxes,
)
from .decorations import _is_circular_frame
from .spherical import destination_point, small_circle

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


def _outer_tangent_endpoints(
    cA: Any, rA: float, cB: Any, rB: float,
) -> np.ndarray | None:
    """Endpoints of the two outer common tangents of circles A and B.

    For two circles in a plane with centers ``cA``, ``cB`` and radii
    ``rA``, ``rB``, returns a 2×2 array of tangent points: the first
    row is the tangent point on each circle for one outer tangent
    line, the second row for the other. Both tangents touch each
    circle on the same side, so they hug the outside of the pair
    without crossing between them.

    Math: for an outer tangent, the line touches at points where the
    radius vector is perpendicular to the tangent direction and the
    two tangent radii are *parallel*. Writing the perpendicular unit
    vector as ``p̂ = (cos α, sin α)``, the perpendicularity condition
    reduces to ``cos(α - ψ) = (rA - rB) / d`` where ``ψ`` is the
    direction from ``cA`` to ``cB`` and ``d = |cB - cA|``. So
    ``α = ψ ± β`` with ``β = arccos((rA - rB) / d)``, and the
    tangent points are ``cA + rA·(cos α, sin α)`` and
    ``cB + rB·(cos α, sin α)``. For equal radii this reduces to
    ``α = ψ ± π/2`` (the perpendicular-to-line-of-centers form).

    Returns ``None`` when the circles are too close for outer
    tangents to exist (one strictly contains the other, or the
    centers coincide).
    """
    dx = cB[0] - cA[0]
    dy = cB[1] - cA[1]
    d = float(np.hypot(dx, dy))
    if d < 1e-9:
        return None
    arg = (rA - rB) / d
    if abs(arg) >= 1.0:
        return None
    psi = float(np.arctan2(dy, dx))
    beta = float(np.arccos(arg))
    alphas = (psi + beta, psi - beta)
    out = np.empty((2, 2, 2), dtype=float)
    for i, alpha in enumerate(alphas):
        ca, sa = np.cos(alpha), np.sin(alpha)
        out[i, 0] = (cA[0] + rA * ca, cA[1] + rA * sa)
        out[i, 1] = (cB[0] + rB * ca, cB[1] + rB * sa)
    return out


def _reformat_ctype(old_ctype: str, new_proj: str) -> str:
    """
    Replace the projection code of a FITS CTYPE string, preserving the
    coordinate frame prefix.

    FITS CTYPEs are 8 characters: ``{FRAME:<4s}-{PROJ:<3s}`` with ``-``
    padding, e.g. ``RA---TAN``, ``DEC--AIT``, ``GLON-MOL``.

    Parameters
    ----------
    old_ctype : str
        Original CTYPE (8 chars expected; shorter strings are accepted).
    new_proj : str
        New projection code (3 chars: ``TAN``, ``SIN``, ``AIT``, ``MOL``,
        ``CAR``, ``SFL``, ``PAR``, etc.).

    Returns
    -------
    str
        The reformatted CTYPE string (always 8 chars).

    Examples
    --------
    >>> _reformat_ctype('RA---AIT', 'TAN')
    'RA---TAN'
    >>> _reformat_ctype('GLON-MOL', 'SIN')
    'GLON-SIN'
    >>> _reformat_ctype('ELAT-CAR', 'TAN')
    'ELAT-TAN'
    """
    frame = old_ctype[:5].split('-')[0].rstrip()
    return f'{frame:<4s}-{new_proj:<3s}'.replace(' ', '-')


def _add_inset_background(ax: Any, color: Any) -> Any:
    """Add an opaque background patch behind an inset's content.

    Uses the frame patch's own path (rectangular or elliptical) so the fill
    follows the frame shape, at a very low zorder behind everything. Being an
    ordinary child patch (not the axes patch), it survives a
    ``savefig(transparent=True)`` — unlike an axes facecolor, which that export
    renders transparent. Falls back to the axes-boundary rectangle if the frame
    path isn't available.
    """
    from matplotlib.patches import PathPatch, Rectangle

    fpatch = getattr(getattr(ax, 'coords', None), 'frame', None)
    fpatch = getattr(fpatch, 'patch', None)
    if fpatch is not None and len(np.asarray(fpatch.get_path().vertices)) > 2:
        patch: Any = PathPatch(
            fpatch.get_path(), transform=fpatch.get_transform(),
            facecolor=color, edgecolor='none', zorder=-1000,
            clip_on=False, gid='_sph_inset_bg')
    else:
        patch = Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes, facecolor=color,
            edgecolor='none', zorder=-1000, gid='_sph_inset_bg')
    ax.add_patch(patch)
    return patch


def reproject_inset_axes(parent_ax: Any, rect: Any, wcs: Any = None,
                         projection: str = 'TAN',
                         center: SkyCoord | tuple[float, float] | None = None, size: Any = None,
                         fig: Any = None, npix: int = 500,
                         inherit_frame: bool = True,
                         direction: str = 'inherit',
                         transform: Any = None,
                         auto_fontsize: bool = True,
                         bg_color: Any = None,
                         tick_style: str = 'auto',
                         tick_rotation: Any = 'tangent',
                         **subplot_kw: Any) -> Any:
    """
    Create a WCSAxes inset at ``rect`` covering a sky region.

    The inset has its own independent WCS. If ``wcs`` is not given
    explicitly, one is constructed from ``center``, ``size``, and
    ``projection``, inheriting the parent's coordinate frame (RA/Dec,
    galactic, ecliptic, ITRS, ...) when ``inherit_frame=True``.

    The inset may sit outside the parent frame, partially overlap it, or
    be placed fully inside it (useful for "HST FOV bubble on a wide-field
    view" figures). Placement is controlled by the ``transform`` kwarg.

    Parameters
    ----------
    parent_ax : matplotlib Axes (typically WCSAxes)
        The main axes whose region is being zoomed. Used only to inherit
        the coordinate frame by default (``inherit_frame=True``), and to
        resolve ``transform='parent'`` placements.
    rect : 4-tuple
        ``(left, bottom, width, height)`` in the coordinate system
        specified by ``transform`` (default: figure fraction).
    wcs : astropy.wcs.WCS or None
        Explicit WCS for the inset. If ``None``, one is built below.
    projection : str
        FITS projection code for the auto-constructed WCS. Common
        choices: ``'TAN'`` (gnomonic, for small-area zoom), ``'SIN'``
        (orthographic, for globe-style zoom), ``'AIT'`` (Aitoff),
        ``'MOL'`` (Mollweide), ``'CAR'`` (plate carrée), ``'ZEA'``
        (Lambert equal-area). Ignored if ``wcs`` is given.
    center : (lon, lat) in degrees, SkyCoord, or None
        Center of the zoom region. Required if ``wcs`` is None. A scalar
        :class:`~astropy.coordinates.SkyCoord` is resolved in the PARENT
        axes' frame (the frame the inset is cut from), not blindly ICRS.
    size : float or (dx, dy) in degrees or None
        Angular extent of the zoom region. Required if ``wcs`` is None.
        A scalar means a square region.
    fig : matplotlib.figure.Figure or None
        Figure to add the inset to. ``None`` → ``parent_ax.figure``.
    npix : int
        Nominal pixel grid for the auto-constructed WCS (both axes).
        Higher values = finer control of set_xlim/ylim cropping; doesn't
        affect rendering quality directly.
    inherit_frame : bool
        If True (default) and ``wcs`` is None, inherit the parent's
        coordinate frame type (e.g. ``GLON``/``GLAT`` if parent is
        galactic). If False, defaults to ``RA``/``DEC``.
    direction : str
        Longitude orientation of the auto-built inset. ``'inherit'`` (default)
        matches the parent axes' on-screen east direction, so a geographic
        (east-right) parent gets a geographic inset and an astro (east-left)
        parent an astro inset — they never silently disagree. Pass ``'sky'`` /
        ``'geographic'`` (or aliases ``'astro'`` / ``'geo'`` / ``'earth'``) to
        force it. Ignored when an explicit ``wcs`` is supplied.
    transform : {None, 'figure', 'parent', 'axes'} or matplotlib Transform
        Coordinate system for ``rect``:

        * ``None`` or ``'figure'`` (default) — figure fraction: ``(0,0)``
          is the figure's lower-left, ``(1,1)`` the upper-right. This
          matches :meth:`matplotlib.figure.Figure.add_axes`.
        * ``'parent'`` or ``'axes'`` — parent-axes fraction: ``(0,0)`` is
          the parent axes' lower-left corner, ``(1,1)`` its upper-right.
          This makes it easy to position the inset within or partially
          overlapping the parent. Example:
          ``rect=[0.60, 0.02, 0.38, 0.35], transform='parent'`` places
          the inset in the parent's lower-right quadrant.
        * A :class:`matplotlib.transforms.Transform` instance — rect is
          given in that coord system and converted internally.

        Note: when the inset overlaps the parent, pass ``facecolor='none'``
        (in ``subplot_kw``) to keep the parent visible beneath it, or a
        color with alpha for a tinted overlay.
    bg_color : color or None, optional
        If given, add an opaque background patch behind the inset content,
        shaped to the frame (rectangular or elliptical). Unlike ``facecolor``
        (an axes background, which ``savefig(transparent=True)`` renders
        transparent), this is an ordinary child patch that survives a
        transparent export — so the inset stays an opaque "card" over the
        parent in dark-figure builds. Default ``None`` (no background artist).
    auto_fontsize : bool
        When ``True`` (default), shrink the inset's tick-label
        fontsize to fit its (typically smaller) display width via
        :func:`skyplothelper.autosize.auto_size_ticklabels`. Matches
        the behavior of ``make_wcs_frame`` and ``make_globe_frame``:
        inset axes usually occupy 20-40% of the parent's display
        area, and the default ``rcParams['xtick.labelsize']`` of
        10 pt looks oversized at that scale. Set ``False`` to keep
        the rcParams default.
    tick_style : {'auto', 'in_frame', 'boundary', 'native'}
        Where to draw the inset's tick labels. ``'auto'`` (default) gives a
        *curved globe-like* inset (SIN/ZEA orthographic zoom) the same clean
        in-frame labels as :func:`make_globe_frame` — astropy's native tick
        marks on a circular frame render poorly — while rectilinear (TAN/CAR)
        and elliptical (AIT/MOL) insets keep astropy's native labels, which
        render cleanly. Pass ``'native'`` to force bare astropy ticks,
        ``'in_frame'`` / ``'boundary'`` to force the overlay styles on any
        projection. See :func:`make_wcs_frame` for the full style semantics.
    tick_rotation : {'tangent', 'tangent_upright', 'horizontal'} or float
        Label rotation for the overlay tick styles; ignored when the
        effective style is ``'native'``. Forwarded to ``_apply_tick_style``
        (same meaning as on :func:`make_wcs_frame`).
    **subplot_kw
        Extra kwargs passed to
        :class:`~astropy.visualization.wcsaxes.WCSAxes`. Useful ones:

        * ``frame_class=EllipticalFrame`` — circular inset boundary
          (recommended for SIN, AIT, MOL, and ZEA projections).
        * ``facecolor=...`` — inset background color (applied to the axes;
          for an opaque fill that survives ``savefig(transparent=True)`` or
          follows an elliptical frame, prefer the ``bg_color`` argument).

    Returns
    -------
    inset_ax : astropy.visualization.wcsaxes.WCSAxes
        The created inset axes. Pixel limits are set to bound the
        requested region.

    Examples
    --------
    TAN zoom outside the parent (default placement)::

        inset = reproject_inset_axes(
            main_ax, rect=[0.7, 0.1, 0.28, 0.35],
            projection='TAN', center=(45, 30), size=5)

    Inset placed inside the parent axes (lower-right quadrant)::

        inset = reproject_inset_axes(
            main_ax, rect=[0.60, 0.02, 0.38, 0.35],
            transform='parent',
            projection='TAN', center=(45, 30), size=2,
            facecolor='white')  # opaque background to stand out

    SIN (orthographic) zoom-globe with circular frame::

        from astropy.visualization.wcsaxes.frame import EllipticalFrame
        inset = reproject_inset_axes(
            main_ax, rect=[0.02, 0.6, 0.3, 0.3],
            projection='SIN', center=(120, -20), size=40,
            frame_class=EllipticalFrame)

    Caller-supplied WCS (full control)::

        my_wcs = WCS(my_fits_header)
        inset = reproject_inset_axes(main_ax, rect=(0, 0, 1, 1),
                                     wcs=my_wcs)
    """
    # A SkyCoord center is resolved in the PARENT axes' frame (the frame the
    # inset is cut from), not blindly ICRS.
    if hasattr(center, 'transform_to'):
        from ..geometry._parsing import _coords_to_frame_deg
        from ..wcs_frame import _get_wcs_frame_name
        center = _coords_to_frame_deg(center, _get_wcs_frame_name(parent_ax))
    from astropy.visualization.wcsaxes import WCSAxes

    if fig is None:
        fig = parent_ax.figure

    # Resolve `rect` to figure-fraction coordinates.
    fig_rect = _resolve_rect(rect, transform, parent_ax, fig)

    # Build the inset WCS if not supplied.
    if wcs is None:
        if center is None or size is None:
            raise ValueError(
                "reproject_inset_axes: must provide both `center` and `size` "
                "if `wcs` is not given.")
        if np.isscalar(size):
            size = (float(size), float(size))  # type: ignore[arg-type]  # np.isscalar TypeGuard over-narrows the Any input
        else:
            size = (float(size[0]), float(size[1]))

        parent_wcs = getattr(parent_ax, 'wcs', None)
        if inherit_frame and parent_wcs is not None:
            ctype0 = _reformat_ctype(parent_wcs.wcs.ctype[0], projection)
            ctype1 = _reformat_ctype(parent_wcs.wcs.ctype[1], projection)
        else:
            ctype0 = f'RA  -{projection:<3s}'.replace(' ', '-')
            ctype1 = f'DEC -{projection:<3s}'.replace(' ', '-')

        dpix_x = size[0] / npix
        dpix_y = size[1] / npix
        w = WCS(naxis=2)
        w.wcs.ctype = [ctype0, ctype1]
        w.wcs.crval = [float(center[0]), float(center[1])]
        w.wcs.crpix = [npix / 2 + 0.5, npix / 2 + 0.5]
        # Negative CDELT1 so RA/lon increases to the left (astro convention).
        w.wcs.cdelt = [-dpix_x, dpix_y]
        w.wcs.cunit = ['deg', 'deg']
        # Inherit RADESYS if set on parent.
        if parent_wcs is not None:
            try:
                radesys = parent_wcs.wcs.radesys
                if radesys:
                    w.wcs.radesys = radesys
            except Exception:
                pass

        # Match the inset's longitude orientation to the parent by default, so
        # a geographic parent doesn't get an astro (mirrored) inset and vice
        # versa. ``direction='inherit'`` (default) copies the parent's on-screen
        # east direction; an explicit 'sky'/'geographic' (or alias) forces it.
        # The WCS is built east-left above; flip CDELT1 if the target wants
        # east-right. (Sign↔direction is projection-dependent, so compare the
        # actual mappings rather than the raw signs.)
        if isinstance(direction, str) and direction.strip().lower() == 'inherit':
            target_right = (_east_increases_right(parent_wcs)
                            if parent_wcs is not None else False)
        else:
            from ..projections.project import resolve_direction
            target_right = resolve_direction(direction) == 'geographic'
        if _east_increases_right(w) != target_right:
            w.wcs.cdelt = [-w.wcs.cdelt[0], w.wcs.cdelt[1]]
        wcs = w

    # Circular frame classes (EllipticalFrame on square data limits, or
    # the package's CircularFrame) only render as a *circle* when the
    # axes display bbox is square — which requires ``aspect='equal'``.
    # Auto-apply it when the user opted into a circular frame and didn't
    # override aspect themselves; explicit user choices win.
    try:
        from astropy.visualization.wcsaxes.frame import EllipticalFrame
        _frame_class = subplot_kw.get('frame_class')
        _is_circular_class = False
        if _frame_class is not None:
            if _frame_class is EllipticalFrame or (
                    isinstance(_frame_class, type)
                    and issubclass(_frame_class, EllipticalFrame)):
                _is_circular_class = True
            try:
                from ..projections.frames import CircularFrame
                if _frame_class is CircularFrame or (
                        isinstance(_frame_class, type)
                        and issubclass(_frame_class, CircularFrame)):
                    _is_circular_class = True
            except ImportError:
                pass
        if _is_circular_class and 'aspect' not in subplot_kw:
            subplot_kw['aspect'] = 'equal'
    except ImportError:
        pass

    # ``facecolor`` passed to the WCSAxes constructor is a no-op for the inset
    # frame (the frame PathPatch keeps its default), so apply it explicitly.
    _facecolor = subplot_kw.pop('facecolor', None)

    inset_ax = WCSAxes(fig, fig_rect, wcs=wcs, **subplot_kw)
    fig.add_axes(inset_ax)
    if _facecolor is not None:
        inset_ax.set_facecolor(_facecolor)

    # Bound the inset to the requested region, when we know the region.
    if center is not None and size is not None:
        clon, clat = float(center[0]), float(center[1])
        cx, cy = wcs.wcs_world2pix(np.array([[clon, clat]]), 0)[0]
        # Bound from the *projected* position of the requested angular edges
        # rather than a linear ``size/cdelt``. The linear estimate over-extends
        # on a nonlinear projection: a SIN hemisphere's limb sits at only
        # ``(180/pi)*sin(size/2)`` px — 2/pi (~64%) of the linear size at a full
        # hemisphere — so the frame ends up bigger than the sky disk. Offsetting
        # the center by half the angular size along each cardinal great circle
        # and projecting those edge points frames SIN/AIT/MOL/ZEA (and TAN)
        # correctly, with no caller-side limit fit needed.
        hx = np.radians(size[0] / 2.0)
        hy = np.radians(size[1] / 2.0)
        edges = np.array([
            destination_point(clon, clat, 0.0, hy),      # N
            destination_point(clon, clat, 180.0, hy),    # S
            destination_point(clon, clat, 90.0, hx),     # E
            destination_point(clon, clat, 270.0, hx),    # W
        ], dtype=float)
        epix = wcs.wcs_world2pix(edges, 0)
        sy_pix = max(abs(epix[0, 1] - cy), abs(epix[1, 1] - cy))  # N/S → y
        sx_pix = max(abs(epix[2, 0] - cx), abs(epix[3, 0] - cx))  # E/W → x
        inset_ax.set_xlim(cx - sx_pix, cx + sx_pix)
        inset_ax.set_ylim(cy - sy_pix, cy + sy_pix)

    # Auto-shrink tick labels to the inset's (smaller) display width.
    # Inset axes typically take 20-40% of the parent's display area;
    # leaving rcParams-default 10-pt labels makes them visibly oversized
    # against the inset frame. Same flow as ``make_wcs_frame``: draw
    # once so labels render, then introspect via auto_size_ticklabels.
    # try/except so a layout edge case never breaks the inset build.
    # A draw populates the frame path (needed to shape the background) and lets
    # auto_size_ticklabels introspect rendered label sizes.
    if bg_color is not None or auto_fontsize:
        try:
            fig.canvas.draw()
        except Exception:
            pass

    # Opt-in opaque background artist. Unlike an axes facecolor (which a
    # ``savefig(transparent=True)`` renders transparent — it zeroes ax.patch),
    # this is an ordinary child patch that survives transparent export, so the
    # inset stays an opaque "card" over the parent in dark-figure builds. It
    # follows the frame shape (rectangular or elliptical) and sits behind all
    # inset content.
    if bg_color is not None:
        _add_inset_background(inset_ax, bg_color)

    # Auto-shrink tick labels to the inset's (smaller) display width.
    auto_fs = None
    if auto_fontsize:
        try:
            from ..autosize import auto_size_ticklabels
            auto_fs = auto_size_ticklabels(inset_ax)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"reproject_inset_axes: auto_fontsize failed "
                f"({type(exc).__name__}: {exc}); falling back to "
                f"rcParams default. Pass auto_fontsize=False to suppress.",
                UserWarning, stacklevel=2)

    # Tick styling. A curved zenithal inset (SIN/ZEA orthographic zoom,
    # frame_shape 'circular') inherits astropy's poorly-rendered native ticks
    # on a circular frame; route it through the same in-frame overlay
    # make_globe_frame uses so the default globe inset looks good. Rectilinear
    # (TAN/CAR) and elliptical (AIT/MOL) insets render cleanly with native
    # ticks, so 'auto' leaves them untouched (output unchanged). The projection
    # is read back from the built/supplied WCS so an explicit ``wcs=`` is
    # handled the same as an auto-built one.
    _apply_inset_tick_style(inset_ax, wcs, tick_style, tick_rotation, auto_fs)

    return inset_ax


def _apply_inset_tick_style(inset_ax: Any, wcs: Any, tick_style: str,
                            tick_rotation: Any,
                            label_fontsize: float | None) -> None:
    """Give a curved (globe-like) inset the same clean in-frame tick labels as
    :func:`make_globe_frame`; leave rectilinear / elliptical insets on astropy's
    native ticks. See the caller for the rationale."""
    from ..projections.registry import _resolve_projection

    # Resolve the projection's frame shape from the WCS ctype (works for both
    # an auto-built and a caller-supplied ``wcs``).
    try:
        ctype0 = str(wcs.wcs.ctype[0])
        tokens = [t for t in ctype0.split('-') if t]
        fits_code = tokens[-1] if tokens else 'TAN'
        _, proj_info = _resolve_projection(fits_code)
        frame_shape = proj_info.frame_shape
    except Exception:
        fits_code, frame_shape = 'TAN', 'rectangular'

    # 'auto' only restyles the globe-like zenithal insets; everything else
    # keeps native ticks (byte-identical to the pre-existing behavior). An
    # explicit style is honored on any projection.
    eff_style = tick_style
    if eff_style == 'auto':
        eff_style = 'in_frame' if frame_shape == 'circular' else 'native'
    if eff_style == 'native':
        return

    # Treat the inset as a bounded field so the overlay derives field-scale
    # tick values from the view extent, not the all-sky 30/15 graticule.
    inset_ax._sph_is_allsky = False
    from ..wcs_frame import _apply_tick_style
    try:
        _apply_tick_style(inset_ax, frame_shape, eff_style, tick_rotation,
                          label_fontsize=label_fontsize, fits_code=fits_code)
    except Exception:
        # Tick styling is cosmetic; never fail the inset build over it.
        pass


def _resolve_rect(rect: Any, transform: Any, parent_ax: Any,
                  fig: Any) -> list[float]:
    """
    Convert ``rect`` (left, bottom, width, height) in the given
    coordinate system to figure-fraction coordinates suitable for
    :meth:`matplotlib.figure.Figure.add_axes`.

    Supported ``transform`` values:

    * ``None`` or ``'figure'`` — already figure fraction (no-op).
    * ``'parent'`` / ``'axes'`` — parent-axes fraction (uses
      ``parent_ax.get_position()``).
    * A :class:`matplotlib.transforms.Transform` — interpreted as
      "rect is in whatever coords this transform maps from"; corners
      are mapped to display then inverted through ``fig.transFigure``.
    """
    from matplotlib.transforms import Transform

    left, bottom, width, height = rect

    if transform is None or transform == 'figure':
        return [float(left), float(bottom), float(width), float(height)]

    if isinstance(transform, str) and transform in ('parent', 'axes'):
        pos = parent_ax.get_position()
        return [
            pos.x0 + float(left) * pos.width,
            pos.y0 + float(bottom) * pos.height,
            float(width) * pos.width,
            float(height) * pos.height,
        ]

    if isinstance(transform, Transform):
        # Force a draw so the transform stack is valid.
        try:
            fig.canvas.draw()
        except Exception:
            pass
        corners = np.array([[left, bottom], [left + width, bottom + height]])
        disp = transform.transform(corners)
        fig_frac = fig.transFigure.inverted().transform(disp)
        return [float(fig_frac[0, 0]), float(fig_frac[0, 1]),
                float(fig_frac[1, 0] - fig_frac[0, 0]),
                float(fig_frac[1, 1] - fig_frac[0, 1])]

    raise ValueError(
        f"Unknown transform={transform!r}; use None, 'figure', 'parent', "
        "'axes', or a matplotlib.transforms.Transform instance.")


def mark_inset_axes(parent_ax: Any, inset_ax: Any, style: str = 'rectangle',
                    center: SkyCoord | tuple[float, float] | None = None, size: Any = None,
                    radius: float | None = None,
                    edgecolor: Any = 'red', linewidth: float = 1.5,
                    linestyle: str = '-',
                    facecolor: Any = 'none', alpha: float = 1.0,
                    zorder: int = 5,
                    n_pts: int = 200, unwrap: bool = True,
                    wrap_fix: str = 'auto', **kwargs: Any) -> Any:
    """
    Draw the inset's ROI on the parent axes, following the parent's projection.

    The ROI boundary is sampled densely in the inset's pixel/world space
    and plotted on the parent via ``parent_ax.get_transform('world')``,
    so it appears curvilinear when the parent projection warrants it
    (e.g. an AIT all-sky map shows a rectangular TAN inset's ROI as
    curved edges on the parent).

    If the ROI boundary crosses the parent projection's native wrap
    (antimeridian), the boundary is automatically broken with ``MOVETO``
    codes so the resulting patch renders as two pieces, one on each edge
    of the map, rather than a spurious line across the whole plot.

    Parameters
    ----------
    parent_ax : WCSAxes
        Must have a ``.wcs`` and support ``get_transform('world')``.
    inset_ax : WCSAxes
        Must have a ``.wcs`` attribute.
    style : {'rectangle', 'circle'}
        Shape of the ROI. ``'rectangle'`` uses the inset's current
        ``(xlim, ylim)`` pixel bounds. ``'circle'`` draws a small
        circle in world coordinates.
    center : (lon, lat), SkyCoord, or None
        For ``style='circle'``: center of the circle in degrees. A scalar
        SkyCoord is resolved in the parent axes' frame.
        Defaults to the inset WCS's ``CRVAL``.
    size : (dx, dy) or None
        Unused for now; ROI rectangle is always derived from the
        inset's pixel limits. Accepted for API symmetry / future use.
    radius : float or None
        For ``style='circle'``: angular radius in degrees. If ``None``,
        estimated as half the inset's x-extent.
    edgecolor, linewidth, linestyle, facecolor, alpha, zorder :
        Styling passed to the resulting patch.
    n_pts : int
        Number of boundary points sampled in total. For rectangles, each
        edge gets ``max(25, n_pts // 4)`` samples — higher values give
        smoother curves under strong parent-projection warping.
    unwrap : bool
        For ``style='rectangle'``: if the inset's world-coord longitudes
        jump by ~360° (antimeridian crossing within the inset), unwrap
        them so the boundary samples are continuous in longitude before
        being projected onto the parent.
    wrap_fix : {'auto', 'on', 'off'}
        How to handle boundaries that cross the parent projection's
        native wrap edge. ``'auto'`` (default) detects jumps by looking
        at parent-pixel displacement between adjacent samples; if any
        exceed ~40% of the parent image width, the boundary is split
        with ``MOVETO`` codes (outline renders cleanly, fill may still
        look imperfect). ``'on'`` forces the detection, ``'off'`` skips
        it entirely. Ignored for ``style='circle'`` (small circles
        rarely cross antimeridians).
    **kwargs
        Extra kwargs forwarded to the underlying
        :class:`matplotlib.patches.Polygon` or
        :class:`matplotlib.patches.PathPatch`.

    Returns
    -------
    patch : matplotlib.patches.Patch
        The ROI patch added to the parent axes (a ``Polygon`` in the
        normal case, a ``PathPatch`` when wrap-breaking was needed).

    Examples
    --------
    Rectangle ROI::

        mark_inset_axes(main_ax, inset, style='rectangle',
                        edgecolor='red', linewidth=1.5)

    Circular ROI centered on an object::

        mark_inset_axes(main_ax, inset, style='circle',
                        center=(83.633, 22.014), radius=0.5,
                        edgecolor='gold', linewidth=2)
    """
    # A SkyCoord center is resolved in the PARENT axes' frame (the frame the
    # inset is cut from), not blindly ICRS.
    if hasattr(center, 'transform_to'):
        from ..geometry._parsing import _coords_to_frame_deg
        from ..wcs_frame import _get_wcs_frame_name
        center = _coords_to_frame_deg(center, _get_wcs_frame_name(parent_ax))
    import matplotlib.patches as mpatches
    from matplotlib.path import Path

    inset_wcs = getattr(inset_ax, 'wcs', None)
    if inset_wcs is None:
        raise ValueError("inset_ax must be a WCSAxes (have a .wcs attribute).")

    # patch is a Polygon or PathPatch depending on the wrap-split branch
    # taken below; the union is exposed as Any so the _sph_inset_* marker
    # attributes can be tagged on either concrete type.
    patch: Any
    if style == 'rectangle':
        x0, x1 = inset_ax.get_xlim()
        y0, y1 = inset_ax.get_ylim()
        n_edge = max(25, n_pts // 4)
        # Trace boundary counter-clockwise starting at bottom-left.
        edges_pix = [
            (np.linspace(x0, x1, n_edge), np.full(n_edge, y0)),
            (np.full(n_edge, x1), np.linspace(y0, y1, n_edge)),
            (np.linspace(x1, x0, n_edge), np.full(n_edge, y1)),
            (np.full(n_edge, x0), np.linspace(y1, y0, n_edge)),
        ]
        px = np.concatenate([e[0] for e in edges_pix])
        py = np.concatenate([e[1] for e in edges_pix])
        world = inset_wcs.wcs_pix2world(np.column_stack([px, py]), 0)
        lon, lat = world[:, 0], world[:, 1]
        if unwrap:
            lon = np.degrees(np.unwrap(np.radians(lon)))

        # Detect antimeridian crossings on the parent and split with MOVETO.
        parent_wcs = getattr(parent_ax, 'wcs', None)
        split_indices = []
        if wrap_fix != 'off' and parent_wcs is not None:
            ppix = parent_wcs.wcs_world2pix(np.column_stack([lon, lat]), 0)
            # Width of parent image in pixels; fall back to 2*CRPIX1 if unset.
            if parent_wcs.array_shape is not None:
                naxis1 = float(parent_wcs.array_shape[1])
            else:
                naxis1 = 2.0 * abs(float(parent_wcs.wcs.crpix[0]))
            wrap_thresh = 0.4 * naxis1
            dx = np.abs(np.diff(ppix[:, 0]))
            split_indices = list(np.where(dx > wrap_thresh)[0])

        if split_indices:
            # Build a Path: MOVETO at start and after each detected jump,
            # LINETO elsewhere. No explicit CLOSEPOLY because splits would
            # close the wrong sub-loops; we close manually per segment.
            n = len(lon)
            codes = np.full(n, Path.LINETO, dtype=np.uint8)
            codes[0] = Path.MOVETO
            for idx in split_indices:
                # Break after index idx (between sample idx and idx+1).
                if idx + 1 < n:
                    codes[idx + 1] = Path.MOVETO
            path = Path(np.column_stack([lon, lat]), codes)
            patch = mpatches.PathPatch(
                path, facecolor=facecolor, edgecolor=edgecolor,
                linewidth=linewidth, linestyle=linestyle,
                alpha=alpha, zorder=zorder,
                transform=parent_ax.get_transform('world'),
                **kwargs,
            )
        else:
            verts = np.column_stack([lon, lat])
            patch = mpatches.Polygon(
                verts, closed=True,
                facecolor=facecolor, edgecolor=edgecolor,
                linewidth=linewidth, linestyle=linestyle,
                alpha=alpha, zorder=zorder,
                transform=parent_ax.get_transform('world'),
                **kwargs,
            )
    elif style == 'circle':
        if center is None:
            center = (float(inset_wcs.wcs.crval[0]),
                      float(inset_wcs.wcs.crval[1]))
        if radius is None:
            x0, x1 = inset_ax.get_xlim()
            sz_pix = abs(x1 - x0)
            radius = 0.5 * sz_pix * abs(inset_wcs.wcs.cdelt[0])
        lon, lat = small_circle(center[0], center[1], radius, n_pts=n_pts)
        verts = np.column_stack([lon, lat])
        patch = mpatches.Polygon(
            verts, closed=True,
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth, linestyle=linestyle,
            alpha=alpha, zorder=zorder,
            transform=parent_ax.get_transform('world'),
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown style {style!r}; use 'rectangle' or 'circle'.")

    parent_ax.add_patch(patch)
    # Tag the patch with a reference to the inset and the style/center/
    # radius so ``connect_inset_axes`` can identify which patch belongs
    # to which inset (matplotlib's ``get_transform()`` returns a fresh
    # composite each call, so an ``is``-identity check is unreliable).
    patch._sph_inset_marker = inset_ax
    patch._sph_inset_marker_style = style
    if style == 'circle':
        patch._sph_inset_marker_center = center
        patch._sph_inset_marker_radius = radius
    return patch


def connect_inset_axes(parent_ax: Any, inset_ax: Any, corners: Any = 'diagonal',
                       color: Any = '0.3', linewidth: float = 0.8,
                       linestyle: str = ':',
                       alpha: float = 0.8, zorder: int = 4,
                       style: str = 'auto',
                       curvature: float = 0.0) -> list[Any]:
    """
    Draw connector lines from the parent-ROI corners to the inset corners.

    Uses :class:`matplotlib.patches.ConnectionPatch`, so the line endpoints
    update automatically when each axes is resized (positions are evaluated
    at draw time via ``transData``). The specific *corners* to connect are
    chosen at call time, however, so if you later drastically reposition
    the inset across the parent, calling this again may give a better
    corner choice.

    Corner names (``'ll'``, ``'lr'``, ``'ul'``, ``'ur'``) refer to the
    corners' **display positions** after all WCS transforms and matplotlib
    orientation are applied — not their pixel indices. This makes the
    auto-selection behave correctly even when the inset or parent uses
    non-standard WCS orientations (e.g. ``CDELT2 < 0`` on a SIN globe).

    Parameters
    ----------
    parent_ax : WCSAxes
        Must have a ``.wcs`` attribute.
    inset_ax : WCSAxes
        Must have a ``.wcs`` attribute.
    corners : 'diagonal' (default), 'crossing', 'matching', or list of (parent_corner, inset_corner) tuples
        Which corners to link. Each corner is one of
        ``'ll'`` (lower-left), ``'lr'``, ``'ul'``, ``'ur'`` — meaning
        the corner that appears in that position *on screen*.

        ``'diagonal'`` (the default) picks a *diagonal corner pair*
        within each axis — perpendicular to the parent-to-inset
        direction — and connects matching corner names. The two
        connectors hug the outsides of both axes and don't traverse
        through either one's interior:

        * Inset to the lower-right or upper-left → ``[('ur', 'ur'), ('ll', 'll')]``
        * Inset to the upper-right or lower-left → ``[('ul', 'ul'), ('lr', 'lr')]``

        When the inset is roughly directly to the side / above / below
        (no diagonal component), ``'diagonal'`` falls back to
        ``'crossing'`` since the diagonal rule degenerates.

        ``'crossing'`` follows the standard mpl-toolkits inset-zoom
        pattern: connectors run from the parent ROI's edge that faces
        the inset to the *adjacent* (near-side) edge of the inset.
        Two parallel connectors form a clean trapezoid:

        * Inset to the right  → ``[('ur', 'ul'), ('lr', 'll')]``
        * Inset to the left   → ``[('ul', 'ur'), ('ll', 'lr')]``
        * Inset above         → ``[('ul', 'll'), ('ur', 'lr')]``
        * Inset below         → ``[('ll', 'ul'), ('lr', 'ur')]``

        ``'matching'`` connects the same-named corner of the parent
        ROI to the inset's same-named corner. The connectors run to
        the inset's *far* edge, so they have to pass behind the
        inset:

        * Inset to the right  → ``[('ur', 'ur'), ('lr', 'lr')]``
        * Inset to the left   → ``[('ul', 'ul'), ('ll', 'll')]``
        * Inset above         → ``[('ul', 'ul'), ('ur', 'ur')]``
        * Inset below         → ``[('ll', 'll'), ('lr', 'lr')]``

        Pass an explicit list of ``(parent_corner, inset_corner)``
        pairs for full control (e.g. ``[('ur', 'll'), ('ll', 'ur')]``
        for true diagonals that cross over the inset).
    color, linewidth, linestyle, alpha, zorder :
        Line styling.
    curvature : float
        Curvature factor for the connectors. ``0.0`` (default)
        gives straight pixel-space lines. (A sky-aware great-circle
        mode is deliberately not offered: the parent ROI corner and
        the matching inset corner are the *same sky point*, so a
        great-circle arc between them degenerates to the straight
        bridge — ``curvature`` is the tunable alternative.)

        Positive values render each connector as a quadratic
        Bezier via matplotlib's
        ``ConnectionPatch(connectionstyle='arc3,rad=...')``, with
        the curve bowing **outward** — away from the line of
        centers between the parent ROI and the inset, on the same
        side as the connector's midpoint. The pair of connectors
        flares apart symmetrically, leaving the parent ROI
        interior visually unobscured. Typical useful values:
        ``0.2`` for a soft curve, ``0.4`` for a pronounced one;
        ``> 0.5`` looks exaggerated for most layouts.

        Negative values invert the convention so the curves bow
        **inward** (toward and possibly across the line of
        centers). Useful for tightly-grouped insets where an
        inward sweep reads more naturally than outward flare.
    style : {'auto', 'rectangular', 'circular'}
        Geometric layout of the connectors.

        * ``'auto'`` (default) inspects the inset frame and routes to
          ``'circular'`` when ``inset_ax`` has a CircularFrame (or an
          EllipticalFrame on a square axes — i.e. a ``make_globe_frame``
          panel); otherwise ``'rectangular'``.
        * ``'rectangular'`` uses the four-corner ``corners=`` logic
          described above.
        * ``'circular'`` treats both the parent ROI and the inset
          as circles inscribed in their respective bounding boxes
          (parent ROI = 4 projected pixel corners of the inset's
          xlim/ylim mapped through the parent WCS; inset = its
          window-extent bbox in figure pixels) and draws the two
          outer common tangent lines between them. Independent of
          the ``corners=`` argument. Pair with
          ``mark_inset_axes(style='circle', ...)`` for the cleanest
          look — the marker circle and the tangent endpoints sit on
          the same parent-ROI circle.

    Returns
    -------
    patches : list of matplotlib.patches.ConnectionPatch
        The connector patches added to the figure (2 elements).

    Notes
    -----
    If you call this before the figure has been laid out (e.g. before
    ``tight_layout`` or the first draw), the auto-corner choice uses
    current display bounds, which may be wrong. Call after layout, or
    re-call after layout changes.

    Examples
    --------
    Default usage::

        connect_inset_axes(main_ax, inset)

    Manual corner pairs (diagonal)::

        connect_inset_axes(main_ax, inset,
                           corners=[('ur', 'll'), ('ll', 'ur')])
    """

    inset_wcs = getattr(inset_ax, 'wcs', None)
    parent_wcs = getattr(parent_ax, 'wcs', None)
    if inset_wcs is None or parent_wcs is None:
        raise ValueError(
            "Both parent_ax and inset_ax must be WCSAxes (have .wcs).")

    # Force a draw so transforms / window extents are up to date.
    fig = parent_ax.figure
    try:
        fig.canvas.draw()
    except Exception:
        pass

    # 4 inset pixel corners, mapped through inset WCS → world → parent pixel.
    ix0, ix1 = inset_ax.get_xlim()
    iy0, iy1 = inset_ax.get_ylim()
    inset_pix = np.array([(ix0, iy0), (ix1, iy0),
                          (ix0, iy1), (ix1, iy1)])
    world = inset_wcs.wcs_pix2world(inset_pix, 0)
    parent_pix = parent_wcs.wcs_world2pix(world, 0)

    # Identify which of the 4 points is LL / LR / UL / UR in DISPLAY space.
    # Using display positions makes auto-selection robust to any WCS or
    # transform orientation flip (CDELT sign, origin convention, etc.).
    inset_disp = inset_ax.transData.transform(inset_pix)
    parent_disp_arr = parent_ax.transData.transform(parent_pix)

    # Resolve `style`: 'auto' inspects the inset frame.
    if style not in ('auto', 'rectangular', 'circular'):
        raise ValueError(
            f"connect_inset_axes: style={style!r} not understood — "
            f"expected one of 'auto', 'rectangular', 'circular'."
        )
    if style == 'auto':
        style = 'circular' if _is_circular_frame(inset_ax) else 'rectangular'

    if style == 'circular':
        # Inset axes: the limb is the EllipticalFrame / CircularFrame
        # outline. With ``aspect='equal'`` and square xlim/ylim the
        # frame is inscribed in the (square) axes window bbox, so
        # ``get_window_extent()`` gives the exact display-coord
        # bounding box of the visible limb.
        inset_bbox = inset_ax.get_window_extent()
        inset_cx = 0.5 * (inset_bbox.x0 + inset_bbox.x1)
        inset_cy = 0.5 * (inset_bbox.y0 + inset_bbox.y1)
        inset_r = 0.5 * min(inset_bbox.width, inset_bbox.height)

        # Parent ROI: look for the user-drawn circular marker on
        # ``parent_ax``. ``mark_inset_axes(style='circle')`` tags its
        # returned patch with ``_sph_inset_marker = inset_ax``, plus
        # the original center/radius. Using the stored sky-coords to
        # regenerate a dense small-circle sample gives a clean,
        # high-resolution point set in display coords; snapping the
        # geometric tangent endpoints to the nearest sample then
        # lands them exactly on the visible marker even when the
        # parent projection distorts the marker into an oval.
        parent_marker = None
        for patch in reversed(parent_ax.patches):
            if getattr(patch, '_sph_inset_marker', None) is inset_ax:
                parent_marker = patch
                break

        parent_disp_pts = None
        if parent_marker is not None and getattr(
                parent_marker, '_sph_inset_marker_style', None) == 'circle':
            try:
                m_center = parent_marker._sph_inset_marker_center
                m_radius = parent_marker._sph_inset_marker_radius
                # Resample densely so closest-point snapping yields a
                # sub-pixel-accurate landing on the marker.
                from .spherical import small_circle
                lon_m, lat_m = small_circle(
                    m_center[0], m_center[1], m_radius, n_pts=720)
                marker_world = np.column_stack([lon_m, lat_m])
                marker_parent_pix = parent_wcs.wcs_world2pix(
                    marker_world, 0)
                parent_disp_pts = parent_ax.transData.transform(
                    marker_parent_pix)
            except Exception:
                parent_disp_pts = None

        if parent_disp_pts is None:
            # Fallback: sample the inset's inscribed pixel-limb and
            # project through the WCS chain. Less accurate (uses the
            # inset's visible limb, which may differ in sky-radius from
            # whatever marker the user drew), but workable when no
            # mark_inset_axes circle exists.
            ic_cx = 0.5 * (ix0 + ix1)
            ic_cy = 0.5 * (iy0 + iy1)
            ic_r = 0.5 * min(abs(ix1 - ix0), abs(iy1 - iy0))
            thetas = np.linspace(0, 2 * np.pi, 720, endpoint=False)
            limb_pix = np.column_stack([
                ic_cx + ic_r * np.cos(thetas),
                ic_cy + ic_r * np.sin(thetas),
            ])
            limb_world = inset_wcs.wcs_pix2world(limb_pix, 0)
            limb_parent_pix = parent_wcs.wcs_world2pix(limb_world, 0)
            parent_disp_pts = parent_ax.transData.transform(limb_parent_pix)

        valid = np.all(np.isfinite(parent_disp_pts), axis=1)
        if not np.any(valid):
            # Last-resort fallback: 4 corner bbox.
            px = parent_disp_arr[:, 0]
            py = parent_disp_arr[:, 1]
            parent_cx = 0.5 * (px.min() + px.max())
            parent_cy = 0.5 * (py.min() + py.max())
            parent_r = 0.5 * min(px.max() - px.min(),
                                  py.max() - py.min())
        else:
            # Best-fit circle: centroid + average distance. Matches a
            # projected (slightly oval) marker more closely than
            # bounding-box-inscribed; for a true circle in display the
            # two are identical.
            lx = parent_disp_pts[valid, 0]
            ly = parent_disp_pts[valid, 1]
            parent_cx = float(np.mean(lx))
            parent_cy = float(np.mean(ly))
            parent_r = float(np.mean(np.hypot(lx - parent_cx,
                                                ly - parent_cy)))

        tangents = _outer_tangent_endpoints(
            (parent_cx, parent_cy), parent_r,
            (inset_cx, inset_cy), inset_r,
        )
        if tangents is None:
            # Concentric, or one circle contains the other — outer
            # tangents don't exist. Fall back to a single line between
            # centers so the figure still draws something visible.
            tangents = np.array([
                [(parent_cx, parent_cy), (inset_cx, inset_cy)]
            ])

        if inset_ax.get_zorder() <= zorder:
            inset_ax.set_zorder(zorder + 1)

        # The parent-side tangent endpoint from circle-circle math lies
        # on the fitted circle, which is a best-fit to the (possibly
        # oval) projected marker. For non-conformal parent projections
        # the marker isn't a perfect circle in display, so the
        # geometric tangent endpoint can drift 1-3 px off the actual
        # marker boundary. Snap each parent endpoint to the closest
        # marker vertex so it lands exactly on the visible curve.
        # Inset endpoint stays at the mathematically-tangent point on
        # the inset limb (which is a true circle in display).
        if parent_marker is not None and parent_disp_pts is not None:
            valid_marker = np.all(np.isfinite(parent_disp_pts), axis=1)
            snap_source = parent_disp_pts[valid_marker]
        else:
            snap_source = None

        # Convert each tangent endpoint from figure-pixel coords (where
        # the geometry was computed) to the respective axes' data coords.
        # Anchoring via ``coordsA='data', axesA=...`` makes the
        # ConnectionPatch dpi-independent (matplotlib re-evaluates
        # data→display at draw time, picking up the current figure dpi)
        # and tracks ``tight_layout`` / resize the same way the
        # rectangular path's corner anchors do — each tangent endpoint
        # stays at its fixed (lon, lat) on the parent ROI and at its
        # fixed (inset_pixel_x, inset_pixel_y) on the inset limb across
        # layout changes.
        parent_inv = parent_ax.transData.inverted()
        inset_inv = inset_ax.transData.inverted()
        patches = []
        for tangent in tangents:
            parent_disp_pt = tangent[0]
            inset_disp_pt = tangent[1]
            if snap_source is not None and len(snap_source):
                dists = np.hypot(
                    snap_source[:, 0] - parent_disp_pt[0],
                    snap_source[:, 1] - parent_disp_pt[1])
                parent_disp_pt = snap_source[int(np.argmin(dists))]
            parent_data_pt = parent_inv.transform(parent_disp_pt)
            inset_data_pt = inset_inv.transform(inset_disp_pt)
            # For two outer tangents the parent endpoints lie on
            # opposite sides of the line of centers in display
            # space. Project each onto the perpendicular to the
            # line-of-centers direction and use the sign to bow the
            # Bezier outward on each side (positive curvature →
            # connectors arc away from the line of centers, keeping
            # the parent ROI interior visually unobscured). Negative
            # curvature inverts the convention if a user wants
            # inward-bowing connectors. ``curvature=0`` keeps the
            # historical straight-line look.
            if curvature:
                centers_vec = np.array([
                    inset_cx - parent_cx, inset_cy - parent_cy])
                perp = np.array([-centers_vec[1], centers_vec[0]])
                offset = parent_disp_pt - np.array(
                    [parent_cx, parent_cy])
                # arc3's positive rad bows the curve to the RIGHT
                # of the chord A→B (matplotlib convention). The
                # parent endpoint is on the +perp side when
                # ``dot(offset, perp) > 0``; for that side the
                # outward-bowing direction is +rad (right of A→B).
                # The opposite tangent gets the opposite sign.
                side = -float(np.sign(np.dot(offset, perp)) or 1.0)
                conn_style = (
                    f"arc3,rad={float(curvature) * side:.4f}")
            else:
                conn_style = "arc3,rad=0"
            cp = ConnectionPatch(
                xyA=(float(parent_data_pt[0]), float(parent_data_pt[1])),
                coordsA='data', axesA=parent_ax,
                xyB=(float(inset_data_pt[0]), float(inset_data_pt[1])),
                coordsB='data', axesB=inset_ax,
                color=color, linewidth=linewidth, linestyle=linestyle,
                alpha=alpha, zorder=zorder, arrowstyle='-',
                connectionstyle=conn_style,
            )
            cp.set_clip_on(False)
            fig.add_artist(cp)
            patches.append(cp)
        return patches

    def _name_by_display(disp4: np.ndarray) -> dict[str, int]:
        # Lower 2 by y (smaller y in display = "lower" visually); upper 2 top.
        order_y = np.argsort(disp4[:, 1])
        lower, upper = order_y[:2], order_y[2:]
        ll = lower[np.argmin(disp4[lower, 0])]
        lr = lower[np.argmax(disp4[lower, 0])]
        ul = upper[np.argmin(disp4[upper, 0])]
        ur = upper[np.argmax(disp4[upper, 0])]
        return {'ll': int(ll), 'lr': int(lr), 'ul': int(ul), 'ur': int(ur)}

    inset_name2idx = _name_by_display(inset_disp)
    parent_name2idx = _name_by_display(parent_disp_arr)

    # Auto-choose which corners to connect based on inset-vs-ROI centroid
    # *in display coords*. ``'diagonal'`` (default) picks an axis-internal
    # *diagonal* corner pair perpendicular to the parent-to-inset
    # direction, then connects matching corner names — clean connectors
    # that hug the outsides of both axes. Degenerates for straight-side
    # placements so falls back to ``'crossing'`` there. ``'crossing'`` is
    # the mpl-toolkits trapezoidal pattern (near-edge to near-edge).
    # ``'matching'`` is the same-named-corners legacy pattern that
    # routes connectors past the inset's far edge.
    if corners in ('diagonal', 'matching', 'crossing', 'auto'):
        roi_cx = np.mean(parent_disp_arr[:, 0])
        roi_cy = np.mean(parent_disp_arr[:, 1])
        inset_bbox = inset_ax.get_window_extent()
        inset_cx = (inset_bbox.x0 + inset_bbox.x1) / 2
        inset_cy = (inset_bbox.y0 + inset_bbox.y1) / 2
        dx = inset_cx - roi_cx
        dy = inset_cy - roi_cy
        # Distinguish "diagonal" from "straight-to-side" placements by
        # comparing the smaller component to the larger. Threshold 0.3
        # = the shorter side must be at least 30% of the longer to
        # count as diagonal. Below that, fall back to crossing.
        diag_aspect = (min(abs(dx), abs(dy))
                       / max(abs(dx), abs(dy), 1e-9))
        is_diagonal = diag_aspect > 0.3
        mode = corners
        if mode == 'auto':
            mode = 'crossing'  # deprecated alias
        if mode == 'diagonal' and not is_diagonal:
            mode = 'crossing'
        if mode == 'diagonal':
            # Use the axis-internal diagonal perpendicular to the
            # parent→inset direction:
            #   dx*dy > 0 (upper-right or lower-left placement)
            #     → use the UL/LR diagonal (corners along "\")
            #   dx*dy < 0 (upper-left or lower-right placement)
            #     → use the UR/LL diagonal (corners along "/")
            if dx * dy > 0:
                pairs = [('ul', 'ul'), ('lr', 'lr')]
            else:
                pairs = [('ur', 'ur'), ('ll', 'll')]
        elif mode == 'crossing':
            if abs(dx) >= abs(dy):
                pairs = ([('ur', 'ul'), ('lr', 'll')] if dx > 0
                         else [('ul', 'ur'), ('ll', 'lr')])
            else:
                pairs = ([('ul', 'll'), ('ur', 'lr')] if dy > 0
                         else [('ll', 'ul'), ('lr', 'ur')])
        else:  # 'matching'
            if abs(dx) >= abs(dy):
                pairs = ([('ur', 'ur'), ('lr', 'lr')] if dx > 0
                         else [('ul', 'ul'), ('ll', 'll')])
            else:
                pairs = ([('ul', 'ul'), ('ur', 'ur')] if dy > 0
                         else [('ll', 'll'), ('lr', 'lr')])
    else:
        pairs = corners

    # Ensure the inset axes draws on top of the figure-level connector
    # patches. matplotlib draws figure children sorted by zorder; both
    # axes default to zorder=0, so a connector at zorder=4 would
    # otherwise sit on top of the inset. Bumping inset_ax above the
    # connector zorder makes the connector appear to slip behind the
    # inset (still above the parent axes content).
    if inset_ax.get_zorder() <= zorder:
        inset_ax.set_zorder(zorder + 1)

    # For the rectangular path, bow each connector outward by
    # projecting its display midpoint onto the perpendicular of the
    # ROI→inset centroid direction; the sign of that projection
    # selects which side the Bezier bulges. ``curvature=0`` keeps
    # the historical straight-line look.
    if curvature:
        roi_cx = float(np.mean(parent_disp_arr[:, 0]))
        roi_cy = float(np.mean(parent_disp_arr[:, 1]))
        inset_bbox = inset_ax.get_window_extent()
        inset_dcx = 0.5 * (inset_bbox.x0 + inset_bbox.x1)
        inset_dcy = 0.5 * (inset_bbox.y0 + inset_bbox.y1)
        centers_vec = np.array(
            [inset_dcx - roi_cx, inset_dcy - roi_cy])
        rect_perp = np.array([-centers_vec[1], centers_vec[0]])
    else:
        rect_perp = None

    patches = []
    for p_name, i_name in pairs:
        p_idx = parent_name2idx[p_name]
        i_idx = inset_name2idx[i_name]
        xyA = tuple(parent_pix[p_idx])
        xyB = tuple(inset_pix[i_idx])
        if curvature and rect_perp is not None:
            mid_disp = 0.5 * (parent_disp_arr[p_idx]
                               + inset_disp[i_idx])
            offset = mid_disp - np.array(
                [0.5 * (roi_cx + inset_dcx),
                 0.5 * (roi_cy + inset_dcy)])
            # Positive curvature → connector bows outward (away from
            # the line of centers, on the same side as the connector's
            # midpoint). matplotlib's arc3 positive rad bows to the
            # right of A→B, so we negate the +perp-side sign.
            side = -float(np.sign(np.dot(offset, rect_perp)) or 1.0)
            conn_style = (
                f"arc3,rad={float(curvature) * side:.4f}")
        else:
            conn_style = "arc3,rad=0"
        cp = ConnectionPatch(
            xyA=xyA, coordsA='data', axesA=parent_ax,
            xyB=xyB, coordsB='data', axesB=inset_ax,
            color=color, linewidth=linewidth, linestyle=linestyle,
            alpha=alpha, zorder=zorder, arrowstyle='-',
            connectionstyle=conn_style,
        )
        cp.set_clip_on(False)
        fig.add_artist(cp)
        patches.append(cp)
    return patches
