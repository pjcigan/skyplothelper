"""Catalog and motion plotting helpers.

``plot_sky_vectors`` and ``plot_displacement`` draw vector arrows on
WCS axes; ``plot_catalog`` plots a tabular catalog with auto-styling.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    LogNorm,
    Normalize,
    PowerNorm,
)
from matplotlib.ticker import (
    FormatStrFormatter,
    Formatter,
    NullFormatter,
    StrMethodFormatter,
)

from .geometry._parsing import _coords_to_frame_deg
from .wcs_frame import _get_wcs_frame_name


def _wcs_deg_per_pixel(wcs: Any) -> float:
    """Sky degrees per WCS pixel, as a single scalar.

    Uses :func:`~astropy.wcs.utils.proj_plane_pixel_scales` rather than
    ``wcs.wcs.cdelt``: a CD-matrix header carries ``cdelt = [1, 1]`` with the
    real scale living in the CD matrix, so reading cdelt directly would silently
    report a scale of 1 deg/pixel. Falls back to cdelt (then 1.0) if the WCS is
    too odd to describe a celestial plane.

    A single scalar is deliberate — see :func:`plot_sky_vectors`.
    """
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = np.abs(np.asarray(
            proj_plane_pixel_scales(wcs.celestial), dtype=float)[:2])
        s = float(np.mean(scales))
        if np.isfinite(s) and s > 0:
            return s
    except Exception:
        pass
    try:
        cd = np.abs(np.asarray(wcs.wcs.cdelt, dtype=float)[:2])
        s = float(np.mean(cd))
        if np.isfinite(s) and s > 0:
            return s
    except Exception:
        pass
    return 1.0


# Vector-magnitude unit → degrees. Shared by ``plot_sky_vectors`` and
# ``sky_quiverkey`` so the arrow field and its key convert identically.
_VECTOR_UNIT_FACTORS = {'deg': 1.0, 'arcmin': 1 / 60., 'arcsec': 1 / 3600.,
                        'mas': 1 / 3.6e6, 'uas': 1 / 3.6e9}


class SkyVectorResult(NamedTuple):
    """Return type for :func:`plot_sky_vectors`.

    Attribute-accessible: ``result.quiver``, ``result.colorbar``, plus
    ``result.scale`` and ``result.deg_per_pix`` — the resolved numeric
    scale (useful for reading back what ``scale='auto'`` chose) and the
    frame pixel scale actually used. Both feed :func:`sky_quiverkey`.

    .. note::
       This is a 4-field tuple. Bind it whole (``result = ...``) and use
       attributes; a two-name unpack (``q, cbar = ...``) will not work.
    """
    quiver:      object  # matplotlib.quiver.Quiver
    colorbar:    object  # matplotlib.colorbar.Colorbar | None
    scale:       float = 1.0
    deg_per_pix: float = 1.0


class CatalogPlot(NamedTuple):
    """Colorbar-carrying return type for :func:`plot_catalog`.

    Returned only when ``plot_catalog`` creates a colorbar (``cbar=True``
    with ``colorby`` set); a plain ``PathCollection`` is returned otherwise.
    Tuple-unpackable (``sc, cb = plot_catalog(...)``) and attribute-accessible
    (``result.scatter`` / ``result.colorbar``).
    """
    scatter:  object  # matplotlib.collections.PathCollection
    colorbar: object  # matplotlib.colorbar.Colorbar | None


class _CatalogSizeInfo(NamedTuple):
    """The ``sizeby`` scaling ``plot_catalog`` applied, for a size legend.

    Stashed on the returned scatter as ``_sph_size_info`` (and readable via
    :meth:`SizeBlock.from_catalog`) so a graduated-size legend reproduces the
    exact on-plot sizes without the caller restating the scaling. ``size_map``
    is ``(raw, scaled, rmin, rmax)`` — the same contract the size legend uses.
    """
    size_map:   tuple[Any, Any, float, float]
    smin:       float
    smax:       float
    size_scale: Any


def plot_sky_vectors(ax: Any, lon: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None,
                     dlon: npt.ArrayLike | None = None,
                     dlat: npt.ArrayLike | None = None,
                     scale: float | str = 1.0,
                     color: str | npt.ArrayLike = 'C0', alpha: float = 0.8,
                     width: float = 0.003,
                     headwidth: float = 3, headlength: float = 4,
                     headaxislength: float = 3,
                     units: str = 'arcsec', cos_dec: bool = True,
                     label: str | None = None, zorder: int = 5,
                     auto_target_deg: float = 2.0,
                     pivot: str = 'middle',
                     color_by_magnitude: bool = False,
                     cmap: Any = None,
                     add_colorbar: bool = False, cbar_label: str | None = None,
                     cbar_orientation: str = 'vertical',
                     cbar_kwargs: dict[str, Any] | None = None,
                     **kwargs: Any) -> SkyVectorResult:
    """
    Plot 2D vectors at sky positions (arrows) on a WCSAxes.

    Generic vector-field-on-sphere renderer — works for proper-motion
    arrows, catalog position differences (e.g. ICRS3 vs Gaia),
    VSH-fit residuals, simulated velocity fields, or any other
    "this point + this vector" data. Correctly handles the RA cos(Dec)
    factor and the WCS transform; quiver-based.

    Parameters
    ----------
    ax : WCSAxes
    lon, lat : array-like, or SkyCoord in ``lon``
        Source positions in degrees (in the WCS native frame). A ``SkyCoord``
        array may be passed as ``lon`` instead, replacing both — the deltas
        are then given as keywords: ``plot_sky_vectors(ax, coords, dlon=...,
        dlat=...)``.
    dlon, dlat : array-like
        Vector components: the longitudinal and latitudinal parts of
        the per-source vector. Units are set by ``units``. Examples:

        * Proper motion: ``dlon = μ_α cosδ`` (mas/yr), ``dlat = μ_δ``
          (mas/yr); set ``cos_dec=True`` (default; the typical
          convention for tabulated PM).
        * Catalog position differences: ``dlon = ra2 - ra1`` (in
          ``units``), ``dlat = dec2 - dec1``; ``cos_dec=False`` if
          the differences are *not* yet cosδ-scaled.
        * VSH-fit residuals: predicted minus observed PM components
          in mas / μas.
    scale : float or str
        Arrow scale factor. Multiplies each vector's magnitude (in
        ``units``); the product is the arrow's on-sky length, converted
        to degrees for plotting — so it is degrees-per-unit-magnitude
        only when ``units='deg'``. That on-sky length is independent of
        the frame's projection, its pixel scale, and ``npix``. Increase
        to make arrows longer. If ``'auto'``, sizes the median arrow to
        ``auto_target_deg`` degrees.
    color : str or array-like
        Arrow color. Single color string OR per-source array (see
        ``color_by_magnitude=`` for the common "color by vector
        magnitude" recipe).
    alpha : float
    width : float
        Arrow shaft width in axes fraction. Default 0.003.
    headwidth, headlength, headaxislength : float
        Arrow head proportions (passed to quiver).
    units : str
        Unit of ``dlon``/``dlat``: 'deg', 'arcmin', 'arcsec', 'mas',
        'uas'.
    cos_dec : bool
        If True (default), ``dlon`` is assumed to already include
        the cos(dec) factor (the typical convention for tabulated
        proper motion: μ_α cosδ). If False, the cos(dec) correction
        is applied internally — useful for raw catalog-position
        differences ``ra2 - ra1`` that have not been cosδ-scaled.
    label : str, optional
    zorder : int
    auto_target_deg : float
        Target angular size (degrees) for the median arrow when
        ``scale='auto'``.
    pivot : {'middle', 'tail', 'tip'}
        Quiver pivot point. Default ``'middle'`` — arrow is
        centered on the data point. ``'tail'`` (matplotlib's
        default) anchors the arrow's start at the data point;
        ``'tip'`` anchors the head. Centered arrows look cleaner
        on all-sky views where tip-anchored arrows would extend
        off the visible field. (Pattern adopted from the VSH
        literature's plot_RDEM convention.)
    color_by_magnitude : bool
        If True, color arrows by their on-sky magnitude
        ``hypot(dlon, dlat)``. The ``color`` kwarg is then ignored;
        pass ``cmap=`` to pick the colormap (default ``'viridis'``).
        Pair with ``add_colorbar=True`` for the canonical
        "magnitude-coded arrows + colorbar" recipe.
    cmap : str or Colormap, optional
        Colormap, used when ``color_by_magnitude=True`` or when
        ``color`` is a numerical array. Default ``'viridis'`` for
        magnitude coloring; ignored otherwise.
    add_colorbar : bool
        If True and the arrows have variable color (either via
        ``color_by_magnitude`` or a numerical ``color`` array),
        attach a colorbar to the axes. Returned as
        ``result.colorbar``.
    cbar_label : str, optional
        Colorbar label (passed to ``cbar.set_label``). Defaults to
        ``f'magnitude ({units})'`` when ``color_by_magnitude=True``.
    cbar_orientation : {'vertical', 'horizontal'}
        Colorbar orientation.
    cbar_kwargs : dict, optional
        Extra kwargs forwarded to ``fig.colorbar``.
    **kwargs
        Additional kwargs passed to ``ax.quiver``.

    Returns
    -------
    SkyVectorResult
        NamedTuple ``(quiver, colorbar, scale, deg_per_pix)``. Bind it
        whole and use attributes — ``result.colorbar`` is ``None`` when
        no colorbar is attached; ``result.scale`` is the resolved numeric
        scale (so you can read back what ``scale='auto'`` chose), and
        ``result.deg_per_pix`` the frame pixel scale used. Pass the whole
        result to :func:`sky_quiverkey` to draw a matching reference key.

    Notes
    -----
    The returned quiver's internal ``U`` / ``V`` are in the frame's *pixel*
    units — the on-sky degree length divided by the local pixel scale — not
    world degrees, because ``ax.quiver(angles='xy', scale_units='xy')``
    measures length in data (pixel) units. So a matplotlib
    :meth:`~matplotlib.axes.Axes.quiverkey` value must be converted the same
    way (``U = value * ufac * scale / deg_per_pix``); use :func:`sky_quiverkey`
    to have that handled from the result rather than doing it by hand.

    Examples
    --------
    >>> # Proper motion in mas/yr with explicit scale
    >>> sph.plot_sky_vectors(ax, ra, dec, pmra, pmdec,
    ...     scale=0.1, units='mas', color='red')

    >>> # Magnitude-coded arrows + auto colorbar (recommended recipe)
    >>> result = sph.plot_sky_vectors(ax, ra, dec, pmra, pmdec,
    ...     scale='auto', units='mas',
    ...     color_by_magnitude=True, cmap='viridis',
    ...     add_colorbar=True, cbar_label='PM (mas/yr)')
    >>> result.colorbar.set_label('μ (mas/yr)')

    >>> # Catalog position differences (not cosδ-scaled)
    >>> dra  = ra_v2 - ra_v1
    >>> ddec = dec_v2 - dec_v1
    >>> sph.plot_sky_vectors(ax, ra_v1, dec_v1, dra, ddec,
    ...     scale='auto', units='arcsec', cos_dec=False,
    ...     color_by_magnitude=True, add_colorbar=True,
    ...     cbar_label='Δposition (arcsec)')

    >>> # VSH residual field
    >>> sph.plot_sky_vectors(ax, ra, dec, dra_resid, ddec_resid,
    ...     scale='auto', units='uas',
    ...     color_by_magnitude=True, add_colorbar=True)
    """
    # A SkyCoord occupies the `lon` slot alone, so the deltas come by keyword:
    # plot_sky_vectors(ax, coords, dlon=..., dlat=...). Deliberately NOT the
    # positional-shift trick used elsewhere — that only shifts one argument and
    # this signature would need three.
    if hasattr(lon, 'transform_to'):
        if lat is not None:
            raise TypeError(
                "plot_sky_vectors: pass either a SkyCoord (with dlon=/dlat= as "
                "keywords) or separate lon/lat arrays — not both.")
        lon, lat = _coords_to_frame_deg(lon, _get_wcs_frame_name(ax))
    if lat is None or dlon is None or dlat is None:
        raise TypeError(
            "plot_sky_vectors requires lat, dlon and dlat (dlon/dlat may be "
            "keywords when the position is given as a SkyCoord).")
    lon_a: np.ndarray = np.atleast_1d(np.asarray(lon, dtype=float))
    lat_a: np.ndarray = np.atleast_1d(np.asarray(lat, dtype=float))
    dlon_a: np.ndarray = np.atleast_1d(np.asarray(dlon, dtype=float))
    dlat_a: np.ndarray = np.atleast_1d(np.asarray(dlat, dtype=float))

    # Convert vector units to degrees
    if units.lower() not in _VECTOR_UNIT_FACTORS:
        raise ValueError(f"Unknown units '{units}'. "
                         f"Choose from: {', '.join(_VECTOR_UNIT_FACTORS)}")
    ufac = _VECTOR_UNIT_FACTORS[units.lower()]

    # Auto-scale: compute scale so median vector → auto_target_deg on sky.
    # ``scale_val`` is the resolved numeric scale (mypy can't narrow the
    # ``float | str`` parameter through the string branch below).
    if isinstance(scale, str) and scale.lower() == 'auto':
        magnitudes = np.sqrt(dlon_a**2 + dlat_a**2)
        median_mag = np.median(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 1.0
        # scale * ufac * median_mag = auto_target_deg (in degrees)
        scale_val = auto_target_deg / (ufac * median_mag) if median_mag * ufac > 0 else 1.0
    else:
        scale_val = float(scale)

    # Convert to projected degrees and apply scale (the actual
    # arrow lengths handed to ``ax.quiver`` in world coords).
    dlon_deg = dlon_a * ufac * scale_val
    dlat_deg = dlat_a * ufac * scale_val

    # Apply cos(dec) correction if not already included
    if not cos_dec:
        cos_lat = np.cos(np.radians(lat_a))
        cos_lat = np.where(cos_lat > 1e-6, cos_lat, 1e-6)
        dlon_deg = dlon_deg / cos_lat

    # ``quiver(..., angles='xy', scale_units='xy', scale=1)`` measures the arrow
    # length in the axes' *data* units, which on a WCSAxes are WCS pixels — not
    # world degrees. Handing it degrees made every arrow pick up a spurious
    # factor of the pixel scale, so ``scale`` only behaved as documented on the
    # default all-sky grid (where deg/pixel happens to be ~0.9); on a zoomed
    # field, or after raising ``npix``, arrows collapsed. Convert to pixels here
    # so the arrow's on-sky length stays a fixed multiple of the vector
    # magnitude, independent of the frame and of ``npix``.
    #
    # Divide both components by the SAME scalar: ``angles='xy'`` derives the
    # arrow's direction by perturbing the world coords by (dlon, dlat), so
    # scaling them alike leaves the direction (and the cos(dec) handling above)
    # untouched, whereas per-axis divisors would skew it. ``scale_units='xy'``
    # likewise measures length against the view diagonal, a single scalar. The
    # frames this draws on have square pixels, where the two axis scales agree.
    deg_per_pix = _wcs_deg_per_pixel(ax.wcs)
    dlon_pix = dlon_deg / deg_per_pix
    dlat_pix = dlat_deg / deg_per_pix

    # A whole field of sub-pixel arrows renders as a grid of dots and looks
    # like "the arrows didn't draw" — but nothing failed, the vectors are just
    # tiny in the frame's units. The usual cause is a unit mismatch: ``units``
    # defaults to arcsec, so a field carrying dimensionless or degree-scale
    # amplitudes is read ~3600x too small. Say so rather than drawing nothing
    # visible.
    _max_pix = float(np.nanmax(np.hypot(dlon_pix, dlat_pix))) \
        if dlon_pix.size else 0.0
    if 0.0 < _max_pix < 1.0:
        warnings.warn(
            f"plot_sky_vectors: every arrow is shorter than one pixel "
            f"(longest ~{_max_pix:.2g} px), so the field will render as dots. "
            f"The magnitudes are being read as {units!r}"
            + (f" and multiplied by scale={scale}" if not isinstance(scale, str)
               else "")
            + ". Pass units= matching your data (e.g. units='deg'), or "
            "scale='auto' to size the median arrow automatically.",
            stacklevel=2)

    transform = ax.get_transform('world')

    # Resolve color-by-magnitude shortcut. When set, ``color=`` is
    # ignored in favor of an internally-computed magnitude C-array.
    if color_by_magnitude:
        c_arr = np.hypot(dlon_a, dlat_a)
        if cmap is None:
            cmap = 'viridis'
    elif hasattr(color, '__len__') and not isinstance(color, str):
        c_arr = np.asarray(color)
    else:
        c_arr = None

    quiver_kwargs = dict(
        transform=transform, angles='xy', scale_units='xy', scale=1,
        alpha=alpha, width=width,
        headwidth=headwidth, headlength=headlength,
        headaxislength=headaxislength, zorder=zorder,
        pivot=pivot,
    )
    if label is not None:
        quiver_kwargs['label'] = label
    if c_arr is None:
        quiver_kwargs['color'] = color
    elif cmap is not None:
        quiver_kwargs['cmap'] = cmap
    quiver_kwargs.update(kwargs)

    if c_arr is not None:
        q = ax.quiver(lon_a, lat_a, dlon_pix, dlat_pix, c_arr, **quiver_kwargs)
    else:
        q = ax.quiver(lon_a, lat_a, dlon_pix, dlat_pix, **quiver_kwargs)

    cbar = None
    if add_colorbar and c_arr is not None:
        fig = ax.figure
        ck = dict(cbar_kwargs) if cbar_kwargs else {}
        if cbar_orientation == 'horizontal':
            ck.setdefault('orientation', 'horizontal')
            ck.setdefault('fraction', 0.025)
            ck.setdefault('pad', 0.04)
        else:
            ck.setdefault('orientation', 'vertical')
            ck.setdefault('fraction', 0.03)
            ck.setdefault('pad', 0.04)
        cbar = fig.colorbar(q, ax=ax, **ck)
        if cbar_label is None and color_by_magnitude:
            cbar_label = f'magnitude ({units})'
        if cbar_label is not None:
            cbar.set_label(cbar_label)

    return SkyVectorResult(quiver=q, colorbar=cbar, scale=float(scale_val),
                           deg_per_pix=float(deg_per_pix))


def sky_quiverkey(result: SkyVectorResult, ax: Any, X: float, Y: float,
                  value: float, label: str, *, units: str = 'deg',
                  **kwargs: Any) -> Any:
    """Draw a reference-arrow key for a :func:`plot_sky_vectors` field, in
    physical units.

    ``plot_sky_vectors`` hands ``ax.quiver`` its arrow lengths in the frame's
    *pixel* units (sky degrees divided by the local pixel scale), so a raw
    :meth:`~matplotlib.axes.Axes.quiverkey` call — whose ``U`` is measured in
    those same pixel units — would need the same conversion, and passing the
    plain degree value silently produces a key short by exactly the pixel
    scale. This helper does the conversion from ``result``, so a key can't
    disagree with the field it annotates.

    Parameters
    ----------
    result : SkyVectorResult
        The return value of :func:`plot_sky_vectors`. Supplies the resolved
        ``scale`` (so ``scale='auto'`` fields work without the caller knowing
        the number) and the ``deg_per_pix`` actually used.
    ax : matplotlib WCSAxes
        The axes the field was drawn on.
    X, Y : float
        Key position, in axes fraction (matplotlib's ``quiverkey`` default
        ``coordinates='axes'``; override via ``coordinates=`` in ``kwargs``).
    value : float
        Reference magnitude, in ``units``. The key arrow is drawn the length
        this magnitude would take in the field.
    label : str
        Text drawn beside the key (e.g. ``'10 mas/yr'``).
    units : str
        Unit of ``value`` — ``'deg'`` / ``'arcmin'`` / ``'arcsec'`` /
        ``'mas'`` / ``'uas'``. Independent of the field's own ``units``:
        ``scale`` multiplies the angular magnitude in degrees, so the key may
        be expressed in whatever unit reads best. Default ``'deg'``.
    **kwargs
        Forwarded to :meth:`~matplotlib.axes.Axes.quiverkey` (``labelpos``,
        ``coordinates``, ``color``, ...).

    Returns
    -------
    matplotlib.quiver.QuiverKey

    Examples
    --------
    >>> res = plot_sky_vectors(ax, ra, dec, pmra, pmdec,   # doctest: +SKIP
    ...                        units='mas', scale='auto')
    >>> sky_quiverkey(res, ax, 0.9, 0.95, 10, '10 mas/yr', units='mas')
    """
    if units.lower() not in _VECTOR_UNIT_FACTORS:
        raise ValueError(f"Unknown units '{units}'. "
                         f"Choose from: {', '.join(_VECTOR_UNIT_FACTORS)}")
    ufac = _VECTOR_UNIT_FACTORS[units.lower()]
    # Match plot_sky_vectors: on-sky length = value * ufac * scale (degrees),
    # then / deg_per_pix into the quiver's pixel units. scale and deg_per_pix
    # are unit-independent, so they come from the result verbatim.
    u_pix = value * ufac * result.scale / result.deg_per_pix
    return ax.quiverkey(result.quiver, X, Y, u_pix, label, **kwargs)


def _wrap_break_lonlat(lons: npt.ArrayLike, lats: npt.ArrayLike,
                       center: float) -> tuple[np.ndarray, np.ndarray]:
    """Wrap ``lons`` into the projection window and NaN-break seam jumps.

    Returns ``(lons, lats)`` with each longitude wrapped to
    ``[center - 180, center + 180)`` and a ``NaN`` inserted between any two
    consecutive samples that straddle the wrap meridian (their wrapped
    longitudes differ by more than 180°). Plotting the result with the
    WCSAxes ``'world'`` transform then draws the path off one frame edge
    and onto the other instead of streaking straight across the canvas.
    """
    w = center + (((np.asarray(lons, dtype=float) - center + 180.0) % 360.0)
                  - 180.0)
    lats = np.asarray(lats, dtype=float)
    out_lon = [w[0]]
    out_lat = [lats[0]]
    for k in range(1, len(w)):
        if abs(w[k] - w[k - 1]) > 180.0:
            out_lon.append(np.nan)
            out_lat.append(np.nan)
        out_lon.append(w[k])
        out_lat.append(lats[k])
    return np.array(out_lon), np.array(out_lat)


def plot_displacement(ax: Any, lon1: SkyCoord | npt.ArrayLike, lat1: SkyCoord | npt.ArrayLike,
                      lon2: npt.ArrayLike | None = None, lat2: npt.ArrayLike | None = None,
                      color: str = 'C0', alpha: float = 0.7,
                      arrowstyle: str = '->', lw: float = 1,
                      connectionstyle: str = 'arc3,rad=0',
                      label: str | None = None, zorder: int = 5,
                      geodesic: bool = True, n_geodesic: int = 24,
                      **kwargs: Any) -> list[Any]:
    """
    Plot displacement vectors between two epochs as arrows.

    Unlike ``plot_sky_vectors`` which uses quiver (uniform arrow style),
    this draws individual arrows with per-arrow styling — better for small
    numbers of sources.

    By default each arrow follows the **great-circle path** between the two
    positions, sampled and drawn through the WCS so it tracks the
    projection's curvature (meaningful for large displacements or in
    distorted regions) and — crucially — renders correctly across the
    projection's wrap seam: the shaft is split at the seam and drawn off
    one frame edge and onto the other, rather than streaking straight
    across the canvas the way a single ``annotate`` arrow does. The
    arrowhead is placed on the final segment at the destination.

    Parameters
    ----------
    ax : WCSAxes
    lon1, lat1 : array-like, or two SkyCoords
        Epoch 1 positions (degrees). Two ``SkyCoord`` objects may be passed as
        ``lon1, lat1`` instead — start and end — replacing all four coordinate
        arguments, the same shape as :meth:`Ruler.from_world`.
    lon2, lat2 : array-like
        Epoch 2 positions (degrees). Omit when passing two SkyCoords.
    color : str
    alpha : float
    arrowstyle : str
        Matplotlib arrowstyle string (``'->'``, ``'-|>'``, ``'fancy'``, etc.)
    lw : float
    connectionstyle : str
        Arrow path shape for the head segment / the legacy straight arrow
        ('arc3,rad=0' for straight, 'arc3,rad=0.1' for curved). When
        ``geodesic=True`` the shaft follows the great circle, so this only
        affects the short head segment.
    label : str, optional
    zorder : int
    geodesic : bool
        If ``True`` (default), draw the great-circle path (seam-aware). If
        ``False``, fall back to a single straight ``annotate`` arrow in
        display space — faster for local (e.g. TAN) fields, but *not*
        seam-aware, so avoid it on all-sky frames where an arrow may cross
        the wrap edge.
    n_geodesic : int
        Number of samples along the great-circle shaft when
        ``geodesic=True``. Default ``24``.
    **kwargs
        Additional kwargs passed to the arrowhead's annotate arrowprops.

    Returns
    -------
    artists : list
        The shaft Line2D and arrowhead annotation artists.

    Examples
    --------
    >>> # epoch-to-epoch arrows from four coordinate arrays
    >>> sph.plot_displacement(ax, ra1, dec1, ra2, dec2, color='C1')

    >>> # or from two SkyCoords (start, end)
    >>> sph.plot_displacement(ax, coords_epoch1, coords_epoch2)
    """
    # Two SkyCoords replace all four coordinate args:
    # plot_displacement(ax, start_coords, end_coords) — the same shape as
    # Ruler.from_world(coord1, coord2), which serves this exact use case.
    if hasattr(lon1, 'transform_to') and hasattr(lat1, 'transform_to'):
        if lon2 is not None or lat2 is not None:
            raise TypeError(
                "plot_displacement: pass either two SkyCoords (start, end) or "
                "four lon/lat arrays — not a mixture.")
        frame_name = _get_wcs_frame_name(ax)
        lon1, lat1_deg = _coords_to_frame_deg(lon1, frame_name)
        lon2, lat2 = _coords_to_frame_deg(lat1, frame_name)
        lat1 = lat1_deg
    if lon2 is None or lat2 is None:
        raise TypeError(
            "plot_displacement requires either two SkyCoords (start, end) or "
            "all four of lon1, lat1, lon2, lat2.")

    lon1 = np.atleast_1d(lon1)
    lat1 = np.atleast_1d(lat1)
    lon2 = np.atleast_1d(lon2)
    lat2 = np.atleast_1d(lat2)

    transform = ax.get_transform('world')
    artists = []

    def _head(xytext: tuple[Any, Any], xy: tuple[Any, Any]) -> Any:
        return ax.annotate('', xy=xy, xytext=xytext,
                           xycoords=transform, textcoords=transform,
                           arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                           lw=lw, alpha=alpha,
                                           connectionstyle=connectionstyle,
                                           **kwargs),
                           zorder=zorder)

    if not geodesic:
        # Legacy straight-arrow path (display-space line; not seam-aware).
        for i in range(len(lon1)):
            artists.append(_head((lon1[i], lat1[i]), (lon2[i], lat2[i])))
            if i == 0 and label:
                ax.plot([], [], color=color, alpha=alpha, lw=lw, label=label)
        return artists

    from .geometry._densify import _slerp
    try:
        center = float(ax.wcs.wcs.crval[0])
    except Exception:
        center = 0.0

    for i in range(len(lon1)):
        # Sample the great-circle arc (``_slerp`` excludes its endpoint —
        # built for chaining densified segments — so append the true end).
        gl, gb = _slerp(float(lon1[i]), float(lat1[i]),
                        float(lon2[i]), float(lat2[i]), int(n_geodesic))
        gl = np.append(gl, float(lon2[i]))
        gb = np.append(gb, float(lat2[i]))
        # Shaft: wrap + seam-break so the polyline doesn't streak across the
        # canvas at the wrap edge, then draw through the WCS.
        wl, wb = _wrap_break_lonlat(gl, gb, center)
        line, = ax.plot(wl, wb, transform=transform, color=color, lw=lw,
                        alpha=alpha, zorder=zorder,
                        label=label if (i == 0 and label) else None)
        artists.append(line)
        # Head on the final arc segment (short, at the destination). Skip if
        # that segment itself straddles the seam (destination on the wrap
        # edge) — the shaft already conveys the direction there.
        d_last = ((float(gl[-1]) - float(gl[-2]) + 180.0) % 360.0) - 180.0
        if abs(d_last) < 180.0:
            artists.append(_head((float(gl[-2]), float(gb[-2])),
                                 (float(gl[-1]), float(gb[-1]))))

    return artists


def _apply_size_scale(
    raw: npt.NDArray[np.floating],
    scale: str | Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
) -> npt.NDArray[np.floating]:
    """Shape raw ``sizeby`` values before the linear ``smin``/``smax`` map.

    A callable receives and returns an array. ``'log'`` clips non-positive
    values up to the smallest positive value (and warns) since the log of
    ``<= 0`` is undefined.
    """
    if callable(scale):
        return np.asarray(scale(raw), dtype=float)
    if scale == 'linear':
        return raw
    if scale == 'sqrt':
        if np.any(raw < 0):
            warnings.warn("size_scale='sqrt': clipping negative sizeby values to 0")
        return np.sqrt(np.clip(raw, 0.0, None))
    if scale == 'log':
        pos = raw[np.isfinite(raw) & (raw > 0)]
        if pos.size == 0:
            warnings.warn("size_scale='log': no positive sizeby values; using linear")
            return raw
        if np.any(np.isfinite(raw) & (raw <= 0)):
            warnings.warn("size_scale='log': clipping non-positive sizeby values "
                          "to the smallest positive value")
        return np.log10(np.where(raw > 0, raw, pos.min()))
    raise ValueError(
        f"size_scale must be 'linear', 'sqrt', 'log', or callable; got {scale!r}")


def _resolve_color_norm(
    color_scale: str | Normalize,
    color_arr: npt.NDArray[np.floating],
    vmin: float | None, vmax: float | None,
) -> Normalize | None:
    """Resolve ``color_scale`` to a single :class:`Normalize` (or ``None``).

    Returns ``None`` for the linear default (the caller then passes
    ``vmin``/``vmax`` to ``scatter`` as before). A ``Normalize`` instance is
    used as-is; ``'log'`` builds a :class:`LogNorm` and ``'sqrt'`` a
    :class:`PowerNorm` (``gamma=0.5``) from the effective limits. We never hand
    ``scatter`` both a norm and ``vmin``/``vmax`` (mpl rejects it).
    """
    if isinstance(color_scale, Normalize):
        return color_scale
    if color_scale == 'linear':
        return None
    if color_scale == 'log':
        finite_pos = color_arr[np.isfinite(color_arr) & (color_arr > 0)]
        lo = vmin if vmin is not None else (
            float(finite_pos.min()) if finite_pos.size else None)
        hi = vmax if vmax is not None else (
            float(finite_pos.max()) if finite_pos.size else None)
        if lo is None or lo <= 0:
            warnings.warn("color_scale='log' needs positive vmin / colorby "
                          "values; falling back to linear")
            return None
        return LogNorm(vmin=lo, vmax=hi)
    if color_scale == 'sqrt':
        finite = color_arr[np.isfinite(color_arr)]
        lo = vmin if vmin is not None else (
            float(finite.min()) if finite.size else None)
        hi = vmax if vmax is not None else (
            float(finite.max()) if finite.size else None)
        if lo is None:
            return None
        if lo < 0:
            warnings.warn("color_scale='sqrt' clips negative colorby values at 0")
            lo = 0.0
        return PowerNorm(gamma=0.5, vmin=lo, vmax=hi)
    raise ValueError(
        "color_scale must be 'linear', 'sqrt', 'log', or a matplotlib "
        f"Normalize; got {color_scale!r}")


def _truncate_cmap(cmap: Any, cmap_range: tuple[float, float] | None,
                   n: int = 256) -> Any:
    """Restrict a colormap to the sub-range ``(lo, hi)`` of ``[0, 1]``.

    Used to keep markers legible against dark/light backgrounds by avoiding
    the extreme ends of the colormap. Returns ``cmap`` unchanged when
    ``cmap_range`` is ``None``.
    """
    if cmap_range is None:
        return cmap
    lo, hi = cmap_range
    base = cmap if isinstance(cmap, Colormap) else plt.get_cmap(cmap)
    return LinearSegmentedColormap.from_list(
        f'{base.name}_trunc', base(np.linspace(lo, hi, n)))


def _add_size_legend(
    ax: Any,
    size_map: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating],
                    float, float],
    smin: float, smax: float, marker: Any, alpha: float, color: Any,
    num: int, title: str, legend_kwargs: dict[str, Any] | None,
) -> Any:
    """Attach a representative-marker legend explaining a ``sizeby`` encoding.

    Representative points are spaced evenly in the *scaled* domain (so the
    markers grow by even visual steps even for skewed data), and labeled with
    the corresponding raw data values via interpolation on the actual
    ``(scaled, raw)`` pairs — which works for any monotonic ``size_scale``,
    including a callable, without needing an analytic inverse.
    """
    from matplotlib.lines import Line2D

    raw, scaled, rmin, rmax = size_map
    if not (rmax > rmin) or num < 1:
        return None
    order = np.argsort(scaled)
    s_sorted, r_sorted = scaled[order], raw[order]
    targets = np.linspace(rmin, rmax, num)
    raw_labels = np.interp(targets, s_sorted, r_sorted)
    sizes = smin + (smax - smin) * (targets - rmin) / (rmax - rmin)
    # Line2D markersize is a diameter in points; scatter ``s`` is an area in
    # points² — convert so the legend markers match the plotted ones.
    handles = [Line2D([], [], linestyle='none', marker=marker,
                      markersize=float(np.sqrt(sz)),
                      markerfacecolor=color, markeredgecolor='none',
                      alpha=alpha)
               for sz in sizes]
    label_text = [f"{v:.3g}" for v in raw_labels]
    leg_kw: dict[str, Any] = dict(title=title, labelspacing=1.3,
                                  loc='lower left', frameon=True,
                                  borderpad=0.8)
    if legend_kwargs:
        leg_kw.update(legend_kwargs)
    leg = ax.legend(handles=handles, labels=label_text, **leg_kw)
    # add_artist so the size legend survives a later ax.legend() for other
    # artists (e.g. an overlaid reference line).
    ax.add_artist(leg)
    return leg


def plot_catalog(ax: Any, catalog: Any, ra_col: str = 'ra', dec_col: str = 'dec',
                 lon_col: str | None = None, lat_col: str | None = None,
                 frame: str | None = None, unit: str = 'deg',
                 marker: str = 'o', color: Any = 'C0', s: float = 20,
                 alpha: float = 0.7,
                 colorby: str | None = None, sizeby: str | None = None,
                 cmap: Any = 'viridis',
                 vmin: float | None = None, vmax: float | None = None,
                 smin: float = 10, smax: float = 200,
                 size_vlim: tuple[float, float] | None = None,
                 size_scale: str | Callable[
                     [npt.NDArray[np.floating]], npt.NDArray[np.floating]
                 ] = 'linear',
                 color_scale: str | Normalize = 'linear',
                 cmap_range: tuple[float, float] | None = None,
                 cbar: bool = False, cbar_label: str = '',
                 cbar_format: str | Formatter | None = None,
                 cbar_ticks: Sequence[float] | None = None,
                 size_legend: bool = False, size_legend_num: int = 4,
                 size_legend_kwargs: dict[str, Any] | None = None,
                 label_col: str | None = None, label_fontsize: float = 8,
                 label_offset: tuple[float, float] = (5, 5),
                 transform: Any = None, **kwargs: Any) -> Any:
    """
    Overlay catalog source positions on a WCSAxes.

    Accepts astropy Tables, pandas DataFrames, dicts, or arrays.
    Automatically handles coordinate column name detection.

    Parameters
    ----------
    ax : WCSAxes
        The plot axes.
    catalog : Table, DataFrame, dict, SkyCoord, or tuple
        Source catalog. For Table/DataFrame/dict, specify column
        names via ``ra_col``/``dec_col`` (or ``lon_col``/``lat_col``).
        For tuple, pass ``(lon_array, lat_array)`` directly.
        A **SkyCoord** (scalar or array) is accepted directly and converted
        into the frame the points are drawn in — ``frame=`` if given, else the
        axes' own frame — so it lands correctly on any axes. This is what
        :func:`~skyplothelper.cone_search` / :func:`~skyplothelper.region_search`
        return, so search → plot needs no ``.ra.deg``/``.dec.deg`` unwrapping.
    ra_col, dec_col : str
        Column names for the longitude / latitude coordinates. Default
        ``'ra'`` / ``'dec'``; auto-detected from common equatorial *and*
        galactic names ('RA', 'ra_deg', 'RAJ2000', 'l', 'GLON', 'b',
        'GLAT', ...) if the given names are not found.
    lon_col, lat_col : str, optional
        Frame-neutral aliases for ``ra_col`` / ``dec_col`` — handy for
        galactic ``l``/``b`` input. When given, they take precedence.
    frame : str, optional
        Coordinate frame the catalog coordinates are in (e.g. ``'icrs'``,
        ``'fk5'``, ``'galactic'``). The points are converted from this
        frame onto the plot, so a galactic catalog lands correctly on an
        equatorial map and vice versa. Default ``None`` — the coordinates
        are assumed already in the axes' native frame (no conversion; the
        back-compatible default). Ignored when ``transform`` is given.
    unit : str
        Angular unit of the coordinates ('deg', 'hourangle', 'rad').
    marker, color, s, alpha :
        Scatter plot styling (passed to ``ax.scatter``).
        ``color`` is ignored when ``colorby`` is set.
        ``s`` is ignored when ``sizeby`` is set.
    colorby : str, optional
        Column name for color-coding markers by value. Uses ``cmap``.
    sizeby : str, optional
        Column name for size-coding markers by value. Mapped to the
        ``[smin, smax]`` range (see ``size_scale``).
    cmap : str or Colormap
        Colormap for ``colorby`` (default 'viridis').
    vmin, vmax : float, optional
        Color scale limits for ``colorby``. Ignored when ``color_scale``
        is a :class:`~matplotlib.colors.Normalize` instance.
    smin, smax : float
        Min/max marker size for ``sizeby`` scaling.
    size_vlim : (float, float), optional
        Fixed raw-value bounds ``(lo, hi)`` for the ``sizeby`` scaling, in
        the data's own units. By default the scaling auto-ranges to this
        call's min/max; pass a shared ``size_vlim`` across several calls
        (e.g. one scatter per marker category, since a scatter has a single
        marker) so equal values render at equal sizes across all of them.
        A graduated-size legend then also spans this range.
    size_scale : {'linear', 'sqrt', 'log'} or callable
        Transform applied to the raw ``sizeby`` values *before* the linear
        ``[smin, smax]`` map (``smin``/``smax`` still define the output
        range; the scale only reshapes the distribution). ``'log'`` clips
        non-positive values to the smallest positive value (with a warning).
        Default ``'linear'``.

        A callable takes the raw ``sizeby`` array and **returns** the
        transformed array (same shape) — it must *call* the transform on its
        argument, not return a function or a scalar::

            size_scale=lambda x: np.sqrt(np.clip(x, 0, None))  # == 'sqrt'
            size_scale=np.sqrt                                 # also valid
            size_scale=lambda x: x**0.3                        # any monotonic shape

        (A common slip is ``lambda x: np.sqrt`` — that returns the *function*
        object, not ``np.sqrt(x)``, and raises a ``float()`` ``TypeError``.)
    color_scale : {'linear', 'sqrt', 'log'} or Normalize
        How ``colorby`` values map onto the colormap. ``'log'`` builds a
        :class:`~matplotlib.colors.LogNorm` and ``'sqrt'`` a
        :class:`~matplotlib.colors.PowerNorm` (``gamma=0.5``) from the
        effective ``vmin``/``vmax`` (mirroring ``size_scale``); a
        :class:`~matplotlib.colors.Normalize` instance is used as-is (and
        ``vmin``/``vmax`` are then ignored). Default ``'linear'``.
    cmap_range : tuple of float, optional
        ``(lo, hi)`` sub-range of ``[0, 1]`` to truncate the colormap to,
        e.g. ``(0.15, 0.95)`` to keep markers off the darkest/lightest ends
        for legibility on colored backgrounds. Default ``None`` (full range).
    cbar : bool
        Add a colorbar when ``colorby`` is set. Default False.
    cbar_label : str
        Colorbar label text.
    cbar_format : str or Formatter, optional
        Tick-label format for the colorbar. A ``str`` is treated as a
        new-style format (``'{x:.2f}'``) if it contains ``'{'``, else an
        old-style format (``'%.2f'``); a :class:`~matplotlib.ticker.Formatter`
        is used directly. Handy to show plain decimals on a ``'log'`` colorbar.
    cbar_ticks : sequence of float, optional
        Explicit colorbar tick locations.
    size_legend : bool
        When ``sizeby`` is active, add a legend of representative markers
        explaining the size encoding (labeled in the data's own units).
        Default False. Attached via ``ax.add_artist`` so it coexists with any
        other legend you create.
    size_legend_num : int
        Number of representative markers in the size legend. Default 4.
    size_legend_kwargs : dict, optional
        Extra kwargs forwarded to ``ax.legend`` for the size legend (e.g.
        ``loc``, ``title``, ``frameon``). ``title`` defaults to the ``sizeby``
        column name.
    label_col : str, optional
        Column name for source labels.
    label_fontsize : float
    label_offset : tuple
        (dx, dy) pixel offset for labels from marker position.
    transform : matplotlib Transform, optional
        Override coordinate transform. Default resolves from ``frame``
        (the axes' native ``'world'`` frame when ``frame`` is ``None``).
    **kwargs
        Additional kwargs passed to ``ax.scatter()``.

    Returns
    -------
    PathCollection or CatalogPlot
        The scatter artist. When a colorbar is created (``cbar=True`` with
        ``colorby`` set) the return is a :class:`CatalogPlot` named tuple
        ``(scatter, colorbar)`` instead — unpackable as
        ``sc, cb = plot_catalog(...)`` and attribute-accessible
        (``.scatter`` / ``.colorbar``). Otherwise just the
        :class:`~matplotlib.collections.PathCollection` is returned.

    Examples
    --------
    >>> sc = plot_catalog(ax, cat, label_col='name')
    >>> sc = plot_catalog(ax, (ra_array, dec_array), color='red', s=5)

    >>> # Color-code by redshift, size by magnitude (colorbar → tuple return)
    >>> sc, cb = plot_catalog(ax, cat, colorby='z', sizeby='mag',
    ...             cmap='plasma', cbar=True, cbar_label='Redshift')

    >>> # Skewed data: sqrt-scaled sizes (+ size legend), log color
    >>> sc, cb = plot_catalog(ax, cat, sizeby='n_sess', size_scale='sqrt',
    ...             size_legend=True, colorby='pos_err', color_scale='log',
    ...             cmap_range=(0.2, 1.0), cbar=True, cbar_format='{x:.2f}')

    >>> # Galactic l/b catalog onto an equatorial map (auto-converted)
    >>> sc = plot_catalog(ax, gal_cat, lon_col='l', lat_col='b',
    ...             frame='galactic')
    """
    # Guard the common misuse `plot_catalog(ax, ra, dec)`: the 2nd argument is
    # a CATALOG (table / DataFrame / dict, or a (ra, dec) pair) and
    # ra_col/dec_col/colorby/sizeby name COLUMNS in it — not coordinate arrays.
    for _name, _val in (("ra_col", ra_col), ("dec_col", dec_col),
                        ("lon_col", lon_col), ("lat_col", lat_col),
                        ("colorby", colorby), ("sizeby", sizeby)):
        if _val is not None and not isinstance(_val, str):
            raise TypeError(
                f"plot_catalog: {_name}= must be a column NAME (str), got a "
                f"{type(_val).__name__}. The 2nd argument is a catalog (astropy "
                f"Table / pandas DataFrame / dict of columns, or a (ra, dec) "
                f"pair) and {_name} names a column in it. For raw coordinate "
                f"arrays use ax.scatter(ra, dec, "
                f"transform=ax.get_transform('world')), or pass a (ra, dec) "
                f"tuple as the catalog.")

    # Extract coordinate arrays
    if isinstance(catalog, (tuple, list)) and len(catalog) == 2:
        ra_arr, dec_arr = np.asarray(catalog[0], dtype=float), \
                          np.asarray(catalog[1], dtype=float)
        labels = None
    elif hasattr(catalog, 'transform_to'):
        # SkyCoord (scalar or array). Converted into the frame the points are
        # about to be drawn in — ``frame=`` when given, else the axes' native
        # frame — so a galactic SkyCoord lands correctly on any axes. This is
        # the natural output of cone_search/region_search/crossmatch, so the
        # search → plot pipeline works without hand-unwrapping .ra/.dec.
        from .geometry._parsing import _coords_to_frame_deg
        from .wcs_frame import _get_wcs_frame_name
        target_frame = frame if frame is not None else _get_wcs_frame_name(ax)
        lon_vals, lat_vals = _coords_to_frame_deg(catalog, target_frame)
        ra_arr = np.atleast_1d(np.asarray(lon_vals, dtype=float))
        dec_arr = np.atleast_1d(np.asarray(lat_vals, dtype=float))
        labels = None
    else:
        # Table, DataFrame, or dict. Auto-detect spans equatorial, galactic,
        # and ecliptic names so an l/b (or lon/lat) catalog resolves without
        # an explicit column. The equatorial names are shared with
        # skyplothelper.catalog's registry (single source of truth, so the
        # two modules can't drift — e.g. VizieR's DE_ICRS); the galactic /
        # ecliptic aliases are plotting-only extras. Matching is
        # case-insensitive, like catalog._find_col.
        from .catalog import _DEC_NAMES, _RA_NAMES
        _common_lon = _RA_NAMES + ('glon', 'l', 'lon', 'lii', 'elon')
        _common_lat = _DEC_NAMES + ('glat', 'b', 'lat', 'bii', 'elat')

        # Try specified column names first, then auto-detect
        cols = (list(catalog.keys()) if hasattr(catalog, 'keys')
                else list(catalog.columns) if hasattr(catalog, 'columns')
                else [])
        if not cols:
            raise TypeError(
                f"plot_catalog: the 2nd argument must be a catalog with named "
                f"columns (astropy Table / pandas DataFrame / dict) or a "
                f"(ra, dec) pair; got a {type(catalog).__name__} with no "
                f"columns. For raw coordinate arrays either pass them as a "
                f"(ra, dec) tuple, or use ax.scatter(ra, dec, "
                f"transform=ax.get_transform('world')).")

        def _find_col(specified: str | None,
                      common_names: tuple[str, ...]) -> str:
            lower = {c.lower(): c for c in cols}
            if specified is not None and specified.lower() in lower:
                return lower[specified.lower()]
            for name in common_names:
                if name in lower:
                    return lower[name]
            raise KeyError(
                f"Column {specified!r} not found and could not auto-detect "
                f"from {cols[:10]}...")

        # lon_col/lat_col are frame-neutral aliases; they win when given.
        lon_spec = lon_col if lon_col is not None else ra_col
        lat_spec = lat_col if lat_col is not None else dec_col
        ra_key = _find_col(lon_spec, _common_lon)
        dec_key = _find_col(lat_spec, _common_lat)

        ra_arr = np.asarray(catalog[ra_key], dtype=float)
        dec_arr = np.asarray(catalog[dec_key], dtype=float)

        if label_col and label_col in cols:
            labels = list(catalog[label_col])
        else:
            labels = None

    # Convert units if needed
    if unit == 'hourangle':
        ra_arr = ra_arr * 15.0  # hours → degrees
    elif unit == 'rad':
        ra_arr = np.degrees(ra_arr)
        dec_arr = np.degrees(dec_arr)

    # Extract colorby / sizeby columns
    color_arr = None
    size_arr = None
    # (raw, scaled, rmin, rmax) for sizeby, kept for the optional size legend.
    size_map: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating],
                     float, float] | None = None

    if colorby is not None and not isinstance(catalog, (tuple, list)):
        if colorby in cols:
            color_arr = np.asarray(catalog[colorby], dtype=float)
        else:
            warnings.warn(f"colorby column '{colorby}' not found in catalog")

    if sizeby is not None and not isinstance(catalog, (tuple, list)):
        if sizeby in cols:
            raw = np.asarray(catalog[sizeby], dtype=float)
            scaled = _apply_size_scale(raw, size_scale)
            # Linear map of the (re)shaped values to [smin, smax]; smin/smax
            # keep defining the output range, size_scale only shapes it.
            if size_vlim is not None:
                # Fixed raw-value bounds so multiple subset calls (e.g. one
                # scatter per marker category, since a PathCollection has a
                # single marker) share one raw→size scale — identical values
                # render at identical sizes across calls.
                bounds = _apply_size_scale(
                    np.asarray(size_vlim, dtype=float), size_scale)
                rmin, rmax = float(np.min(bounds)), float(np.max(bounds))
            else:
                rmin, rmax = float(np.nanmin(scaled)), float(np.nanmax(scaled))
            if rmax > rmin:
                size_arr = smin + (smax - smin) * (scaled - rmin) / (rmax - rmin)
            else:
                size_arr = np.full_like(scaled, (smin + smax) / 2)
            size_map = (raw, scaled, rmin, rmax)
        else:
            warnings.warn(f"sizeby column '{sizeby}' not found in catalog")

    # Build scatter kwargs
    if transform is None:
        # frame=None → the axes' native world frame (back-compatible default).
        # An explicit frame ('galactic', 'fk5', ...) makes WCSAxes convert the
        # input coordinates from that frame onto the plot — so e.g. galactic
        # l/b land correctly on an equatorial map (and vice versa).
        transform = ax.get_transform('world' if frame is None else frame)

    scatter_kw = dict(marker=marker, alpha=alpha,
                      transform=transform, **kwargs)

    if color_arr is not None:
        scatter_kw['c'] = color_arr
        # Optionally truncate the colormap to a sub-range for legibility.
        scatter_kw['cmap'] = _truncate_cmap(cmap, cmap_range)
        # Resolve color_scale to a single norm. Hand scatter EITHER a norm
        # (log / explicit Normalize) OR vmin/vmax — never both (mpl rejects it).
        norm = _resolve_color_norm(color_scale, color_arr, vmin, vmax)
        if norm is not None:
            scatter_kw['norm'] = norm
        else:
            if vmin is not None:
                scatter_kw['vmin'] = vmin
            if vmax is not None:
                scatter_kw['vmax'] = vmax
    else:
        scatter_kw['c'] = color

    scatter_kw['s'] = size_arr if size_arr is not None else s

    sc = ax.scatter(ra_arr, dec_arr, **scatter_kw)

    # Stash the size scaling on the artist so a graduated-size legend
    # (SizeBlock.from_catalog / MultiLegend.add_size_from) reproduces the
    # exact on-plot sizes without the caller restating smin/smax/scale.
    if size_map is not None:
        sc._sph_size_info = _CatalogSizeInfo(size_map, smin, smax, size_scale)

    # Colorbar
    cb = None
    if cbar and color_arr is not None:
        import matplotlib.axes as maxes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05,
                                   axes_class=maxes.Axes)
        cb = plt.colorbar(sc, cax=cax)
        if cbar_label:
            cb.set_label(cbar_label)
        if cbar_ticks is not None:
            cb.set_ticks(list(cbar_ticks))
        if cbar_format is not None:
            if isinstance(cbar_format, str):
                fmt: Formatter = (StrMethodFormatter(cbar_format)
                                  if '{' in cbar_format
                                  else FormatStrFormatter(cbar_format))
            else:
                fmt = cbar_format
            cb.formatter = fmt
            # On a log colorbar the minor ticks would still print scientific
            # labels; silence them so only the requested format shows.
            cb.ax.yaxis.set_minor_formatter(NullFormatter())
            cb.update_ticks()

    # Size legend — representative markers labeled in the data's own units.
    if size_legend:
        if size_map is None:
            warnings.warn("size_legend=True ignored: no valid sizeby data")
        else:
            _add_size_legend(ax, size_map, smin, smax, marker, alpha,
                             color if color_arr is None else '0.4',
                             size_legend_num, sizeby or '',
                             size_legend_kwargs)

    # Labels
    label_color = color if color_arr is None else '0.3'
    if labels is not None:
        dx, dy = label_offset
        for i, lab in enumerate(labels):
            if lab is None or (isinstance(lab, str) and not lab.strip()):
                continue
            # For WCSAxes, the anchor point's coordinate system is specified
            # via ``xycoords=`` (not ``transform=``, which sets the transform
            # for ``xytext``). Pass the 'world' transform as xycoords so the
            # (ra, dec) anchor is interpreted in world degrees.
            ax.annotate(str(lab), (ra_arr[i], dec_arr[i]),
                       xycoords=transform,
                       xytext=(dx, dy), textcoords='offset points',
                       fontsize=label_fontsize, color=label_color,
                       clip_on=True)

    # Bare scatter by default (back-compatible); expose the colorbar via a
    # tuple-unpackable named tuple only when one was actually created.
    if cb is not None:
        return CatalogPlot(sc, cb)
    return sc


