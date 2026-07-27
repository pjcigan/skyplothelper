"""Thin matplotlib wrappers for plotting your own data onto a sky axes.

Every one of these is the matplotlib verb you already know, with three chores
absorbed:

1. **The world transform.** ``ax.scatter(ra, dec, transform=ax.get_transform(
   'world'))`` becomes ``sph.scatter(ax, ra, dec)``. (``annotate`` is the
   inconsistent one in matplotlib — its transform goes in ``xycoords=``, not
   ``transform=`` — which the wrapper hides.)
2. **SkyCoord input.** Pass a ``SkyCoord`` (scalar or array) instead of two
   degree arrays; it is converted into the axes' own frame, so a galactic
   catalog lands correctly on an equatorial map.
3. **The antimeridian seam.** The line-drawing verbs break the path where it
   crosses the wrap edge, instead of streaking a horizontal line across the
   whole map. Opt out with ``wrap=False``.

The functions take the axes first (``sph.scatter(ax, ...)``), matching the rest
of the package. On axes that skyplothelper builds, the same calls are also
available as methods — ``ax.sky_scatter(...)`` — which read more like ordinary
matplotlib. Those methods are one-line delegations to these functions, so
behavior can't drift between the two spellings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .geometry._parsing import _coords_or_arrays_deg

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    import numpy.typing as npt
    from astropy.coordinates import SkyCoord

__all__ = [
    'world_transform', 'to_lonlat',
    'scatter', 'plot', 'text', 'annotate', 'errorbar', 'fill',
    'fill_between', 'step',
    'contour', 'contourf', 'pcolormesh', 'tricontourf', 'hist2d',
]

# Verbs whose artists are a connected path, and so can streak across the map at
# the wrap edge. Point/patch verbs (scatter, text, ...) can't, so they skip the
# seam machinery entirely.
_LINE_VERBS = frozenset({'plot', 'step', 'fill', 'fill_between'})


def world_transform(ax: Any) -> Any:
    """The world-coordinate transform for a WCSAxes.

    ``ax.get_transform('world')`` with a clear error when *ax* isn't a sky
    axes. Exposed because the idiom appears constantly in user code and in
    these docs; the wrappers in this module apply it for you.
    """
    if not hasattr(ax, 'get_transform'):
        raise TypeError(
            "world_transform() needs a WCSAxes (a skyplothelper sky frame); "
            f"got a plain {type(ax).__name__}. Build one with "
            "sph.make_wcs_frame(...) / sph.allsky_figure(...).")
    return ax.get_transform('world')


def to_lonlat(coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, *,
              frame: str | None = None, ax: Any = None,
              ) -> tuple[Any, Any]:
    """Normalize any sky-coordinate input to ``(lon, lat)`` in degrees.

    The supported one-liner for "I have coordinates in some form — give me
    plain degrees". Accepts what the rest of the package accepts, so the
    result can go to matplotlib, to a non-skyplothelper library, or into your
    own math.

    Parameters
    ----------
    coords : SkyCoord or array-like
        A :class:`~astropy.coordinates.SkyCoord` (scalar or array), or the
        longitude values, in which case *lat* holds the latitudes in degrees.
    lat : array-like, optional
        Latitudes in degrees. Omit when *coords* is a SkyCoord.
    frame : str, optional
        Convert into this frame (``'galactic'``, ``'ecliptic'``, ``'icrs'``
        and the usual short aliases). Default ``None`` — **the coordinates
        come back in their own frame, unconverted**. Plain numbers are always
        taken at face value.
    ax : WCSAxes, optional
        Convert into *this axes'* frame — i.e. "what numbers do I pass to this
        plot?". Ignored when *frame* is given.

    Returns
    -------
    lon, lat : float or ndarray
        Degrees. Scalar input gives scalars, array input gives arrays.

    Notes
    -----
    The default deliberately does **not** convert to ICRS. A bare
    ``to_lonlat(coord)`` is a lossless read-out of what you already have;
    silently coercing the frame is the failure mode this API exists to avoid.
    Ask for a conversion explicitly with *frame* or *ax*.

    Examples
    --------
    >>> lon, lat = sph.to_lonlat(catalog_skycoord)        # own frame, as-is
    >>> lon, lat = sph.to_lonlat(gal_coord, frame='icrs')  # convert
    >>> lon, lat = sph.to_lonlat(gal_coord, ax=ax)         # into the axes frame
    >>> lon, lat = sph.to_lonlat(ra_deg, dec_deg)          # pass-through
    """
    target = frame
    if target is None and ax is not None:
        target = _axes_frame(ax)
    lon_out, lat_out = _coords_or_arrays_deg(coords, lat, target,
                                             'sph.to_lonlat')
    lon_arr = np.asarray(lon_out, dtype=float)
    lat_arr = np.asarray(lat_out, dtype=float)
    if lon_arr.ndim == 0:
        return float(lon_arr), float(lat_arr)
    return lon_arr, lat_arr


def _axes_frame(ax: Any) -> str:
    from .wcs_frame import _get_wcs_frame_name
    return _get_wcs_frame_name(ax)


def _as_point(a: Any, b: Any) -> tuple[Any, Any]:
    """Allow a single ``(lon, lat)`` pair for the scalar-position verbs.

    ``sph.annotate(ax, 'M31', (110, 20))`` is the natural spelling, because
    matplotlib's own ``annotate``/``text`` take ``xy`` as a tuple. Only applied
    where a position is unambiguously scalar — for the array verbs a 2-tuple
    would be ambiguous with a ``(lons, lats)`` pair.
    """
    if (b is None and not hasattr(a, 'transform_to')
            and isinstance(a, (tuple, list)) and len(a) == 2
            and np.ndim(a[0]) == 0 and np.ndim(a[1]) == 0):
        return a[0], a[1]
    return a, b


def _resolve(ax: Any, a: Any, b: Any, frame: str | None,
             caller: str) -> tuple[np.ndarray, np.ndarray]:
    """``(SkyCoord | lon, lat)`` → degree arrays in the frame we will draw in.

    ``frame=None`` means the axes' own frame, which is what the ``'world'``
    transform expects.
    """
    target = frame if frame is not None else _axes_frame(ax)
    lon, lat = _coords_or_arrays_deg(a, b, target, caller)
    return np.atleast_1d(np.asarray(lon, dtype=float)), \
        np.atleast_1d(np.asarray(lat, dtype=float))


def _split_at_seam(ax: Any, lon: np.ndarray, lat: np.ndarray,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN breaks where a path crosses the projection's wrap edge.

    The crossing is found in **display space** — project the points and look
    for a step that spans a large fraction of the axes — rather than by
    re-deriving where the seam "should" be from the longitudes.

    That matters because the analytic version is subtly wrong at the boundary:
    wrapping with ``((lon - center + 180) % 360) - 180`` sends lon=180 to
    *minus* 180, while the projection renders it at *plus* 180. On a frame
    centered at 0, a path stepping across exactly 180 then gets its break one
    position off and still streaks. Display space is ground truth — whatever
    the projection, center, or wrapping convention, a seam crossing is a jump
    across the canvas, and nothing else is.
    """
    if lon.size < 2:
        return lon, lat
    try:
        disp = world_transform(ax).transform(
            np.column_stack([lon, lat]))
        width = float(ax.get_window_extent().width)
    except Exception:
        return lon, lat                      # not renderable yet — leave as-is
    if not np.isfinite(width) or width <= 0:
        return lon, lat
    dx = np.abs(np.diff(disp[:, 0]))
    # Half the axes width: far larger than any real step in a sampled track,
    # far smaller than the near-full-width jump a seam crossing produces.
    breaks = np.nonzero(np.isfinite(dx) & (dx > 0.5 * width))[0]
    if breaks.size == 0:
        return lon, lat
    # Insert a NaN after each crossing; matplotlib lifts the pen at NaN.
    out_lon = np.insert(lon.astype(float), breaks + 1, np.nan)
    out_lat = np.insert(lat.astype(float), breaks + 1, np.nan)
    return out_lon, out_lat


def _xy_kwargs(ax: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add the world transform unless the caller pinned their own."""
    kwargs.setdefault('transform', world_transform(ax))
    return kwargs


def _draw(ax: Any, verb: str, a: Any, b: Any, frame: str | None,
          wrap: bool, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    lon, lat = _resolve(ax, a, b, frame, f'sph.{verb}')
    if wrap and verb in _LINE_VERBS:
        lon, lat = _split_at_seam(ax, lon, lat)
    return getattr(ax, verb)(lon, lat, *args, **_xy_kwargs(ax, kwargs))


# --- point / path verbs ---------------------------------------------------

def scatter(ax: Any, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, *,
            frame: str | None = None, **kwargs: Any) -> Any:
    """``ax.scatter`` on sky coordinates.

    Parameters
    ----------
    ax : WCSAxes
    coords : SkyCoord or array-like
        Either a ``SkyCoord`` (scalar or array) or the longitude array, in
        which case *lat* holds the latitudes in degrees.
    lat : array-like, optional
        Latitudes in degrees. Omit when *coords* is a SkyCoord.
    frame : str, optional
        Frame the coordinates are in. Defaults to the axes' own frame, which
        is what the world transform expects; set it to plot e.g. galactic
        values onto an equatorial map.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.scatter`.

    Examples
    --------
    >>> sph.scatter(ax, ra_deg, dec_deg, s=6, c='C1')
    >>> sph.scatter(ax, skycoord_array, s=6)
    """
    return _draw(ax, 'scatter', coords, lat, frame, False, (), kwargs)


def plot(ax: Any, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, *,
         frame: str | None = None, wrap: bool = True,
         **kwargs: Any) -> Any:
    """``ax.plot`` on sky coordinates, split at the antimeridian.

    ``wrap=True`` (default) breaks the path where it crosses the projection's
    wrap edge, so a track spanning the seam does not streak across the map.
    Pass ``wrap=False`` for the raw matplotlib behavior.

    Examples
    --------
    >>> sph.plot(ax, ra_deg, dec_deg, lw=1)
    >>> sph.plot(ax, skycoord_track, ls='--', wrap=False)
    """
    return _draw(ax, 'plot', coords, lat, frame, wrap, (), kwargs)


def step(ax: Any, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, *,
         frame: str | None = None, wrap: bool = True, **kwargs: Any) -> Any:
    """``ax.step`` on sky coordinates. See :func:`plot` for ``wrap``."""
    return _draw(ax, 'step', coords, lat, frame, wrap, (), kwargs)


def fill(ax: Any, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, *,
         frame: str | None = None, wrap: bool = True, **kwargs: Any) -> Any:
    """``ax.fill`` on sky coordinates. See :func:`plot` for ``wrap``.

    For real spherical polygons (great-circle edges, pole handling, set
    algebra) prefer :func:`~skyplothelper.add_spherical_polygon` — this is the
    plain matplotlib fill with the transform applied.
    """
    return _draw(ax, 'fill', coords, lat, frame, wrap, (), kwargs)


def errorbar(ax: Any, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, *,
             frame: str | None = None, **kwargs: Any) -> Any:
    """``ax.errorbar`` on sky coordinates.

    ``xerr`` / ``yerr`` are in degrees of longitude / latitude. Note that a
    longitude error is *not* an on-sky angle away from the equator (it shrinks
    by cos(lat)); pass ``xerr`` already divided by cos(lat) if you want true
    angular error bars.
    """
    return _draw(ax, 'errorbar', coords, lat, frame, False, (), kwargs)


def fill_between(ax: Any, coords: npt.ArrayLike,
                 lat1: npt.ArrayLike | None = None,
                 lat2: npt.ArrayLike | None = None, *,
                 frame: str | None = None, wrap: bool = True,
                 **kwargs: Any) -> Any:
    """``ax.fill_between`` on sky coordinates.

    *coords* is the longitude array (a SkyCoord makes no sense here — two
    latitude bounds are required), *lat1* / *lat2* the bounding latitudes.
    """
    if hasattr(coords, 'transform_to'):
        raise TypeError(
            "sph.fill_between needs a longitude array plus two latitude "
            "bounds, so a SkyCoord (which carries only one latitude) can't "
            "be used here. Pass lon, lat1, lat2 arrays.")
    lon = np.atleast_1d(np.asarray(coords, dtype=float))
    y1 = np.atleast_1d(np.asarray(lat1, dtype=float))
    if wrap:
        # Only the longitudes can cross the seam; break both bounds together.
        lon_s, y1_s = _split_at_seam(ax, lon, y1)
        if lat2 is not None and np.ndim(lat2) > 0:
            _, y2_s = _split_at_seam(ax, lon, np.atleast_1d(
                np.asarray(lat2, dtype=float)))
            lat2 = y2_s
        lon, y1 = lon_s, y1_s
    args = () if lat2 is None else (lat2,)
    return ax.fill_between(lon, y1, *args, **_xy_kwargs(ax, kwargs))


# --- text verbs -----------------------------------------------------------

def text(ax: Any, coords: SkyCoord | npt.ArrayLike,
         lat: npt.ArrayLike | str | None = None, s: Any = None, *,
         frame: str | None = None, **kwargs: Any) -> Any:
    """``ax.text`` at a sky position.

    The position may be given as ``lon, lat`` degrees, as a single
    ``(lon, lat)`` pair, or as a scalar
    :class:`~astropy.coordinates.SkyCoord` (which carries its own frame and is
    converted into the axes').

    Examples
    --------
    >>> sph.text(ax, 83.6, 22.0, 'Crab')
    >>> sph.text(ax, (83.6, 22.0), 'Crab', ha='left')
    >>> sph.text(ax, SkyCoord.from_name('M1'), 'Crab')
    """
    if s is None and (hasattr(coords, 'transform_to')
                      or isinstance(coords, (tuple, list))):
        s, lat = lat, None      # sph.text(ax, coord_or_pair, 'label')
    coords, lat = _as_point(coords, lat)
    lon_a, lat_a = _resolve(ax, coords, lat, frame, 'sph.text')
    if s is None:
        raise TypeError("sph.text: the label string is required.")
    return ax.text(float(lon_a[0]), float(lat_a[0]), s,
                   **_xy_kwargs(ax, kwargs))


def annotate(ax: Any, textstr: str, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, *,
             frame: str | None = None, **kwargs: Any) -> Any:
    """``ax.annotate`` at a sky position.

    Absorbs matplotlib's inconsistency here: the world transform belongs in
    ``xycoords=``, **not** ``transform=``. Generalizing the ``scatter`` idiom
    to ``annotate`` silently mis-places the label; this wrapper does it right.

    The position may be ``lon, lat`` degrees, a single ``(lon, lat)`` pair, or
    a scalar :class:`~astropy.coordinates.SkyCoord`.

    Examples
    --------
    >>> sph.annotate(ax, 'M31', (10.68, 41.27))
    >>> sph.annotate(ax, 'M31', 10.68, 41.27,
    ...              xytext=(20, 20), textcoords='offset points',
    ...              arrowprops=dict(arrowstyle='->'))
    >>> sph.annotate(ax, 'M31', SkyCoord.from_name('M31'))
    """
    coords, lat = _as_point(coords, lat)
    lon_a, lat_a = _resolve(ax, coords, lat, frame, 'sph.annotate')
    kwargs.setdefault('xycoords', world_transform(ax))
    return ax.annotate(textstr, xy=(float(lon_a[0]), float(lat_a[0])),
                       **kwargs)


# --- mesh / field verbs ---------------------------------------------------

def contour(ax: Any, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, values: Any = None, *,
            frame: str | None = None, **kwargs: Any) -> Any:
    """``ax.contour`` on a sky grid."""
    lon_a, lat_a = _resolve(ax, coords, lat, frame, 'sph.contour')
    return ax.contour(lon_a, lat_a, values, **_xy_kwargs(ax, kwargs))


def contourf(ax: Any, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, values: Any = None, *,
             frame: str | None = None, **kwargs: Any) -> Any:
    """``ax.contourf`` on a sky grid."""
    lon_a, lat_a = _resolve(ax, coords, lat, frame, 'sph.contourf')
    return ax.contourf(lon_a, lat_a, values, **_xy_kwargs(ax, kwargs))


def pcolormesh(ax: Any, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, values: Any = None, *,
               frame: str | None = None, **kwargs: Any) -> Any:
    """``ax.pcolormesh`` on a sky grid."""
    lon_a, lat_a = _resolve(ax, coords, lat, frame, 'sph.pcolormesh')
    return ax.pcolormesh(lon_a, lat_a, values, **_xy_kwargs(ax, kwargs))


def tricontourf(ax: Any, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, values: Any = None, *,
                frame: str | None = None, **kwargs: Any) -> Any:
    """``ax.tricontourf`` on scattered sky points."""
    lon_a, lat_a = _resolve(ax, coords, lat, frame, 'sph.tricontourf')
    return ax.tricontourf(lon_a, lat_a, values, **_xy_kwargs(ax, kwargs))


def hist2d(ax: Any, coords: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None, *,
           frame: str | None = None, **kwargs: Any) -> Any:
    """``ax.hist2d`` of sky positions.

    .. warning::
       Rectangular lon/lat bins do **not** have equal sky area — a bin's area
       falls as cos(lat), so a raw density map is biased and increasingly so
       away from the equator. For a quantitative sky density use the
       equal-area HEALPix path
       (:func:`~skyplothelper.sources_to_healpix_plot`); use this for a quick
       look or over a small field where the distortion is negligible.

    Examples
    --------
    >>> sph.hist2d(ax, ra, dec, bins=40, cmap='sph.deepsky')
    >>> sph.hist2d(ax, catalog_skycoord, bins=40)

    >>> # for a quantitative density map, prefer the equal-area path:
    >>> sph.sources_to_healpix_plot(ra, dec, nside=64, ax=ax)
    """
    lon_a, lat_a = _resolve(ax, coords, lat, frame, 'sph.hist2d')
    return ax.hist2d(lon_a, lat_a, **_xy_kwargs(ax, kwargs))


# --- ax.sky_* methods -----------------------------------------------------

# Attached ONLY to axes skyplothelper builds. Deliberately not monkeypatched
# onto WCSAxes globally: that would leak sph behavior into every astropy user's
# axes in the same process. The functions above stay the single implementation
# — these are bound references to them, so the two spellings cannot drift.
_METHOD_VERBS = ('scatter', 'plot', 'step', 'fill', 'fill_between',
                 'errorbar', 'text', 'annotate', 'contour', 'contourf',
                 'pcolormesh', 'tricontourf', 'hist2d')


def _attach_sky_methods(ax: Any) -> Any:
    """Bind ``ax.sky_<verb>`` for each wrapper. Returns *ax* for chaining."""
    import types
    for verb in _METHOD_VERBS:
        fn = globals().get(verb)
        if fn is not None:
            setattr(ax, f'sky_{verb}', types.MethodType(fn, ax))
    return ax
