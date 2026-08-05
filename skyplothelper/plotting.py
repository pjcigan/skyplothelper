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

from .geometry._parsing import _coords_or_arrays_deg, _resolve_sky_frame

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
    'set_extent', 'set_xlim', 'set_ylim', 'zoom_to', 'set_view',
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

    Two complementary detectors are unioned, because neither alone is
    sufficient:

    * **Display space** — project the points and look for a step that spans a
      large fraction of the axes. This is ground truth at the exact-180
      boundary: the analytic wrap ``((lon - center + 180) % 360) - 180`` sends
      lon=180 to *minus* 180 while the projection renders it at *plus* 180, so a
      path stepping across exactly 180 would get its break one position off. It
      is also projection-agnostic. But its threshold is *half the full canvas
      width*, so on a pseudocylindrical frame (MOL, Robinson, …) that narrows
      toward the poles it MISSES a high-latitude crossing — the jump there spans
      only the *local* width, well under half the canvas, and the line streaks.
    * **Analytic** — a center-relative ``|delta lon| > 180`` between neighbors.
      This catches the near-pole crossings the display test misses, at any
      latitude, since it doesn't depend on the on-screen width.

    Taking the union gives correct breaks everywhere: display space fixes the
    boundary/degenerate cases, analytic fixes the narrow-frame poles. Shared by
    every sky line verb (``sph.plot`` & co.) and the geographic overlays, so a
    coastline on a flat planet map breaks at the seam the same way a great
    circle does.
    """
    if lon.size < 2:
        return lon, lat
    breaks: set[int] = set()
    # Display-space detector (see above): a seam crossing is a near-full-width
    # jump across the canvas — far larger than any real step in a sampled track.
    try:
        disp = world_transform(ax).transform(np.column_stack([lon, lat]))
        width = float(ax.get_window_extent().width)
        if np.isfinite(width) and width > 0:
            dx = np.abs(np.diff(disp[:, 0]))
            breaks.update(
                np.nonzero(np.isfinite(dx) & (dx > 0.5 * width))[0].tolist())
    except Exception:
        pass                                 # not renderable yet — analytic still runs
    # Analytic detector (see above): catches the narrow-frame near-pole crossings.
    try:
        from .wcs_frame import _get_wcs_center_lon
        center = float(_get_wcs_center_lon(ax))
        lon_norm = ((np.asarray(lon, float) - center + 180.0) % 360.0) - 180.0
        dlon = np.abs(np.diff(lon_norm))
        breaks.update(np.nonzero(np.isfinite(dlon) & (dlon > 180.0))[0].tolist())
    except Exception:
        pass
    if not breaks:
        return lon, lat
    # Insert a NaN after each crossing; matplotlib lifts the pen at NaN.
    idx = np.array(sorted(breaks))
    out_lon = np.insert(lon.astype(float), idx + 1, np.nan)
    out_lat = np.insert(lat.astype(float), idx + 1, np.nan)
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


# --- view / limits --------------------------------------------------------
#
# A WCSAxes draws in PIXEL data coordinates, so matplotlib's own
# ``ax.set_xlim`` / ``ax.set_ylim`` take pixels, not degrees — the same reason
# ``ax.scatter`` needs the world transform. These wrappers set the *view* in
# world (lon/lat) degrees instead. The subtlety is that on any curved
# projection (MOL, AIT, a SIN globe, Robinson, an oblique/rotated WCS) a lon/lat
# box is NOT a pixel rectangle: its meridians and parallels are curves, so the
# four corners don't bound it. The robust primitive therefore samples the whole
# *perimeter* of the requested box, projects every sample to pixels, and takes
# the min/max pixel box. On a rectilinear frame (CAR / Mercator) that pixel box
# IS the lon/lat box exactly; on a curved frame (MOL, AIT, a globe, oblique) it
# is the bounding box of the projected region — so the view FRAMES the region
# (showing a little extra near the corners) rather than cropping an exact box,
# the same behavior as cartopy's set_extent off PlateCarree. Off-limb
# (non-finite) samples on a globe are dropped.


def _axes_world_to_pixel(ax: Any, lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """World (deg, axes frame) → pixel/data coords, via the same ``'world'``
    transform the plotting verbs use (so it works on FITS and non-FITS frames
    alike). Off-projection points come back non-finite for the caller to drop."""
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    disp = world_transform(ax).transform(np.column_stack([lon, lat]))
    data = ax.transData.inverted().transform(disp)
    return np.asarray(data[:, 0], float), np.asarray(data[:, 1], float)


def _to_axes_frame(ax: Any, lon: Any, lat: Any, frame: str | None) -> tuple[Any, Any]:
    """Convert degree arrays given in *frame* into the axes' own frame. A no-op
    when *frame* is None or already matches the axes frame."""
    if frame is None:
        return lon, lat
    src = _resolve_sky_frame(frame)
    dst = _resolve_sky_frame(_axes_frame(ax))
    if src == dst:
        return lon, lat
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    c = SkyCoord(np.asarray(lon, float) * u.deg,
                 np.asarray(lat, float) * u.deg, frame=src).transform_to(dst)
    return c.spherical.lon.to_value(u.deg), c.spherical.lat.to_value(u.deg)


def _short_lon_pair(a: float, b: float) -> tuple[float, float]:
    """Two longitudes → ``(lon0, lon1)`` whose ``linspace`` sweeps the SHORTER
    arc between them, so a box straddling 0°/360° (or given in °W) still traces
    the small box the user meant rather than the long way around the sphere."""
    a = float(a)
    dl = ((float(b) - a + 180.0) % 360.0) - 180.0
    return a, a + dl


def _world_box_to_pixels(ax: Any, lon0: float, lon1: float, lat0: float,
                         lat1: float, *, frame: str | None = None,
                         samples: int = 181) -> tuple[float, float, float, float]:
    """Perimeter-sample the lon/lat box → bounding ``(x0, x1, y0, y1)`` in pixels."""
    lo = np.linspace(lon0, lon1, samples)
    la = np.linspace(lat0, lat1, samples)
    ones = np.ones(samples)
    edge_lon = np.concatenate([lo, lo, lon0 * ones, lon1 * ones])
    edge_lat = np.concatenate([lat0 * ones, lat1 * ones, la, la])
    edge_lon, edge_lat = _to_axes_frame(ax, edge_lon, edge_lat, frame)
    px, py = _axes_world_to_pixel(ax, edge_lon, edge_lat)
    good = np.isfinite(px) & np.isfinite(py)
    if not good.any():
        raise ValueError(
            "sph view: none of the requested lon/lat box projects to a finite "
            "pixel on this frame — is the region entirely off the visible "
            "hemisphere of a globe, or the frame not yet built?")
    return (float(px[good].min()), float(px[good].max()),
            float(py[good].min()), float(py[good].max()))


def _apply_box(ax: Any, x0: float, x1: float, y0: float,
               y1: float) -> tuple[float, float, float, float]:
    """Set the pixel view window, preserving the frame's native orientation
    (the east-left/right handedness lives in the world→display transform, not
    in the pixel limits, so we never invert here)."""
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return (x0, x1, y0, y1)


def set_extent(ax: Any, extent: Any, *, frame: str | None = None,
               lon_west: bool = False, pad: float = 0.0,
               ) -> tuple[float, float, float, float]:
    """Set the visible field of view to a lon/lat box, cartopy-style.

    The world-coordinate answer to "zoom this WCS map to a region" (a WCS axes
    otherwise takes ``set_xlim``/``set_ylim`` in *pixels*, not degrees). Frames
    the region on **any** projection by sampling the box perimeter, not just its
    corners (see the module note above): the pixel window is the lon/lat box
    exactly on a rectilinear frame (CAR/Mercator) and its bounding box on a
    curved one (MOL/AIT/globe/oblique) — so on curved frames it shows a little
    beyond the box near the corners. Returns the pixel ``(x0, x1, y0, y1)`` set.

    Parameters
    ----------
    ax : WCSAxes
    extent : sequence of 4 floats
        ``[lon_min, lon_max, lat_min, lat_max]`` in degrees. Longitudes are
        east-positive unless *lon_west* is set; a box that crosses 0°/360° is
        handled (the shorter arc between the two longitudes is used).
    frame : str, optional
        Frame the *extent* is given in (``'galactic'``, ``'icrs'``, …). Default
        ``None`` — the axes' own frame. Converted to the axes frame for you.
    lon_west : bool
        Interpret the two longitudes as **west**-longitude (°W); they are run
        through :func:`~skyplothelper.lon_west_to_east` before use. Pair with a
        frame built with ``lon_west=True`` for a fully west-labeled regional map.
    pad : float
        Margin in **degrees** added on all sides before conversion (default 0).

    Examples
    --------
    >>> sph.set_extent(ax, [-125, -66, 24, 50])           # continental US, °E
    >>> sph.set_extent(ax, [125, 66, 24, 50], lon_west=True, pad=3)
    >>> sph.set_extent(ax, [-10, 10, -5, 5], frame='galactic')  # galactic center
    """
    lon_a, lon_b, lat_a, lat_b = (float(v) for v in extent)
    if lon_west:
        from .projections.project import lon_west_to_east
        lon_a = float(lon_west_to_east(lon_a))
        lon_b = float(lon_west_to_east(lon_b))
    lat0, lat1 = min(lat_a, lat_b), max(lat_a, lat_b)
    lon0, lon1 = _short_lon_pair(lon_a, lon_b)
    if pad:
        lat0 = max(-90.0, lat0 - pad)
        lat1 = min(90.0, lat1 + pad)
        s = np.sign(lon1 - lon0) or 1.0
        lon0 -= s * pad
        lon1 += s * pad
    return _apply_box(ax, *_world_box_to_pixels(ax, lon0, lon1, lat0, lat1,
                                                frame=frame))


def set_xlim(ax: Any, lon0: float, lon1: float, *, frame: str | None = None,
             lon_west: bool = False, pad: float = 0.0) -> tuple[float, float]:
    """Set the longitude range of the view (leaving the latitude range as is).

    A convenience for the common cylindrical (plate-carrée / ``CAR``) case,
    where a longitude range maps to a fixed pixel-x range. On a curved
    projection the x-extent of a meridian varies with latitude, so this samples
    at the current center latitude — exact on ``CAR``, approximate elsewhere;
    prefer :func:`set_extent` there. Returns the pixel ``(x0, x1)`` it set.
    """
    from .wcs_frame import _get_wcs_center_lat
    if lon_west:
        from .projections.project import lon_west_to_east
        lon0 = float(lon_west_to_east(lon0))
        lon1 = float(lon_west_to_east(lon1))
    lon0, lon1 = _short_lon_pair(lon0, lon1)
    if pad:
        s = np.sign(lon1 - lon0) or 1.0
        lon0 -= s * pad
        lon1 += s * pad
    clat = float(_get_wcs_center_lat(ax))
    px, _ = _axes_world_to_pixel(ax, [lon0, lon1], [clat, clat])
    x0, x1 = float(np.nanmin(px)), float(np.nanmax(px))
    ax.set_xlim(x0, x1)
    return (x0, x1)


def set_ylim(ax: Any, lat0: float, lat1: float, *,
             pad: float = 0.0) -> tuple[float, float]:
    """Set the latitude range of the view (leaving the longitude range as is).

    The latitude companion to :func:`set_xlim`; samples at the current center
    longitude. Exact on ``CAR``, approximate on curved projections (prefer
    :func:`set_extent`). Returns the pixel ``(y0, y1)`` it set.
    """
    from .wcs_frame import _get_wcs_center_lon
    lat0, lat1 = min(lat0, lat1), max(lat0, lat1)
    if pad:
        lat0 = max(-90.0, lat0 - pad)
        lat1 = min(90.0, lat1 + pad)
    clon = float(_get_wcs_center_lon(ax))
    _, py = _axes_world_to_pixel(ax, [clon, clon], [lat0, lat1])
    y0, y1 = float(np.nanmin(py)), float(np.nanmax(py))
    ax.set_ylim(y0, y1)
    return (y0, y1)


def zoom_to(ax: Any, lon: SkyCoord | npt.ArrayLike, lat: npt.ArrayLike | None = None,
            *, pad: float = 5.0, frame: str | None = None,
            lon_west: bool = False) -> tuple[float, float, float, float]:
    """Frame the view around a set of points (autoscale-to-content).

    Fits the field of view to the bounding box of *lon*/*lat* (or a SkyCoord)
    with a *degree* margin — "show me all these things with a little room". The
    longitude bounding box is computed with wrap handling, so a cluster
    straddling 0°/360° still frames tightly. Returns the pixel box it set.

    Examples
    --------
    >>> sph.zoom_to(ax, ra, dec, pad=2)             # frame a catalog +2 deg
    >>> sph.zoom_to(ax, site_skycoords, pad=5)      # frame observatories
    >>> sph.zoom_to(ax, covis_region, pad=3)        # frame a CompoundRegion
    """
    if hasattr(lon, 'representative_point') and hasattr(lon, 'bounds'):
        # A CompoundRegion: frame its lon/lat bounding box.
        if getattr(lon, 'is_empty', False):
            raise ValueError("sph.zoom_to: the region is empty — nothing to frame.")
        lo0, lo1, la0, la1 = lon.bounds
        return set_extent(ax, [lo0, lo1, la0, la1], frame=frame, pad=pad)
    if hasattr(lon, 'transform_to'):                 # SkyCoord → axes frame
        lon_d, lat_d = _resolve(ax, lon, None, None, 'sph.zoom_to')
        box_frame = None
    else:
        lon_d = np.atleast_1d(np.asarray(lon, dtype=float))
        lat_d = np.atleast_1d(np.asarray(lat, dtype=float))
        box_frame = frame
    if lon_west:
        from .projections.project import lon_west_to_east
        lon_d = np.atleast_1d(lon_west_to_east(lon_d))
    ref = float(lon_d[0])                             # unwrap around 1st point
    lon_un = ref + (((lon_d - ref + 180.0) % 360.0) - 180.0)
    return set_extent(ax, [float(lon_un.min()), float(lon_un.max()),
                           float(np.min(lat_d)), float(np.max(lat_d))],
                      frame=box_frame, pad=pad)


def set_view(ax: Any, center: Any, fov: Any, *, frame: str | None = None,
             lon_west: bool = False, pad: float = 0.0,
             ) -> tuple[float, float, float, float]:
    """Set the view to a *center* at a chosen angular *field of view*.

    The post-construction analog of a frame's ``center`` + ``fov_deg``.
    *center* is ``(lon, lat)`` (or a lone longitude, keeping the frame's center
    latitude); *fov* is a scalar full-width in degrees or ``(dlon, dlat)``.
    Returns the pixel box it set.

    Examples
    --------
    >>> sph.set_view(ax, (-105, 35), 40)             # 40 deg wide about NM
    >>> sph.set_view(ax, (266.4, -29.0), (12, 8))    # galactic center field
    """
    if np.ndim(center) == 0:
        from .wcs_frame import _get_wcs_center_lat
        clon = float(center)
        clat = float(_get_wcs_center_lat(ax))
    else:
        clon, clat = float(center[0]), float(center[1])
    if np.ndim(fov) == 0:
        dlon = dlat = float(fov)
    else:
        dlon, dlat = float(fov[0]), float(fov[1])
    return set_extent(ax, [clon - dlon / 2.0, clon + dlon / 2.0,
                           clat - dlat / 2.0, clat + dlat / 2.0],
                      frame=frame, lon_west=lon_west, pad=pad)


# --- ax.sky_* methods -----------------------------------------------------

# Attached ONLY to axes skyplothelper builds. Deliberately not monkeypatched
# onto WCSAxes globally: that would leak sph behavior into every astropy user's
# axes in the same process. The functions above stay the single implementation
# — these are bound references to them, so the two spellings cannot drift.
_METHOD_VERBS = ('scatter', 'plot', 'step', 'fill', 'fill_between',
                 'errorbar', 'text', 'annotate', 'contour', 'contourf',
                 'pcolormesh', 'tricontourf', 'hist2d',
                 'set_extent', 'set_xlim', 'set_ylim', 'zoom_to', 'set_view')


def _attach_sky_methods(ax: Any) -> Any:
    """Bind ``ax.sky_<verb>`` for each wrapper. Returns *ax* for chaining."""
    import types
    for verb in _METHOD_VERBS:
        fn = globals().get(verb)
        if fn is not None:
            setattr(ax, f'sky_{verb}', types.MethodType(fn, ax))
    return ax
