"""Plotting helpers for orthographic globe axes.

Hemisphere-aware versions of scatter/line/pcolormesh/contour, plus
``imscatter`` family for placing icon images at sky positions.
"""

from __future__ import annotations

import warnings  # noqa: F401
from typing import TYPE_CHECKING, Any

import matplotlib.patheffects as PathEffects  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.image import imread  # noqa: F401
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

try:
    import cartopy.crs as ccrs
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

from ..geometry._parsing import _coords_or_arrays_deg
from ..wcs_frame import _get_wcs_frame_name
from .spherical import great_circle_arc, orthographic_visibility

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

# ===== imscatter family =====


def _rotate_image_ccw(image: Any, deg: float) -> npt.NDArray[Any]:
    """Rotate an RGB(A) / grayscale image array counter-clockwise by *deg*,
    expanding the canvas to fit (corners filled transparent / zero).

    Uses Pillow, which matplotlib already requires — so image rotation needs no
    extra dependency. Equivalent to
    ``scipy.ndimage.rotate(image, deg, reshape=True)`` in shape and on-screen
    direction. Float ``[0, 1]`` images round-trip through ``uint8`` (which also
    means the rotation can't overshoot the valid range, unlike a cubic spline).
    """
    from PIL import Image

    a = np.asarray(image)
    if deg % 360.0 == 0.0:
        return a
    is_float = np.issubdtype(a.dtype, np.floating)
    u = ((np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8) if is_float
         else a.astype(np.uint8))
    out = np.asarray(
        Image.fromarray(u).rotate(deg, resample=Image.Resampling.BICUBIC, expand=True))
    return out.astype(a.dtype) / 255.0 if is_float else out


def _broadcast_zoom(zoom: Any, n: int) -> npt.NDArray[np.float64]:
    """Normalize *zoom* to a length-*n* float array (a scalar broadcasts).

    Array-valued zoom is the raster counterpart of ``scatter(s=...)``: it lets
    one call size icons by a physical quantity (dish diameter, flux) instead of
    forcing the caller into a Python loop.
    """
    z = np.atleast_1d(np.asarray(zoom, dtype=float))
    if z.size == 1:
        return np.full(n, float(z[0]))
    if z.size != n:
        raise ValueError(
            'zoom must be a scalar or an array matching the length of x '
            f'and y (got {z.size} zoom values for {n} points)')
    return z


def imscatter(x: npt.ArrayLike, y: npt.ArrayLike, image: Any,
              ax: Any = None, zoom: Any = 1,
              zorder: int | None = None, autoscale: bool = True,
              **kwargs: Any) -> list[Any]:
    """
    Plot image icons at specified data coordinates on a matplotlib axis.

    Similar to ax.scatter() but renders a small image (e.g. a radio dish
    icon, a satellite, etc.) at each (x, y) position instead of a marker.

    Parameters
    ----------
    x, y : float or array-like
        Data coordinates for icon placement. Should already be in the
        axis's native coordinate system (e.g. projected pixel coords
        for cartopy axes, not raw lon/lat).
    image : str or ndarray
        Path to an image file, or a pre-loaded image array.
    ax : matplotlib Axes, optional
        Target axes. Defaults to current axes.
    zoom : float or array-like, optional
        Zoom/scale factor for the icon. Default 1. An array matching the
        length of ``x``/``y`` sizes each icon individually — the raster
        counterpart of ``ax.scatter(s=...)``, e.g. scaling dishes by their
        diameter.
    zorder : int or None, optional
        Drawing order for the icons.
    autoscale : bool, optional
        If True (default), expand the view to include the placed icons. This
        respects any ``set_xlim``/``set_ylim`` you set beforehand — the icons
        no longer override an explicit limit. Set False to leave the view
        untouched entirely.
    **kwargs
        Forwarded to each ``matplotlib.offsetbox.AnnotationBbox`` (e.g.
        ``alpha``, ``pad``, ``box_alignment``, ``clip_on``).

    Returns
    -------
    artists : list of AnnotationBbox
        The artist objects added to the axes.

    Notes
    -----
    Based on https://stackoverflow.com/a/22570069
    """

    if ax is None:
        ax = plt.gca()
    if isinstance(image, str):
        image = plt.imread(image)
    x, y = np.atleast_1d(x, y)
    zooms = _broadcast_zoom(zoom, len(x))
    artists = []
    # One OffsetImage per point: they all reference the same underlying array,
    # so this is cheap, and it lets zoom vary per icon.
    for x0, y0, z0 in zip(x, y, zooms):
        im = OffsetImage(image, zoom=float(z0))
        ab = AnnotationBbox(im, (x0, y0), xycoords='data',
                            frameon=False, zorder=zorder, **kwargs)
        artists.append(ax.add_artist(ab))
    if autoscale:
        # autoscale_view() (not autoscale()) so an explicit set_xlim/set_ylim
        # made before this call is respected — autoscale() would force
        # autoscaling back on and clobber the user's limits.
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale_view()
    return artists


def imscatter_rotated(x: npt.ArrayLike, y: npt.ArrayLike, image: Any,
                      rotations: npt.ArrayLike | None = None, ax: Any = None,
                      zoom: Any = 1, zorder: int | None = None,
                      autoscale: bool = True, *,
                      aim_at: Any = None, rest_angle: float = 0.0,
                      flip: Any = 'auto', target_coords: str = 'data',
                      coord_type: str = 'pixel', frame: str | None = None,
                      **kwargs: Any) -> list[Any]:
    """
    Plot image icons at data coordinates with per-point rotation.

    Rotation is CCW, and can be given directly (``rotations=``) or solved so
    each icon points at a target (``aim_at=``) — the raster counterpart of the
    vector markers' ``aim_at=`` (see :func:`skyplothelper.aim_angles`).

    Parameters
    ----------
    x, y : float or array-like
        Data coordinates for icon placement (in axis projection coords).
    image : str or ndarray
        Path to image file, or pre-loaded image array.
    rotations : None, float, or array-like, optional
        Rotation angle(s) in degrees (CCW). None = no rotation.
        Scalar applies the same rotation to all icons. Array must match
        length of x, y. Mutually exclusive with ``aim_at``.
    ax : matplotlib Axes, optional
        Target axes. Defaults to current axes.
    zoom : float or array-like, optional
        Zoom/scale factor for icons. Default 1. An array matching the length
        of ``x``/``y`` sizes each icon individually. See :func:`imscatter`.
    zorder : int or None, optional
        Drawing order.
    autoscale : bool, optional
        If True (default), expand the view to include the placed icons,
        respecting any ``set_xlim``/``set_ylim`` set beforehand. Set False to
        leave the view untouched. See :func:`imscatter`.
    aim_at : (x, y) or SkyCoord, optional
        Point each icon's boresight at this target. The rotation for every
        icon is solved as ``aim_angle - rest_angle``, where ``aim_angle`` is
        the on-screen direction from that icon to the target (resolved in
        display pixels, so a non-square axes doesn't skew it). Mutually
        exclusive with ``rotations``.
    rest_angle : float, optional
        The icon's **native boresight**: the direction the un-rotated image
        already points, in degrees CCW from screen-right. Default ``0``
        (pointing right). This is what ``aim_at`` rotates *away from*, so it
        must be measured for each icon; the bundled icons' values are recorded
        in ``examples/data/README.md`` (e.g. the radio dish points at 130°).
    flip : {'auto', True, False}, optional
        Mirror the icon horizontally when the target lies on its far side.
        ``'auto'`` (default) does so when ``(aim_angle - 90) * (rest_angle -
        90) < 0`` — without it an aimed icon rolls past vertical and reads
        upside-down. Mirroring sends the boresight ``rest -> 180 - rest``.
        ``True``/``False`` force the behavior. Only used with ``aim_at``.
    target_coords : {'data', 'display', 'axes', 'figure', 'world'}, optional
        How a numeric ``aim_at`` tuple is interpreted. Default ``'data'``,
        matching this function's own ``x``/``y`` (note this differs from
        :func:`~skyplothelper.aim_angles`, whose default is ``'display'``).
        A :class:`~astropy.coordinates.SkyCoord` is always a world position.
    coord_type : {'pixel', 'world'}, optional
        How ``x``/``y`` are interpreted when resolving the aim angle. Default
        ``'pixel'`` (data coords).
    frame : str or None, optional
        Celestial frame for world coords given as numeric tuples.
    **kwargs
        Forwarded to each ``matplotlib.offsetbox.AnnotationBbox``.

    Returns
    -------
    artists : list of AnnotationBbox

    Raises
    ------
    ValueError
        If both ``rotations`` and ``aim_at`` are given, if ``flip`` is
        invalid, or if a ``rotations``/``zoom`` array length doesn't match
        ``x``/``y``.

    Examples
    --------
    >>> # Point a dish icon (native boresight 130 deg) at a source
    >>> imscatter_rotated([x0], [y0], dish, aim_at=(src_x, src_y),
    ...                   rest_angle=130.0, ax=ax)
    """
    if ax is None:
        ax = plt.gca()
    if isinstance(image, str):
        image = plt.imread(image)

    if aim_at is not None and rotations is not None:
        raise ValueError(
            'pass either rotations= or aim_at=, not both — aim_at solves the '
            'rotations for you')
    if flip not in (True, False, 'auto'):
        raise ValueError(f"flip must be True, False, or 'auto', got {flip!r}")

    x, y = np.atleast_1d(x, y)
    npts = len(x)
    zooms = _broadcast_zoom(zoom, npts)
    flips = np.zeros(npts, dtype=bool)

    if aim_at is not None:
        # Lazy import: overlays already reaches into globe (ruler ->
        # globe.spherical), so keep this direction lazy to avoid an
        # import cycle at package init.
        from ..overlays.instruments import _aim_angle

        # Solved per point: each icon sits at a different place, so each has
        # its own on-screen direction to the target.
        phis = np.array([
            _aim_angle(ax, (x0, y0), aim_at, coord_type=coord_type,
                       frame=frame, target_coords=target_coords)
            for x0, y0 in zip(x, y)])
        rests = np.full(npts, float(rest_angle))
        if flip == 'auto':
            # The target is on the icon's far side when the two angles fall on
            # opposite sides of vertical; mirroring then keeps the icon leaning
            # toward the target instead of rolling past upright.
            flips = (phis - 90.0) * (rests - 90.0) < 0
        else:
            flips[:] = bool(flip)
        # A horizontal mirror reflects the boresight about the vertical.
        rests = np.where(flips, 180.0 - rests, rests)
        rotations = phis - rests
    elif rotations is None:
        rotations = np.zeros(npts, dtype=float)
    elif np.isscalar(rotations):
        rotations = np.zeros(npts, dtype=float) + rotations
    else:
        rotations = np.asarray(rotations, dtype=float)
        if len(rotations) != npts:
            raise ValueError(
                'rotations must be None, scalar, or array matching '
                'length of x and y')

    rotations = np.atleast_1d(rotations)
    artists = []
    for x0, y0, r0, z0, f0 in zip(x, y, rotations, zooms, flips):
        img = image[:, ::-1] if f0 else image
        rotated_img = _rotate_image_ccw(img, float(r0))
        im = OffsetImage(rotated_img, zoom=float(z0))
        ab = AnnotationBbox(im, (x0, y0), xycoords='data',
                            frameon=False, zorder=zorder, **kwargs)
        artists.append(ax.add_artist(ab))
    if autoscale:
        # autoscale_view() (not autoscale()) preserves an explicit
        # set_xlim/set_ylim made before this call — see imscatter().
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale_view()
    return artists


def imscatter_globe(ax: Any, coords: Any, coords_crs: Any,
                    center_longitude: float, image: Any,
                    zoom: Any = 1, zorder: int | None = None,
                    autoscale: bool = True, *, rest_angle: float = 45.0,
                    **kwargs: Any) -> tuple[
                        list[Any], list[Any]]:
    """
    Plot image icons on a cartopy globe projection with automatic
    latitude-based rotation and hemisphere-aware horizontal flipping.

    Icons are rotated so they appear to "stand upright" on the globe
    surface (tilted by 90° minus their latitude). Icons on the right
    side of the central longitude use a horizontally-flipped version
    of the image so they face the correct direction.

    .. note::
       The mirroring here is a **hemisphere** rule (which side of the central
       longitude a site falls on), not the **target** rule behind
       :func:`imscatter_rotated`'s ``flip=``. The two are unrelated despite
       both being horizontal mirrors.

    Parameters
    ----------
    ax : cartopy GeoAxes
        The globe projection axes (e.g. Orthographic).
    coords : ndarray, shape (N, 2) or (N, 3)
        Site coordinates, one row per site. For lon/lat use shape (N, 2)
        with ``coords_crs=ccrs.PlateCarree()``; for geocentric XYZ use
        shape (N, 3) with ``coords_crs=ccrs.Geocentric()``.
    coords_crs : cartopy CRS
        The coordinate reference system of the input coordinates.
    center_longitude : float
        The central longitude of the globe projection, in degrees.
    image : str or ndarray
        Path to icon image, or pre-loaded image array.
    zoom : float or array-like, optional
        Zoom/scale factor. Default 1. An array matching the number of
        ``coords`` rows sizes each site's icon individually.
    zorder : int or None, optional
        Drawing order.
    autoscale : bool, optional
        Passed to the underlying :func:`imscatter_rotated` calls. Default
        True; set False to leave the view limits untouched.
    rest_angle : float, optional
        The icon's **native boresight** — the direction the un-rotated image
        already points, in degrees CCW from screen-right. Default ``45``,
        which is the upper-right-pointing icon this function has always
        assumed. Naming it lets an icon with a different boresight (e.g. the
        bundled radio dish at 130°) stand upright without the caller
        pre-rotating the image; the bundled values are recorded in
        ``examples/data/README.md``.
    **kwargs
        Forwarded to each ``matplotlib.offsetbox.AnnotationBbox``.

    Returns
    -------
    artists_left, artists_right : lists of AnnotationBbox
        Artist objects for left-side and right-side icons.

    Notes
    -----
    Requires cartopy.

    Icons on the back side of the globe will still be plotted at invalid
    projected coordinates — a future improvement would pre-filter them
    using the projection's valid domain.
    """
    if not _HAS_CARTOPY:
        raise ImportError(
            'cartopy is required for imscatter_globe(). '
            'Install with: conda install -c conda-forge cartopy')

    if isinstance(image, str):
        img = plt.imread(image)
    else:
        img = np.copy(image)
    img_flip = np.fliplr(img)

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(
            "coords must have shape (N, 2) for lon/lat or (N, 3) for XYZ; "
            f"got {coords.shape}.")

    # Project coordinates to the axis projection and to lat/lon. cartopy's
    # transform_points takes separate component arrays, so pass the columns
    # (x, y[, z]) rather than the (N, k) array itself.
    components = coords.T
    site_axproj = ax.projection.transform_points(coords_crs, *components)
    site_latlon = ccrs.PlateCarree().transform_points(coords_crs, *components)

    # Rotation angle = 90 - latitude (so equator icons are horizontal, pole
    # icons are vertical), corrected for the icon's native boresight. The
    # historical rule assumed an upper-right-pointing icon; subtracting
    # (rest_angle - 45) generalizes it to any boresight while reproducing the
    # old behavior exactly at the rest_angle=45 default. The right-side branch
    # below still negates this (a mirror reverses the sense of rotation), which
    # stays correct for any rest_angle.
    rotations = (90.0 - site_latlon[:, 1]) - (float(rest_angle) - 45.0)
    zooms = _broadcast_zoom(zoom, coords.shape[0])

    # Split into left/right of the central meridian for the horizontal flip.
    # Test the SIGNED longitude difference wrapped to [-180, 180): a site is on
    # the left iff it lies west of center. Comparing wrapped [0, 360)
    # longitudes against the antimeridian instead fails whenever the
    # west-of-center interval crosses 0 (i.e. for any center_longitude < 180),
    # which silently sent every icon down the mirrored right-hand branch.
    dlon = (site_latlon[:, 0] - center_longitude + 180.) % 360. - 180.
    left_mask = dlon < 0.
    right_mask = ~left_mask

    left_inds = np.where(left_mask)[0]
    right_inds = np.where(right_mask)[0]

    artists_left = []
    artists_right = []

    if len(left_inds) > 0:
        left_x, left_y = site_axproj[:, :2][left_inds].T
        artists_left = imscatter_rotated(
            left_x, left_y,
            img, rotations=rotations[left_inds],
            zoom=zooms[left_inds], ax=ax, zorder=zorder,
            autoscale=autoscale, **kwargs)

    if len(right_inds) > 0:
        right_x, right_y = site_axproj[:, :2][right_inds].T
        artists_right = imscatter_rotated(
            right_x, right_y,
            img_flip, rotations=-rotations[right_inds],
            zoom=zooms[right_inds], ax=ax, zorder=zorder,
            autoscale=autoscale, **kwargs)

    return artists_left, artists_right

# =============================================================================
# Orthographic Grid & Visibility
# =============================================================================


# ===== plot_*_globe (hemisphere-aware wrappers) =====

def plot_scatter_globe(ax: Any, lons: SkyCoord | npt.ArrayLike, lats: Any = None,
                       wcs: Any = None, hemisphere_only: bool = True,
                       center_lon: float | None = None,
                       center_lat: float | None = None,
                       **kwargs: Any) -> Any:
    """
    Scatter plot on a WCSAxes globe, optionally culling back-hemisphere points.

    Parameters
    ----------
    ax : WCSAxes
        Axes created by make_globe_frame().
    lons, lats : array_like, or SkyCoord in ``lons``
        Coordinates in degrees (in the WCS native frame, typically ITRS).
        A ``SkyCoord`` array may be passed as ``lons`` instead, replacing
        both; it is converted into the axes' frame.
    wcs : WCS or None
        If None, extracted from ax.wcs.
    hemisphere_only : bool
        If True (default), cull back-hemisphere points before plotting.
        ``False`` does **not** draw the far side: the orthographic projection
        maps far-hemisphere points to NaN and matplotlib drops them, so the
        back is never rendered — ``False`` only skips the explicit pre-cull.
        To *show* the far side, reach for a decoration that carries the
        mirror-to-near-side machinery: :func:`highlight_great_circle` draws
        the full ring with its far half dashed, and :func:`plot_baselines`
        fades the mirrored back hemisphere via its ``back_hemisphere_*``
        options.
    center_lon, center_lat : float or None
        Projection center for visibility culling. If None, extracted from WCS.
    **kwargs
        Passed to ax.scatter().

    Returns
    -------
    sc : PathCollection or None
        The scatter artist, or None if no visible points.

    Examples
    --------
    >>> sph.plot_scatter_globe(ax, lons, lats, s=8, c='C1')
    >>> sph.plot_scatter_globe(ax, site_coords)      # SkyCoord array
    """
    lons, lats = _coords_or_arrays_deg(
        lons, lats, _get_wcs_frame_name(ax), 'plot_scatter_globe/plot_line_globe')

    if wcs is None:
        wcs = ax.wcs

    if hemisphere_only:
        if center_lon is None:
            center_lon = wcs.wcs.crval[0]
        if center_lat is None:
            center_lat = wcs.wcs.crval[1]
        vis = orthographic_visibility(lons, lats, center_lon, center_lat)
        # Subset any per-point array kwargs (sizes ``s``, colors ``c``, …) by
        # the SAME visibility mask — otherwise a full-catalog-length size/color
        # array mismatches the culled lons/lats and ax.scatter raises.
        npoints = lons.shape[0]
        for key in ('s', 'c', 'color', 'sizes', 'linewidths', 'edgecolors',
                    'alpha'):
            val = kwargs.get(key)
            if val is None:
                continue
            arr = np.asarray(val)
            if arr.ndim >= 1 and arr.shape[0] == npoints:
                kwargs[key] = arr[vis]
        lons, lats = np.asarray(lons[vis]), np.asarray(lats[vis])

    if len(lons) == 0:
        return None

    # Scatter in world coordinates via the WCSAxes 'world' transform.
    kwargs.setdefault('transform', ax.get_transform('world'))

    return ax.scatter(lons, lats, **kwargs)


def plot_line_globe(ax: Any, lons: SkyCoord | npt.ArrayLike, lats: Any = None,
                    wcs: Any = None, hemisphere_only: bool = True,
                    center_lon: float | None = None,
                    center_lat: float | None = None, densify: bool = True,
                    n_interp: int = 5, **kwargs: Any) -> Any:
    """
    Plot a line (e.g., path, boundary) on a WCSAxes globe, splitting at
    the hemisphere boundary to avoid cross-globe jumps.

    Parameters
    ----------
    ax : WCSAxes
        Axes created by make_globe_frame().
    lons, lats : array_like, or SkyCoord in ``lons``
        Coordinates in degrees.
        A ``SkyCoord`` array may be passed as ``lons`` instead, replacing
        both; it is converted into the axes' frame.
    wcs : WCS or None
        If None, extracted from ax.wcs.
    hemisphere_only : bool
        If True (default), NaN-mask back-hemisphere segments. ``False`` does
        not draw the far side either — the orthographic projection drops
        far-hemisphere points as NaN regardless; it only skips the explicit
        mask. To render the far side use :func:`highlight_great_circle`
        (full ring, far half dashed) or :func:`plot_baselines`
        (faded back hemisphere).
    center_lon, center_lat : float or None
        Projection center. If None, extracted from WCS.
    densify : bool
        If True, interpolate between consecutive points for smoother curves.
    n_interp : int
        Points to insert between each pair when densifying.
    **kwargs
        Passed to ax.plot().

    Returns
    -------
    lines : list of Line2D

    Examples
    --------
    >>> sph.plot_line_globe(ax, lons, lats, lw=1.2)
    >>> sph.plot_line_globe(ax, track_coords, ls='--')
    """
    lons, lats = _coords_or_arrays_deg(
        lons, lats, _get_wcs_frame_name(ax), 'plot_scatter_globe/plot_line_globe')

    if wcs is None:
        wcs = ax.wcs

    # Densify by great-circle interpolation
    if densify and len(lons) > 1:
        new_lons, new_lats = [lons[0]], [lats[0]]
        for i in range(len(lons) - 1):
            if np.isnan(lons[i]) or np.isnan(lons[i+1]):
                new_lons.append(lons[i+1])
                new_lats.append(lats[i+1])
                continue
            seg_lons, seg_lats = great_circle_arc(
                lons[i], lats[i], lons[i+1], lats[i+1], n_pts=n_interp + 2)
            new_lons.extend(seg_lons[1:])
            new_lats.extend(seg_lats[1:])
        lons = np.array(new_lons)
        lats = np.array(new_lats)

    if hemisphere_only:
        if center_lon is None:
            center_lon = wcs.wcs.crval[0]
        if center_lat is None:
            center_lat = wcs.wcs.crval[1]
        vis = orthographic_visibility(lons, lats, center_lon, center_lat)
        lons = np.where(vis, lons, np.nan)
        lats = np.where(vis, lats, np.nan)

    kwargs.setdefault('transform', ax.get_transform('world'))
    return ax.plot(lons, lats, **kwargs)


def plot_pcolormesh_globe(ax: Any, lon_grid: npt.ArrayLike,
                          lat_grid: npt.ArrayLike, data: npt.ArrayLike,
                          wcs: Any = None,
                          hemisphere_only: bool = True,
                          center_lon: float | None = None,
                          center_lat: float | None = None,
                          **kwargs: Any) -> Any:
    """
    Pseudocolor mesh plot on a WCSAxes globe.

    Parameters
    ----------
    ax : WCSAxes
        Axes created by make_globe_frame().
    lon_grid, lat_grid : 2D array_like
        Coordinate grids in degrees (e.g., from np.meshgrid).
    data : 2D array_like
        Data values.
    wcs : WCS or None
        If None, extracted from ax.wcs.
    hemisphere_only : bool
        If True, mask back-hemisphere cells with NaN.
    center_lon, center_lat : float or None
        Projection center.
    **kwargs
        Passed to ax.pcolormesh().

    Returns
    -------
    mesh : QuadMesh
    """
    lon_grid = np.asarray(lon_grid, dtype=float)
    lat_grid = np.asarray(lat_grid, dtype=float)
    data = np.asarray(data, dtype=float).copy()

    if wcs is None:
        wcs = ax.wcs

    if hemisphere_only:
        if center_lon is None:
            center_lon = wcs.wcs.crval[0]
        if center_lat is None:
            center_lat = wcs.wcs.crval[1]
        vis = orthographic_visibility(lon_grid, lat_grid, center_lon, center_lat)
        data = np.where(vis[:data.shape[0], :data.shape[1]], data, np.nan)

    kwargs.setdefault('transform', ax.get_transform('world'))
    kwargs.setdefault('shading', 'auto')
    return ax.pcolormesh(lon_grid, lat_grid, data, **kwargs)


def plot_contour_globe(ax: Any, lon_grid: npt.ArrayLike,
                       lat_grid: npt.ArrayLike, data: npt.ArrayLike,
                       wcs: Any = None,
                       hemisphere_only: bool = True,
                       center_lon: float | None = None,
                       center_lat: float | None = None, filled: bool = False,
                       **kwargs: Any) -> Any:
    """
    Contour plot on a WCSAxes globe.

    Parameters
    ----------
    ax : WCSAxes
        Axes created by make_globe_frame().
    lon_grid, lat_grid : 2D array_like
        Coordinate grids in degrees.
    data : 2D array_like
        Data values.
    filled : bool
        If True, use contourf; otherwise contour.
    **kwargs
        Passed to ax.contour() or ax.contourf().

    Returns
    -------
    cs : QuadContourSet
    """
    lon_grid = np.asarray(lon_grid, dtype=float)
    lat_grid = np.asarray(lat_grid, dtype=float)
    data = np.asarray(data, dtype=float).copy()

    if wcs is None:
        wcs = ax.wcs

    if hemisphere_only:
        if center_lon is None:
            center_lon = wcs.wcs.crval[0]
        if center_lat is None:
            center_lat = wcs.wcs.crval[1]
        vis = orthographic_visibility(lon_grid, lat_grid, center_lon, center_lat)
        data = np.where(vis, data, np.nan)

    kwargs.setdefault('transform', ax.get_transform('world'))
    func = ax.contourf if filled else ax.contour
    return func(lon_grid, lat_grid, data, **kwargs)




# ===== Internal helpers =====

def _is_globe_axes(ax: Any) -> bool:
    """
    Heuristic: return True if ``ax`` is a globe-like WCSAxes (orthographic
    SIN, zenithal ZEA, or similar) where great-circle lines may cross the
    back hemisphere.
    """
    wcs = getattr(ax, 'wcs', None)
    if wcs is None:
        return False
    try:
        ctype = wcs.wcs.ctype[0].upper()
    except Exception:
        return False
    for proj in ('SIN', 'ZEA', 'ARC', 'STG', 'AZP', 'SZP'):
        if ctype.endswith(proj):
            return True
    return False


def _wrap_fix_lons(lons: npt.ArrayLike, lats: npt.ArrayLike,
                   threshold_deg: float = 180.) -> tuple[
                       np.ndarray, np.ndarray]:
    """
    Insert NaN breaks where consecutive longitudes jump by more than
    ``threshold_deg`` (an antimeridian crossing). Returns new arrays with
    NaNs inserted, suitable for ``ax.plot(...)``.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if len(lons) < 2:
        return lons, lats
    dlon = np.diff(lons)
    jumps = np.where(np.abs(dlon) > threshold_deg)[0]
    if len(jumps) == 0:
        return lons, lats
    out_lons, out_lats = [lons[0]], [lats[0]]
    for i in range(1, len(lons)):
        if (i - 1) in jumps:
            out_lons.append(np.nan)
            out_lats.append(np.nan)
        out_lons.append(lons[i])
        out_lats.append(lats[i])
    return np.array(out_lons), np.array(out_lats)

