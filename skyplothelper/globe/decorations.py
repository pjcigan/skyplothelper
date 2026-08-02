"""Globe decorations: grids, borders, compass, scale bars.

``plot_ortho_grid`` draws a lon/lat grid on an orthographic globe;
``add_checkered_border`` draws the classic checkered globe edge;
``add_compass_rose`` draws a compass; ``add_scale_bar_*`` draw scale
bars in either Plate Carrée or orthographic coordinates.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import rcParams

from .._stroke import _stroke_path_effects
from ..constants import planet_radii
from .spherical import (
    _ortho_project,
    destination_point,
    great_circle_arc,
    lonlat_to_xyz,
    orthographic_forward,
    orthographic_visibility,
    xyz_to_lonlat,
)

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

# ===== Grid / border =====


def _resolve_globe_center(ax: Any, lon_0: float | None,
                          lat_0: float | None) -> tuple[float, float]:
    """Resolve the orthographic projection center for a globe decoration.

    On a WCSAxes (``make_globe_frame`` / ``make_planet_frame``) default
    ``lon_0`` / ``lat_0`` to the axes' own center so a decoration matches the
    map without the caller repeating it; a plain mpl axes falls back to
    ``(0, 0)``. An explicitly-passed value always wins.

    Uses the shared center accessors (``_get_wcs_center_lon`` / ``_lat``), which
    honor ``ax._sph_center_*`` — the ONLY center source on a non-FITS frame
    (Robinson & co.), where ``ax.wcs is None`` and a raw CRVAL read would
    silently fall back to ``(0, 0)``.
    """
    c_lon, c_lat = 0.0, 0.0
    # WCSAxes carry a ``.wcs`` attribute (``None`` on the non-FITS frames);
    # plain mpl axes don't — those keep the (0, 0) fallback.
    if hasattr(ax, 'wcs') or hasattr(ax, '_sph_center_lon'):
        try:
            from ..wcs_frame import _get_wcs_center_lat, _get_wcs_center_lon
            c_lon = float(_get_wcs_center_lon(ax))
            c_lat = float(_get_wcs_center_lat(ax))
        except Exception:
            pass
    return (c_lon if lon_0 is None else lon_0,
            c_lat if lat_0 is None else lat_0)


def _ortho_front_affine(ax: Any, lon_0: float, lat_0: float,
                        R: float = 1.0) -> tuple[Any, Any] | None:
    """Affine mapping ``orthographic_forward`` (x, y) → WCS pixel, or ``None``.

    A SIN/orthographic WCS is undefined beyond the limb, so far-side world
    coords → NaN through ``world_to_pixel`` and a ``get_transform('world')``
    overlay draws only the front. But orthographic-from-the-same-center makes
    the manual ``orthographic_forward`` output and the WCS pixel grid related
    by an EXACT affine (residual ~1e-13 px), so calibrate it on
    front-hemisphere sample points (where the WCS IS defined) and use it to
    blit far-side lines onto the disk in pixel/data coords. Returns
    ``(cx, cy)`` coefficient triples (``px = cx[0]·x + cx[1]·y + cx[2]``), or
    ``None`` when *ax* isn't a globe WCSAxes / calibration fails. Shared by
    :func:`plot_ortho_grid` (back-hemisphere graticule) and
    :func:`highlight_meridian_tracer`.
    """
    if not (hasattr(ax, 'wcs') and getattr(ax, 'wcs', None) is not None):
        return None
    try:
        gl, ga = np.meshgrid(np.arange(-180.0, 181.0, 30.0),
                             np.arange(-80.0, 81.0, 20.0))
        lo, la = gl.ravel(), ga.ravel()
        m = orthographic_visibility(lo, la, lon_0, lat_0)
        lo, la = lo[m], la[m]
        xy = orthographic_forward(lo, la, lon_0, lat_0, R)
        mx, my = xy[:, 0], xy[:, 1]
        pix = ax.wcs.wcs_world2pix(np.column_stack([lo, la]), 0)
        ok = (np.isfinite(pix).all(axis=1)
              & np.isfinite(mx) & np.isfinite(my))
        mx, my, pix = mx[ok], my[ok], pix[ok]
        if mx.size >= 3:
            basis = np.column_stack([mx, my, np.ones_like(mx)])
            cx = np.linalg.lstsq(basis, pix[:, 0], rcond=None)[0]
            cy = np.linalg.lstsq(basis, pix[:, 1], rcond=None)[0]
            return (cx, cy)
    except Exception:
        return None
    return None


def plot_ortho_grid(ax: Any, lon_0: float | None = None,
                    lat_0: float | None = None,
                    R: float = 1., n_pts: int = 500,
                    lon_spacing: float = 15, lat_spacing: float = 15,
                    front_color: Any = None, front_lw: float = 0.7,
                    front_ls: str = '-',
                    back_color: Any = '0.6', back_lw: float = 0.4,
                    back_ls: str = '--',
                    back_alpha: float = 0.85, show_back: bool = True,
                    circle_color: Any = 'k', circle_lw: float = 1.5,
                    equator_color: Any = None, equator_lw: float | None = None,
                    prime_meridian_color: Any = None,
                    prime_meridian_lw: float = 1.5,
                    lon_cmap: Any = None,
                    lon_cmap_range: tuple[float, float] = (0, 360),
                    lon_cmap_lw: float | None = None) -> None:
    """
    Plot a complete orthographic projection grid with front/back hemisphere
    styling, highlighted equator/prime meridian, and optional per-longitude
    colormap.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes (should have equal aspect ratio).
    lon_0, lat_0 : float or None
        Projection center in degrees (used for the front/back hemisphere
        test). ``None`` (default) auto-detects: on a WCSAxes the axes'
        own center (CRVAL) is used, so ``plot_ortho_grid(ax)`` matches a
        globe built with any ``center_LONdeg`` / ``center_LATdeg``; on a
        plain mpl axes it falls back to ``(0, 0)``.
    R : float
        Sphere radius for plot coordinates.
    n_pts : int
        Points per grid line.
    lon_spacing, lat_spacing : float
        Grid line spacing in degrees.
    front_color, front_lw, front_ls : str/float
        Style for front-hemisphere grid lines. ``front_color=None``
        (default) uses ``rcParams['grid.color']`` so the graticule looks
        like a standard plot grid; pass a color (e.g. ``'steelblue'``)
        for a highlighted front hemisphere.
    back_color, back_lw, back_ls : str/float
        Style for back-hemisphere grid lines. Defaults (``'0.6'`` mid-gray,
        ``0.4`` lw, dashed) keep the far side clearly visible but secondary to
        the front; pass lighter values to fade it further.
    back_alpha : float
        Alpha for back hemisphere lines (default ``0.85`` — mostly opaque, so
        the far graticule reads clearly).
    show_back : bool
        Whether to draw the far-hemisphere grid lines. Works on both a plain
        mpl axes and a :func:`~skyplothelper.globe.frame.make_globe_frame`
        WCSAxes: a SIN/orthographic WCS is undefined beyond the limb (far-side
        ``(lon, lat)`` → NaN through ``world_to_pixel``), so the far lines are
        instead blitted onto the disk through an affine calibrated against the
        front hemisphere (exact, since orthographic-from-the-same-center is an
        affine relation) — matching the plain-axes output.
    circle_color, circle_lw : str/float
        Style for the outer limb circle.
    equator_color : color or None
        Highlight color for the equator. None = default grid color.
    equator_lw : float or None
        Line width for equator highlight.
    prime_meridian_color : color or None
        Highlight color for the prime meridian (lon=0). ``None``
        (default) draws it in the normal grid color (no highlight);
        pass a color (e.g. ``'#33AA33'``) to emphasize it.
    prime_meridian_lw : float or None
        Line width for prime meridian highlight.
    lon_cmap : str, Colormap, or None
        Colormap applied to longitude lines. Overrides front_color for
        meridians (prime meridian highlight still takes precedence).
    lon_cmap_range : tuple (vmin, vmax)
        Normalization range for longitude colormap, in degrees.
    lon_cmap_lw : float or None
        Line width for cmap-colored lines. None = front_lw.
    """
    if lon_cmap is not None:
        if isinstance(lon_cmap, str):
            lon_cmap = plt.get_cmap(lon_cmap)
        lon_norm = mcolors.Normalize(vmin=lon_cmap_range[0],
                                      vmax=lon_cmap_range[1])
    if lon_cmap_lw is None:
        lon_cmap_lw = front_lw

    # Route through ``ax.get_transform('world')``
    # on WCSAxes (e.g. ``make_globe_frame``) so user-facing lon/lat values
    # land on the pixel grid correctly. For plain mpl axes set up as
    # orthographic [-1, 1] R units, fall back to the manual forward-projection
    # path that this helper has always used.
    is_wcs = hasattr(ax, 'wcs') and getattr(ax, 'wcs', None) is not None
    plot_transform = ax.get_transform('world') if is_wcs else ax.transData

    # Resolve the projection center used for the front/back visibility test.
    # On a WCSAxes (e.g. make_globe_frame) default to the axes' own center
    # (CRVAL) so plot_ortho_grid(ax) gets front/back right without the caller
    # repeating center_LON/center_LAT — otherwise the hemisphere mask is
    # computed for (0, 0) while the lines are drawn at the true center, which
    # paints the "front" styling onto the wrong (far) hemisphere. Plain mpl
    # axes default to (0, 0) as before.
    lon_0, lat_0 = _resolve_globe_center(ax, lon_0, lat_0)
    # Default the graticule color to the theme's grid color so the globe grid
    # looks like a standard plot; loud highlights (steelblue front, green prime
    # meridian, lon_cmap) are opt-in via the kwargs.
    if front_color is None:
        front_color = rcParams['grid.color']

    def _fwd(lon: npt.ArrayLike,
             lat: npt.ArrayLike) -> tuple[Any, Any]:
        lat_r = np.radians(lat)
        lon_r = np.radians(np.asarray(lon, dtype=float) - lon_0)
        lat0_r = np.radians(lat_0)
        x = R * np.cos(lat_r) * np.sin(lon_r)
        y = R * (np.cos(lat0_r) * np.sin(lat_r) -
                 np.sin(lat0_r) * np.cos(lat_r) * np.cos(lon_r))
        return x, y

    # A SIN/orthographic WCS is undefined beyond the limb, so far-side
    # (lon, lat) → NaN through ``world_to_pixel`` and the back-hemisphere lines
    # vanish on a WCSAxes (``make_globe_frame``). The shared affine helper maps
    # the manual orthographic output onto the WCS pixel grid (calibrated on the
    # front hemisphere, exact), so the far-side lines can be blitted onto the
    # disk in pixel/data coords. ``None`` → not a globe WCS / calibration
    # failed → the back path falls back to the world transform (prior behavior).
    _back_affine = (_ortho_front_affine(ax, lon_0, lat_0, R)
                    if (is_wcs and show_back) else None)

    def _plot_curve(lons_arr: npt.ArrayLike, lats_arr: npt.ArrayLike,
                    vis_mask: npt.NDArray[np.bool_], on_back: bool = False,
                    **kwargs: Any) -> None:
        """Plot a lat/lon curve, masking points where vis_mask is False."""
        # Far-side lines on a globe WCSAxes: route through the calibrated
        # affine onto the pixel/data grid (the WCS itself NaNs them out).
        if is_wcs and on_back and _back_affine is not None:
            mx, my = _fwd(lons_arr, lats_arr)
            cx, cy = _back_affine
            px = (cx[0] * mx + cx[1] * my + cx[2]).copy()
            py = (cy[0] * mx + cy[1] * my + cy[2]).copy()
            px[~vis_mask] = np.nan
            py[~vis_mask] = np.nan
            ax.plot(px, py, transform=ax.transData, **kwargs)
            return
        if is_wcs:
            xs = np.asarray(lons_arr, dtype=float).copy()
            ys = np.asarray(lats_arr, dtype=float).copy()
        else:
            xs, ys = _fwd(lons_arr, lats_arr)
            xs, ys = xs.copy(), ys.copy()
        xs[~vis_mask] = np.nan
        ys[~vis_mask] = np.nan
        ax.plot(xs, ys, transform=plot_transform, **kwargs)

    lons_full = np.linspace(-180, 180, n_pts)
    lats_full = np.linspace(-90, 90, n_pts)

    # --- Latitude lines ---
    for lat in np.arange(-90, 91, lat_spacing):
        lats_line = np.full_like(lons_full, lat)
        vis = orthographic_visibility(lons_full, lat, lon_0, lat_0)

        is_eq = abs(lat) < 0.01
        fc = equator_color if (is_eq and equator_color) else front_color
        flw = equator_lw if (is_eq and equator_lw) else front_lw
        _plot_curve(lons_full, lats_line, vis,
                    color=fc, lw=flw, ls=front_ls, zorder=2)

        if show_back:
            bc = equator_color if (is_eq and equator_color) else back_color
            blw = (equator_lw * 0.6) if (is_eq and equator_lw) else back_lw
            _plot_curve(lons_full, lats_line, ~vis, on_back=True,
                        color=bc, lw=blw, ls=back_ls,
                        alpha=back_alpha, zorder=1)

    # --- Longitude lines ---
    for lon in np.arange(-180, 180, lon_spacing):
        lons_line = np.full_like(lats_full, lon)
        vis = orthographic_visibility(lon, lats_full, lon_0, lat_0)

        is_pm = abs(lon % 360) < 0.01

        if is_pm and prime_meridian_color:
            fc = prime_meridian_color
            flw = prime_meridian_lw or front_lw
        elif lon_cmap is not None:
            fc = lon_cmap(lon_norm(lon % 360))
            flw = lon_cmap_lw
        else:
            fc = front_color
            flw = front_lw
        _plot_curve(lons_line, lats_full, vis,
                    color=fc, lw=flw, ls=front_ls, zorder=2)

        if show_back:
            if is_pm and prime_meridian_color:
                bc = prime_meridian_color
                blw = (prime_meridian_lw or front_lw) * 0.6
            elif lon_cmap is not None:
                bc = lon_cmap(lon_norm(lon % 360))
                blw = lon_cmap_lw * 0.6
            else:
                bc = back_color
                blw = back_lw
            _plot_curve(lons_line, lats_full, ~vis, on_back=True,
                        color=bc, lw=blw, ls=back_ls,
                        alpha=back_alpha * 0.7, zorder=1)

    # Limb circle — only needed on plain mpl axes; WCSAxes' frame_class
    # (CircularFrame for SIN globes) already draws the limb.
    if not is_wcs:
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(R * np.cos(theta), R * np.sin(theta),
                color=circle_color, lw=circle_lw, zorder=3)


def _great_circle_ring(pole_lon: float, pole_lat: float,
                       n_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """``(lons, lats)`` of the great circle whose POLE is ``(pole_lon, pole_lat)``.

    The great circle is the locus of points 90° from the pole. Build two unit
    vectors spanning the plane ⟂ to the pole and sweep a full turn through them.
    """
    pole = np.asarray(lonlat_to_xyz(pole_lon, pole_lat), dtype=float)
    pole = pole / np.linalg.norm(pole)
    # A reference axis not (near-)parallel to the pole, to seed the in-plane basis.
    ref = (np.array([0.0, 0.0, 1.0]) if abs(pole[2]) < 0.9
           else np.array([1.0, 0.0, 0.0]))
    u = np.cross(pole, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(pole, u)            # already unit (pole, u orthonormal)
    t = np.linspace(0.0, 2.0 * np.pi, n_pts)
    ring = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    lons, lats = xyz_to_lonlat(ring)
    return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)


def _resolve_great_circle_pole(
    pole: tuple[float, float] | None,
    points: Any,
    inclination: float | None,
    node: float | None,
) -> tuple[float, float]:
    """Resolve any of the great-circle specifications to a single ``(lon, lat)``
    pole — the one quantity that fully defines a great circle."""
    n_specs = sum((pole is not None, points is not None,
                   inclination is not None or node is not None))
    if n_specs != 1:
        raise ValueError(
            "specify exactly one of pole=(lon, lat), points=((lon1, lat1), "
            "(lon2, lat2)), or inclination=/node=")
    if pole is not None:
        return float(pole[0]), float(pole[1])
    if points is not None:
        (lon1, lat1), (lon2, lat2) = points
        a = np.asarray(lonlat_to_xyz(lon1, lat1), dtype=float)
        b = np.asarray(lonlat_to_xyz(lon2, lat2), dtype=float)
        normal = np.cross(a, b)
        mag = np.linalg.norm(normal)
        if mag < 1e-9:
            raise ValueError(
                "points= must be two distinct, non-antipodal points")
        plon, plat = xyz_to_lonlat(normal / mag)
        return float(np.asarray(plon)), float(np.asarray(plat))
    # inclination / ascending node (orbit-style). The orbital-plane normal sits
    # at lon = node - 90, lat = 90 - inclination (i=0 → north pole = equator).
    incl = 0.0 if inclination is None else float(inclination)
    nod = 0.0 if node is None else float(node)
    return nod - 90.0, 90.0 - incl


def _blit_front_back_curve(
    ax: Any, lons: npt.ArrayLike, lats: npt.ArrayLike, lon_0: float,
    lat_0: float, R: float, is_wcs: bool, affine: Any, color: Any, lw: float,
    front_ls: str, back_ls: str, zorder: float, **kwargs: Any,
) -> None:
    """Project a sphere-coord curve onto the globe and draw it front-solid /
    back-dashed. Shared by :func:`highlight_great_circle` (and thus the meridian
    tracer): the near half goes through the normal transform, the far half is
    blitted onto the disk via the front-calibrated affine (the same machinery
    :func:`plot_ortho_grid` uses), so a WCSAxes globe shows BOTH halves."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    vis = orthographic_visibility(lons, lats, lon_0, lat_0)
    if is_wcs and affine is not None:
        xy = orthographic_forward(lons, lats, lon_0, lat_0, R)
        cx, cy = affine
        xs = cx[0] * xy[:, 0] + cx[1] * xy[:, 1] + cx[2]
        ys = cy[0] * xy[:, 0] + cy[1] * xy[:, 1] + cy[2]
        transform = ax.transData
    elif is_wcs:
        # No affine (non-globe WCS / calibration failed): world transform —
        # only the front draws (far side NaNs out).
        xs, ys = lons, lats
        transform = ax.get_transform('world')
    else:
        xy = orthographic_forward(lons, lats, lon_0, lat_0, R)
        xs, ys = xy[:, 0], xy[:, 1]
        transform = ax.transData
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    for mask, ls in ((vis, front_ls), (~vis, back_ls)):
        cx_, cy_ = xs.copy(), ys.copy()
        cx_[~mask] = np.nan
        cy_[~mask] = np.nan
        ax.plot(cx_, cy_, color=color, lw=lw, ls=ls, zorder=zorder,
                transform=transform, **kwargs)


def highlight_great_circle(ax: Any, *,
                           pole: tuple[float, float] | None = None,
                           points: Any = None,
                           inclination: float | None = None,
                           node: float | None = None,
                           lon_0: float | None = None,
                           lat_0: float | None = None,
                           color: Any = None, lw: float = 1.5,
                           front_ls: str = '-', back_ls: str = '--',
                           R: float = 1.0, n_pts: int = 500,
                           zorder: float = 3, **kwargs: Any) -> None:
    """Trace an arbitrary great circle all the way around a globe.

    The general version of :func:`highlight_meridian_tracer` (a meridian is just
    the great circle through the poles): traces the complete ring at **matched**
    color and weight, near half solid / far half dashed (the only front/back
    difference is the linestyle). Works on a plain orthographic mpl axes and on a
    :func:`~skyplothelper.globe.frame.make_globe_frame` WCSAxes alike — on the
    WCSAxes the far half is blitted onto the disk via the shared front-calibrated
    affine (:func:`_ortho_front_affine`), so BOTH halves render.

    Specify the circle in exactly ONE of three ways:

    Parameters
    ----------
    ax : matplotlib Axes or WCSAxes
        Plain orthographic axes or a ``make_globe_frame`` globe.
    pole : (lon, lat), optional
        The great circle's pole in degrees — the circle is every point 90° from
        it (e.g. ``pole=(0, 90)`` is the equator).
    points : ((lon1, lat1), (lon2, lat2)), optional
        Two points the great circle passes through (must be distinct and
        non-antipodal).
    inclination, node : float, optional
        Orbit-style: ``inclination`` from the equator (degrees) and the
        ascending-``node`` longitude where the circle crosses the equator going
        north. ``inclination=0`` is the equator; ``inclination=90`` a polar ring.
    lon_0, lat_0 : float or None
        Projection center (degrees). ``None`` (default) auto-detects the axes'
        own center (CRVAL on a WCSAxes; ``(0, 0)`` on a plain axes).
    color : color or None
        Ring color (same front and back). ``None`` → ``rcParams['axes.edgecolor']``.
    lw : float
        Line width (same front and back).
    front_ls, back_ls : str
        Linestyles for the near (front) and far (back) halves — the *only*
        front/back distinction.
    R : float
        Sphere radius for plain-axes data coords. Default 1.
    n_pts : int
        Points around the full ring.
    zorder : float
        Draw order (default 3 — above a ``plot_ortho_grid`` graticule).
    **kwargs
        Forwarded to ``ax.plot`` for both halves.

    Examples
    --------
    >>> highlight_great_circle(ax, pole=(45, 20))                 # by pole
    >>> highlight_great_circle(ax, points=((0, 0), (90, 30)))     # through 2 pts
    >>> highlight_great_circle(ax, inclination=60, node=120)      # orbit-style
    """
    is_wcs = hasattr(ax, 'wcs') and getattr(ax, 'wcs', None) is not None
    lon_0, lat_0 = _resolve_globe_center(ax, lon_0, lat_0)
    if color is None:
        color = rcParams['axes.edgecolor']
    affine = _ortho_front_affine(ax, lon_0, lat_0, R) if is_wcs else None
    pole_lon, pole_lat = _resolve_great_circle_pole(
        pole, points, inclination, node)
    lons, lats = _great_circle_ring(pole_lon, pole_lat, n_pts)
    _blit_front_back_curve(ax, lons, lats, lon_0, lat_0, R, is_wcs, affine,
                           color, lw, front_ls, back_ls, zorder, **kwargs)


def highlight_meridian_tracer(ax: Any, meridian_lon: float = 0.0,
                              lon_0: float | None = None,
                              lat_0: float | None = None,
                              color: Any = None, lw: float = 1.5,
                              front_ls: str = '-', back_ls: str = '--',
                              R: float = 1.0, n_pts: int = 500,
                              zorder: float = 3, **kwargs: Any) -> None:
    """Highlight one meridian as a COMPLETE great circle traced around a globe.

    Unlike :func:`plot_ortho_grid`'s ``prime_meridian`` highlight — which draws
    only ``lon=0`` (half a great circle, pole-to-pole on the near face) and
    de-emphasizes its far half — this traces the full ring (the meridian *and*
    its antimeridian, pole-over-pole-over-pole) at **matched** color and weight,
    front solid / far dashed (the only front/back difference is the linestyle).

    Works on both a plain mpl axes (orthographic ``[-R, R]`` data coords) and a
    :func:`~skyplothelper.globe.frame.make_globe_frame` WCSAxes. On the WCSAxes
    the far-side world coords are undefined beyond the limb (NaN through
    ``world_to_pixel``), so the ring is projected with :func:`orthographic_forward`
    and blitted onto the disk via the shared front-calibrated affine
    (:func:`_ortho_front_affine`) in pixel/data coords — the same machinery
    :func:`plot_ortho_grid` uses for its back hemisphere.

    Parameters
    ----------
    ax : matplotlib Axes or WCSAxes
        Plain orthographic axes or a ``make_globe_frame`` globe.
    meridian_lon : float
        Longitude of the meridian to trace (degrees). Its antimeridian
        (``meridian_lon + 180``) completes the great circle.
    lon_0, lat_0 : float or None
        Projection center (degrees). ``None`` (default) auto-detects the axes'
        own center (CRVAL on a WCSAxes; ``(0, 0)`` on a plain axes).
    color : color or None
        Tracer color (same for front and back). ``None`` (default) uses
        ``rcParams['axes.edgecolor']`` so it reads as a bold, theme-aware ring.
    lw : float
        Line width (same for front and back).
    front_ls, back_ls : str
        Linestyles for the near (front) and far (back) halves. The *only*
        front/back distinction — color and width are matched.
    R : float
        Sphere radius for plain-axes data coords. Default 1.
    n_pts : int
        Points around the full ring.
    zorder : float
        Draw order (default 3 — above a ``plot_ortho_grid`` graticule).
    **kwargs
        Forwarded to ``ax.plot`` for both halves.

    See Also
    --------
    highlight_great_circle : the general version for an ARBITRARY great circle
        (by pole, two points, or inclination/node). A meridian is just the
        great circle through the poles, so this is a thin wrapper over it.
    """
    # A meridian is the great circle through both poles; its great-circle pole
    # lies on the equator, 90° away in longitude.
    highlight_great_circle(
        ax, pole=(meridian_lon + 90.0, 0.0), lon_0=lon_0, lat_0=lat_0,
        color=color, lw=lw, front_ls=front_ls, back_ls=back_ls, R=R,
        n_pts=n_pts, zorder=zorder, **kwargs)


# =============================================================================
# Cartographic Decorations (merged from globe_extras.py)
# =============================================================================



# ===== Quantity helpers =====

def _to_angular_deg(q: Any, lat: float = 0., distance: Any = None,
                    body: str = 'earth',
                    radius_km: float | None = None) -> float | None:
    """
    Convert a quantity to angular degrees, given optional physical context.

    Accepted inputs
    ---------------
    * A plain float or int — treated as degrees (for legacy API compatibility).
    * An :class:`astropy.units.Quantity` with angular units (``deg``,
      ``arcmin``, ``arcsec``, ``rad``, ...): converted directly.
    * An :class:`astropy.units.Quantity` with length units (``km``,
      ``m``, ``mi``, ``pc``, ``kpc``, ``Mpc``, ``ly``, ``AU``, ...):

      - If ``distance`` is given (as a length Quantity) — treat as a
        sky-plane separation at that distance::

            angle = length / distance  (small-angle approximation)

      - Else treat as a surface distance on a spherical body with the
        given ``body`` radius (Earth by default); convert at latitude
        ``lat`` so east-west scale accounts for cos(lat).

    Returns ``None`` when ``q`` is ``None``. The returned value is a
    plain float in degrees.
    """
    if q is None:
        return None

    try:
        import astropy.units as u
    except ImportError:
        # Without astropy, accept only plain numbers as degrees.
        return float(q)

    # Plain number → assume degrees.
    if not isinstance(q, u.Quantity):
        return float(q)

    # Angular Quantity → direct conversion.
    try:
        return float(q.to(u.deg).value)
    except u.UnitConversionError:
        pass

    # Length Quantity → need context.
    try:
        if distance is not None:
            d = distance if isinstance(distance, u.Quantity) else distance * u.pc
            # Small-angle: angle_rad = length / distance  (same length unit).
            common = q.unit
            length_val = q.to(common).value
            dist_val = d.to(common).value
            return float(np.degrees(length_val / dist_val))
        else:
            R = planet_radii.get(body.lower(), 6371.0) if radius_km is None else float(radius_km)
            length_km = float(q.to(u.km).value)
            km_per_deg = R * np.cos(np.radians(float(lat))) * np.pi / 180
            return length_km / km_per_deg
    except u.UnitConversionError as e:
        raise ValueError(
            f"Cannot convert Quantity {q} to degrees: need angular units, "
            f"or length units with `distance=` (sky plot) or `lat=`+`body=` "
            f"(planetary surface)."
        ) from e


def _format_quantity_label(q: Any, fallback_unit: str = 'km') -> str:
    """
    Render a Quantity like ``500*u.km`` as "500 km" for a scale-bar label.

    Uses integer formatting when possible, otherwise a compact float.
    If ``q`` is a plain number, falls back to ``fallback_unit`` (legacy).
    """
    try:
        import astropy.units as u
    except ImportError:
        v = float(q)
        return (f'{v:,.0f} {fallback_unit}' if v >= 10
                else f'{v:g} {fallback_unit}')

    if not isinstance(q, u.Quantity):
        v = float(q)
        return (f'{v:,.0f} {fallback_unit}' if v >= 10
                else f'{v:g} {fallback_unit}')

    v = float(q.value)
    unit_str = q.unit.to_string('generic')
    if v >= 10 and v == int(v):
        return f'{int(v):,} {unit_str}'
    if v >= 10:
        return f'{v:,.1f} {unit_str}'
    return f'{v:g} {unit_str}'




# ===== Checkered border + compass rose =====

def _is_circular_frame(ax: Any) -> bool:
    """Return True if ``ax`` is a WCSAxes whose frame outline is a circle.

    Two cases land here:

    * ``CircularFrame`` from ``skyplothelper.projections.frames`` —
      used by ``make_wcs_frame`` for zenithal projections (SIN, ARC,
      ZEA, STG, ...).
    * astropy's ``EllipticalFrame`` with square ``xlim`` / ``ylim``
      AND a square axes-bbox in display coords — used by
      ``make_globe_frame`` and by circular insets from
      ``reproject_inset_axes``. AIT / MOL also use
      ``EllipticalFrame`` but with a 2:1 span ratio, producing a true
      ellipse which is intentionally out of scope here.
      Display-bbox squareness is checked alongside data-coord
      squareness because an EllipticalFrame on a stretched axes
      (square data limits but rectangular display bbox, e.g. from
      ``aspect='auto'`` plus a non-square axes rect) still renders
      as a visual ellipse — only an actually-square display bbox
      gives a true circle.
    """
    coords = getattr(ax, 'coords', None)
    frame_obj = getattr(coords, 'frame', None) if coords is not None else None
    if frame_obj is None:
        return False
    try:
        from ..projections.frames import CircularFrame
        if isinstance(frame_obj, CircularFrame):
            return True
    except ImportError:
        pass
    try:
        from astropy.visualization.wcsaxes.frame import EllipticalFrame
        if isinstance(frame_obj, EllipticalFrame):
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            xspan = abs(xlim[1] - xlim[0])
            yspan = abs(ylim[1] - ylim[0])
            if xspan > 0 and yspan > 0:
                ratio = max(xspan, yspan) / min(xspan, yspan)
                if ratio >= 1.05:
                    return False
            else:
                return False
            try:
                bbox = ax.get_window_extent()
                w, h = float(bbox.width), float(bbox.height)
                if w <= 0 or h <= 0:
                    return False
                disp_ratio = max(w, h) / min(w, h)
                if disp_ratio >= 1.05:
                    return False
            except Exception:
                pass
            return True
    except ImportError:
        pass
    return False


def add_checkered_border(ax: Any, n_segments: int | None = None,
                         segment_spacing_deg: float | None = None,
                         segment_spacing: Any = None,
                         lat: float = 0., distance: Any = None,
                         body: str = 'earth',
                         colors: tuple[Any, Any] = ('k', 'w'),
                         width_frac: float = 0.02,
                         zorder: int = 10, edgecolor: Any = 'k',
                         edgewidth: float = 0.5,
                         frame: str = 'auto',
                         center: tuple[float, float] | None = None,
                         radius: float | None = None) -> Any:
    """
    Add a classic B&W checkered border around a map axis.

    Two layouts are supported:

    * **Rectangular** (default for plain mpl axes, WCSAxes with
      RectangularFrame): four straight edges + corner caps, sized in
      the axes' own coordinate system.
    * **Circular**: alternating arc-wedges along a circular limb. The
      natural fit for zenithal projections (SIN / ARC / ZEA / STG)
      whose frame is bounded by a circle, and for plain-mpl globe
      panels (``plot_ortho_grid``-style) where the limb is a circle
      of explicit radius.

    The layout is auto-detected by default: WCSAxes whose frame
    outline is a ``CircularFrame`` get the circular layout, everything
    else gets the rectangular one. Pass ``frame='circular'`` (with
    ``radius=`` for plain mpl) or ``frame='rectangular'`` to force.

    Supports angular *or* physical segment spacing. Pass a plain
    number (degrees), or an :class:`astropy.units.Quantity` in any
    of:

    * Angular units (``deg``, ``arcmin``, ``arcsec``, ...): used
      directly.
    * Length units on a planet surface (``km``, ``mi``, ``m``, ...):
      combined with ``lat`` and ``body`` to get degrees-east-west.
    * Length units in sky-plane projection (``pc``, ``kpc``, ``Mpc``,
      ``ly``, ``AU``, ...): combined with ``distance`` (a length
      Quantity) to get the angular size via small-angle approximation.

    Parameters
    ----------
    ax : matplotlib Axes (including WCSAxes)
        The axes to decorate. If the axes has a ``.wcs`` attribute
        with valid ``cdelt`` values, ``segment_spacing_deg`` is
        interpreted in degrees-of-sky; otherwise it's in data units.
    n_segments : int or None
        Segments per edge for the rectangular layout, or total
        segments around the limb for the circular layout. Overrides
        ``segment_spacing_deg`` and ``segment_spacing`` if given.
    segment_spacing_deg : float or None
        Desired spacing between segment breakpoints, in degrees of
        sky (WCSAxes) or data units (plain rectangular axes), or in
        degrees of azimuth around the limb (circular). Default 30°.
    segment_spacing : astropy.units.Quantity or None
        Alternative: spacing as an astropy Quantity. Takes precedence
        over ``segment_spacing_deg`` when both are given. Examples:

        * ``segment_spacing = 10 * u.deg`` — equivalent to
          ``segment_spacing_deg=10``.
        * ``segment_spacing = 500 * u.km, lat=40`` — one segment per
          500 km on an Earth map at 40° latitude.
        * ``segment_spacing = 5 * u.pc, distance=200 * u.pc`` — one
          segment per 5 pc of projected sky at a source 200 pc away.
    lat : float
        Reference latitude in degrees for Earth-km conversion.
        Ignored if ``segment_spacing`` is angular or if ``distance``
        is given.
    distance : astropy.units.Quantity or None
        Distance to a source for pc/kpc/Mpc conversion. When set,
        physical lengths in ``segment_spacing`` are treated as
        sky-plane separations at that distance.
    body : str
        Planet body name for the Earth-km conversion radius (looked
        up in :data:`planet_radii`). Default ``'earth'``. Ignored if
        ``distance`` is set.
    colors : tuple of 2 colors
        Alternating fill colors.
    width_frac : float
        Border width as a fraction of the shorter axis dimension
        (rectangular) or as a fraction of the limb radius (circular).
    zorder : int
        Z-order for the border patches.
    edgecolor, edgewidth :
        Outline styling for each checker piece.
    frame : {'auto', 'rectangular', 'circular'}
        Layout selector. ``'auto'`` (default) detects ``CircularFrame``
        on WCSAxes; everything else falls back to ``'rectangular'``.
        For plain-mpl globe panels the limb shape can't be detected
        and ``frame='circular'`` should be passed explicitly (together
        with ``radius=`` if it doesn't equal half the shorter axes
        span).
    center : tuple or None
        ``(cx, cy)`` of the circular limb in data coordinates. Only
        used by the circular layout. Defaults to the center of the
        current xlim/ylim.
    radius : float or None
        Limb radius in data coordinates. Only used by the circular
        layout. Defaults to half the shorter span of the current
        xlim/ylim, which is right for both
        :func:`make_globe_frame` (WCSAxes with CircularFrame, pixel
        coords) and the standard :func:`plot_ortho_grid` layout
        (plain mpl, R=1.0 centered at 0, 0).

    Examples
    --------
    On a main map axis (e.g. Plate Carrée, data in degrees)::

        add_checkered_border(main_ax, segment_spacing_deg=10)

    On a WCSAxes inset returned by ``reproject_inset_axes``::

        inset = reproject_inset_axes(main_ax, [0.55, 0.02, 0.43, 0.45],
                                     transform='parent',
                                     projection='TAN', center=(-112, 33),
                                     size=(18, 10))
        add_checkered_border(inset, segment_spacing_deg=5)

    Earth map with 200-km segments at mid-latitudes::

        import astropy.units as u
        add_checkered_border(main_ax, segment_spacing=200*u.km, lat=40)

    Globe panel (auto-detected when ``ax`` is a ``make_globe_frame``
    WCSAxes)::

        gx = make_globe_frame(111, center_LONdeg=0, center_LATdeg=23)
        add_checkered_border(gx, segment_spacing_deg=15)

    Plain-mpl globe panel with explicit limb::

        fig, ax = plt.subplots()
        plot_ortho_grid(ax, lon_0=0, lat_0=0, R=1.0)
        add_checkered_border(ax, frame='circular', radius=1.0,
                             segment_spacing_deg=15)
    """
    if frame == 'auto':
        layout = 'circular' if _is_circular_frame(ax) else 'rectangular'
    elif frame in ('rectangular', 'circular'):
        layout = frame
    else:
        raise ValueError(
            f"add_checkered_border: frame={frame!r} not understood — "
            f"expected one of 'auto', 'rectangular', 'circular'."
        )

    if layout == 'circular':
        return _add_checkered_border_circular(
            ax, n_segments=n_segments,
            segment_spacing_deg=segment_spacing_deg,
            segment_spacing=segment_spacing,
            lat=lat, distance=distance, body=body,
            colors=colors, width_frac=width_frac, zorder=zorder,
            edgecolor=edgecolor, edgewidth=edgewidth,
            center=center, radius=radius,
        )
    return _add_checkered_border_rectangular(
        ax, n_segments=n_segments,
        segment_spacing_deg=segment_spacing_deg,
        segment_spacing=segment_spacing,
        lat=lat, distance=distance, body=body,
        colors=colors, width_frac=width_frac, zorder=zorder,
        edgecolor=edgecolor, edgewidth=edgewidth,
    )


def _add_checkered_border_rectangular(
    ax: Any, n_segments: int | None, segment_spacing_deg: float | None,
    segment_spacing: Any, lat: float, distance: Any, body: str,
    colors: tuple[Any, Any], width_frac: float, zorder: int,
    edgecolor: Any, edgewidth: float,
) -> None:
    import matplotlib.patches as mpatches

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xspan, yspan = xlim[1] - xlim[0], ylim[1] - ylim[0]
    bw = min(xspan, yspan) * width_frac

    if n_segments is None:
        # Resolve angular spacing (degrees) from whichever arg was given.
        if segment_spacing is not None:
            sp = _to_angular_deg(segment_spacing, lat=lat,
                                  distance=distance, body=body)
        else:
            sp = segment_spacing_deg or 30

        # On WCSAxes, xlim/ylim are pixels → convert via CDELT to get
        # the actual degree-span so sp (in degrees) maps to a pixel count.
        wcs = getattr(ax, 'wcs', None)
        if wcs is not None:
            try:
                deg_per_unit_x = abs(float(wcs.wcs.cdelt[0]))
                deg_per_unit_y = abs(float(wcs.wcs.cdelt[1]))
                xspan_deg = xspan * deg_per_unit_x
                yspan_deg = yspan * deg_per_unit_y
            except Exception:
                xspan_deg, yspan_deg = xspan, yspan
        else:
            xspan_deg, yspan_deg = xspan, yspan
        n_x = max(2, round(xspan_deg / sp))
        n_y = max(2, round(yspan_deg / sp))
    else:
        n_x = n_y = n_segments

    dx, dy = xspan / n_x, yspan / n_y

    edges = []
    for i in range(n_x):
        edges.append(((xlim[0] + i*dx, ylim[0] - bw), dx, bw, colors[i % 2]))
        edges.append(((xlim[0] + i*dx, ylim[1]),      dx, bw, colors[i % 2]))
    for j in range(n_y):
        edges.append(((xlim[0] - bw, ylim[0] + j*dy), bw, dy, colors[j % 2]))
        edges.append(((xlim[1],      ylim[0] + j*dy), bw, dy, colors[j % 2]))
    for cx, cy in [(xlim[0]-bw, ylim[0]-bw), (xlim[1], ylim[0]-bw),
                   (xlim[0]-bw, ylim[1]),     (xlim[1], ylim[1])]:
        edges.append(((cx, cy), bw, bw, colors[0]))

    for (xy, w, h, fc) in edges:
        p = mpatches.Rectangle(xy, w, h, facecolor=fc, edgecolor=edgecolor,
                               linewidth=edgewidth, zorder=zorder, clip_on=False)
        ax.add_patch(p)


def _add_checkered_border_circular(
    ax: Any, n_segments: int | None, segment_spacing_deg: float | None,
    segment_spacing: Any, lat: float, distance: Any, body: str,
    colors: tuple[Any, Any], width_frac: float, zorder: int,
    edgecolor: Any, edgewidth: float,
    center: tuple[float, float] | None, radius: float | None,
) -> None:
    """Build N alternating arc-wedges along a circular limb.

    The limb represents a great circle on the sky (the horizon, 90°
    from a zenithal projection's center, or the equator on a globe).
    Walking once around the limb is one full 360° great-circle
    traverse — so an angular spacing of N degrees translates directly
    to ``360 / N`` segments, the same as for one straight edge of the
    rectangular layout but applied around the whole limb instead of
    along four sides.
    """
    import matplotlib.patches as mpatches

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if center is None:
        cx = 0.5 * (xlim[0] + xlim[1])
        cy = 0.5 * (ylim[0] + ylim[1])
    else:
        cx, cy = center
    if radius is None:
        radius = 0.5 * min(xlim[1] - xlim[0], ylim[1] - ylim[0])

    bw = radius * width_frac * 2  # match rectangular's visible weight

    if n_segments is None:
        if segment_spacing is not None:
            sp = _to_angular_deg(segment_spacing, lat=lat,
                                  distance=distance, body=body)
        else:
            sp = segment_spacing_deg or 30
        # sp is non-None here (segment_spacing was not None, or the
        # degrees fallback); _to_angular_deg's None is unreachable.
        assert sp is not None
        n_seg = max(4, int(round(360.0 / sp)))
        # Round to an even count so the alternating colors close
        # consistently at theta=0 / 360.
        if n_seg % 2:
            n_seg += 1
    else:
        n_seg = int(n_segments)
        if n_seg % 2:
            n_seg += 1

    dtheta = 360.0 / n_seg
    for i in range(n_seg):
        theta1 = i * dtheta
        theta2 = theta1 + dtheta
        wedge = mpatches.Wedge(
            (cx, cy), radius + bw, theta1, theta2, width=bw,
            facecolor=colors[i % 2],
            edgecolor=edgecolor, linewidth=edgewidth,
            zorder=zorder, clip_on=False,
        )
        ax.add_patch(wedge)


# ===== Pole rod =====

def _outgoing_limb_crossing(px: float, py: float, ex: float, ey: float,
                            cx: float, cy: float, R: float) -> tuple[float, float]:
    """Point where the segment P→E (P inside the disk, E outside) crosses the
    limb circle of radius ``R`` about ``(cx, cy)`` — i.e. the outgoing root of
    ``|P + s(E−P) − C| = R`` for ``s ∈ [0, 1]``. Returns ``P`` if it cannot be
    solved (degenerate segment / no crossing)."""
    dx, dy = ex - px, ey - py
    fx, fy = px - cx, py - cy
    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - R * R
    disc = b * b - 4.0 * a * c
    if a < 1e-12 or disc < 0.0:
        return px, py
    s = float(np.clip((-b + np.sqrt(disc)) / (2.0 * a), 0.0, 1.0))
    return px + s * dx, py + s * dy


def add_pole_rod(ax: Any, *,
                 length: float = 1.5,
                 color: Any = '#F4F0E6',
                 linewidth: float = 2.5,
                 linestyle: str = '-',
                 stroke_color: Any = '0.15',
                 stroke_lw: float = 4.0,
                 occlude_back: bool = True,
                 zorder_front: float = 10,
                 zorder_back: float = -5,
                 end_marker: str | None = None,
                 end_marker_size: float = 8,
                 end_marker_color: Any = None,
                 solid_capstyle: str = 'round',
                 **plot_kwargs: Any) -> list[Any]:
    """Draw the body's rotation-axis rod on an orthographic (SIN) globe.

    The rod is a flat colored line passing through the projected
    north and south poles of the data frame, with each tip at
    ``length`` × the body radius from the body center (in pixel
    space) — so a pole sits at ``length = 1`` and ``length > 1`` makes
    the rod protrude past it. It is the 2D counterpart to a 3D
    ``add_pole_rod`` on an mplot3d sphere:
    designed for "tilted globe" views built with
    :func:`make_globe_frame` (SIN projection) — for example, showing
    Earth's rotation axis on an ITRS plot with a ``lonpole=-23.44``
    obliquity tilt.

    Defaults are tuned for legibility on both light and dark
    backgrounds: a warm bone-white core with a thin near-black stroke
    outline (via :class:`matplotlib.patheffects.withStroke`). Pass
    ``stroke_color=None`` to disable the stroke, or override ``color``
    / ``stroke_color`` for a different palette.

    Front / back occlusion
    ----------------------
    For nonzero ``CRVAL2``, one pole is in front of the body and the
    other behind. With ``occlude_back=True`` (default), the back-pole
    half is TRIMMED at the limb — only its stub poking outside the disk
    is drawn (at ``zorder_back``), on the opposite side of the body from
    the front extension. (The inside-disk portion is behind an opaque
    body anyway; not drawing it is robust to the thin coverage gap that
    rasters/meshes leave at the limb, where the far pole sits on a tilted
    view — otherwise that sliver pokes through in front.) With
    ``occlude_back=False``, the rod is drawn as a single line crossing
    both extensions, "x-ray" style.

    Parameters
    ----------
    ax : WCSAxes
        WCSAxes with a SIN (orthographic) projection — e.g. from
        :func:`make_globe_frame`. SIN is the only zenithal projection
        where the rotation axis projects to a straight 2D line that
        can be reasoned about as a 3D rod through the body.
    length : float, optional
        Distance of each rod tip from the body center, in body radii.
        A pole is at ``length = 1``, so use ``length > 1`` for the rod
        to protrude past the poles. Default 1.5. Total rod length
        (end-to-end) is ``2 * length * R_body``.
    color : color spec, optional
        Rod core color. Default ``'#F4F0E6'`` — a warm bone white.
    linewidth : float, optional
        Rod core line width in points. Default 2.5.
    linestyle : str, optional
        Matplotlib linestyle. Default ``'-'``.
    stroke_color : color spec or None, optional
        Color of the stroke drawn behind the core. Default ``'0.15'``
        — a near-black gray that reads as a thin outline against the
        bone-white core. Set to ``None`` to disable the stroke.
    stroke_lw : float, optional
        Total stroke width in points (the core draws on top, so the
        visible stroke on each side is ``(stroke_lw - linewidth) / 2``).
        Default ``4.0`` — a 0.75 pt stroke each side at default
        ``linewidth=2.5``.
    occlude_back : bool, optional
        Whether to hide the back-pole half behind the body's texture
        (using ``zorder_back``). Default True.
    zorder_front, zorder_back : float, optional
        zorders for the front and back halves. Defaults 10 and -5 —
        the back-half zorder is below typical body textures (matplotlib
        defaults around 1-2), so the body covers the back rod's
        inside-disk portion.
    end_marker : str or None, optional
        Matplotlib marker style for the two rod ends (e.g. ``'o'``,
        ``'^'``). Default ``None`` (no markers).
    end_marker_size : float, optional
        Marker size in points. Default 8.
    end_marker_color : color spec or None, optional
        Marker core color. ``None`` (default) uses ``color``. The
        stroke inherits from ``stroke_color``.
    solid_capstyle : str, optional
        Line cap style. Default ``'round'`` — matches the
        cartopy_test_withpole.png style and hides the join at the disk
        center when ``occlude_back=True``.
    **plot_kwargs
        Additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    artists : list of Line2D
        The Line2D artists created (rod halves + optional markers).

    Raises
    ------
    TypeError
        If ``ax`` is not a WCSAxes.
    ValueError
        If the axes' projection is not SIN (orthographic).

    Examples
    --------
    Earth with obliquity tilt, rotation axis shown with default
    polished look::

        from skyplothelper import make_globe_frame, add_pole_rod
        ax = make_globe_frame(111, radesys='ITRS', center_LONdeg=30,
                              center_LATdeg=15, lonpole=-23.44)
        # ... draw Earth texture, grid, coastlines ...
        add_pole_rod(ax)

    Plain red rod, no stroke::

        add_pole_rod(ax, color='red', stroke_color=None)
    """
    wcs = getattr(ax, 'wcs', None)
    if wcs is None:
        raise TypeError(
            "add_pole_rod requires a WCSAxes (got a plain Axes with no .wcs)"
        )

    ctype = wcs.wcs.ctype[0]
    proj_code = ctype.split('-')[-1] if '-' in ctype else ctype
    if proj_code != 'SIN':
        raise ValueError(
            f"add_pole_rod requires a SIN (orthographic) projection; "
            f"got CTYPE1={ctype!r} (projection code {proj_code!r}). "
            "SIN is the only zenithal projection where the rotation "
            "axis projects to a straight 2D line; other projections "
            "would yield a curved rod or hide the back hemisphere."
        )

    lat_0 = float(wcs.wcs.crval[1])
    n_view_z = np.sin(np.radians(lat_0))  # +ve = N pole towards viewer

    # Disk center in pixel space — CRPIX is 1-indexed in FITS.
    xc = float(wcs.wcs.crpix[0]) - 1.0
    yc = float(wcs.wcs.crpix[1]) - 1.0

    # SIN projects only the front hemisphere — the back-hemisphere pole
    # would return NaN. Project only the front pole via WCS and derive
    # the back pole by reflection through the disk center (exact for
    # SIN's parallel projection: the 3D rotation axis passes through
    # the body center, so the two poles are 3D antipodes and project
    # to 2D antipodes through the projected center).
    front_lat = 90.0 if n_view_z >= 0 else -90.0
    pix_front = wcs.wcs_world2pix(np.array([[0.0, front_lat]]), 0)[0]
    x_front_pix, y_front_pix = float(pix_front[0]), float(pix_front[1])
    x_back_pix = 2.0 * xc - x_front_pix
    y_back_pix = 2.0 * yc - y_front_pix

    if front_lat == 90.0:
        xn_pix, yn_pix = x_front_pix, y_front_pix
        xs_pix, ys_pix = x_back_pix, y_back_pix
    else:
        xs_pix, ys_pix = x_front_pix, y_front_pix
        xn_pix, yn_pix = x_back_pix, y_back_pix

    # Looking straight down the rotation axis — front pole collapses to
    # the disk center, so there's no rod direction to render.
    if not np.isfinite(x_front_pix) or not np.isfinite(y_front_pix):
        return []
    if np.hypot(x_front_pix - xc, y_front_pix - yc) < 1e-9:
        return []

    # Extension endpoints: for the parallel SIN projection, scaling the
    # (pole - center) vector by ``length`` is the exact 2D image of the
    # 3D rod's tip at parameter t=length·R_body.
    xn_end = xc + length * (xn_pix - xc)
    yn_end = yc + length * (yn_pix - yc)
    xs_end = xc + length * (xs_pix - xc)
    ys_end = yc + length * (ys_pix - yc)

    # Limb radius in pixels: a point 90° from the sub-observer point projects
    # onto the limb circle. Project one along the center meridian (the far pole
    # often sits very close to the limb on a tilted view, so we need this to
    # trim the back rod exactly at the edge).
    _limb_lat = (lat_0 - 89.999 if lat_0 - 89.999 >= -90.0
                 else lat_0 + 89.999)
    _limb_pix = wcs.wcs_world2pix(
        np.array([[float(wcs.wcs.crval[0]), _limb_lat]]), 0)[0]
    R_limb = (float(np.hypot(_limb_pix[0] - xc, _limb_pix[1] - yc))
              if np.all(np.isfinite(_limb_pix)) else np.nan)

    marker_color = end_marker_color if end_marker_color is not None else color
    common_kw = dict(color=color, linewidth=linewidth, linestyle=linestyle,
                     clip_on=False, solid_capstyle=solid_capstyle)
    common_kw.update(plot_kwargs)

    artists = []

    if occlude_back and abs(n_view_z) > 1e-9:
        # Two-segment rod: the front segment is drawn pole→tip on top; the back
        # segment is TRIMMED to the part outside the limb (its visible stub).
        # Relying on a low zorder to let the body texture hide the back rod's
        # inside-disk portion is fragile: the far pole sits near the limb, and
        # rasters/meshes leave a thin coverage gap right at the edge, so that
        # inside sliver pokes through (worst on a tilted globe). Not drawing it
        # is occluder-independent — and identical for an opaque body, which is
        # what occlude_back=True means.
        if n_view_z > 0:
            front_pole_xy, front_end_xy = (xn_pix, yn_pix), (xn_end, yn_end)
            back_pole_xy, back_end_xy = (xs_pix, ys_pix), (xs_end, ys_end)
        else:
            front_pole_xy, front_end_xy = (xs_pix, ys_pix), (xs_end, ys_end)
            back_pole_xy, back_end_xy = (xn_pix, yn_pix), (xn_end, yn_end)

        if np.isfinite(R_limb):
            back_start_xy = _outgoing_limb_crossing(
                back_pole_xy[0], back_pole_xy[1],
                back_end_xy[0], back_end_xy[1], xc, yc, R_limb)
        else:
            back_start_xy = back_pole_xy
        artists.extend(ax.plot(
            [back_start_xy[0], back_end_xy[0]],
            [back_start_xy[1], back_end_xy[1]],
            zorder=zorder_back, **common_kw,
        ))
        artists.extend(ax.plot(
            [front_pole_xy[0], front_end_xy[0]],
            [front_pole_xy[1], front_end_xy[1]],
            zorder=zorder_front, **common_kw,
        ))
    else:
        # Single line spanning both extension ends.
        artists.extend(ax.plot(
            [xs_end, xn_end], [ys_end, yn_end],
            zorder=zorder_front, **common_kw,
        ))

    if end_marker is not None:
        # Each end marker takes the same zorder as the rod segment that
        # terminates at it — so markers on the back end are hidden when
        # the back rod is.
        if occlude_back and abs(n_view_z) > 1e-9:
            z_n = zorder_front if n_view_z > 0 else zorder_back
            z_s = zorder_back if n_view_z > 0 else zorder_front
        else:
            z_n = z_s = zorder_front
        for (mx, my), mz in (((xn_end, yn_end), z_n),
                             ((xs_end, ys_end), z_s)):
            artists.extend(ax.plot(
                [mx], [my], marker=end_marker, markersize=end_marker_size,
                color=marker_color, linestyle='none',
                zorder=mz, clip_on=False,
            ))

    # Stroke outline: paint each core artist on top of a wider stroke in
    # ``stroke_color``. Applied via PathEffects so the core remains
    # the topmost layer of each Line2D. Only meaningful when the stroke
    # is wider than the core line.
    if stroke_lw is not None and stroke_lw > linewidth:
        stroke_effect = _stroke_path_effects(stroke_color, stroke_lw)
        if stroke_effect is not None:
            for artist in artists:
                artist.set_path_effects(stroke_effect)

    return artists


def add_compass_rose(ax: Any, x: float = 0.95, y: float = 0.08,
                     size: float = 36, size_units: str = 'points',
                     transform: Any = None, color: Any = 'k',
                     color_alt: Any = 'white',
                     label_color: Any = 'k',
                     fontsize: float = 8, fontweight: str = 'bold',
                     zorder: int = 15,
                     style: str = 'simple', show_labels: bool = True,
                     pad: float = 2.,
                     stroke_color: Any = 'white', stroke_lw: float = 0.8) -> Any:
    """
    Add a compass rose to a plot at a fixed aspect ratio.

    The rose is drawn in *point* units inside an AnchoredOffsetbox, so it
    always appears round/symmetric regardless of the axes' aspect ratio,
    data limits, or a wide/tall figure layout. The anchor point ``(x, y)``
    is interpreted in ``transform`` coordinates (default: axes fraction).

    Parameters
    ----------
    ax : matplotlib Axes
    x, y : float
        Anchor point for the compass center, in ``transform`` coords.
        Default places it in the lower-right corner of the axes
        (axes fraction 0.95, 0.08).
    size : float
        Overall diameter of the compass rose in ``size_units``. This is
        the tip-to-tip dimension of the N/S or E/W points (labels extend
        slightly past this).
    size_units : {'points', 'inches', 'fraction'}
        How to interpret ``size``.

        * ``'points'`` (default) — diameter in typographic points (1/72 in).
          Recommended — stays a consistent visible size across figures.
        * ``'inches'`` — diameter in inches.
        * ``'fraction'`` — diameter as a fraction of the shorter axes
          dimension, evaluated at call time. (Does not auto-update on
          figure resize; use ``'points'`` for that.)
    transform : matplotlib.transforms.Transform or None
        Transform for ``(x, y)``. ``None`` → ``ax.transAxes``. Use
        ``ax.transData`` or ``ax.get_transform('world')`` (on WCSAxes)
        to anchor at a data/world position instead.
    color : matplotlib color
        Color of the compass star edges and filled points.
    color_alt : matplotlib color
        Fill color of the *hollow* (alternating) star points in the
        ``'simple'`` style — the second tone of the classic two-tone
        rose. Default ``'white'``. Has no effect on the ``'arrow'``
        style (which is single-color). Matches the ``color_alt``
        argument of :func:`add_surface_compass`.
    label_color : matplotlib color
        Color of the N/S/E/W labels.
    fontsize : int
        Font size for direction labels.
    fontweight : str
        Font weight for direction labels.
    zorder : int
        Z-order of the compass rose group.
    style : {'simple', 'arrow'}
        ``'simple'`` — four-pointed compass star with alternating
        filled/hollow points (classic cartographic style).
        ``'arrow'`` — single north-pointing arrow.
    show_labels : bool
        Whether to draw the N/S/E/W direction labels.
    pad : float
        Padding inside the anchor box in points (space between the rose
        and its bounding box).
    stroke_color : color spec or None
        Stroke color drawn behind the N/S/E/W labels (the compass-star
        polygons keep their crisp edges). Default ``'white'`` — the
        classical cartographic look: black text on a thin white stroke
        for legibility on textured backgrounds. Set to ``None`` to
        disable the stroke.
    stroke_lw : float
        Total stroke width in points. Default ``0.8`` — matches the
        ratio used by :func:`add_scale_bar_curved_parallel` at the same
        default ``fontsize=8``.

    Returns
    -------
    anchor : matplotlib.offsetbox.AnchoredOffsetbox
        The anchor artist holding the compass rose. Can be used to
        adjust placement or remove the rose later.

    Notes
    -----
    Because the compass is drawn in point units inside a DrawingArea,
    all internal distances preserve aspect ratio automatically — the
    rose stays circular whether the plot is 1:1, 4:1, or 1:4. The
    anchor point still follows ``transform`` (so a data-space anchor
    will shift with the data limits), but the rose's own geometry
    never stretches.

    Examples
    --------
    >>> add_compass_rose(ax)                          # default 36pt at lower-right
    >>> add_compass_rose(ax, x=0.05, y=0.9, size=48)  # upper-left, larger
    >>> add_compass_rose(ax, size=0.1, size_units='fraction')  # 10% of axes
    >>> add_compass_rose(ax, style='arrow', size=50)  # north arrow only
    """
    import matplotlib.patches as mpatches
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
    from matplotlib.text import Text

    if transform is None:
        transform = ax.transAxes
    fig = ax.figure

    # Convert size to points (DrawingArea's native unit).
    if size_units == 'points':
        size_pt = float(size)
    elif size_units == 'inches':
        size_pt = float(size) * 72.0
    elif size_units == 'fraction':
        bbox = ax.get_window_extent()
        size_px = size * min(bbox.width, bbox.height)
        size_pt = size_px * 72.0 / fig.dpi
    else:
        raise ValueError(
            f"Unknown size_units {size_units!r}; "
            "use 'points', 'inches', or 'fraction'.")

    # Reserve room for labels outside the tip-to-tip diameter.
    label_pad_pt = fontsize * 1.2 if show_labels else 0.0
    total = size_pt + 2 * label_pad_pt
    da = DrawingArea(total, total, 0, 0)

    label_effects = _stroke_path_effects(stroke_color, stroke_lw)

    # Center of the drawing area (in points).
    cx, cy = total / 2, total / 2
    r = size_pt / 2              # tip radius
    r_inner = r * 0.3            # inner notch

    if style == 'arrow':
        # Shaft: thin rectangle from center to 60% of r
        shaft_w = r * 0.12
        shaft_len = r * 0.65
        shaft = mpatches.Rectangle(
            (cx - shaft_w/2, cy - shaft_len * 0.25),
            shaft_w, shaft_len,
            facecolor=color, edgecolor=color, lw=0.5)
        da.add_artist(shaft)
        # Arrowhead triangle
        head = mpatches.Polygon(
            [(cx - r * 0.22, cy + shaft_len * 0.4),
             (cx + r * 0.22, cy + shaft_len * 0.4),
             (cx, cy + r)],
            closed=True, facecolor=color, edgecolor=color, lw=0.5)
        da.add_artist(head)
        if show_labels:
            label = Text(cx, cy + r + label_pad_pt * 0.1, 'N',
                         ha='center', va='bottom',
                         fontsize=fontsize, fontweight=fontweight,
                         color=label_color)
            if label_effects is not None:
                label.set_path_effects(label_effects)
            da.add_artist(label)
    elif style == 'simple':
        # Four-pointed star: N filled, S hollow, E filled, W hollow (classic look)
        star_specs = [
            # (dx_pts, dy_pts, filled?)
            ([0, -r_inner, 0, r_inner, 0], [r, 0, r*0.15, 0, r], True),       # N
            ([0, -r_inner, 0, r_inner, 0], [-r, 0, -r*0.15, 0, -r], False),   # S
            ([r, 0, r*0.15, 0, r], [0, r_inner, 0, -r_inner, 0], True),       # E
            ([-r, 0, -r*0.15, 0, -r], [0, r_inner, 0, -r_inner, 0], False),   # W
        ]
        for dx_arr, dy_arr, filled in star_specs:
            verts = list(zip([cx + dx for dx in dx_arr],
                             [cy + dy for dy in dy_arr]))
            fc = color if filled else color_alt
            poly = mpatches.Polygon(verts, closed=True,
                                    facecolor=fc, edgecolor=color, lw=0.8)
            da.add_artist(poly)

        if show_labels:
            lp = r + label_pad_pt * 0.15  # labels just outside tip
            for txt, lx, ly, ha, va in [
                ('N', cx, cy + lp, 'center', 'bottom'),
                ('S', cx, cy - lp, 'center', 'top'),
                ('E', cx + lp, cy, 'left', 'center'),
                ('W', cx - lp, cy, 'right', 'center')]:
                t = Text(lx, ly, txt, ha=ha, va=va,
                         fontsize=fontsize, fontweight=fontweight,
                         color=label_color)
                if label_effects is not None:
                    t.set_path_effects(label_effects)
                da.add_artist(t)
    else:
        raise ValueError(f"Unknown style {style!r}; use 'simple' or 'arrow'.")

    anchor = AnchoredOffsetbox(
        loc='center', child=da, pad=pad / 72.0,    # pad is in font-size units
        frameon=False,
        bbox_to_anchor=(x, y), bbox_transform=transform,
        borderpad=0,
    )
    anchor.set_zorder(zorder)
    ax.add_artist(anchor)
    return anchor


# Unit compass-rose geometry in the local tangent plane (+y = North,
# +x = East), tips at distance 1. Each blade is a kite ``[tip, ccw_side, inner
# notch, cw_side]``; it is split along its spine (tip→notch) into a CCW (dark)
# and a CW (light) half for the classic two-tone pinwheel. r_inner = 0.3.
_ROSE_KITES = [   # (name, [tip, ccw_side, notch, cw_side])
    ('N', [(0.0, 1.0), (-0.3, 0.0), (0.0, 0.15), (0.3, 0.0)]),
    ('E', [(1.0, 0.0), (0.0, 0.3), (0.15, 0.0), (0.0, -0.3)]),
    ('S', [(0.0, -1.0), (0.3, 0.0), (0.0, -0.15), (-0.3, 0.0)]),
    ('W', [(-1.0, 0.0), (0.0, -0.3), (-0.15, 0.0), (0.0, 0.3)]),
]
# A single CONNECTED north+east double-arrow whose two shafts meet in a clean
# right angle at the anchor (inner corner at the origin), so the tails join
# instead of crossing/notching. N shaft x∈[0, sw]; E shaft y∈[0, sw].
_ARROW_FRAME_VERTS = [
    (0.0, 0.0),                # inner corner (right angle at the anchor)
    (0.0, 0.42),               # up the N shaft's inner (left) edge
    (-0.13, 0.42), (0.05, 1.0), (0.23, 0.42),   # N arrowhead (left, tip, right)
    (0.10, 0.42), (0.10, 0.10),                 # N shaft outer edge → corner
    (0.42, 0.10),                               # along the E shaft's top edge
    (0.42, 0.23), (1.0, 0.05), (0.42, -0.13),   # E arrowhead (top, tip, bottom)
    (0.42, 0.0),               # E shaft's inner (bottom) edge → back to corner
]
_LABEL_BEARINGS = {'N': 0.0, 'E': 90.0, 'S': 180.0, 'W': 270.0}


def _tangent_offsets_to_lonlat(lon0: float, lat0: float,
                               offsets: Any, size_rad: float
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Map local tangent-plane offsets ``(x=east, y=north)`` (in units of
    ``size_rad`` radians) to ``(lons, lats)`` on the sphere via great-circle
    bearings/distances from the anchor."""
    lons, lats = [], []
    for x, y in offsets:
        if x == 0.0 and y == 0.0:
            lons.append(lon0)
            lats.append(lat0)
            continue
        bearing = float(np.degrees(np.arctan2(x, y)))   # +y=N→0°, +x=E→90°
        dist = float(np.hypot(x, y)) * size_rad
        dlon, dlat = destination_point(lon0, lat0, bearing, dist)
        lons.append(float(dlon))
        lats.append(float(dlat))
    return np.asarray(lons), np.asarray(lats)


def add_surface_compass(ax: Any, lon: SkyCoord | float, lat: Any = None, *,
                        size_deg: float = 8.0, style: str = 'star',
                        full: bool = False, color: Any = 'k', lw: float = 1.5,
                        center_marker: str | None = None,
                        center_size: float = 4.0, color_alt: Any = 'white',
                        show_labels: bool = True, label_color: Any = None,
                        fontsize: float = 8.0, fontweight: str = 'bold',
                        label_pad: float = 0.35, stroke_color: Any = 'white',
                        stroke_lw: float = 2.0, zorder: float = 15,
                        **kwargs: Any) -> dict[str, Any]:
    """Draw a compass ON the surface at ``(lon, lat)`` of a projected sky/globe.

    The in-projection companion to :func:`add_compass_rose` (which is a fixed
    axes-fraction overlay). This one anchors at a *world* position and is drawn
    in surface coordinates, so it follows the projection's local geometry and
    perspective — it warps with a globe (and culls on the back hemisphere),
    bends on an all-sky AIT/MOL frame, etc. Handy for marking the local N/E
    frame at a feature on a planet surface.

    Works on any WCSAxes (SIN/ZEA globes, AIT/MOL all-sky, TAN fields, …).

    Parameters
    ----------
    ax : WCSAxes
        Any sky/globe frame (``make_globe_frame`` / ``make_planet_frame`` /
        ``make_wcs_frame``).
    lon, lat : float, or SkyCoord in ``lon``
        Anchor position on the surface, in the axes' world frame (degrees).
        A scalar :class:`~astropy.coordinates.SkyCoord` may be passed as
        ``lon`` instead (converted into the axes' frame); the next positional
        argument is then read as ``size_deg``.
    size_deg : float
        Overall compass size as an angular distance along the surface
        (degrees) — the full tip-to-tip span (e.g. ``size_deg=20`` is a rose
        ~20° across; each arm reaches ``size_deg / 2`` from the anchor). Matches
        the tip-to-tip convention of :func:`add_compass_rose`'s ``size``.
    style : {'star', 'lines', 'arrow'}
        ``'star'`` (default) — a four-point compass rose whose blades are each
        split into a dark (``color``) and light (``color_alt``) half — the
        classic two-tone map rose — rendered as spherical polygons through the
        region pipeline (so it warps and clips with the projection).
        ``'lines'`` — simple geodesic arm(s) along the cardinal directions.
        ``'arrow'`` — north + east arrows (the local frame as arrows).
    full : bool
        For ``style='lines'`` only: ``False`` (default) draws just N and E
        arms (the local frame); ``True`` draws a full N/E/S/W cross. (The
        ``'star'`` rose and ``'arrow'`` have fixed geometry.)
    color : color
        Compass color (the dark blade halves / lines / edges / arrows).
    color_alt : color
        The second (light) blade half of the two-tone ``'star'`` rose.
        Default ``'white'``. Matches the ``color_alt`` argument of
        :func:`add_compass_rose`.
    lw : float
        Line width for ``'lines'`` arms and polygon edges.
    center_marker : str or None
        Marker drawn at the anchor (default ``None``); e.g. ``'o'``.
    center_size : float
        Anchor marker size (points).
    show_labels : bool
        Whether to draw the cardinal labels. A label whose position projects
        off the visible surface (back hemisphere) is skipped.
    label_color : color or None
        Label color. ``None`` → ``color``.
    fontsize, fontweight : label text style.
    label_pad : float
        Label offset beyond the tip, as a fraction of ``size_deg``.
    stroke_color, stroke_lw : legibility-outline stroke behind the rose / arms
        AND the labels — an exterior outline to lift the compass off a busy
        surface (independent of the two-tone ``color`` / ``color_alt`` fill).
        ``stroke_color=None`` disables it.
    zorder : float
        Draw order.
    **kwargs
        Forwarded to the underlying renderer (``add_spherical_polygon`` for
        ``'star'``/``'arrow'``; ``ax.plot`` for ``'lines'``).

    Returns
    -------
    dict
        ``{'shapes': [...], 'labels': [...Text...], 'center': artist|None}`` —
        ``shapes`` holds the rose/arm artists.

    Examples
    --------
    >>> add_surface_compass(ax, 30, 15)                      # two-tone rose
    >>> add_surface_compass(ax, 30, 15, style='lines', full=True)
    >>> add_surface_compass(ax, 30, 15, style='arrow', size_deg=10)  # N+E arrows
    """
    if style not in ('star', 'lines', 'arrow'):
        raise ValueError(
            f"style must be 'star', 'lines', or 'arrow', got {style!r}")
    if not (hasattr(ax, 'get_transform') and hasattr(ax, 'wcs')
            and getattr(ax, 'wcs', None) is not None):
        raise TypeError(
            "add_surface_compass requires a WCSAxes (a sky/globe frame)")
    # A SkyCoord anchor occupies the `lon` slot; `lat` then holds the next
    # positional (size_deg), matching the convention used by the shape helpers.
    if hasattr(lon, 'transform_to'):
        from ..geometry._parsing import _coords_to_frame_deg
        from ..wcs_frame import _get_wcs_frame_name
        if lat is not None:
            size_deg = lat
        lon, lat = _coords_to_frame_deg(lon, _get_wcs_frame_name(ax))
    if label_color is None:
        label_color = color
    label_pe = _stroke_path_effects(stroke_color, stroke_lw)
    # size_deg is the full tip-to-tip span; the unit geometry has tips at
    # radius 1, so the arm radius is half of it.
    size_rad = np.radians(size_deg) * 0.5
    world = ax.get_transform('world')

    out: dict[str, Any] = {'shapes': [], 'labels': [], 'center': None}

    if center_marker is not None:
        out['center'] = ax.scatter(
            [lon], [lat], s=center_size ** 2, marker=center_marker,
            color=color, zorder=zorder, transform=world, clip_on=False)

    def _emit_polygon(verts: Any, fc: Any) -> None:
        plons, plats = _tangent_offsets_to_lonlat(lon, lat, verts, size_rad)
        poly_kw: dict[str, Any] = dict(facecolor=fc, edgecolor=color, lw=lw,
                                       zorder=zorder)
        # The legibility stroke backs the rose/arms too, not only the labels —
        # a two-tone rose still needs an exterior outline to lift off a busy
        # surface (e.g. ocean/ice). For the 'star' style this also thinly
        # outlines each blade's inner spine, which reads fine at typical sizes.
        if label_pe is not None:
            poly_kw['path_effects'] = label_pe
        poly_kw.update(kwargs)
        shapes = _add_compass_polygon(ax, plons, plats, poly_kw)
        out['shapes'].extend(shapes if isinstance(shapes, (list, tuple))
                             else [shapes])

    if style == 'star':
        # Each blade [tip, ccw_side, notch, cw_side] is split along its spine
        # into a dark (CCW) and light (CW) half — the two-tone pinwheel.
        for _name, (tip, ccw, notch, cw) in _ROSE_KITES:
            _emit_polygon([tip, ccw, notch], color)        # dark half
            _emit_polygon([tip, notch, cw], color_alt)   # light half
        label_dirs = _LABEL_BEARINGS
    elif style == 'arrow':
        # One connected N+E frame so the shafts join at a clean right angle.
        _emit_polygon(_ARROW_FRAME_VERTS, color)
        label_dirs = {'N': 0.0, 'E': 90.0}
    else:  # 'lines'
        dirs = ({'N': 0.0, 'E': 90.0} if not full else dict(_LABEL_BEARINGS))
        for bearing in dirs.values():
            end_lon, end_lat = destination_point(lon, lat, bearing, size_rad)
            arc_lon, arc_lat = great_circle_arc(lon, lat, end_lon, end_lat,
                                                n_pts=20)
            line_kw: dict[str, Any] = dict(
                color=color, lw=lw, zorder=zorder, transform=world,
                clip_on=False)
            if label_pe is not None:
                line_kw['path_effects'] = label_pe
            line_kw.update(kwargs)
            line = ax.plot(np.asarray(arc_lon, dtype=float),
                           np.asarray(arc_lat, dtype=float), **line_kw)
            out['shapes'].extend(line)
        label_dirs = dirs

    if show_labels:
        for name, bearing in label_dirs.items():
            lab_lon, lab_lat = destination_point(
                lon, lat, bearing, size_rad * (1.0 + label_pad))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                px, py = ax.wcs.world_to_pixel_values(lab_lon, lab_lat)
            # Skip a label that projects off the visible surface (back
            # hemisphere) — matplotlib would warn on a non-finite position.
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            txt = ax.text(
                lab_lon, lab_lat, name, color=label_color, fontsize=fontsize,
                fontweight=fontweight, ha='center', va='center',
                zorder=zorder + 1, transform=world, clip_on=False)
            if label_pe is not None:
                txt.set_path_effects(label_pe)
            out['labels'].append(txt)

    return out


def _add_compass_polygon(ax: Any, lons: np.ndarray, lats: np.ndarray,
                         poly_kw: dict[str, Any]) -> Any:
    """Render one compass-rose facet as a spherical polygon (region pipeline),
    so it warps and clips with the projection."""
    from ..geometry.shapes import add_spherical_polygon
    return add_spherical_polygon(ax, lons, lats, **poly_kw)


# ===== Scale bar helpers =====

def _nice_scale_length(target_km: float) -> Any:
    """Round to a 'nice' value (1, 2, 5 × 10^n)."""
    exp = np.floor(np.log10(target_km))
    base = target_km / (10 ** exp)
    nice = 1. if base < 1.5 else (2. if base < 3.5 else (5. if base < 7.5 else 10.))
    return nice * (10 ** exp)


def _compute_checker_segments(
    length: float, n_segments: int | None = None,
    segment_km: float | None = None, default_n: int = 4,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Divide a scale bar of total length ``length`` into checkered segments.

    Handles three modes:
      * ``segment_km`` given: fixed segment length; last segment may be
        partial (e.g. 2.5 segments of 10 km = 25 km total).
      * ``n_segments`` given: equal segments totaling ``length``.
      * neither given: ``default_n`` equal segments.

    Returns
    -------
    breakpoints : ndarray, shape (M+1,)
        Cumulative lengths from 0 to ``length``, where M is the number
        of drawn segments (possibly with a fractional last segment).
    widths : ndarray, shape (M,)
        Width of each segment.
    """
    if segment_km is not None:
        # Fixed segment length, allow fractional final segment.
        if segment_km <= 0:
            raise ValueError("segment_km must be > 0")
        n_full = int(np.floor(length / segment_km + 1e-9))
        remainder = length - n_full * segment_km
        widths_list = [segment_km] * n_full
        if remainder > 1e-6 * length:
            widths_list.append(remainder)
        if not widths_list:  # length < segment_km
            widths_list = [length]
    elif n_segments is not None:
        if n_segments < 1:
            raise ValueError("n_segments must be >= 1")
        widths_list = [length / n_segments] * int(n_segments)
    else:
        widths_list = [length / default_n] * default_n

    widths = np.array(widths_list, dtype=float)
    breakpoints = np.concatenate(([0.], np.cumsum(widths)))
    return breakpoints, widths




# ===== Scale bars (cylindrical + curved-parallel + dispatcher) =====

def _sublabel_color(ax: Any, override: Any = None,
                    light: str = '0.35') -> Any:
    """Color for a scale bar's *secondary* text (sub-tick values, callouts).

    These sit deliberately quieter than the bar's main label — smaller AND
    lighter — so they are not resolved to the primary ink, which would
    flatten that hierarchy. Only the dark case was broken: a light-theme
    gray like ``'0.35'`` disappears on a dark canvas.
    """
    if override is not None:
        return override
    from ..style import muted_ink
    return muted_ink(ax, light=light)


def add_scale_bar_cylindrical(ax: Any, lat: float = 0.,
                               lon: float | None = None,
                               length_km: float | None = None,
                               body: str = 'earth',
                               position: str = 'lower-right',
                               pad_frac: float = 0.05, color: Any = 'k',
                               colors: Any = None, sublabel_color: Any = None,
                               fontsize: float = 8, zorder: int = 15,
                               style: str = 'plain',
                               n_segments: int | None = None,
                               segment_km: float | None = None,
                               edgecolor: Any = None,
                               edgewidth: float = 0.6,
                               length: Any = None, segment: Any = None,
                               distance: Any = None,
                               stroke_color: Any = 'white',
                               stroke_lw: float = 0.8) -> None:
    """
    Add a distance scale bar to a cylindrical-family projection map.

    Works on any projection whose X axis is linear in longitude:
    Plate Carrée (CAR), Mercator (MER), Cylindrical Equal-area (CEA),
    Cylindrical Perspective (CYP), and the corresponding cartopy
    ``PlateCarree`` (axes data units in degrees). Locally also a
    reasonable approximation on near-tangent-plane sky inset views.

    Supports Earth km (default) *and* sky physical units (pc, kpc, ...)
    when a ``distance`` is provided. Lengths can be passed either as plain
    numbers (legacy ``length_km`` / ``segment_km``, always km) or as astropy
    Quantities (new ``length`` / ``segment``, any unit).

    Parameters
    ----------
    ax : matplotlib Axes
    lat : float
        Latitude (degrees) at which to evaluate the scale. The bar will be
        drawn at this latitude by default. Ignored for sky mode
        (``distance`` given).
    lon : float or None
        Longitude (degrees) to center the bar on. If ``None``, position is
        derived from ``position``.
    length_km : float or None
        Total bar length in km (legacy API; Earth-km only). Auto-computed
        from ``position`` if ``None`` and ``length`` is also ``None``.
    length : astropy.units.Quantity or None
        Bar length as an astropy Quantity. Takes precedence over
        ``length_km``. Accepts angular units (used directly) or length
        units (km with Earth context, or pc / kpc / ... with ``distance``
        given for sky plots).
    segment_km : float or None
        Length (km) of each checker segment (legacy). Used only when
        ``segment`` is not given.
    segment : astropy.units.Quantity or None
        Segment length as a Quantity. Takes precedence over
        ``segment_km``. Same unit rules as ``length``.
    distance : astropy.units.Quantity or None
        Distance to a source for sky-mode conversion (length ↔ angular).
        When given, ``length`` / ``segment`` length-units are interpreted
        as sky-plane separations at this distance. The bar label uses the
        same unit the user passed.
    body : str
        Planet body for radius lookup (Earth-km mode). Default ``'earth'``.
        Ignored when ``distance`` is given.
    position : str
        ``'lower-right'`` / ``'lower-left'`` / ``'upper-right'`` /
        ``'upper-left'``. Ignored if ``lon`` is given.
    pad_frac, color, colors, fontsize, zorder, style, n_segments, edgecolor, edgewidth :
        Styling. ``style='plain'`` for a single filled bar,
        ``'checkered'`` for surveyor-rod style; ``colors=(c1, c2)`` sets
        alternating colors.
    stroke_color : color spec or None
        Stroke color drawn behind the bar's text labels (main length
        label, tick labels, latitude callout) for legibility on
        textured backgrounds. Default ``'white'`` — the classical
        cartographic convention (black text on a thin white stroke).
        Set to ``None`` to disable.
    stroke_lw : float
        Total stroke width in points. Default ``0.8`` — matches the
        ratio used by :func:`add_scale_bar_curved_parallel`.

    Examples
    --------
    Legacy Earth-km (unchanged)::

        add_scale_bar_cylindrical(ax, lat=40, length_km=500)
        add_scale_bar_cylindrical(ax, lat=40, length_km=500,
                                   style='checkered', segment_km=100)

    Quantity-based Earth km (equivalent)::

        import astropy.units as u
        add_scale_bar_cylindrical(ax, lat=40, length=500*u.km,
                                   style='checkered', segment=100*u.km)

    Sky plot with known distance (pc)::

        add_scale_bar_cylindrical(
            inset, length=10*u.pc, distance=150*u.pc,
            style='checkered', segment=2*u.pc,
            position='lower-right')

    Mixed: length in arcmin::

        add_scale_bar_cylindrical(ax, length=30*u.arcmin)
    """
    import matplotlib.patches as mpatches

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xspan, yspan = xlim[1] - xlim[0], ylim[1] - ylim[0]

    # Detect whether the axes uses pixel-space data coords (WCSAxes) or
    # degree-space (plain matplotlib axes with lat/lon axis limits).
    # Everything below is computed in degrees; at draw time we scale by
    # the appropriate factor to the axes' native data units.
    wcs = getattr(ax, 'wcs', None)
    if wcs is not None:
        try:
            units_per_deg_x = 1.0 / abs(float(wcs.wcs.cdelt[0]))
        except Exception:
            units_per_deg_x = 1.0
    else:
        units_per_deg_x = 1.0

    # Detect Quantity-mode arguments.
    length_is_quantity = False
    seg_is_quantity = False
    try:
        import astropy.units as u
        if length is not None and isinstance(length, u.Quantity):
            length_is_quantity = True
        if segment is not None and isinstance(segment, u.Quantity):
            seg_is_quantity = True
    except ImportError:
        u = None

    if length_is_quantity:
        bar_deg = _to_angular_deg(length, lat=lat, distance=distance, body=body)
        label = _format_quantity_label(length)
        seg_deg = (_to_angular_deg(segment, lat=lat, distance=distance,
                                    body=body)
                   if seg_is_quantity else None)
        km_per_deg = None  # sentinel — legacy km path not used
    else:
        # Legacy path: length_km in km on the given body.
        R = planet_radii.get(body.lower(), 6371.0)
        km_per_deg = R * np.cos(np.radians(lat)) * np.pi / 180
        if length_km is None:
            xspan_deg = xspan / units_per_deg_x  # correct for pixel xlim
            length_km = _nice_scale_length(km_per_deg * xspan_deg * 0.15)
        bar_deg = length_km / km_per_deg
        label = (f'{length_km:,.0f} km' if length_km >= 10
                 else f'{length_km:g} km')
        seg_deg = None

    # bar_deg is a concrete float past this point: in the quantity path
    # `length` is a Quantity (so _to_angular_deg never returns None), and
    # the legacy path computes it directly. The Optional is just the
    # static type of _to_angular_deg's None-on-None-input contract.
    assert bar_deg is not None

    bar_h = yspan * 0.008  # fraction of yspan — already in axes units
    xpad, ypad = xspan * pad_frac, yspan * pad_frac
    # Convert bar_deg and seg_deg to AXES-NATIVE units (pixels on WCSAxes,
    # degrees on plain axes).
    bar_x = bar_deg * units_per_deg_x
    bx = (xlim[1] - xpad - bar_x) if 'right' in position else (xlim[0] + xpad)
    by = (ylim[0] + ypad) if 'lower' in position else (ylim[1] - ypad - bar_h)
    if lon is not None:
        # `lon` is given in degrees of world coord. On a WCSAxes we need
        # to convert to pixel coords via wcs_world2pix at a representative
        # latitude (use `lat` or CRVAL2).
        if wcs is not None:
            try:
                world = np.array([[float(lon),
                                    float(lat) if abs(lat) > 1e-9
                                    else float(wcs.wcs.crval[1])]])
                px = wcs.wcs_world2pix(world, 0)[0, 0]
                bx = px - bar_x / 2
            except Exception:
                bx = lon - bar_x / 2
        else:
            bx = lon - bar_x / 2

    if edgecolor is None:
        edgecolor = color

    # Pre-compute checker breakpoints/widths in DEGREES for both paths.
    # Empty (not None) when not checkered: downstream ``len(...) > 2``
    # guards then read False just as the prior None-sentinel did.
    breakpoints_deg: list[float] = []
    widths_deg: list[float] = []
    if style == 'checkered':
        if colors is None:
            colors = (color, 'white')
        if length_is_quantity:
            if seg_deg is not None:
                n_full = int(bar_deg // seg_deg)
                remainder = bar_deg - n_full * seg_deg
                widths_deg = [seg_deg] * n_full
                if remainder > 1e-12:
                    widths_deg.append(remainder)
                if not widths_deg:
                    widths_deg = [bar_deg]
            else:
                n = n_segments if n_segments is not None else 4
                widths_deg = [bar_deg / n] * n
        else:
            # Legacy path: both length_km and km_per_deg are concrete
            # floats here (set in the non-quantity branch above).
            assert length_km is not None and km_per_deg is not None
            breakpoints_km, widths_km = _compute_checker_segments(
                length_km, n_segments=n_segments, segment_km=segment_km)
            widths_deg = [w / km_per_deg for w in widths_km]
        breakpoints_deg = [0.0]
        for w in widths_deg:
            breakpoints_deg.append(breakpoints_deg[-1] + w)

    # Draw the bar.
    if style == 'plain':
        ax.add_patch(mpatches.Rectangle(
            (bx, by), bar_x, bar_h, facecolor=color, edgecolor=color,
            zorder=zorder, clip_on=False))
    elif style == 'checkered':
        for i, w_deg in enumerate(widths_deg):
            seg_x0 = bx + breakpoints_deg[i] * units_per_deg_x
            w_units = w_deg * units_per_deg_x
            fc = colors[i % 2]
            ax.add_patch(mpatches.Rectangle(
                (seg_x0, by), w_units, bar_h,
                facecolor=fc, edgecolor=edgecolor, linewidth=edgewidth,
                zorder=zorder, clip_on=False))
        ax.add_patch(mpatches.Rectangle(
            (bx, by), bar_x, bar_h,
            facecolor='none', edgecolor=edgecolor, linewidth=edgewidth,
            zorder=zorder + 0.1, clip_on=False))
    else:
        raise ValueError(f"Unknown style {style!r}; use 'plain' or 'checkered'.")

    # End ticks.
    th = bar_h * 3
    for tx in [bx, bx + bar_x]:
        ax.plot([tx, tx], [by - th/2, by + th/2 + bar_h],
                color=color, lw=0.8, zorder=zorder, clip_on=False)

    label_effects = _stroke_path_effects(stroke_color, stroke_lw)

    # Main length label above the bar.
    ax.annotate(label, xy=(bx + bar_x/2, by + bar_h), xytext=(0, 3),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=fontsize, color=color, zorder=zorder, clip_on=False,
                path_effects=label_effects)

    # Sub-segment breakpoint labels (only meaningful when user specified
    # a fixed segment size and there are more than 2 segments).
    show_subticks = (style == 'checkered' and len(widths_deg) > 2
                      and ((seg_is_quantity and segment is not None)
                           or segment_km is not None))
    if show_subticks:
        if seg_is_quantity:
            seg_val = float(segment.value)
            for i in range(1, len(widths_deg)):
                cum_val = seg_val * i
                tx = bx + breakpoints_deg[i] * units_per_deg_x
                tick_label = (f'{int(cum_val)}'
                              if cum_val == int(cum_val) else f'{cum_val:g}')
                ax.annotate(tick_label, xy=(tx, by), xytext=(0, -3),
                            textcoords='offset points', ha='center', va='top',
                            fontsize=fontsize - 2,
                            color=_sublabel_color(ax, sublabel_color),
                            zorder=zorder, clip_on=False,
                            path_effects=label_effects)
        else:
            # Reached only via show_subticks, whose non-quantity branch
            # requires segment_km is not None.
            assert segment_km is not None
            for i in range(1, len(widths_deg)):
                cum_km = segment_km * i
                tx = bx + breakpoints_deg[i] * units_per_deg_x
                tick_label = (f'{int(cum_km)}'
                              if cum_km == int(cum_km) else f'{cum_km:g}')
                ax.annotate(tick_label, xy=(tx, by), xytext=(0, -3),
                            textcoords='offset points', ha='center', va='top',
                            fontsize=fontsize - 2,
                            color=_sublabel_color(ax, sublabel_color),
                            zorder=zorder, clip_on=False,
                            path_effects=label_effects)

    # Latitude callout — only for Earth-km mode (non-trivial lat, no distance).
    if not length_is_quantity and abs(lat) > 0.01 and distance is None:
        y_offset_pts = -4 - (fontsize if show_subticks else 0)
        ax.annotate(f'(at {lat:.0f}\u00b0)', xy=(bx + bar_x/2, by),
                    xytext=(0, y_offset_pts), textcoords='offset points',
                    ha='center', va='top', fontsize=fontsize - 2,
                    color=_sublabel_color(ax, sublabel_color, light='0.4'),
                    zorder=zorder, clip_on=False,
                    path_effects=label_effects)


def add_scale_bar_curved_parallel(ax: Any, lon_0: float = 0., lat_0: float = 0.,
                        length_km: float | None = None,
                        body: str = 'earth', R_sphere: float = 1.,
                        position: str = 'lower-center', pad_frac: float = 0.08,
                        color: Any = 'k', colors: Any = None,
                        sublabel_color: Any = None,
                        fontsize: float = 8, lw: float = 2.5,
                        zorder: int = 15, style: str = 'plain',
                        n_segments: int | None = None,
                        segment_km: float | None = None,
                        thickness: float = 0.02, edgecolor: Any = None,
                        edgewidth: float = 0.6,
                        stroke_color: Any = 'w', stroke_lw: float = 0.8) -> None:
    """
    Add a distance scale bar that follows a curved parallel.

    Samples N points along a small-circle (parallel of latitude),
    projects them via the axes' world transform, and renders the
    resulting curve as a polyline (``style='plain'``) or alternating
    polygons between two parallels (``style='checkered'``).

    Works on any WCSAxes whose world transform maps (lon, lat) to
    display coordinates — including:

    * Zenithal FITS projections (SIN / TAN / ARC / ZEA / STG) on
      ``make_globe_frame`` / ``make_wcs_frame``. This was the
      function's original target ("ortho").
    * Pseudocylindrical FITS projections with linear longitude but
      curved parallels (AIT / MOL / SFL / PAR / BON).
    * Custom-transform pseudocylindrical projections in skyplothelper
      (``robinson`` / ``kavrayskiy`` / ``mcbryde`` / ``winkel_tripel``
      / ``eckert4`` / ``mollweide``) — even though these set
      ``ax.wcs = None``, ``ax.get_transform('world')`` still routes
      (lon, lat) through the custom mpl Projection.
    * Plain matplotlib axes set up as orthographic [-1, 1] R units
      (e.g. an axes prepared by ``plot_ortho_grid``): the function
      falls back to a manual ``_ortho_project`` forward.

    Parameters
    ----------
    ax : matplotlib Axes
    lon_0, lat_0 : float
        Orthographic projection center (degrees).
    length_km : float or None
        Total bar length in km. Auto-computed if ``None``.
    body : str
        Planet body for radius lookup (default ``'earth'``).
    R_sphere : float
        Sphere radius in the projected plot (usually 1.0).
    position : str
        ``'lower-left'``, ``'lower-center'``, or ``'lower-right'``.
    pad_frac : float
        Vertical offset from ``lat_0`` as a fraction of 90°.
    color : matplotlib color
        Primary color for ``style='plain'``, or first alternating color
        for ``style='checkered'``.
    colors : tuple of 2 colors or None
        Alternating colors for ``style='checkered'`` (default
        ``(color, 'white')``).
    fontsize : int
    lw : float
        Line width for ``style='plain'``.
    zorder : int
    style : {'plain', 'checkered'}
        ``'plain'`` draws a thick curved line along the arc;
        ``'checkered'`` draws alternating polygon-filled lat bands.
    n_segments, segment_km : int or float or None
        See :func:`add_scale_bar_cylindrical`. If ``segment_km`` is
        given, the final segment may be partial.
    thickness : float
        Radial thickness of the checkered bar, as fraction of
        ``R_sphere`` (≈ half-height in plot units). Only used for
        ``style='checkered'``.
    edgecolor : matplotlib color or None
        Edge color for checkered segments. ``None`` → ``color``.
    edgewidth : float
        Edge line width for checkered segments.
    stroke_color : color spec or None
        Stroke color drawn behind the bar's label text (the classic
        thin white stroke for legibility on textured backgrounds).
        Default ``'w'`` — preserves the prior hardcoded behavior.
        Set to ``None`` to disable.
    stroke_lw : float
        Total stroke width in points. Default ``0.8`` — matches the
        prior hardcoded value.

    Examples
    --------
    >>> add_scale_bar_curved_parallel(ax, lon_0=0, lat_0=30)
    >>> add_scale_bar_curved_parallel(ax, lon_0=0, lat_0=30, style='checkered')
    >>> add_scale_bar_curved_parallel(ax, lon_0=0, lat_0=30, length_km=2500,
    ...                     style='checkered', segment_km=1000)  # 2.5 segments
    """
    import matplotlib.patches as mpatches

    R_km = planet_radii.get(body.lower(), 6371.0)
    if length_km is None:
        length_km = _nice_scale_length(R_km * np.pi * 0.1)

    km_per_deg = R_km * np.cos(np.radians(lat_0)) * np.pi / 180
    bar_deg = length_km / km_per_deg

    bar_center = lon_0 + (-25 if 'left' in position else
                           (25 if 'right' in position else 0))
    bar_lat = lat_0 - 90 * pad_frac

    if edgecolor is None:
        edgecolor = color

    # Routing for the (lon, lat) → display transform. WCSAxes (any
    # flavor — astropy FITS WCS like SIN/AIT/MOL/SFL/PAR/BON, or
    # custom-mpl-projection like robinson/kavrayskiy/mcbryde/
    # winkel_tripel/eckert4/mollweide) expose ``get_transform('world')``
    # which correctly maps (lon, lat) → display regardless of whether
    # an astropy WCS is attached. Plain mpl axes set up as
    # orthographic [-1, 1] R units use the manual ``_ortho_project``
    # forward.
    try:
        from astropy.visualization.wcsaxes import WCSAxes as _WCSAxes
        is_wcsaxes = isinstance(ax, _WCSAxes)
    except ImportError:
        is_wcsaxes = False
    if is_wcsaxes:
        plot_transform = ax.get_transform('world')
    else:
        plot_transform = ax.transData
    plots_lonlat_native = is_wcsaxes

    def _proj(lons: npt.ArrayLike,
              lats: npt.ArrayLike) -> tuple[Any, Any]:
        """Return (x, y) coords in the right space for ``ax.plot`` / ``Polygon``.

        WCSAxes (any flavor): pass lon/lat through unchanged — the
        world transform handles the projection. Plain mpl axes:
        forward-project to [-1, 1] R-sphere units via
        ``_ortho_project``.
        """
        if plots_lonlat_native:
            return (np.asarray(lons, dtype=float),
                    np.asarray(lats, dtype=float))
        return _ortho_project(lons, lats, lon_0, lat_0, R_sphere)

    if style == 'plain':
        lons = np.linspace(bar_center - bar_deg/2, bar_center + bar_deg/2, 100)
        bar_lats = np.full_like(lons, bar_lat)
        x, y = _proj(lons, bar_lats)
        ax.plot(x, y, color=color, lw=lw, solid_capstyle='butt',
                transform=plot_transform, zorder=zorder)
        # Endpoint ticks: small perpendicular lines. On a WCSAxes globe
        # the tick height is given as a small ±lat offset; on plain
        # axes it stays in R_sphere units.
        if plots_lonlat_native:
            tk_lat = R_sphere * 0.03 * (180.0 / np.pi)  # ≈ 1.7° at R=1
            for ei in (0, -1):
                ax.plot([lons[ei], lons[ei]],
                        [bar_lats[ei] - tk_lat, bar_lats[ei] + tk_lat],
                        color=color, lw=lw * 0.5,
                        transform=plot_transform, zorder=zorder)
        else:
            tk = R_sphere * 0.03
            for i in [0, -1]:
                ax.plot([x[i], x[i]], [y[i] - tk, y[i] + tk],
                        color=color, lw=lw * 0.5,
                        transform=plot_transform, zorder=zorder)
        label_lon = lons[len(lons)//2]
        label_lat = bar_lats[len(lons)//2]
        label_x, label_y = _proj([label_lon], [label_lat])
        label_x, label_y = float(np.asarray(label_x)[0]), float(np.asarray(label_y)[0])

    elif style == 'checkered':
        if colors is None:
            colors = (color, 'white')
        breakpoints_km, widths_km = _compute_checker_segments(
            length_km, n_segments=n_segments, segment_km=segment_km)

        # Convert bar thickness to angular latitude offset (approximate,
        # but visually clean for small bars near lat_0).
        t_half = thickness * 0.5
        # Latitude offset for the band (symmetric around bar_lat).
        # We work in the tangent plane near (lon_0, bar_lat), where
        # dx/d(lon) ~ cos(lat) and dy/d(lat) ~ 1 (in R_sphere units).
        # Latitude offset in degrees for +/- t_half in plot-units ~= thickness:
        dlat_deg = np.degrees(t_half / R_sphere)

        left_lon = bar_center - bar_deg/2
        for i, w_km in enumerate(widths_km):
            seg_lon0 = left_lon + breakpoints_km[i] / km_per_deg
            seg_lon1 = left_lon + breakpoints_km[i + 1] / km_per_deg
            n_s = max(20, int(40 * (seg_lon1 - seg_lon0) / bar_deg))
            seg_lons = np.linspace(seg_lon0, seg_lon1, n_s)
            # Build a closed polygon: top edge left→right, bottom edge right→left.
            top_lats = np.full_like(seg_lons, bar_lat + dlat_deg)
            bot_lats = np.full_like(seg_lons, bar_lat - dlat_deg)
            xt, yt = _proj(seg_lons, top_lats)
            xb, yb = _proj(seg_lons, bot_lats)
            xy = np.column_stack([
                np.concatenate([xt, xb[::-1]]),
                np.concatenate([yt, yb[::-1]]),
            ])
            fc = colors[i % 2]
            poly = mpatches.Polygon(
                xy, closed=True, facecolor=fc,
                edgecolor=edgecolor, linewidth=edgewidth,
                transform=plot_transform, zorder=zorder)
            ax.add_patch(poly)

        # Endpoint ticks (small perpendicular lines extending slightly above
        # and below the bar) and label position.
        tk = thickness * 1.2
        end_lons = np.array([bar_center - bar_deg/2, bar_center + bar_deg/2])
        end_lat_hi = np.full_like(end_lons, bar_lat + dlat_deg + tk * 0.3 * 180 / np.pi)
        end_lat_lo = np.full_like(end_lons, bar_lat - dlat_deg - tk * 0.3 * 180 / np.pi)
        for ei in (0, 1):
            xh, yh = _proj([end_lons[ei]], [end_lat_hi[ei]])
            xl, yl = _proj([end_lons[ei]], [end_lat_lo[ei]])
            ax.plot([float(np.asarray(xl)[0]), float(np.asarray(xh)[0])],
                    [float(np.asarray(yl)[0]), float(np.asarray(yh)[0])],
                    color=color, lw=0.8,
                    transform=plot_transform, zorder=zorder)

        mid_x, mid_y = _proj([bar_center], [bar_lat - dlat_deg])
        label_x, label_y = float(np.asarray(mid_x)[0]), float(np.asarray(mid_y)[0])
    else:
        raise ValueError(f"Unknown style {style!r}; use 'plain' or 'checkered'.")

    label_effects = _stroke_path_effects(stroke_color, stroke_lw)

    label = f'{length_km:,.0f} km' if length_km >= 10 else f'{length_km:g} km'
    ax.annotate(label, xy=(label_x, label_y), xycoords=plot_transform,
                xytext=(0, -8), textcoords='offset points',
                ha='center', va='top', fontsize=fontsize, color=color,
                fontweight='bold', zorder=zorder,
                path_effects=label_effects)

    # Segment tick labels for meaningful checkered breakpoints.
    if style == 'checkered' and segment_km is not None and len(widths_km) > 2:
        for bkp_km in breakpoints_km[1:-1]:
            tick_lon = bar_center - bar_deg/2 + bkp_km / km_per_deg
            tx, ty = _proj([tick_lon], [bar_lat + dlat_deg])
            tick_label = (f'{int(bkp_km)}' if bkp_km == int(bkp_km)
                           else f'{bkp_km:g}')
            ax.annotate(tick_label,
                        xy=(float(np.asarray(tx)[0]), float(np.asarray(ty)[0])),
                        xycoords=plot_transform,
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=fontsize - 2,
                        color=_sublabel_color(ax, sublabel_color),
                        zorder=zorder,
                        path_effects=label_effects)


# ===== Scale bar top-level dispatcher =====

_ZENITHAL_FITS_CODES = frozenset({'SIN', 'TAN', 'ARC', 'ZEA', 'STG',
                                   'AZP', 'SZP', 'NCP', 'AIR'})
_CYLINDRICAL_FITS_CODES = frozenset({'CAR', 'MER', 'CEA', 'CYP'})
_PSEUDOCYLINDRICAL_FITS_CODES = frozenset({'AIT', 'MOL', 'SFL', 'PAR', 'BON'})
_CARTOPY_ZENITHAL_NAMES = frozenset({'Orthographic', 'NearsidePerspective'})
_CARTOPY_CYLINDRICAL_NAMES = frozenset({'PlateCarree'})


def _detect_scale_bar_backend(ax: Any) -> tuple[str, str]:
    """Inspect ``ax`` and return (kind, source) where ``kind`` is
    ``'curved_parallel'``, ``'cylindrical'``, or ``'unsupported'`` and
    ``source`` is a short string describing the detected projection —
    used in the error message for unsupported cases.

    ``'curved_parallel'`` covers projections where the parallel of
    latitude renders as a curve in plot coords (or a straight line on
    a globe, which is the limiting case): zenithal (SIN/TAN/ARC/...),
    pseudocylindrical FITS (AIT/MOL/SFL/PAR/BON), skyplothelper's
    custom-transform pseudocylindricals (robinson/kavrayskiy/mcbryde/
    winkel_tripel/eckert4/mollweide), and cartopy
    Orthographic/NearsidePerspective.

    ``'cylindrical'`` is reserved for projections with straight,
    horizontal parallels (CAR/MER/CEA/CYP, cartopy PlateCarree).
    """
    wcs = getattr(ax, 'wcs', None)
    if wcs is not None:
        try:
            ctype = wcs.wcs.ctype[0]
        except Exception:
            ctype = ''
        proj_code = ctype.split('-')[-1] if '-' in ctype else ctype
        if proj_code in _ZENITHAL_FITS_CODES:
            return 'curved_parallel', f"WCSAxes {proj_code}"
        if proj_code in _PSEUDOCYLINDRICAL_FITS_CODES:
            return 'curved_parallel', f"WCSAxes {proj_code}"
        if proj_code in _CYLINDRICAL_FITS_CODES:
            return 'cylindrical', f"WCSAxes {proj_code}"
        return 'unsupported', f"WCSAxes {proj_code or '(unknown)'}"

    # WCSAxes with no astropy WCS — skyplothelper's custom-transform
    # projections (robinson, kavrayskiy, mcbryde, winkel_tripel,
    # eckert4, mollweide). All of these are pseudocylindrical:
    # parallels render as curves, so route to the curved-parallel
    # helper which samples N points along the parallel.
    try:
        from astropy.visualization.wcsaxes import WCSAxes as _WCSAxes
        if isinstance(ax, _WCSAxes):
            return 'curved_parallel', "WCSAxes (custom-transform)"
    except ImportError:
        pass

    # Cartopy GeoAxes carries a ``.projection`` attribute that is a
    # ``cartopy.crs.CRS`` instance. Plain mpl axes also have a
    # ``.projection`` (the string '3d' / 'polar' / etc.), so we type-
    # check against the cartopy module rather than duck-typing.
    proj = getattr(ax, 'projection', None)
    if proj is not None and not isinstance(proj, str):
        try:
            import cartopy.crs as ccrs
            if isinstance(proj, ccrs.CRS):
                name = type(proj).__name__
                if name in _CARTOPY_ZENITHAL_NAMES:
                    return 'curved_parallel', f"cartopy {name}"
                if name in _CARTOPY_CYLINDRICAL_NAMES:
                    return 'cylindrical', f"cartopy {name}"
                return 'unsupported', f"cartopy {name}"
        except ImportError:
            pass

    # Plain matplotlib axes — treat as a generic cylindrical grid
    # with degrees in data coords (the existing fallback in
    # add_scale_bar_cylindrical's ``units_per_deg_x = 1.0`` path).
    return 'cylindrical', 'plain mpl axes'


def add_scale_bar(ax: Any, **kwargs: Any) -> Any:
    """Add a distance scale bar, auto-detecting the projection family.

    Dispatches to :func:`add_scale_bar_curved_parallel` for any
    projection whose parallel of latitude renders as a curve (or
    line) in plot coords — covering zenithal (SIN / TAN / ARC / ZEA
    / STG / ...), pseudocylindrical FITS (AIT / MOL / SFL / PAR /
    BON), skyplothelper's custom-transform pseudocylindricals
    (robinson / kavrayskiy / mcbryde / winkel_tripel / eckert4 /
    mollweide), and cartopy ``Orthographic`` / ``NearsidePerspective``.

    Dispatches to :func:`add_scale_bar_cylindrical` for cylindrical
    projections with linear longitude AND straight horizontal
    parallels (CAR / MER / CEA / CYP, and cartopy ``PlateCarree``).
    Plain matplotlib axes (no WCS, no cartopy projection) are treated
    as cylindrical with degree-valued data coords.

    For unsupported projections — including cartopy ``Mercator`` /
    ``LambertConformal`` etc. — raises a clear :class:`ValueError`
    pointing at `matplotlib-scalebar`_, which is projection-agnostic
    and works on any matplotlib axes.

    .. _matplotlib-scalebar: https://pypi.org/project/matplotlib-scalebar/

    Parameters
    ----------
    ax : matplotlib Axes
        WCSAxes, cartopy GeoAxes, or a plain matplotlib axes.
    **kwargs
        Forwarded to the resolved underlying function. The common
        styling kwargs (``length_km``, ``length``, ``position``,
        ``color``, ``style``, ``segment_km`` / ``segment``,
        ``stroke_color`` / ``stroke_lw``, etc.) are accepted by both
        helpers. Projection-specific kwargs (``lon_0`` / ``lat_0`` /
        ``R_sphere`` for curved-parallel; ``lat`` / ``lon`` /
        ``distance`` for cylindrical) are forwarded as-is — calling
        with a kwarg the resolved helper does not accept will raise
        ``TypeError`` from that helper.

    Returns
    -------
    Whatever the underlying helper returns (currently ``None``).

    Raises
    ------
    ValueError
        If the axes' projection cannot be routed to either helper.

    Examples
    --------
    Auto-routed on a SIN globe::

        ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=23.44)
        add_scale_bar(ax, length_km=2000, style='checkered',
                      segment_km=500)

    Auto-routed on a CAR all-sky map::

        ax = make_wcs_frame(111, projection='CAR', center=0)
        add_scale_bar(ax, lat=40, length_km=500,
                      style='checkered', segment_km=100)

    Auto-routed on an AIT all-sky map (pseudocylindrical FITS)::

        ax = make_wcs_frame(111, projection='AIT', center=0)
        add_scale_bar(ax, length_km=2000, style='checkered',
                      segment_km=500)

    Auto-routed on a Robinson map (custom-transform projection)::

        ax = make_wcs_frame(111, projection='robinson', center=0)
        add_scale_bar(ax, length_km=2000)
    """
    kind, source = _detect_scale_bar_backend(ax)
    if kind == 'curved_parallel':
        return add_scale_bar_curved_parallel(ax, **kwargs)
    if kind == 'cylindrical':
        return add_scale_bar_cylindrical(ax, **kwargs)
    raise ValueError(
        f"add_scale_bar: projection {source!r} is not directly "
        "supported. Supported: WCSAxes SIN / TAN / ARC / ZEA / STG / "
        "CAR / MER / CEA / CYP / AIT / MOL / SFL / PAR / BON, "
        "skyplothelper's custom-transform projections (robinson / "
        "kavrayskiy / mcbryde / winkel_tripel / eckert4 / mollweide), "
        "and cartopy Orthographic / NearsidePerspective / "
        "PlateCarree. For other cartopy projections (Mercator, "
        "LambertConformal, ...) use the projection-agnostic "
        "`matplotlib-scalebar` package."
    )


# =============================================================================
# Coordinate Conversion Utilities
# =============================================================================
