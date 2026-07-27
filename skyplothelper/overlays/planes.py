"""Galactic / ecliptic / supergalactic plane and great-circle overlays.

The ``make_*plane_in_RArange`` helpers generate plane coordinates as
``SkyCoord`` sequences wrapped for plotting; ``add_great_circle`` and
``add_plane_overlay`` are the rendering entry points for WCSAxes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord

from .._stroke import _stroke_path_effects
from ..geometry._parsing import _coords_to_frame_deg
from ..wcs_frame import _get_wcs_center_lon, _get_wcs_frame_name


def make_gplane_in_RArange(RAmax: float, Nelements: int,
                           lat_offset: float = 0.) -> SkyCoord:
    """
    Create Galactic Plane coordinates wrapped for plotting around a given RA max.
    """
    gplane = SkyCoord(l=np.linspace(-180, 180, Nelements),
                      b=np.zeros(Nelements) + lat_offset,
                      frame='galactic', unit='deg').icrs
    gswitchind = np.where(np.diff(gplane.ra.wrap_at('%id' % RAmax)) < 0)[0][0] + 1
    return gplane[np.concatenate([range(gswitchind, len(gplane)),
                                   range(0, gswitchind)])]


def make_ecplane_in_RArange(RAmax: float, Nelements: int,
                            frame: str = 'geocentrictrueecliptic',
                            lat_offset: float = 0.) -> SkyCoord:
    """Create Ecliptic Plane coordinates wrapped for plotting."""
    ecplane = SkyCoord(lon=np.linspace(-180, 180, Nelements),
                       lat=np.zeros(Nelements) + lat_offset,
                       frame=frame, unit='deg').icrs
    eswitchind = np.where(np.diff(ecplane.ra.wrap_at('%id' % RAmax)) < 0)[0][0] + 1
    return ecplane[np.concatenate([range(eswitchind, len(ecplane)),
                                    range(0, eswitchind)])]


def make_sgplane_in_RArange(RAmax: float, Nelements: int,
                            lat_offset: float = 0.) -> SkyCoord:
    """Create Supergalactic Plane coordinates wrapped for plotting."""
    sgplane = SkyCoord(sgl=np.linspace(-180, 180, Nelements),
                       sgb=np.zeros(Nelements) + lat_offset,
                       frame='supergalactic', unit='deg').icrs
    sswitchind = np.where(np.diff(sgplane.ra.wrap_at('%id' % RAmax)) < 0)[0][0] + 1
    return sgplane[np.concatenate([range(sswitchind, len(sgplane)),
                                    range(0, sswitchind)])]


# ===== Unified entry points =====

def _find_world_polyline_splits(ax: Any, lons: npt.ArrayLike,
                                lats: npt.ArrayLike, *,
                                lon_jump_deg: float = 90.,
                                pixel_jump_fraction: float = 0.25) -> np.ndarray:
    """Indices at which to split a (lons, lats) polyline before plotting.

    Returns the indices that should be passed to ``np.split`` so the
    polyline is broken wherever consecutive samples cross a projection
    discontinuity.  Two split sources are combined:

    * **Lon-space jumps** (``|diff(lon)| > lon_jump_deg``) — catch
      antimeridian wraps in the natural-order array (no-op after
      ``np.argsort``-by-wrapped-lon, but kept for callers that don't
      sort).
    * **Pixel-space jumps** (``|diff(x_pix)|`` or ``|diff(y_pix)|`` >
      ``pixel_jump_fraction`` × frame extent) — catch the multi-face
      projections (HPX, XPH, CSC, TSC, QSC) where adjacent samples in
      lon land on opposite faces, producing 100s-of-pixel jumps that
      lon-space alone cannot detect. NaN pixel values (off-projection
      points) also trigger a split.
    """
    splits = []

    if hasattr(ax, 'wcs') and ax.wcs is not None:
        try:
            x_pix, y_pix = ax.wcs.world_to_pixel_values(lons, lats)
            x_pix = np.asarray(x_pix, dtype=float)
            y_pix = np.asarray(y_pix, dtype=float)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_thresh = pixel_jump_fraction * abs(xlim[1] - xlim[0])
            y_thresh = pixel_jump_fraction * abs(ylim[1] - ylim[0])
            with np.errstate(invalid='ignore'):
                dx = np.abs(np.diff(x_pix))
                dy = np.abs(np.diff(y_pix))
            pix_split = np.where(
                (dx > x_thresh) | (dy > y_thresh)
                | np.isnan(dx) | np.isnan(dy)
            )[0] + 1
            splits.extend(pix_split.tolist())
        except Exception:
            pass

    dlon = np.abs(np.diff(np.asarray(lons, dtype=float)))
    lon_split = np.where(dlon > lon_jump_deg)[0] + 1
    splits.extend(lon_split.tolist())

    if splits:
        return np.unique(splits)
    return np.array([], dtype=int)


def add_great_circle(ax: Any, pole_lon: float = 0., pole_lat: float = 90.,
                     frame: str = 'galactic',
                     n_points: int = 500, lat_offset: float = 0.,
                     color: Any = 'k', lw: float = 1,
                     ls: str = '-', alpha: float = 1.,
                     label: str | None = None, zorder: int = 5,
                     stroke_color: Any = None, stroke_lw: float = 2.5,
                     **kwargs: Any) -> list[Any]:
    """
    Add a great circle (or small circle at a latitude offset) to a WCSAxes.

    The circle is defined as latitude = ``lat_offset`` in the given coordinate
    frame. For lat_offset=0 this traces the equator (great circle); for
    nonzero values it traces a small circle (parallel).

    For custom great circles not aligned with standard frames, specify
    ``pole_lon`` and ``pole_lat`` to define the pole of the great circle
    in ICRS coordinates, and set ``frame='pole'``.

    Parameters
    ----------
    ax : WCSAxes
        The axes to draw on (must be a WCSAxes instance)
    pole_lon, pole_lat : float
        Pole of the great circle in ICRS degrees (only used if frame='pole')
    frame : str
        'galactic', 'ecliptic', 'supergalactic', or 'pole' for custom
    n_points : int
        Number of sample points along the circle
    lat_offset : float
        Latitude offset in degrees (0 = great circle, nonzero = small circle)
    color, lw, ls, alpha, label, zorder : plot styling
    stroke_color : color spec or None
        Optional stroke color drawn under the circle. Default
        ``None`` (no stroke). Useful for visibility on busy
        backgrounds: ``stroke_color='white'`` on a dark sky, ``'k'``
        on a bright canvas.
    stroke_lw : float
        Total stroke width in points. Default ``2.5``.
    **kwargs : additional plot kwargs

    Returns
    -------
    lines : list
        The matplotlib line objects

    Examples
    --------
    >>> add_great_circle(ax, frame='galactic', color='gray', ls='--')
    >>> add_great_circle(ax, frame='ecliptic', color='gold', lw=2)
    >>> add_great_circle(ax, frame='galactic', lat_offset=10, color='gray',
    ...                  ls=':', alpha=0.5)  # b=+10° parallel
    """
    lons_sample = np.linspace(0, 360, n_points, endpoint=False)
    lats_sample = np.full(n_points, lat_offset)

    if frame.lower() == 'galactic':
        coords = SkyCoord(l=lons_sample, b=lats_sample, frame='galactic',
                          unit='deg')
    elif frame.lower() in ('ecliptic', 'geocentrictrueecliptic'):
        coords = SkyCoord(lon=lons_sample, lat=lats_sample,
                          frame='geocentrictrueecliptic', unit='deg')
    elif frame.lower() == 'supergalactic':
        coords = SkyCoord(sgl=lons_sample, sgb=lats_sample,
                          frame='supergalactic', unit='deg')
    elif frame.lower() == 'pole':
        # Custom great circle: generate points in a frame where the
        # given pole is at the north pole, then rotate
        pole = SkyCoord(pole_lon, pole_lat, unit='deg', frame='icrs')
        # Points at angular distance 90°-lat_offset from pole
        pa = np.linspace(0, 360, n_points, endpoint=False)
        sep = 90. - lat_offset
        coords = pole.directional_offset_by(pa * u.deg, sep * u.deg)
    else:
        raise ValueError(f"Unknown frame '{frame}'. Use 'galactic', "
                         "'ecliptic', 'supergalactic', or 'pole'.")

    # Convert to WCS native frame (shared dispatch — see _coords_to_frame_deg)
    plot_lon, plot_lat = _coords_to_frame_deg(coords, _get_wcs_frame_name(ax))

    # Wrap into the projection window, then split at wrap / face jumps.
    #
    # Only a GREAT circle (lat_offset == 0) is sorted by longitude: it's
    # monotonic in RA, so the sort just rotates the wrap seam to the array ends
    # (one clean segment). A SMALL circle (lat_offset != 0) that doesn't enclose
    # the pole is double-valued in RA, so sorting would interleave its two
    # branches into a fan (a filled zig-zag wedge) — it must stay in the
    # linspace path order. ``_find_world_polyline_splits`` breaks antimeridian
    # wraps in natural order too (its lon-jump split is documented as kept for
    # un-sorted callers).
    center_lon = _get_wcs_center_lon(ax)
    wrapped = ((plot_lon - center_lon + 180) % 360) + center_lon - 180
    if lat_offset == 0:
        sort_idx = np.argsort(wrapped)
        wrapped = wrapped[sort_idx]
        plot_lat = plot_lat[sort_idx]

    split_pts = _find_world_polyline_splits(ax, wrapped, plot_lat)

    lon_segments = np.split(wrapped, split_pts)
    lat_segments = np.split(plot_lat, split_pts)

    transform = ax.get_transform('world')
    stroke_effect = _stroke_path_effects(stroke_color, stroke_lw)
    lines = []
    for i, (seg_lon, seg_lat) in enumerate(zip(lon_segments, lat_segments)):
        if len(seg_lon) < 2:
            continue
        lab = label if i == 0 else None
        ln = ax.plot(seg_lon, seg_lat, transform=transform,
                     color=color, lw=lw, ls=ls, alpha=alpha,
                     label=lab, zorder=zorder, **kwargs)
        if stroke_effect is not None:
            for line in ln:
                line.set_path_effects(stroke_effect)
        lines.extend(ln)

    return lines


def add_plane_overlay(ax: Any, plane: str = 'galactic', color: Any = None,
                      lw: float = 1, ls: str = '-',
                      alpha: float = 1., label: str | None = None,
                      parallels: Sequence[float] | None = None,
                      parallel_ls: str = ':', parallel_alpha: float = 0.4,
                      parallel_lw: float | None = None,
                      parallel_color: Any = None,
                      **kwargs: Any) -> list[Any]:
    """
    Add a coordinate plane overlay (galactic, ecliptic, or supergalactic)
    to a WCSAxes, with optional parallels.

    Parameters
    ----------
    ax : WCSAxes
    plane : str
        'galactic', 'ecliptic', or 'supergalactic'
    color : str, optional
        Line color. Defaults: galactic='dimgray', ecliptic='goldenrod',
        supergalactic='steelblue'
    lw : float
    ls : str
    alpha : float
    label : str, optional
        Legend label. If None, auto-generates (e.g., 'Galactic plane')
    parallels : list of float, optional
        Latitude offsets for additional parallel lines (e.g., [-10, 10])
    parallel_ls : str
        Line style for parallels
    parallel_alpha : float
        Alpha for parallels
    parallel_lw : float, optional
        Line width for parallels. Default ``None`` uses ``0.7 * lw``.
    parallel_color : color, optional
        Color for parallels. Default ``None`` reuses the main ``color``.
    **kwargs : additional plot kwargs

    Returns
    -------
    lines : list
        All line objects created

    Examples
    --------
    >>> add_plane_overlay(ax, 'galactic')
    >>> add_plane_overlay(ax, 'ecliptic', parallels=[-23.4, 23.4])
    >>> add_plane_overlay(ax, 'supergalactic', lw=2, ls='--')
    """
    default_colors = {
        'galactic': 'dimgray', 'ecliptic': 'goldenrod',
        'supergalactic': 'steelblue',
    }
    default_labels = {
        'galactic': 'Galactic plane', 'ecliptic': 'Ecliptic plane',
        'supergalactic': 'Supergalactic plane',
    }

    plane_key = plane.lower()
    if color is None:
        color = default_colors.get(plane_key, 'k')
    if label is None:
        label = default_labels.get(plane_key, plane)

    lines = add_great_circle(ax, frame=plane_key, lat_offset=0.,
                             color=color, lw=lw, ls=ls, alpha=alpha,
                             label=label, **kwargs)

    if parallels:
        for lat_off in parallels:
            lines.extend(add_great_circle(
                ax, frame=plane_key, lat_offset=lat_off,
                color=parallel_color if parallel_color is not None else color,
                lw=parallel_lw if parallel_lw is not None else lw * 0.7,
                ls=parallel_ls, alpha=parallel_alpha, **kwargs))

    return lines

