"""Shared kwarg-resolution helpers for the unified region-renderer API.

Every closed-region helper accepts the same core surface:

    backend = 'patch'        # the matplotlib artist (most helpers only accept 'patch')
    clip    = 'auto'         # 'auto' | 'd3' | 'project_shape' | 'simple' | 'none'
    geodesic = 'auto'        # 'auto' | True | False
    resolution = <helper-default>
    complement = False

The functions below normalize each kwarg and raise informative errors
on unsupported values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

_CLIP_VALUES = ('auto', 'd3', 'project_shape', 'simple', 'none')


def _resolve_clip(clip: str, *, helper_name: str = 'helper',
                  auto_default: str = 'd3') -> str:
    """Normalize ``clip=`` to one of the canonical non-'auto' values.

    Parameters
    ----------
    clip : str
    helper_name : str
        Used in error messages.
    auto_default : str
        What ``'auto'`` resolves to. Closed-region patch helpers use
        ``'d3'``; HEALPix's pcolormesh/imshow backends use ``'simple'``.
    """
    if not isinstance(clip, str):
        raise TypeError(
            f"{helper_name}: clip= must be a string, got "
            f"{type(clip).__name__}")
    c = clip.lower()
    if c not in _CLIP_VALUES:
        raise ValueError(
            f"{helper_name}: clip={clip!r} not recognized; "
            f"expected one of {_CLIP_VALUES}")
    if c == 'auto':
        return auto_default
    return c


def _resolve_backend(backend: str, *, helper_name: str = 'helper',
                     valid: tuple[str, ...] = ('patch',)) -> str:
    """Normalize ``backend=`` to a canonical singular value.

    Plural / singular forms are accepted interchangeably (e.g. both
    ``'patch'`` and ``'patches'`` resolve to ``'patch'``). Raises if
    the resolved value isn't in *valid*.
    """
    if not isinstance(backend, str):
        raise TypeError(
            f"{helper_name}: backend= must be a string, got "
            f"{type(backend).__name__}")
    raw = backend.lower()
    # Substring check normalizes plural/singular variants.
    if 'patch' in raw:
        b = 'patch'
    elif 'pcolormesh' in raw:
        b = 'pcolormesh'
    elif 'imshow' in raw:
        b = 'imshow'
    elif 'contour' in raw:
        b = 'contour'
    else:
        b = raw
    if b not in valid:
        raise ValueError(
            f"{helper_name}: backend={backend!r} not supported here; "
            f"expected one of {valid}")
    return b


def _dispatch_projection(ax: Any, lons: npt.ArrayLike, lats: npt.ArrayLike,
                         clip: str, *, expected_frac: float | None = None,
                         lat_center: float | None = None,
                         radius_deg: float | None = None,
                         min_piece_area: float | None = None) -> list[Any]:
    """Dispatch a closed-polygon projection based on the resolved
    ``clip`` value. Returns a list of matplotlib Paths in pixel coords.

    Centralizes the if/elif branching that every region helper would
    otherwise have to repeat. The ``expected_frac`` /
    ``lat_center`` / ``radius_deg`` / ``min_piece_area`` kwargs are
    forwarded only to the pipelines that consume them (``d3`` and
    ``project_shape``); the other branches ignore them.

    ``min_piece_area`` (px²) lets small primitives like deep-field
    surveys override ``_stitch_and_project``'s default sliver filter
    (5.0 px²) so they render even on coarse-pixel all-sky frames.
    """
    # Local imports to avoid circulars (this module is imported by
    # every shape helper in turn).
    from ._antimeridian import _antimeridian_clip, _stitch_and_project
    from ._frame_geom import _get_frame_polygon, _get_projection_center
    from ._projection import _project_shape, _simple_project

    if clip == 'd3':
        lon_center = _get_projection_center(ax)
        frame_poly = _get_frame_polygon(ax)
        segments = _antimeridian_clip(lons, lats, lon_center)
        kw: dict[str, Any] = {'expected_frac': expected_frac}
        if min_piece_area is not None:
            kw['min_piece_area'] = min_piece_area
        return _stitch_and_project(segments, ax, frame_poly, **kw)
    if clip == 'simple':
        return _simple_project(ax, lons, lats)
    # 'project_shape' (legacy default)
    proj_kw: dict[str, Any] = {}
    if lat_center is not None:
        proj_kw['lat_center'] = lat_center
    if radius_deg is not None:
        proj_kw['radius_deg'] = radius_deg
    return _project_shape(ax, lons, lats, **proj_kw)


def _resolve_geodesic_for_clip(geodesic: bool | str, clip: str) -> bool | str:
    """Pair ``(geodesic, clip)`` to determine the effective geodesic mode.

    When ``clip='d3'`` and the user left ``geodesic='auto'``, flip to
    ``True`` (force slerp on every edge). The reason: D3 clip itself
    handles the antimeridian, but the upstream densifier
    (``_densify_polygon_edges``) uses a centroid-direction heuristic
    that mispicks for sparse polygons spanning ~360° in lon. Forcing
    slerp on every edge sidesteps that heuristic and gives clean
    rendering for those cases.
    """
    if clip == 'd3' and geodesic == 'auto':
        return True
    return geodesic


def _prepare_region_vertices(
    lons: npt.ArrayLike, lats: npt.ArrayLike, *, clip: str,
    lon_center: float, resolution: int = 0,
    geodesic: bool | str = 'auto', geodesic_threshold: float = 10.0,
    compute_expected_frac: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float | None]:
    """Close a polygon ring, optionally densify its edges, and (for the
    d3 clip path) optionally compute its expected area fraction.

    Factors the preamble the shapes renderers share before handing a
    region to ``Projector.project_polygon``. The ``expected_frac`` hint
    lets the d3 stitcher decide whether shapely's clipped output is the
    small region or its complement; it is only meaningful for
    ``clip='d3'`` and is otherwise ``None``.

    Parameters
    ----------
    lons, lats : array-like
        Polygon vertices in degrees.
    clip : str
        Resolved clip mode (``'d3'`` / ``'project_shape'`` /
        ``'simple'``).
    lon_center : float
        Projection center longitude, used by the ``expected_frac``
        span calculation.
    resolution : int
        Per-edge densification points. ``0`` (default) skips
        densification — appropriate for the vertex generators that
        already emit smooth boundaries (circles, ellipses, rectangles).
    geodesic, geodesic_threshold :
        Forwarded to ``_densify_polygon_edges`` when ``resolution>0``.
    compute_expected_frac : bool
        When True and ``clip=='d3'``, compute the expected area
        fraction; otherwise return ``None`` for it.

    Returns
    -------
    (lons, lats, expected_frac)
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if not (np.isclose(lons[0], lons[-1]) and np.isclose(lats[0], lats[-1])):
        lons = np.append(lons, lons[0])
        lats = np.append(lats, lats[0])

    if resolution and resolution > 0:
        from ._densify import _densify_polygon_edges
        lons, lats = _densify_polygon_edges(
            lons, lats, resolution=resolution,
            geodesic=geodesic, geodesic_threshold=geodesic_threshold)

    expected_frac = None
    if compute_expected_frac and clip == 'd3':
        sin_range = abs(np.sin(np.radians(np.max(lats)))
                        - np.sin(np.radians(np.min(lats))))
        lon_norm = ((lons - lon_center + 180) % 360) - 180
        lon_span = np.ptp(lon_norm)
        if lon_span > 180:
            lon_span = 360 - lon_span
        expected_frac = (lon_span / 360.0) * sin_range / 2.0

    return lons, lats, expected_frac
