"""Spherical shape vertex generators and renderers.

Public functions take ``(lon, lat)`` floats or a SkyCoord (per
``_parsing._parse_coord``) and angular sizes as float (degrees) or
astropy Quantity (per ``_parse_angle``). Renderers ``add_*`` return the
matplotlib PathPatch artists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

try:
    from cartopy import geodesic
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

from ._api import (
    _prepare_region_vertices,
    _resolve_backend,
    _resolve_clip,
    _resolve_geodesic_for_clip,
)
from ._frame_geom import _safe_intersection
from ._parsing import _parse_angle, _parse_coord, _parse_coords
from ._projector import WCSAxesProjector

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

# ===== Vertex generators =====

def geodesic_circle(
    lon: SkyCoord | float, lat: Any = None, radius_deg: Any = None, resolution: int = 200,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create a geodesic circle on the sphere.

    Parameters
    ----------
    lon : float or SkyCoord
        Center longitude in degrees, or a SkyCoord.
    lat : float or None
        Center latitude in degrees (ignored when *lon* is a SkyCoord,
        in which case the value passed here is used as *radius_deg*).
    radius_deg : float, Quantity, or None
        Radius in degrees.
    resolution : int
        Number of boundary points.
    """
    _orig_lat = lat
    # preserve_frame: no axes/WCS here, so a non-ICRS SkyCoord must keep its own
    # frame — these builders return vertices in the input coordinate's frame.
    lon, lat, shifted = _parse_coord(lon, lat, preserve_frame=True)
    if shifted:
        radius_deg = _orig_lat
    radius_deg = _parse_angle(radius_deg)
    if radius_deg is None:
        raise ValueError("radius_deg is required")

    if CARTOPY_AVAILABLE:
        geod = geodesic.Geodesic()
        # ``n_samples=`` here is cartopy Geodesic.circle's own kwarg, not ours.
        pts = geod.circle(lon, lat, radius_deg * 111320, n_samples=resolution)
        return pts[:, 0], pts[:, 1]

    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    lon_rad, lat_rad = np.radians(lon), np.radians(lat)
    radius_rad = np.radians(radius_deg)

    lats_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(radius_rad)
        + np.cos(lat_rad) * np.sin(radius_rad) * np.cos(angles)
    )
    dlon = np.arctan2(
        np.sin(angles) * np.sin(radius_rad) * np.cos(lat_rad),
        np.cos(radius_rad) - np.sin(lat_rad) * np.sin(lats_rad),
    )
    lons_out = np.degrees(lon_rad + dlon)
    lats_out = np.degrees(lats_rad)
    lons_out = ((lons_out + 180) % 360) - 180
    return lons_out, lats_out



# ===== Geodesic circle renderer =====

def add_geodesic_circle(ax: Any, lon: SkyCoord | float, lat: Any = None,
                        radius_deg: Any = None, resolution: int = 200,
                        complement: bool = False, clip: str = 'auto',
                        backend: str = 'patch', **kwargs: Any) -> Any:
    """
    Add a geodesic circle to WCSAxes.

    Parameters
    ----------
    ax : WCSAxes
    lon : float or SkyCoord
        Center longitude in degrees, or a SkyCoord.
    lat : float or None
        Center latitude in degrees.  When *lon* is a SkyCoord this
        positional slot holds *radius_deg* instead.
    radius_deg : float, Quantity, or None
        Radius in degrees.
    resolution : int
        Number of boundary samples.
    complement : bool
        If True, fill everything EXCEPT the circle.  Edges (if specified)
        are drawn on the circle boundary, not on the frame boundary.
    clip : str
        Projection-seam handling pipeline: ``'auto'`` (default,
        resolves to ``'d3'`` for closed-region patches),
        ``'d3'`` / ``'project_shape'`` / ``'simple'``. See
        ``add_spherical_polygon`` for the full description.
    backend : str
        Matplotlib artist to produce. Currently only ``'patch'`` is
        supported (raises on other values).
    **kwargs
        Passed to PathPatch (facecolor, edgecolor, alpha, etc.).

    Examples
    --------
    >>> import skyplothelper as sph
    >>> ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)
    >>> sph.add_geodesic_circle(ax, 180.0, 0.0, radius_deg=15)   # a 15 deg FoV
    """
    clip = _resolve_clip(clip, helper_name='add_geodesic_circle')
    backend = _resolve_backend(backend, helper_name='add_geodesic_circle',
                                valid=('patch',))

    _orig_lat = lat
    lon, lat, shifted = _parse_coord(lon, lat, wcs=ax.wcs)
    if shifted:
        radius_deg = _orig_lat
    radius_deg = _parse_angle(radius_deg)
    if radius_deg is None:
        raise ValueError("radius_deg is required")

    proj = WCSAxesProjector(ax)
    lons, lats = geodesic_circle(lon, lat, radius_deg, resolution)
    geom = proj.project_polygon(lons, lats, clip=clip,
                                lat_center=lat, radius_deg=radius_deg)
    return proj.render_region(geom, complement=complement, **kwargs)



# ===== Spherical polygon renderer =====

def add_spherical_polygon(ax: Any, lons: SkyCoord | npt.ArrayLike, lats: Any = None,
                          resolution: int = 100,
                          geodesic: bool | str = 'auto',
                          geodesic_threshold: float = 10.0,
                          complement: bool = False, clip: str = 'auto',
                          backend: str = 'patch',
                          min_piece_area: float | None = None,
                          **kwargs: Any) -> Any:
    """
    Add an arbitrary spherical polygon to WCSAxes.

    Edges are automatically densified (interpolated) so that:
    - Edges follow projection curvature correctly
    - Boundary crossings are detected and split properly
    - Even sparse polygons (triangles, etc.) render across boundaries

    Parameters
    ----------
    ax : WCSAxes
        The axes to add the polygon to.
    lons : array-like or SkyCoord
        Vertex longitudes in degrees, or a SkyCoord array.
    lats : array-like or None
        Vertex latitudes in degrees (ignored when *lons* is a SkyCoord).
    resolution : int, optional
        Number of interpolated points per edge (default: 100).
        Higher values give smoother edges.  Set to 0 to disable
        densification (raw vertices only).
    geodesic : bool or 'auto', optional
        Edge interpolation method.  If True, always use great-circle
        (geodesic) interpolation — accurate for large survey footprints
        (JWST, LSST, etc.).  If False, use linear interpolation in
        lon/lat.  If 'auto' (default), use geodesic for edges longer
        than ``geodesic_threshold`` and linear for shorter edges.
        When ``clip='d3'``, ``geodesic='auto'`` is automatically
        promoted to ``True`` to sidestep the centroid-direction
        heuristic in the densifier.
    geodesic_threshold : float, optional
        Edge length in degrees above which geodesic interpolation is
        used when ``geodesic='auto'``.  Default 10°.
    complement : bool
        If True, fill everything EXCEPT the polygon.  Edges (if specified)
        are drawn on the polygon boundary, not the frame boundary.
    clip : str
        Projection-seam handling pipeline. ``'auto'`` (default) selects
        the helper's principled default (``'d3'`` for closed-region
        patches). Other options:

        * ``'d3'`` — Sutherland-Hodgman pre-clip against the
          antimeridian, then project. Best for polygons that cross or
          straddle the projection seam.
        * ``'project_shape'`` — the original
          ``_project_shape`` pipeline (faster, but with a few known
          artifacts at high lat / antipodal corners).
        * ``'simple'`` — raw vertex projection without seam clipping.
          Diagnostic; produces straight-line artifacts at seam.
        * ``'none'`` — invalid for ``backend='patch'``; raises.
    backend : str
        Matplotlib artist to produce. Currently only ``'patch'`` is
        supported on this helper (raises on other values). The kwarg
        exists for API symmetry with the other region helpers; future
        backends (``'contour'`` etc.) may be added without changing
        the public signature. Plural / singular forms are accepted
        interchangeably (``'patches'`` → ``'patch'``).
    min_piece_area : float, optional
        Drop projected sub-polygons smaller than this area (in pixels²)
        after clipping — filters sub-pixel slivers along the wrap edge.
        ``None`` (default) uses the built-in sub-pixel threshold.
    stroke_color : color, optional
        Draw a legibility stroke (outline halo) around the polygon in this
        color — the shared ``stroke_color`` / ``stroke_lw`` knob used across the
        package's overlays. ``None`` (default) draws no stroke. Also accepted by
        ``add_geodesic_circle`` / ``add_rectangle`` / ``add_square`` /
        ``add_ellipse`` / ``add_annulus`` (they share this render path).
    stroke_lw : float, optional
        Stroke width in points (default: a sensible outline width).
    **kwargs
        Passed to matplotlib PathPatch (facecolor, edgecolor, alpha, etc.).

    Examples
    --------
    >>> import skyplothelper as sph
    >>> ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)
    >>> sph.add_spherical_polygon(ax, [150, 210, 210, 150], [-20, -20, 20, 20],
    ...                           facecolor='C0', alpha=0.2)   # edges = geodesics
    """
    proj = WCSAxesProjector(ax)
    # On a bounded (zoomed) field frame, the d3 seam/complement stitcher can
    # flip a frame-crossing polygon's fill to its complement — there's no wrap
    # seam there to justify the d3 machinery. Route the 'auto' default to
    # 'project_shape' (direct projection, verified correct on fields); an
    # explicit clip= is respected, and all-sky / globe frames keep d3.
    if (isinstance(clip, str) and clip.lower() == 'auto'
            and proj.world_bounds() is not None):
        clip = 'project_shape'
    clip = _resolve_clip(clip, helper_name='add_spherical_polygon')
    backend = _resolve_backend(backend, helper_name='add_spherical_polygon',
                                valid=('patch',))
    if clip == 'none':
        raise ValueError(
            "add_spherical_polygon: clip='none' is not meaningful for "
            "backend='patch'; use clip='simple' to skip seam handling "
            "while still projecting the vertices.")
    geodesic = _resolve_geodesic_for_clip(geodesic, clip)

    lons, lats = _parse_coords(lons, lats, wcs=ax.wcs)
    if len(lons) != len(lats):
        raise ValueError("lons and lats must have the same length")
    if len(lons) < 3:
        raise ValueError("Need at least 3 vertices")
    # Close the ring, densify edges for correct projection curvature,
    # and (for d3) compute the expected-area fraction the stitcher uses
    # to disambiguate the polygon from its complement.
    lons, lats, expected_frac = _prepare_region_vertices(
        lons, lats, clip=clip, lon_center=proj.center,
        resolution=resolution, geodesic=geodesic,
        geodesic_threshold=geodesic_threshold,
        compute_expected_frac=True)

    geom = proj.project_polygon(lons, lats, clip=clip,
                                expected_frac=expected_frac,
                                min_piece_area=min_piece_area)
    # Match the upstream sub-pixel filter so a lowered min_piece_area
    # (deep-field surveys) keeps its slivers through the render step too.
    render_min_area = (1.0 if min_piece_area is None
                       else min(1.0, min_piece_area))
    return proj.render_region(geom, complement=complement,
                              min_area=render_min_area, **kwargs)



# ===== Tangent-plane vertex generators (rectangle, ellipse) =====
#
# These build their boundary in the gnomonic tangent plane, where a
# straight line deprojects to a great circle. Densifying each edge
# linearly in tangent-plane coords therefore yields geodesic edges by
# construction — so, unlike add_spherical_polygon (arbitrary lon/lat
# vertices), these shapes have no linear-vs-geodesic choice to expose
# and intentionally take no ``geodesic=`` kwarg.

def _gnomonic_deproject(
    xi: npt.ArrayLike, eta: npt.ArrayLike, lon0_rad: float, lat0_rad: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    De-project tangent-plane coordinates to spherical (lon, lat) in degrees.

    Uses the gnomonic (tangent-plane) inverse projection.  Exact on the
    sphere — handles large offsets correctly (up to ~80° from center).

    Parameters
    ----------
    xi, eta : ndarray
        Tangent-plane offsets in radians.
    lon0_rad, lat0_rad : float
        Tangent point in radians.

    Returns
    -------
    lons, lats : ndarray
        Spherical coordinates in degrees.
    """
    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    rho = np.sqrt(xi**2 + eta**2)
    c = np.arctan(rho)
    safe = rho > 1e-15
    lat = np.where(safe,
        np.arcsin(np.cos(c) * np.sin(lat0_rad)
                  + eta * np.sin(c) * np.cos(lat0_rad) / rho),
        lat0_rad)
    lon = np.where(safe,
        lon0_rad + np.arctan2(
            xi * np.sin(c),
            rho * np.cos(c) * np.cos(lat0_rad)
            - eta * np.sin(c) * np.sin(lat0_rad)),
        lon0_rad)
    return np.degrees(lon), np.degrees(lat)


def rectangle(
    lon: SkyCoord | float, lat: Any = None, width: Any = None, height: Any = None,
    angle: Any = 0, resolution: int = 50,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate vertices of a rectangle on the sphere.

    Parameters
    ----------
    lon : float or SkyCoord
        Center longitude in degrees, or a SkyCoord.
    lat : float or None
        Center latitude in degrees (ignored when *lon* is a SkyCoord,
        in which case the value is used as *width*).
    width, height : float, Quantity, or None
        Full dimensions in degrees (on the sky, not in RA).
    angle : float or Quantity
        Position angle in degrees, measured from north through east.
    resolution : int
        Number of points per edge (for smooth projected curves).

    Returns
    -------
    lons, lats : ndarray
        Vertex arrays (closed: first == last).
    """
    _orig_lat = lat
    # preserve_frame: no axes/WCS here, so a non-ICRS SkyCoord must keep its own
    # frame — these builders return vertices in the input coordinate's frame.
    lon, lat, shifted = _parse_coord(lon, lat, preserve_frame=True)
    if shifted:
        width = _orig_lat
    width = _parse_angle(width)
    height = _parse_angle(height)
    angle = _parse_angle(angle) if angle is not None else 0
    if width is None:
        raise ValueError("width is required")
    if height is None:
        raise ValueError("height is required")
    lon0, lat0 = np.radians(lon), np.radians(lat)
    pa = np.radians(angle)
    hw, hh = np.radians(width / 2), np.radians(height / 2)

    # Rectangle corners in tangent-plane coordinates (radians),
    # rotated by position angle
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    corners_xi = np.array([-hw, hw, hw, -hw])
    corners_eta = np.array([-hh, -hh, hh, hh])
    rot_xi = corners_xi * cos_pa - corners_eta * sin_pa
    rot_eta = corners_xi * sin_pa + corners_eta * cos_pa

    # Densify each edge
    all_xi, all_eta = [], []
    for i in range(4):
        j = (i + 1) % 4
        all_xi.append(np.linspace(rot_xi[i], rot_xi[j], resolution, endpoint=False))
        all_eta.append(np.linspace(rot_eta[i], rot_eta[j], resolution, endpoint=False))

    xi = np.concatenate(all_xi)
    eta = np.concatenate(all_eta)
    lons, lats = _gnomonic_deproject(xi, eta, lon0, lat0)
    lons = np.append(lons, lons[0])
    lats = np.append(lats, lats[0])
    return lons, lats


def ellipse(
    lon: SkyCoord | float, lat: Any = None, semi_major: Any = None, semi_minor: Any = None,
    angle: Any = 0, resolution: int = 200,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate vertices of an ellipse on the sphere.

    Parameters
    ----------
    lon : float or SkyCoord
        Center longitude in degrees, or a SkyCoord.
    lat : float or None
        Center latitude in degrees (ignored when *lon* is a SkyCoord,
        in which case the value is used as *semi_major*).
    semi_major, semi_minor : float, Quantity, or None
        Semi-axis lengths in degrees (on the sky).
    angle : float or Quantity
        Position angle of the semi-major axis in degrees,
        measured from north through east.
    resolution : int
        Number of boundary points.

    Returns
    -------
    lons, lats : ndarray
        Vertex arrays (closed: first == last).
    """
    _orig_lat = lat
    # preserve_frame: no axes/WCS here, so a non-ICRS SkyCoord must keep its own
    # frame — these builders return vertices in the input coordinate's frame.
    lon, lat, shifted = _parse_coord(lon, lat, preserve_frame=True)
    if shifted:
        semi_major = _orig_lat
    semi_major = _parse_angle(semi_major)
    semi_minor = _parse_angle(semi_minor)
    angle = _parse_angle(angle) if angle is not None else 0
    if semi_major is None:
        raise ValueError("semi_major is required")
    if semi_minor is None:
        raise ValueError("semi_minor is required")
    lon0, lat0 = np.radians(lon), np.radians(lat)
    pa = np.radians(angle)
    a, b = np.radians(semi_major), np.radians(semi_minor)

    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    xi_raw = a * np.cos(theta)
    eta_raw = b * np.sin(theta)
    xi = xi_raw * cos_pa - eta_raw * sin_pa
    eta = xi_raw * sin_pa + eta_raw * cos_pa

    lons, lats = _gnomonic_deproject(xi, eta, lon0, lat0)
    lons = np.append(lons, lons[0])
    lats = np.append(lats, lats[0])
    return lons, lats


# ============================================================
# Public API: new shape functions
# ============================================================


# ===== Rectangle / square / ellipse / annulus renderers =====

def add_rectangle(ax: Any, lon: SkyCoord | float, lat: Any = None, width: Any = None,
                  height: Any = None, angle: Any = 0, resolution: int = 50,
                  complement: bool = False, clip: str = 'auto',
                  backend: str = 'patch', **kwargs: Any) -> Any:
    """
    Add a rectangle to WCSAxes.

    Parameters
    ----------
    ax : WCSAxes
    lon : float or SkyCoord
        Center longitude in degrees, or a SkyCoord.
    lat : float or None
        Center latitude in degrees.  When *lon* is a SkyCoord this
        positional slot holds *width* instead.
    width, height : float, Quantity, or None
        Full dimensions in degrees on the sky.
    angle : float or Quantity
        Position angle in degrees, from north through east.
    resolution : int
        Points per edge (default 50).
    complement : bool
        If True, fill everything except the rectangle.
    clip : str
        Projection-seam handling pipeline (``'auto'`` / ``'d3'`` /
        ``'project_shape'`` / ``'simple'``); see
        ``add_spherical_polygon`` for details.
    backend : str
        Currently only ``'patch'`` is supported.
    **kwargs
        Passed to PathPatch.
    """
    clip = _resolve_clip(clip, helper_name='add_rectangle')
    backend = _resolve_backend(backend, helper_name='add_rectangle',
                                valid=('patch',))

    _orig_lat = lat
    lon, lat, shifted = _parse_coord(lon, lat, wcs=ax.wcs)
    if shifted:
        width = _orig_lat
    width = _parse_angle(width)
    height = _parse_angle(height)
    angle = _parse_angle(angle) if angle is not None else 0
    if width is None:
        raise ValueError("width is required")
    if height is None:
        raise ValueError("height is required")

    proj = WCSAxesProjector(ax)
    lons, lats = rectangle(lon, lat, width, height, angle, resolution)
    geom = proj.project_polygon(lons, lats, clip=clip)
    return proj.render_region(geom, complement=complement, **kwargs)


def add_square(ax: Any, lon: SkyCoord | float, lat: Any = None, size: Any = None,
               angle: Any = 0, resolution: int = 50, complement: bool = False,
               clip: str = 'auto', backend: str = 'patch',
               **kwargs: Any) -> Any:
    """
    Add a square to WCSAxes.

    Convenience wrapper around ``add_rectangle`` with width == height.

    Parameters
    ----------
    ax : WCSAxes
    lon : float or SkyCoord
        Center longitude in degrees, or a SkyCoord.
    lat : float or None
        Center latitude in degrees.  When *lon* is a SkyCoord this
        positional slot holds *size* instead.
    size : float, Quantity, or None
        Side length in degrees on the sky.
    angle : float or Quantity
        Position angle in degrees, from north through east.
    clip, backend :
        See ``add_rectangle``.
    **kwargs
        Passed to PathPatch.
    """
    _orig_lat = lat
    lon, lat, shifted = _parse_coord(lon, lat, wcs=ax.wcs)
    if shifted:
        size = _orig_lat
    size = _parse_angle(size)
    angle = _parse_angle(angle) if angle is not None else 0
    if size is None:
        raise ValueError("size is required")
    return add_rectangle(ax, lon, lat, size, size, angle=angle,
                         resolution=resolution, complement=complement,
                         clip=clip, backend=backend, **kwargs)


def add_ellipse(ax: Any, lon: SkyCoord | float, lat: Any = None, semi_major: Any = None,
                semi_minor: Any = None, angle: Any = 0, resolution: int = 200,
                complement: bool = False, clip: str = 'auto',
                backend: str = 'patch', **kwargs: Any) -> Any:
    """
    Add an ellipse to WCSAxes.

    Parameters
    ----------
    ax : WCSAxes
    lon : float or SkyCoord
        Center longitude in degrees, or a SkyCoord.
    lat : float or None
        Center latitude in degrees.  When *lon* is a SkyCoord this
        positional slot holds *semi_major* instead.
    semi_major, semi_minor : float, Quantity, or None
        Semi-axis lengths in degrees on the sky.
    angle : float or Quantity
        Position angle of the semi-major axis in degrees,
        measured from north through east.
    resolution : int
        Number of boundary points (default 200).
    complement : bool
        If True, fill everything except the ellipse.
    clip, backend :
        See ``add_spherical_polygon``.
    **kwargs
        Passed to PathPatch.
    """
    clip = _resolve_clip(clip, helper_name='add_ellipse')
    backend = _resolve_backend(backend, helper_name='add_ellipse',
                                valid=('patch',))

    _orig_lat = lat
    lon, lat, shifted = _parse_coord(lon, lat, wcs=ax.wcs)
    if shifted:
        semi_major = _orig_lat
    semi_major = _parse_angle(semi_major)
    semi_minor = _parse_angle(semi_minor)
    angle = _parse_angle(angle) if angle is not None else 0
    if semi_major is None:
        raise ValueError("semi_major is required")
    if semi_minor is None:
        raise ValueError("semi_minor is required")

    proj = WCSAxesProjector(ax)
    lons, lats = ellipse(lon, lat, semi_major, semi_minor, angle, resolution)
    geom = proj.project_polygon(lons, lats, clip=clip,
                                lat_center=lat, radius_deg=semi_major)
    return proj.render_region(geom, complement=complement, **kwargs)


def add_annulus(ax: Any, lon: SkyCoord | float, lat: Any = None, inner_radius: Any = None,
                outer_radius: Any = None, resolution: int = 200,
                complement: bool = False, clip: str = 'auto',
                backend: str = 'patch', **kwargs: Any) -> Any:
    """
    Add an annulus (ring) to WCSAxes.

    Parameters
    ----------
    ax : WCSAxes
    lon : float or SkyCoord
        Center longitude in degrees, or a SkyCoord.
    lat : float or None
        Center latitude in degrees.  When *lon* is a SkyCoord this
        positional slot holds *inner_radius* instead.
    inner_radius, outer_radius : float, Quantity, or None
        Inner and outer radii in degrees.
    resolution : int
        Boundary point count per circle (default 200).
    complement : bool
        If True, fill everything except the annulus.
    clip, backend :
        See ``add_spherical_polygon``. The clip choice is applied to
        both the outer and inner circle projections before the
        shapely difference op produces the ring.
    **kwargs
        Passed to PathPatch.
    """
    clip = _resolve_clip(clip, helper_name='add_annulus')
    backend = _resolve_backend(backend, helper_name='add_annulus',
                                valid=('patch',))

    _orig_lat = lat
    lon, lat, shifted = _parse_coord(lon, lat, wcs=ax.wcs)
    if shifted:
        inner_radius = _orig_lat
    inner_radius = _parse_angle(inner_radius)
    outer_radius = _parse_angle(outer_radius)
    if inner_radius is None:
        raise ValueError("inner_radius is required")
    if outer_radius is None:
        raise ValueError("outer_radius is required")

    proj = WCSAxesProjector(ax)
    frame_poly = proj.frame_polygon

    # Outer / inner circle geometry, projected to shapely.
    o_lons, o_lats = geodesic_circle(lon, lat, outer_radius, resolution)
    outer = proj.project_polygon(o_lons, o_lats, clip=clip,
                                 lat_center=lat, radius_deg=outer_radius)

    i_lons, i_lats = geodesic_circle(lon, lat, inner_radius, resolution)
    inner = proj.project_polygon(i_lons, i_lats, clip=clip,
                                 lat_center=lat, radius_deg=inner_radius)

    if outer is None:
        return []

    # Ring = outer - inner, clipped to the frame.
    ring = outer.difference(inner) if inner is not None else outer
    ring = _safe_intersection(ring, frame_poly)

    if complement:
        ring = frame_poly.difference(ring)

    # The ring carries its hole as a shapely interior; render it
    # straight out via render_region (which only runs _shapely_to_paths,
    # never the hole-collapsing _paths_to_geom round-trip). complement
    # was already inverted inline, so this is a plain region render.
    return proj.render_region(ring, **kwargs)

