"""CompoundRegion class for set-algebraic spherical regions.

Accumulate spherical shapes and render their boolean combination
(union / difference / intersection / xor / complement). Each shape
is projected to pixel space via ``_project_shape`` (or via D3
pre-clipping for cross-frame bands), then combined with the
accumulated geometry using shapely boolean operations.

The CompoundRegion API mirrors the standalone ``add_*`` shape and band
functions, but with a fluent interface that returns ``self``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt  # noqa: F401  (used by render paths)
import numpy as np
import numpy.typing as npt
from matplotlib.patches import PathPatch

try:
    from shapely.geometry import LineString, Point, Polygon  # noqa: F401
    from shapely.ops import unary_union  # noqa: F401
    from shapely.validation import make_valid  # noqa: F401
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from ._api import _resolve_clip, _resolve_geodesic_for_clip
from ._densify import _densify_polygon_edges
from ._frame_geom import (
    _fix_hairline_kwargs,
    _shapely_to_paths,
)
from ._parsing import (
    _coords_to_frame_deg,
    _parse_angle,
    _parse_coord,
    _parse_coords,
)
from .shapes import ellipse, geodesic_circle, rectangle

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


def _cleanup_for_render(geom: Any, min_area: float = 1e-6,
                        close_eps: float = 0.5) -> Any:
    """Clean a Shapely geometry for visual rendering.

    Set-algebra results (especially compound regions whose recipe
    includes overlapping ``subtract_frame_band`` calls) can carry
    artifacts that are invisible in the geometry's nominal "area"
    but produce spurious lines or seams when rendered:

    * ``GeometryCollection`` containing ``LineString`` / ``Point``
      pieces alongside the real polygons.
    * Zero-area ``Polygon`` slivers from numerical edge-snapping.
    * Two near-touching but technically-disjoint polygons left over
      where two differently-densified band edges cross — a
      sub-pixel gap shows up as a thin seam between filled regions.

    The cleanup pipeline:

    1. ``buffer(0)`` collapses ``GeometryCollection`` to
       ``Polygon`` / ``MultiPolygon`` and resolves self-touch
       topology.
    2. Filter polygon parts with area below ``min_area`` (pixel²).
       Default ``1e-6`` px² catches only numerical-noise zero-area
       artifacts (typical artifact areas are ~10⁻¹⁵ px²) without
       dropping genuinely small regions like deep-field surveys
       at all-sky scale.
    3. Morphological close: ``buffer(+close_eps).buffer(-close_eps)``
       — joins polygons separated by less than ``close_eps`` pixels.

    This is purely cosmetic and is applied at render time only;
    the underlying ``CompoundRegion._geom`` is not mutated. User
    code that needs the precise geometry for downstream queries
    should access ``._geom`` directly.
    """
    from shapely.geometry import MultiPolygon
    if geom is None or geom.is_empty:
        return geom

    cleaned = geom.buffer(0)
    if cleaned.is_empty:
        return cleaned

    # Filter sub-threshold polygons.
    if cleaned.geom_type == 'Polygon':
        if cleaned.area <= min_area:
            return MultiPolygon()
    elif cleaned.geom_type == 'MultiPolygon':
        kept = [p for p in cleaned.geoms if p.area > min_area]
        if not kept:
            return MultiPolygon()
        cleaned = kept[0] if len(kept) == 1 else MultiPolygon(kept)
    elif cleaned.geom_type == 'GeometryCollection':
        polys = [g for g in cleaned.geoms
                 if g.geom_type in ('Polygon', 'MultiPolygon')
                 and g.area > min_area]
        if not polys:
            return MultiPolygon()
        cleaned = polys[0] if len(polys) == 1 else MultiPolygon(
            p for g in polys for p in (
                g.geoms if g.geom_type == 'MultiPolygon' else [g]))

    # Morphological close to merge near-touching pieces.
    if close_eps > 0:
        cleaned = cleaned.buffer(close_eps).buffer(-close_eps)

    return cleaned


class CompoundRegion:
    """
    Accumulate spherical shapes and render their boolean combination.

    Each shape is projected to pixel space via ``_project_shape`` (or D3
    pre-clipping for cross-frame bands), then combined with the accumulated
    geometry using shapely boolean operations.

    Methods return ``self`` for chaining::

        CompoundRegion(ax).add_circle(0, 0, 30).subtract_circle(0, 0, 10).render(...)

    Three boolean operations are available for each shape type:

    - **add** (union): Merge the shape into the region.
    - **subtract** (difference): Cut the shape out of the region.
    - **intersect** (intersection): Keep only the overlap with the shape.

    Shape types and their methods:

    +-----------------------+---------------------------------------------+
    | Shape                 | Methods                                     |
    +=======================+=============================================+
    | Geodesic circle       | ``add_circle``, ``subtract_circle``,        |
    |                       | ``intersect_circle``                        |
    +-----------------------+---------------------------------------------+
    | Spherical polygon     | ``add_polygon``, ``subtract_polygon``,      |
    |                       | ``intersect_polygon``                       |
    +-----------------------+---------------------------------------------+
    | ICRS latitude band    | ``add_latitude_band``,                      |
    |                       | ``subtract_latitude_band``                  |
    +-----------------------+---------------------------------------------+
    | Cross-frame band      | ``add_frame_band``,                         |
    | (galactic, ecliptic)  | ``subtract_frame_band``                     |
    +-----------------------+---------------------------------------------+

    Additional methods: ``complement()`` (flip region), ``render(**kwargs)``
    (draw on axes), ``area_frac`` (sky coverage fraction), ``is_empty``.

    Parameters
    ----------
    ax : WCSAxes
        The axes to render on.

    Examples
    --------
    Union of overlapping circles::

        region = CompoundRegion(ax)
        region.add_circle(30, -10, 20)
        region.add_circle(55, 5, 18)
        region.render(facecolor='teal', alpha=0.5)

    Survey footprint with exclusion zone::

        region = CompoundRegion(ax)
        region.add_polygon(survey_ra, survey_dec)
        region.subtract_circle(bright_star_ra, bright_star_dec, mask_radius)
        region.render(facecolor='orange', edgecolor='blue', alpha=0.4)

    Galactic band with Galactic Center excised::

        region = CompoundRegion(ax)
        region.add_frame_band(-20, 20, frame='galactic')
        region.subtract_circle(266.417, -29.008, 5)
        region.render(facecolor='teal', alpha=0.5)
    """

    def __init__(self, ax_or_projector: Any) -> None:
        """Construct a :class:`CompoundRegion`.

        Parameters
        ----------
        ax_or_projector : WCSAxes or :class:`Projector`
            A matplotlib WCSAxes (the historical / default form — wrapped
            internally in a :class:`WCSAxesProjector`) **or** a concrete
            :class:`Projector` instance. The latter form lets non-mpl
            backends (e.g. the plotly side's ``SkyplothelperProjector``)
            drive the same compound-region algebra without an mpl axes
            in the loop.
        """
        from ._projector import Projector, WCSAxesProjector
        if isinstance(ax_or_projector, Projector):
            self.projector = ax_or_projector
            # ``self.ax`` stays available on mpl projectors for back-compat
            # with ``render()`` / ``render_boundary()`` / ``contains_point``;
            # non-mpl projectors leave it ``None``.
            self.ax = getattr(ax_or_projector, 'ax', None)
        else:
            self.ax = ax_or_projector
            self.projector = WCSAxesProjector(ax_or_projector)
        self._geom = None
        # ``_frame_poly`` is now a thin alias for the projector's frame
        # polygon; preserved as an instance attribute so existing tests
        # and downstream code that read ``region._frame_poly`` directly
        # continue to work.
        self._frame_poly = self.projector.frame_polygon

    @classmethod
    def from_points(cls, ax_or_projector: Any,
                    lons: SkyCoord | npt.ArrayLike, lats: Any = None, *,
                    hull: str = 'convex', ratio: float = 0.3,
                    resolution: int = 200) -> CompoundRegion:
        """Build a region enclosing a scatter of points, via a convex or
        concave hull.

        The hull is computed in a gnomonic tangent plane about the points'
        centroid — so its straight edges are great circles (a true spherical
        hull for a localized footprint) — then added as a spherical polygon.
        Handy for defining a survey / instrument / detection footprint from
        real sources, which then feeds :meth:`solid_angle`, :meth:`clip`,
        :func:`~skyplothelper.region_search`, etc.

        Parameters
        ----------
        ax_or_projector : WCSAxes or Projector
            The frame the region lives on.
        lons, lats : array-like or SkyCoord
            Point coordinates in degrees (or a SkyCoord array in ``lons``).
        hull : {'convex', 'concave'}
            Hull type. ``'concave'`` (alpha-shape-like) hugs the points more
            tightly than the convex hull.
        ratio : float
            Concave-hull tightness in [0, 1] (shapely ``concave_hull`` ratio):
            near 0 = tightest / most detailed, 1 = the convex hull. Ignored
            when ``hull='convex'``.
        resolution : int
            Great-circle densification per hull edge when the hull is added as
            a spherical polygon.

        Returns
        -------
        region : CompoundRegion

        Notes
        -----
        The points must fit within a hemisphere of their centroid (the gnomonic
        hull is undefined otherwise); split very large point sets into separate
        footprints.
        """
        import shapely
        from shapely.geometry import MultiPoint

        region = cls(ax_or_projector)
        lons, lats = _parse_coords(lons, lats, wcs=region._parse_wcs)
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        if lons.size < 3:
            raise ValueError(
                "CompoundRegion.from_points needs at least 3 points.")

        # Unit vectors + centroid direction.
        lon_r, lat_r = np.radians(lons), np.radians(lats)
        xyz = np.column_stack([np.cos(lat_r) * np.cos(lon_r),
                               np.cos(lat_r) * np.sin(lon_r),
                               np.sin(lat_r)])
        c = xyz.mean(axis=0)
        c = c / np.linalg.norm(c)

        # Local tangent frame (east, north) at the centroid.
        ref = (np.array([0., 0., 1.]) if abs(c[2]) < 0.99
               else np.array([1., 0., 0.]))
        east = np.cross(ref, c)
        east /= np.linalg.norm(east)
        north = np.cross(c, east)

        # Gnomonic projection of the near-side points (centroid . point > 0),
        # where straight lines are great circles → a planar hull is a spherical
        # (great-circle) hull.
        denom = xyz @ c
        good = denom > 1e-6
        if int(good.sum()) < 3:
            raise ValueError(
                "from_points: the points span more than a hemisphere; the "
                "gnomonic hull is undefined. Split the set into smaller "
                "footprints.")
        gx = (xyz[good] @ east) / denom[good]
        gy = (xyz[good] @ north) / denom[good]
        mp = MultiPoint(np.column_stack([gx, gy]))
        if hull == 'concave':
            poly = shapely.concave_hull(mp, ratio=ratio)
        elif hull == 'convex':
            poly = mp.convex_hull
        else:
            raise ValueError(
                f"hull must be 'convex' or 'concave', got {hull!r}")
        if getattr(poly, 'geom_type', None) != 'Polygon':
            raise ValueError(
                "from_points: the hull did not form a single polygon (need "
                ">= 3 non-collinear points).")

        # Un-project the hull boundary back to lon/lat and add it.
        hx, hy = np.asarray(poly.exterior.coords).T
        d = (c[None, :] + hx[:, None] * east[None, :]
             + hy[:, None] * north[None, :])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        hlat = np.degrees(np.arcsin(np.clip(d[:, 2], -1.0, 1.0)))
        hlon = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        region.add_polygon(hlon, hlat, resolution=resolution)
        return region

    @classmethod
    def from_polygons(cls, ax_or_projector: Any, polygons: Any, *,
                      resolution: int = 200) -> CompoundRegion:
        """Build a region from many polygons at once via a single union.

        Faster than repeated :meth:`add_polygon` (which unions incrementally,
        O(n^2) for many rings).

        Parameters
        ----------
        ax_or_projector : WCSAxes or Projector
        polygons : sequence of (lons, lats)
            Each item is a ring: two array-likes of degrees (or a SkyCoord in
            the first slot). Rings are auto-closed.
        resolution : int
            Great-circle densification per edge.
        """
        import shapely

        region = cls(ax_or_projector)
        geoms = []
        for lons, lats in polygons:
            g = region._project_polygon(lons, lats, resolution=resolution)
            if g is not None and not g.is_empty:
                geoms.append(g)
        region._geom = shapely.unary_union(geoms) if geoms else None
        return region

    @classmethod
    def from_healpix_mask(cls, ax_or_projector: Any, mask: npt.ArrayLike, *,
                          nest: bool = False) -> CompoundRegion:
        """Build a region from a boolean HEALPix mask (the union of its True
        pixels).

        Pixel angles are interpreted in the axes' sky frame. The inverse of
        :meth:`to_healpix_mask`. Cost scales with the number of True pixels —
        degrade a fine mask first if needed.
        """
        import healpy as hp
        import shapely

        from ..healpix import healpix_pixel_corners

        mask = np.asarray(mask, dtype=bool)
        nside = hp.npix2nside(mask.size)
        pix = np.flatnonzero(mask)
        if pix.size == 0:
            raise ValueError("from_healpix_mask: the mask is empty (all False).")
        region = cls(ax_or_projector)
        lons_list, lats_list = healpix_pixel_corners(pix, nside, nest=nest)
        geoms = []
        for plon, plat in zip(lons_list, lats_list):
            g = region.projector.project_polygon(
                np.asarray(plon, dtype=float), np.asarray(plat, dtype=float),
                clip='d3')
            if g is not None and not g.is_empty:
                geoms.append(g)
        region._geom = shapely.unary_union(geoms) if geoms else None
        return region

    def to_healpix_mask(self, nside: int, *, nest: bool = False) -> np.ndarray:
        """Rasterize the region to a boolean HEALPix mask.

        Returns a boolean array of length ``12 * nside**2`` — True where a pixel
        *center* falls inside the region. Pixel angles are interpreted in the
        region's own sky frame. Turns a footprint into a survey mask (feeds
        HEALPix coverage maps, and the inverse of :meth:`from_healpix_mask`).
        """
        import healpy as hp

        npix = hp.nside2npix(int(nside))
        lon, lat = hp.pix2ang(int(nside), np.arange(npix), nest=nest,
                              lonlat=True)
        return self.contains_points(lon, lat)

    @property
    def centroid(self) -> tuple[float, float]:
        """``(lon, lat)`` of the region's area centroid in degrees (region
        frame); ``(nan, nan)`` if empty. Requires a FITS-WCS frame."""
        if self._geom is None or self._geom.is_empty:
            return (float('nan'), float('nan'))
        wcs = getattr(self.projector, 'wcs', None)
        if wcs is None:
            raise NotImplementedError(
                "centroid requires a FITS-WCS frame (no pixel->world inverse "
                "on non-FITS projections here).")
        c = self._geom.centroid
        lon, lat = wcs.pixel_to_world_values(c.x, c.y)
        return (float(lon), float(lat))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(lon_min, lon_max, lat_min, lat_max)`` of the region in degrees
        (region frame). Longitude bounds are naive (no antimeridian-wrap
        handling); requires a FITS-WCS frame."""
        if self._geom is None or self._geom.is_empty:
            return (float('nan'),) * 4
        wcs = getattr(self.projector, 'wcs', None)
        if wcs is None:
            raise NotImplementedError("bounds requires a FITS-WCS frame.")
        parts = (self._geom.geoms if self._geom.geom_type.startswith('Multi')
                 else [self._geom])
        xs: list[float] = []
        ys: list[float] = []
        for g in parts:
            x, y = g.exterior.coords.xy
            xs.extend(x)
            ys.extend(y)
        lon, lat = wcs.pixel_to_world_values(np.asarray(xs), np.asarray(ys))
        return (float(np.min(lon)), float(np.max(lon)),
                float(np.min(lat)), float(np.max(lat)))

    def _boundary_rings_lonlat(self) -> list[Any]:
        """Per-polygon boundary rings unprojected to sky coords: a list of
        ``((ext_lon, ext_lat), [(hole_lon, hole_lat), ...])``. FITS-WCS only."""
        if self._geom is None or self._geom.is_empty:
            return []
        wcs = getattr(self.projector, 'wcs', None)
        if wcs is None:
            raise NotImplementedError(
                "region export requires a FITS-WCS frame.")
        parts = (self._geom.geoms if self._geom.geom_type.startswith('Multi')
                 else [self._geom])
        out = []
        for g in parts:
            if getattr(g, 'geom_type', None) != 'Polygon':
                continue
            ex, ey = g.exterior.coords.xy
            ext = wcs.pixel_to_world_values(np.asarray(ex), np.asarray(ey))
            holes = []
            for ring in g.interiors:
                hx, hy = ring.coords.xy
                holes.append(
                    wcs.pixel_to_world_values(np.asarray(hx), np.asarray(hy)))
            out.append((ext, holes))
        return out

    def to_ds9(self, path: str | None = None,
               frame: str | None = None) -> str:
        """Export the region to DS9 region format (``.reg``).

        Exterior rings become ``polygon(...)``; interior holes become excluded
        ``-polygon(...)``. Coordinates are in the region's sky frame (or the
        ``frame`` override). Returns the text; also writes it to ``path`` if
        given. Requires a FITS-WCS frame.
        """
        fname = (frame or getattr(self.projector, 'wcs_frame', None)
                 or 'icrs').lower()
        ds9_frame = {'icrs': 'icrs', 'fk5': 'fk5', 'fk4': 'fk4',
                     'galactic': 'galactic',
                     'ecliptic': 'ecliptic'}.get(fname, 'icrs')
        lines = ['# Region file format: DS9 version 4.1',
                 'global color=green', ds9_frame]
        for (ext_lon, ext_lat), holes in self._boundary_rings_lonlat():
            coords = ','.join(f'{lo:.6f},{la:.6f}'
                              for lo, la in zip(ext_lon, ext_lat))
            lines.append(f'polygon({coords})')
            for hlon, hlat in holes:
                hc = ','.join(f'{lo:.6f},{la:.6f}'
                              for lo, la in zip(hlon, hlat))
                lines.append(f'-polygon({hc})')
        text = '\n'.join(lines) + '\n'
        if path is not None:
            with open(path, 'w') as fh:
                fh.write(text)
        return text

    def to_regions(self, frame: str | None = None) -> Any:
        """Convert to an astropy-``regions`` object — a ``Regions`` container of
        ``PolygonSkyRegion`` (one per exterior ring).

        Requires the optional ``regions`` package (``pip install regions``).
        Interior holes are not represented in the returned polygons.
        """
        try:
            from regions import PolygonSkyRegion, Regions
        except ImportError as e:
            raise ImportError(
                "CompoundRegion.to_regions requires the optional 'regions' "
                "package (pip install regions).") from e
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        fname = (frame or getattr(self.projector, 'wcs_frame', None)
                 or 'icrs').lower()
        regs = []
        for (ext_lon, ext_lat), _holes in self._boundary_rings_lonlat():
            sc = SkyCoord(np.asarray(ext_lon) * u.deg,
                          np.asarray(ext_lat) * u.deg, frame=fname)
            regs.append(PolygonSkyRegion(vertices=sc))
        return Regions(regs)

    def to_crtf(self, path: str | None = None,
                frame: str | None = None) -> str:
        """Export to CASA Region Text Format (CRTF, ``.crtf``) — polygons that
        CASA (``viewer``, ``tclean`` masks, etc.) can read.

        Each exterior ring becomes a ``poly[[lon deg, lat deg], ...]`` line.
        Interior holes are not represented. Coordinates are in the region's sky
        frame (or the ``frame`` override). Returns the text; also writes to
        ``path`` if given. Requires a FITS-WCS frame.
        """
        fname = (frame or getattr(self.projector, 'wcs_frame', None)
                 or 'icrs').lower()
        crtf_frame = {'icrs': 'ICRS', 'fk5': 'J2000', 'fk4': 'B1950',
                      'galactic': 'GALACTIC',
                      'ecliptic': 'ECLIPTIC'}.get(fname, 'ICRS')
        lines = ['#CRTFv0']
        for (ext_lon, ext_lat), _holes in self._boundary_rings_lonlat():
            pts = ', '.join(f'[{lo:.6f}deg, {la:.6f}deg]'
                            for lo, la in zip(ext_lon, ext_lat))
            lines.append(f'poly[{pts}] coord={crtf_frame}')
        text = '\n'.join(lines) + '\n'
        if path is not None:
            with open(path, 'w') as fh:
                fh.write(text)
        return text

    @staticmethod
    def _read_region_text(source: str) -> str:
        """Return region text from either a path or a raw string."""
        import os
        if ('\n' not in source) and os.path.exists(source):
            with open(source) as fh:
                return fh.read()
        return source

    @classmethod
    def from_ds9(cls, ax_or_projector: Any, source: str, *,
                 frame: str | None = None) -> CompoundRegion:
        """Build a region from a DS9 region file/text — the inverse of
        :meth:`to_ds9`.

        Parses ``polygon(...)`` and ``circle(x, y, r)`` shapes (and their
        excluded ``-polygon`` / ``-circle`` forms → subtract). Coordinates are
        read in the file's frame line (or the ``frame`` override) and converted
        to the axes frame. Other DS9 shapes are ignored; for full-fidelity
        parsing use astropy ``regions`` + :meth:`from_regions`.
        """
        import re

        import astropy.units as u
        from astropy.coordinates import SkyCoord

        _alias = {'j2000': 'fk5', 'b1950': 'fk4'}
        file_frame = frame
        region = cls(ax_or_projector)
        for raw in cls._read_region_text(source).splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            low = line.lower()
            if low in ('icrs', 'fk5', 'fk4', 'galactic', 'ecliptic',
                       'j2000', 'b1950'):
                file_frame = frame or low
                continue
            exclude = line.startswith('-')
            body = line[1:].strip() if exclude else line
            m = re.match(r'(polygon|circle)\s*\(([^)]*)\)', body, re.I)
            if not m:
                continue
            kind = m.group(1).lower()
            nums = [float(v) for v in
                    re.split(r'[,\s]+', m.group(2).strip()) if v]
            fr = _alias.get((file_frame or 'icrs').lower(),
                            (file_frame or 'icrs').lower())
            if kind == 'polygon' and len(nums) >= 6:
                sc = SkyCoord(nums[0::2] * u.deg, nums[1::2] * u.deg, frame=fr)
                (region.subtract_polygon if exclude
                 else region.add_polygon)(sc)
            elif kind == 'circle' and len(nums) >= 3:
                sc = SkyCoord(nums[0] * u.deg, nums[1] * u.deg, frame=fr)
                (region.subtract_circle if exclude
                 else region.add_circle)(sc, radius_deg=nums[2])
        return region

    @classmethod
    def from_crtf(cls, ax_or_projector: Any, source: str, *,
                  frame: str | None = None) -> CompoundRegion:
        """Build a region from CASA CRTF text — the inverse of :meth:`to_crtf`.

        Parses ``poly[[lon deg, lat deg], ...]`` (and ``circle[[x, y], r deg]``)
        lines; a leading ``-`` subtracts. Frame from each line's ``coord=`` (or
        the ``frame`` override).
        """
        import re

        import astropy.units as u
        from astropy.coordinates import SkyCoord

        _alias = {'j2000': 'fk5', 'b1950': 'fk4'}
        region = cls(ax_or_projector)
        for raw in cls._read_region_text(source).splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            exclude = line.startswith('-')
            cm = re.search(r'coord\s*=\s*(\w+)', line, re.I)
            fname = (frame or (cm.group(1) if cm else 'icrs')).lower()
            fr = _alias.get(fname, fname)
            if re.match(r'-?\s*poly', line, re.I):
                pairs = re.findall(
                    r'\[\s*([-\d.eE]+)\s*deg\s*,\s*([-\d.eE]+)\s*deg\s*\]', line)
                if len(pairs) < 3:
                    continue
                lon = [float(a) for a, _ in pairs]
                lat = [float(b) for _, b in pairs]
                sc = SkyCoord(lon * u.deg, lat * u.deg, frame=fr)
                (region.subtract_polygon if exclude
                 else region.add_polygon)(sc)
        return region

    @classmethod
    def from_regions(cls, ax_or_projector: Any,
                     region_obj: Any) -> CompoundRegion:
        """Build a region from astropy-``regions`` sky-region object(s).

        Accepts a single ``SkyRegion``, a list, or a ``Regions`` container.
        ``PolygonSkyRegion`` and ``CircleSkyRegion`` are supported (union;
        ``meta['include'] == 0`` subtracts). The inverse of :meth:`to_regions`;
        requires the optional ``regions`` package for the input objects.
        """
        region = cls(ax_or_projector)
        try:
            objs = list(region_obj)
        except TypeError:
            objs = [region_obj]
        for r in objs:
            meta = getattr(r, 'meta', {}) or {}
            include = int(meta.get('include', 1)) if hasattr(meta, 'get') else 1
            tname = type(r).__name__
            if tname == 'PolygonSkyRegion':
                (region.subtract_polygon if not include
                 else region.add_polygon)(r.vertices)
            elif tname == 'CircleSkyRegion':
                rad = r.radius.to('deg').value
                (region.subtract_circle if not include
                 else region.add_circle)(r.center, radius_deg=rad)
        return region

    @property
    def _parse_wcs(self) -> Any:
        """Convenience accessor for the projector's WCS (``None`` for
        non-mpl projectors). Used by the various ``_parse_coord`` /
        ``_parse_coords`` calls below — those helpers accept
        ``wcs=None`` and skip sexagesimal/hourangle parsing in that
        case, which is exactly what plotly-side users want."""
        return getattr(self.projector, 'wcs', None)

    @property
    def _parse_frame(self) -> Any:
        """The projector's sky frame NAME.

        Every backend reports this (it is part of the Projector interface),
        including ones with no WCS object — the plotly projector carries a
        frame string instead. Relying on ``_parse_wcs`` alone silently meant
        ICRS there, so a non-ICRS SkyCoord was mis-converted on the plotly
        side while the mpl side was correct.
        """
        return getattr(self.projector, 'wcs_frame', None)

    def _clip_to_frame(self, geom: Any) -> Any:
        if geom is None or geom.is_empty:
            return None
        try:
            clipped = geom.intersection(self._frame_poly)
            return clipped if not clipped.is_empty else None
        except Exception:
            return geom

    def _apply(self, geom: Any, operation: str) -> None:
        geom = self._clip_to_frame(geom)
        if geom is None:
            return
        if self._geom is None:
            if operation == 'union':
                self._geom = geom
            elif operation == 'difference':
                self._geom = self._frame_poly.difference(geom)
            elif operation == 'intersection':
                self._geom = geom
            elif operation == 'symmetric_difference':
                self._geom = geom
        else:
            if operation == 'union':
                self._geom = self._geom.union(geom)
            elif operation == 'difference':
                self._geom = self._geom.difference(geom)
            elif operation == 'intersection':
                self._geom = self._geom.intersection(geom)
            elif operation == 'symmetric_difference':
                self._geom = self._geom.symmetric_difference(geom)
        if self._geom is not None:
            self._geom = self._clip_to_frame(self._geom)

    def _project_band_geom(self, lons: Any, lats: Any, **kw: Any) -> Any:
        """Project a band / cap polygon, preferring the bounded-field clip path.

        Global bands and large caps have most of their vertices off a zoomed
        field frame, where the all-sky seam / complement heuristics in
        ``project_polygon`` misfire (empty, or a bad complement flip that fills
        the whole field). ``project_field_clipped`` clips the spherical polygon
        to the field's world box first and returns ``None`` on all-sky frames,
        where we fall back to the standard projection unchanged."""
        clipped = self.projector.project_field_clipped(lons, lats)
        if clipped is not None:
            return clipped
        return self.projector.project_polygon(lons, lats, **kw)

    def _project_circle(self, lon: SkyCoord | float, lat: Any = None,
                        radius_deg: Any = None, resolution: int = 200,
                        clip: str = 'auto') -> Any:
        clip = _resolve_clip(clip, helper_name='CompoundRegion.add_circle')
        _orig_lat = lat
        lon, lat, shifted = _parse_coord(
            lon, lat, wcs=getattr(self.projector, 'wcs', None))
        if shifted:
            radius_deg = _orig_lat
        radius_deg = _parse_angle(radius_deg)
        if radius_deg is None:
            raise ValueError("radius_deg is required")
        lons, lats = geodesic_circle(lon, lat, radius_deg, resolution)
        # Expected solid-angle fraction lets the projector run a
        # complement-detection flip when the stitched-polygon
        # orientation comes out inverted (small discs straddling the
        # projection wrap edge are the canonical case).
        expected_frac = (1.0 - np.cos(np.radians(radius_deg))) / 2.0
        return self.projector.project_polygon(
            lons, lats, clip=clip,
            lat_center=lat, radius_deg=radius_deg,
            expected_frac=expected_frac)

    def _project_polygon(self, lons: npt.ArrayLike, lats: npt.ArrayLike | None = None,
                         resolution: int = 100, geodesic: bool | str = 'auto',
                         geodesic_threshold: float = 10.0,
                         clip: str = 'auto') -> Any:
        clip = _resolve_clip(clip, helper_name='CompoundRegion.add_polygon')
        geodesic = _resolve_geodesic_for_clip(geodesic, clip)
        lons, lats = _parse_coords(
            lons, lats, wcs=getattr(self.projector, 'wcs', None))
        if not (np.isclose(lons[0], lons[-1]) and np.isclose(lats[0], lats[-1])):
            lons = np.append(lons, lons[0])
            lats = np.append(lats, lats[0])
        if resolution > 0:
            lons, lats = _densify_polygon_edges(
                lons, lats, resolution=resolution,
                geodesic=geodesic, geodesic_threshold=geodesic_threshold)
        return self.projector.project_polygon(lons, lats, clip=clip)

    def _project_band(self, lat_min: Any, lat_max: Any, resolution: int = 360,
                      clip: str = 'auto') -> Any:
        clip = _resolve_clip(clip,
                              helper_name='CompoundRegion.add_latitude_band')
        lat_min = _parse_angle(lat_min)
        lat_max = _parse_angle(lat_max)
        _EPS = 1e-4
        n_side = max(3, resolution // 20)
        # Anchor the polygon's natural seam at the projection's
        # antimeridian (lon = lon_center ± 180), not at lon = ±180. On a
        # center=180 frame the latter wraps to the central meridian, so
        # render_boundary draws the split-seam as a visible interior
        # edge instead of suppressing it as a frame-coincident edge.
        lon_center = self.projector.center
        lon_lo = lon_center - 180 + _EPS
        lon_hi = lon_center + 180 - _EPS
        lons = np.concatenate([
            np.linspace(lon_lo, lon_hi, resolution),
            np.full(n_side - 1, lon_hi),
            np.linspace(lon_hi, lon_lo, resolution),
            np.full(n_side - 1, lon_lo),
        ])
        lats = np.concatenate([
            np.full(resolution, lat_min),
            np.linspace(lat_min, lat_max, n_side)[1:],
            np.full(resolution, lat_max),
            np.linspace(lat_max, lat_min, n_side)[1:],
        ])
        lat_c = (lat_min + lat_max) / 2
        r = (lat_max - lat_min) / 2
        sin_range = abs(np.sin(np.radians(lat_max))
                        - np.sin(np.radians(lat_min)))
        expected_frac = sin_range / 2.0
        return self._project_band_geom(
            lons, lats, clip=clip,
            expected_frac=expected_frac,
            lat_center=lat_c, radius_deg=r)

    def _project_frame_band(self, lat_min: Any, lat_max: Any, frame: str,
                            resolution: int = 500, clip: str = 'auto') -> Any:
        clip = _resolve_clip(clip,
                              helper_name='CompoundRegion.add_frame_band')
        lat_min = _parse_angle(lat_min)
        lat_max = _parse_angle(lat_max)
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        fr = 'geocentrictrueecliptic' if frame == 'ecliptic' else frame

        # Build each polar cap as a geodesic circle around the source
        # frame's pole (expressed in ICRS), then take frame - caps —
        # mirroring how _project_great_circle_band builds its caps. The
        # earlier (lon, lat) cap polygon closed itself with connector
        # edges at src_lon = eps / 360 - eps; for a polar cap those left
        # a thin unsubtracted slit that survived frame - caps as a
        # visible seam sliver running to the pole. A closed circle around
        # the pole has no such connector, so the band comes out clean.
        def _project_cap(pole_lat: float, radius_deg: float) -> Any:
            radius_deg = max(1e-4, min(180 - 1e-4, radius_deg))
            pole = SkyCoord(0 * u.deg, pole_lat * u.deg, frame=fr).icrs
            c_lons, c_lats = geodesic_circle(
                pole.ra.deg, pole.dec.deg, radius_deg, resolution)
            exp = (1.0 - np.cos(np.radians(radius_deg))) / 2.0
            g = self._project_band_geom(
                c_lons, c_lats, clip=clip,
                lat_center=pole.dec.deg, radius_deg=radius_deg,
                expected_frac=exp)
            return g if (g is not None and not g.is_empty) else None

        caps = []
        if lat_min > -89.9:
            # South cap: everything at lat <= lat_min, i.e. within
            # (90 + lat_min)° of the source-frame south pole.
            g = _project_cap(-90.0, 90.0 + lat_min)
            if g is not None:
                caps.append(g)
        if lat_max < 89.9:
            # North cap: everything at lat >= lat_max, i.e. within
            # (90 - lat_max)° of the source-frame north pole.
            g = _project_cap(90.0, 90.0 - lat_max)
            if g is not None:
                caps.append(g)

        if caps:
            return self._frame_poly.difference(unary_union(caps))
        return self._frame_poly

    # --- Circle operations ---

    def add_circle(self, lon: SkyCoord | float, lat: Any = None, radius_deg: Any = None,
                   resolution: int = 200, clip: str = 'auto') -> CompoundRegion:
        """Union a geodesic circle into the region."""
        self._apply(self._project_circle(lon, lat, radius_deg, resolution,
                                           clip=clip), 'union')
        return self

    def subtract_circle(self, lon: SkyCoord | float, lat: Any = None, radius_deg: Any = None,
                        resolution: int = 200,
                        clip: str = 'auto') -> CompoundRegion:
        """Subtract a geodesic circle from the region."""
        self._apply(self._project_circle(lon, lat, radius_deg, resolution,
                                           clip=clip), 'difference')
        return self

    def intersect_circle(self, lon: SkyCoord | float, lat: Any = None, radius_deg: Any = None,
                         resolution: int = 200,
                         clip: str = 'auto') -> CompoundRegion:
        """Intersect the region with a geodesic circle."""
        self._apply(self._project_circle(lon, lat, radius_deg, resolution,
                                           clip=clip), 'intersection')
        return self

    # --- Polygon operations ---

    def add_polygon(self, lons: npt.ArrayLike, lats: npt.ArrayLike | None = None,
                    resolution: int = 100, geodesic: bool | str = 'auto',
                    geodesic_threshold: float = 10.0,
                    clip: str = 'auto') -> CompoundRegion:
        """Union a spherical polygon into the region."""
        self._apply(self._project_polygon(lons, lats, resolution, geodesic,
                                           geodesic_threshold, clip=clip),
                     'union')
        return self

    def subtract_polygon(self, lons: npt.ArrayLike,
                         lats: npt.ArrayLike | None = None, resolution: int = 100,
                         geodesic: bool | str = 'auto', geodesic_threshold: float = 10.0,
                         clip: str = 'auto') -> CompoundRegion:
        """Subtract a spherical polygon from the region."""
        self._apply(self._project_polygon(lons, lats, resolution, geodesic,
                                           geodesic_threshold, clip=clip),
                     'difference')
        return self

    def intersect_polygon(self, lons: npt.ArrayLike,
                          lats: npt.ArrayLike | None = None, resolution: int = 100,
                          geodesic: bool | str = 'auto', geodesic_threshold: float = 10.0,
                          clip: str = 'auto') -> CompoundRegion:
        """Intersect the region with a spherical polygon."""
        self._apply(self._project_polygon(lons, lats, resolution, geodesic,
                                           geodesic_threshold, clip=clip),
                     'intersection')
        return self

    # --- ICRS latitude band operations ---

    def add_latitude_band(self, lat_min: Any, lat_max: Any, resolution: int = 360,
                          clip: str = 'auto') -> CompoundRegion:
        """Union an ICRS latitude band into the region."""
        self._apply(self._project_band(lat_min, lat_max, resolution,
                                         clip=clip), 'union')
        return self

    def intersect_latitude_band(self, lat_min: Any, lat_max: Any,
                                resolution: int = 360,
                                clip: str = 'auto') -> CompoundRegion:
        """Intersect the region with an ICRS latitude band."""
        self._apply(self._project_band(lat_min, lat_max, resolution,
                                         clip=clip), 'intersection')
        return self

    def subtract_latitude_band(self, lat_min: Any, lat_max: Any,
                               resolution: int = 360,
                               clip: str = 'auto') -> CompoundRegion:
        """Subtract an ICRS latitude band from the region."""
        self._apply(self._project_band(lat_min, lat_max, resolution,
                                         clip=clip), 'difference')
        return self

    # --- Cross-frame band operations ---

    def add_frame_band(self, lat_min: Any, lat_max: Any, frame: str = 'galactic',
                       resolution: int = 500,
                       clip: str = 'auto') -> CompoundRegion:
        """Union a cross-frame latitude band (D3-style projection)."""
        self._apply(self._project_frame_band(lat_min, lat_max, frame, resolution,
                                                clip=clip), 'union')
        return self

    def subtract_frame_band(self, lat_min: Any, lat_max: Any,
                            frame: str = 'galactic', resolution: int = 500,
                            clip: str = 'auto') -> CompoundRegion:
        """Subtract a cross-frame latitude band."""
        self._apply(self._project_frame_band(lat_min, lat_max, frame, resolution,
                                                clip=clip), 'difference')
        return self

    # --- Cross-frame lon/lat box ---

    def _project_lonlat_box(self, lat_min: Any, lat_max: Any, lon_min: Any,
                            lon_max: Any, frame: str = 'galactic',
                            resolution: int = 100, clip: str = 'auto') -> Any:
        """Project a closed lon/lat-aligned box defined in *frame*.

        Densifies all four edges (two lat-constant + two
        lon-constant) in the source frame, converts to ICRS via
        :class:`~astropy.coordinates.SkyCoord`, and pipes through the
        same antimeridian-clip + stitch-and-project machinery used by
        :meth:`_project_frame_band`. Returns the projected geometry as
        a shapely polygon, or ``None`` if the box doesn't intersect
        the visible region.

        Polar-touching edges (``lat_max >= 89.9`` or
        ``lat_min <= -89.9``) collapse to single points in the source
        frame — these edges are skipped in the densification and the
        box outline walks only the two non-degenerate meridional
        edges. Longitude wrap (``lon_max < lon_min`` meaning the box
        wraps the antimeridian of the source frame) is normalized by
        bumping ``lon_max`` by 360°.
        """
        clip = _resolve_clip(clip,
                              helper_name='CompoundRegion.add_lonlat_box')
        lat_min = _parse_angle(lat_min)
        lat_max = _parse_angle(lat_max)
        lon_min = _parse_angle(lon_min)
        lon_max = _parse_angle(lon_max)
        if lat_min >= lat_max:
            raise ValueError("lat_min must be less than lat_max")
        if lon_max < lon_min:
            # Box wraps the source-frame antimeridian — normalize so
            # the linspace below traces the box, not its complement.
            lon_max += 360.0

        from astropy import units as u
        from astropy.coordinates import SkyCoord

        fr = 'geocentrictrueecliptic' if frame == 'ecliptic' else frame

        touches_north = lat_max >= 89.9
        touches_south = lat_min <= -89.9

        n_side = max(5, resolution // 5)

        # Walk the box outline counter-clockwise (in source frame):
        # south edge → east edge (lon=lon_max) → north edge → west
        # edge (lon=lon_min). Polar-touching edges collapse to points
        # and are skipped — the remaining two meridional edges meet
        # at the pole and produce a valid closed region.
        src_lons_parts = []
        src_lats_parts = []
        if not touches_south:
            src_lons_parts.append(np.linspace(lon_min, lon_max, resolution))
            src_lats_parts.append(np.full(resolution, lat_min))
        src_lons_parts.append(np.full(n_side, lon_max))
        src_lats_parts.append(np.linspace(lat_min, lat_max, n_side))
        if not touches_north:
            src_lons_parts.append(np.linspace(lon_max, lon_min, resolution))
            src_lats_parts.append(np.full(resolution, lat_max))
        src_lons_parts.append(np.full(n_side, lon_min))
        src_lats_parts.append(np.linspace(lat_max, lat_min, n_side))

        src_lons = np.concatenate(src_lons_parts) % 360.0
        src_lats = np.concatenate(src_lats_parts)

        coords = SkyCoord(src_lons * u.deg, src_lats * u.deg, frame=fr)
        # Transform to the *axes* frame (not blindly to ICRS) so the
        # pixel-projection step below feeds ``ax.wcs.world_to_pixel``
        # coords it understands. Falls back to ICRS when the axes
        # frame can't be detected — preserves behavior on plain
        # ICRS WCSAxes.
        # ``wcs_frame`` is part of the Projector interface — every backend
        # (mpl WCSAxes, plotly, pixel/offset) reports the axes' sky frame.
        ax_lons, ax_lats = _coords_to_frame_deg(coords,
                                                self.projector.wcs_frame)

        # Expected-area fraction for the complement-detection heuristic
        # in _stitch_and_project: spherical area of the box divided by
        # full sphere area (4π → /2 below since frame_poly is a
        # half-area normalized projection unit).
        lon_span = (lon_max - lon_min)
        sin_range = abs(np.sin(np.radians(lat_max))
                        - np.sin(np.radians(lat_min)))
        expected_frac = (lon_span / 360.0) * sin_range / 2.0

        return self.projector.project_polygon(
            ax_lons, ax_lats, clip=clip,
            expected_frac=expected_frac)

    def add_lonlat_box(self, lat_min: Any, lat_max: Any, lon_min: Any,
                       lon_max: Any, frame: str = 'galactic',
                       resolution: int = 100,
                       clip: str = 'auto') -> CompoundRegion:
        """Union a closed lon/lat-aligned box defined in *frame*.

        Cross-frame extension of :meth:`add_latitude_band` —
        ``add_latitude_band`` covers all 360° of longitude in the
        axes' own frame; ``add_lonlat_box`` covers a longitude slice
        ``[lon_min, lon_max]`` in an arbitrary source frame. Useful
        for surveys whose footprint is naturally described as a
        rectangle in a non-axes frame (e.g. eROSITA's
        ``l=180..360, b=-90..+90`` half-sky in galactic coords).

        Polar-touching edges (``lat_max >= 89.9`` /
        ``lat_min <= -89.9``) are handled: the corresponding edge
        collapses to a point and is omitted from the outline walk.
        Longitude wrap (``lon_max < lon_min``) is normalized.
        """
        self._apply(
            self._project_lonlat_box(lat_min, lat_max, lon_min, lon_max,
                                       frame, resolution, clip=clip),
            'union')
        return self

    def subtract_lonlat_box(self, lat_min: Any, lat_max: Any, lon_min: Any,
                            lon_max: Any, frame: str = 'galactic',
                            resolution: int = 100,
                            clip: str = 'auto') -> CompoundRegion:
        """Subtract a closed lon/lat-aligned box defined in *frame*."""
        self._apply(
            self._project_lonlat_box(lat_min, lat_max, lon_min, lon_max,
                                       frame, resolution, clip=clip),
            'difference')
        return self

    def intersect_lonlat_box(self, lat_min: Any, lat_max: Any, lon_min: Any,
                             lon_max: Any, frame: str = 'galactic',
                             resolution: int = 100,
                             clip: str = 'auto') -> CompoundRegion:
        """Intersect the region with a closed lon/lat-aligned box
        defined in *frame*."""
        self._apply(
            self._project_lonlat_box(lat_min, lat_max, lon_min, lon_max,
                                       frame, resolution, clip=clip),
            'intersection')
        return self

    # --- Complement ---

    # --- Symmetric difference operations ---

    def xor_circle(self, lon: SkyCoord | float, lat: Any = None, radius_deg: Any = None,
                   resolution: int = 200, clip: str = 'auto') -> CompoundRegion:
        """Symmetric difference with a geodesic circle (toggle region)."""
        self._apply(self._project_circle(lon, lat, radius_deg, resolution,
                                           clip=clip),
                    'symmetric_difference')
        return self

    def xor_polygon(self, lons: npt.ArrayLike, lats: npt.ArrayLike | None = None,
                    resolution: int = 100, geodesic: bool | str = 'auto',
                    geodesic_threshold: float = 10.0,
                    clip: str = 'auto') -> CompoundRegion:
        """Symmetric difference with a spherical polygon."""
        self._apply(self._project_polygon(lons, lats, resolution, geodesic,
                                           geodesic_threshold, clip=clip),
                    'symmetric_difference')
        return self

    def xor_latitude_band(self, lat_min: Any, lat_max: Any, resolution: int = 360,
                          clip: str = 'auto') -> CompoundRegion:
        """Symmetric difference with an ICRS latitude band."""
        self._apply(self._project_band(lat_min, lat_max, resolution,
                                         clip=clip),
                    'symmetric_difference')
        return self

    def xor_frame_band(self, lat_min: Any, lat_max: Any, frame: str = 'galactic',
                       resolution: int = 500,
                       clip: str = 'auto') -> CompoundRegion:
        """Symmetric difference with a cross-frame latitude band."""
        self._apply(self._project_frame_band(lat_min, lat_max, frame, resolution,
                                                clip=clip),
                    'symmetric_difference')
        return self

    # --- Rectangle operations ---

    def _project_rectangle(self, lon: SkyCoord | float, lat: Any = None, width: Any = None,
                           height: Any = None, angle: Any = 0, resolution: int = 50,
                           clip: str = 'auto', geodesic: bool | str = 'auto') -> Any:
        _orig_lat = lat
        lon, lat, shifted = _parse_coord(lon, lat, wcs=self._parse_wcs,
                                        frame_name=self._parse_frame)
        if shifted:
            width = _orig_lat
        width = _parse_angle(width)
        height = _parse_angle(height)
        angle = _parse_angle(angle) if angle is not None else 0
        if width is None:
            raise ValueError("width is required")
        if height is None:
            raise ValueError("height is required")
        lons, lats = rectangle(lon, lat, width, height, angle, resolution)
        return self._project_polygon(lons, lats, resolution=0,
                                       clip=clip, geodesic=geodesic)

    def add_rectangle(self, lon: SkyCoord | float, lat: Any = None, width: Any = None,
                      height: Any = None, angle: Any = 0, resolution: int = 50,
                      clip: str = 'auto',
                      geodesic: bool | str = 'auto') -> CompoundRegion:
        """Union a rectangle into the region."""
        self._apply(self._project_rectangle(lon, lat, width, height, angle,
                                              resolution, clip=clip,
                                              geodesic=geodesic), 'union')
        return self

    def subtract_rectangle(self, lon: SkyCoord | float, lat: Any = None, width: Any = None,
                           height: Any = None, angle: Any = 0, resolution: int = 50,
                           clip: str = 'auto',
                           geodesic: bool | str = 'auto') -> CompoundRegion:
        """Subtract a rectangle from the region."""
        self._apply(self._project_rectangle(lon, lat, width, height, angle,
                                              resolution, clip=clip,
                                              geodesic=geodesic), 'difference')
        return self

    def intersect_rectangle(self, lon: SkyCoord | float, lat: Any = None, width: Any = None,
                            height: Any = None, angle: Any = 0, resolution: int = 50,
                            clip: str = 'auto',
                            geodesic: bool | str = 'auto') -> CompoundRegion:
        """Intersect the region with a rectangle."""
        self._apply(self._project_rectangle(lon, lat, width, height, angle,
                                              resolution, clip=clip,
                                              geodesic=geodesic),
                     'intersection')
        return self

    def xor_rectangle(self, lon: SkyCoord | float, lat: Any = None, width: Any = None,
                      height: Any = None, angle: Any = 0, resolution: int = 50,
                      clip: str = 'auto',
                      geodesic: bool | str = 'auto') -> CompoundRegion:
        """Symmetric difference with a rectangle."""
        self._apply(self._project_rectangle(lon, lat, width, height, angle,
                                              resolution, clip=clip,
                                              geodesic=geodesic),
                     'symmetric_difference')
        return self

    # --- Square operations ---

    def add_square(self, lon: SkyCoord | float, lat: Any = None, size: Any = None,
                   angle: Any = 0, resolution: int = 50, clip: str = 'auto',
                   geodesic: bool | str = 'auto') -> CompoundRegion:
        """Union a square into the region."""
        _orig_lat = lat
        lon, lat, shifted = _parse_coord(lon, lat, wcs=self._parse_wcs,
                                        frame_name=self._parse_frame)
        if shifted:
            size = _orig_lat
        size = _parse_angle(size)
        angle = _parse_angle(angle) if angle is not None else 0
        if size is None:
            raise ValueError("size is required")
        return self.add_rectangle(lon, lat, size, size, angle, resolution,
                                   clip=clip, geodesic=geodesic)

    def subtract_square(self, lon: SkyCoord | float, lat: Any = None, size: Any = None,
                        angle: Any = 0, resolution: int = 50, clip: str = 'auto',
                        geodesic: bool | str = 'auto') -> CompoundRegion:
        """Subtract a square from the region."""
        _orig_lat = lat
        lon, lat, shifted = _parse_coord(lon, lat, wcs=self._parse_wcs,
                                        frame_name=self._parse_frame)
        if shifted:
            size = _orig_lat
        size = _parse_angle(size)
        angle = _parse_angle(angle) if angle is not None else 0
        if size is None:
            raise ValueError("size is required")
        return self.subtract_rectangle(lon, lat, size, size, angle, resolution,
                                        clip=clip, geodesic=geodesic)

    def intersect_square(self, lon: SkyCoord | float, lat: Any = None, size: Any = None,
                         angle: Any = 0, resolution: int = 50, clip: str = 'auto',
                         geodesic: bool | str = 'auto') -> CompoundRegion:
        """Intersect the region with a square."""
        _orig_lat = lat
        lon, lat, shifted = _parse_coord(lon, lat, wcs=self._parse_wcs,
                                        frame_name=self._parse_frame)
        if shifted:
            size = _orig_lat
        size = _parse_angle(size)
        angle = _parse_angle(angle) if angle is not None else 0
        if size is None:
            raise ValueError("size is required")
        return self.intersect_rectangle(lon, lat, size, size, angle, resolution,
                                         clip=clip, geodesic=geodesic)

    def xor_square(self, lon: SkyCoord | float, lat: Any = None, size: Any = None,
                   angle: Any = 0, resolution: int = 50, clip: str = 'auto',
                   geodesic: bool | str = 'auto') -> CompoundRegion:
        """Symmetric difference with a square."""
        _orig_lat = lat
        lon, lat, shifted = _parse_coord(lon, lat, wcs=self._parse_wcs,
                                        frame_name=self._parse_frame)
        if shifted:
            size = _orig_lat
        size = _parse_angle(size)
        angle = _parse_angle(angle) if angle is not None else 0
        if size is None:
            raise ValueError("size is required")
        return self.xor_rectangle(lon, lat, size, size, angle, resolution,
                                   clip=clip, geodesic=geodesic)

    # --- Ellipse operations ---

    def _project_ellipse(self, lon: SkyCoord | float, lat: Any = None, semi_major: Any = None,
                         semi_minor: Any = None, angle: Any = 0,
                         resolution: int = 200, clip: str = 'auto',
                         geodesic: bool | str = 'auto') -> Any:
        _orig_lat = lat
        lon, lat, shifted = _parse_coord(lon, lat, wcs=self._parse_wcs,
                                        frame_name=self._parse_frame)
        if shifted:
            semi_major = _orig_lat
        semi_major = _parse_angle(semi_major)
        semi_minor = _parse_angle(semi_minor)
        angle = _parse_angle(angle) if angle is not None else 0
        if semi_major is None:
            raise ValueError("semi_major is required")
        if semi_minor is None:
            raise ValueError("semi_minor is required")
        lons, lats = ellipse(lon, lat, semi_major, semi_minor, angle, resolution)
        return self._project_polygon(lons, lats, resolution=0,
                                       clip=clip, geodesic=geodesic)

    def add_ellipse(self, lon: SkyCoord | float, lat: Any = None, semi_major: Any = None,
                    semi_minor: Any = None, angle: Any = 0, resolution: int = 200,
                    clip: str = 'auto',
                    geodesic: bool | str = 'auto') -> CompoundRegion:
        """Union an ellipse into the region."""
        self._apply(self._project_ellipse(lon, lat, semi_major, semi_minor,
                                           angle, resolution,
                                           clip=clip, geodesic=geodesic),
                     'union')
        return self

    def subtract_ellipse(self, lon: SkyCoord | float, lat: Any = None, semi_major: Any = None,
                         semi_minor: Any = None, angle: Any = 0,
                         resolution: int = 200, clip: str = 'auto',
                         geodesic: bool | str = 'auto') -> CompoundRegion:
        """Subtract an ellipse from the region."""
        self._apply(self._project_ellipse(lon, lat, semi_major, semi_minor,
                                           angle, resolution,
                                           clip=clip, geodesic=geodesic),
                     'difference')
        return self

    def intersect_ellipse(self, lon: SkyCoord | float, lat: Any = None, semi_major: Any = None,
                          semi_minor: Any = None, angle: Any = 0,
                          resolution: int = 200, clip: str = 'auto',
                          geodesic: bool | str = 'auto') -> CompoundRegion:
        """Intersect the region with an ellipse."""
        self._apply(self._project_ellipse(lon, lat, semi_major, semi_minor,
                                           angle, resolution,
                                           clip=clip, geodesic=geodesic),
                     'intersection')
        return self

    def xor_ellipse(self, lon: SkyCoord | float, lat: Any = None, semi_major: Any = None,
                    semi_minor: Any = None, angle: Any = 0, resolution: int = 200,
                    clip: str = 'auto',
                    geodesic: bool | str = 'auto') -> CompoundRegion:
        """Symmetric difference with an ellipse."""
        self._apply(self._project_ellipse(lon, lat, semi_major, semi_minor,
                                           angle, resolution,
                                           clip=clip, geodesic=geodesic),
                     'symmetric_difference')
        return self

    # --- Annulus operations ---

    def _project_annulus(self, lon: SkyCoord | float, lat: Any = None,
                         inner_radius: Any = None, outer_radius: Any = None,
                         resolution: int = 200, clip: str = 'auto') -> Any:
        _orig_lat = lat
        lon, lat, shifted = _parse_coord(lon, lat, wcs=self._parse_wcs,
                                        frame_name=self._parse_frame)
        if shifted:
            inner_radius = _orig_lat
        inner_radius = _parse_angle(inner_radius)
        outer_radius = _parse_angle(outer_radius)
        if inner_radius is None:
            raise ValueError("inner_radius is required")
        if outer_radius is None:
            raise ValueError("outer_radius is required")
        outer = self._project_circle(lon, lat, outer_radius, resolution,
                                       clip=clip)
        inner = self._project_circle(lon, lat, inner_radius, resolution,
                                       clip=clip)
        if outer is None:
            return None
        if inner is not None:
            return outer.difference(inner)
        return outer

    def add_annulus(self, lon: SkyCoord | float, lat: Any = None, inner_radius: Any = None,
                    outer_radius: Any = None, resolution: int = 200,
                    clip: str = 'auto') -> CompoundRegion:
        """Union an annulus (ring) into the region."""
        self._apply(self._project_annulus(lon, lat, inner_radius, outer_radius,
                                           resolution, clip=clip), 'union')
        return self

    def subtract_annulus(self, lon: SkyCoord | float, lat: Any = None,
                         inner_radius: Any = None, outer_radius: Any = None,
                         resolution: int = 200,
                         clip: str = 'auto') -> CompoundRegion:
        """Subtract an annulus from the region."""
        self._apply(self._project_annulus(lon, lat, inner_radius, outer_radius,
                                           resolution, clip=clip), 'difference')
        return self

    def intersect_annulus(self, lon: SkyCoord | float, lat: Any = None,
                          inner_radius: Any = None, outer_radius: Any = None,
                          resolution: int = 200,
                          clip: str = 'auto') -> CompoundRegion:
        """Intersect the region with an annulus."""
        self._apply(self._project_annulus(lon, lat, inner_radius, outer_radius,
                                           resolution, clip=clip),
                     'intersection')
        return self

    def xor_annulus(self, lon: SkyCoord | float, lat: Any = None, inner_radius: Any = None,
                    outer_radius: Any = None, resolution: int = 200,
                    clip: str = 'auto') -> CompoundRegion:
        """Symmetric difference with an annulus."""
        self._apply(self._project_annulus(lon, lat, inner_radius, outer_radius,
                                           resolution, clip=clip),
                     'symmetric_difference')
        return self

    # --- Longitude band operations ---

    def _project_longitude_band(self, lon_min: Any, lon_max: Any,
                                lat_min: Any = None, lat_max: Any = None,
                                resolution: int = 360, clip: str = 'auto') -> Any:
        clip = _resolve_clip(clip,
                              helper_name='CompoundRegion.add_longitude_band')
        lon_min = _parse_angle(lon_min)
        lon_max = _parse_angle(lon_max)
        lat_min = _parse_angle(lat_min)
        lat_max = _parse_angle(lat_max)
        _EPS = 1e-4
        lat_lo = lat_min if lat_min is not None else (-90 + _EPS)
        lat_hi = lat_max if lat_max is not None else ( 90 - _EPS)
        n_side = max(3, resolution // 20)
        lons = np.concatenate([
            np.full(resolution, lon_min),
            np.linspace(lon_min, lon_max, n_side),
            np.full(resolution, lon_max),
            np.linspace(lon_max, lon_min, n_side),
        ])
        lats = np.concatenate([
            np.linspace(lat_lo, lat_hi, resolution),
            np.full(n_side, lat_hi),
            np.linspace(lat_hi, lat_lo, resolution),
            np.full(n_side, lat_lo),
        ])
        lat_c = (lat_lo + lat_hi) / 2
        r = (lat_hi - lat_lo) / 2
        sin_range = abs(np.sin(np.radians(lat_hi))
                        - np.sin(np.radians(lat_lo)))
        lon_span = abs(lon_max - lon_min)
        expected_frac = (lon_span / 360.0) * sin_range / 2.0
        return self._project_band_geom(
            lons, lats, clip=clip,
            expected_frac=expected_frac,
            lat_center=lat_c, radius_deg=r)

    def add_longitude_band(self, lon_min: Any, lon_max: Any, lat_min: Any = None,
                           lat_max: Any = None, resolution: int = 360,
                           clip: str = 'auto') -> CompoundRegion:
        """Union a longitude (RA) band into the region."""
        self._apply(self._project_longitude_band(lon_min, lon_max, lat_min,
                                                   lat_max, resolution,
                                                   clip=clip), 'union')
        return self

    def intersect_longitude_band(self, lon_min: Any, lon_max: Any,
                                 lat_min: Any = None, lat_max: Any = None,
                                 resolution: int = 360,
                                 clip: str = 'auto') -> CompoundRegion:
        """Intersect the region with a longitude (RA) band."""
        self._apply(self._project_longitude_band(lon_min, lon_max, lat_min,
                                                   lat_max, resolution,
                                                   clip=clip),
                     'intersection')
        return self

    def subtract_longitude_band(self, lon_min: Any, lon_max: Any,
                                lat_min: Any = None, lat_max: Any = None,
                                resolution: int = 360,
                                clip: str = 'auto') -> CompoundRegion:
        """Subtract a longitude band from the region."""
        self._apply(self._project_longitude_band(lon_min, lon_max, lat_min,
                                                   lat_max, resolution,
                                                   clip=clip), 'difference')
        return self

    # --- Great-circle band operations ---

    def _project_great_circle_band(self, ra_pole: Any, dec_pole: Any = None,
                                   half_width: Any = None, resolution: int = 500,
                                   clip: str = 'auto') -> Any:
        clip = _resolve_clip(clip,
                              helper_name='CompoundRegion.add_great_circle_band')
        _orig_dec = dec_pole
        ra_pole, dec_pole, shifted = _parse_coord(ra_pole, dec_pole,
                                                    wcs=self._parse_wcs,
                                        frame_name=self._parse_frame)
        if shifted:
            half_width = _orig_dec
        half_width = _parse_angle(half_width)
        if half_width is None:
            raise ValueError("half_width is required")

        inner_r = max(90 - half_width, 1e-4)
        outer_r = min(90 + half_width, 180 - 1e-4)

        i_lons, i_lats = geodesic_circle(ra_pole, dec_pole, inner_r, resolution)
        o_lons, o_lats = geodesic_circle(ra_pole, dec_pole, outer_r, resolution)

        inner_frac = (1.0 - np.cos(np.radians(inner_r))) / 2.0
        outer_frac = (1.0 - np.cos(np.radians(outer_r))) / 2.0
        inner = self._project_band_geom(
            i_lons, i_lats, clip=clip,
            lat_center=dec_pole, radius_deg=inner_r,
            expected_frac=inner_frac)
        outer = self._project_band_geom(
            o_lons, o_lats, clip=clip,
            lat_center=dec_pole, radius_deg=outer_r,
            expected_frac=outer_frac)

        if outer and inner:
            return outer.difference(inner)
        elif outer:
            return outer
        return None

    def add_great_circle_band(self, ra_pole: Any, dec_pole: Any = None,
                              half_width: Any = None, resolution: int = 500,
                              clip: str = 'auto') -> CompoundRegion:
        """Union a great-circle band into the region."""
        self._apply(self._project_great_circle_band(ra_pole, dec_pole,
                                                      half_width, resolution,
                                                      clip=clip), 'union')
        return self

    def subtract_great_circle_band(self, ra_pole: Any, dec_pole: Any = None,
                                   half_width: Any = None, resolution: int = 500,
                                   clip: str = 'auto') -> CompoundRegion:
        """Subtract a great-circle band from the region."""
        self._apply(self._project_great_circle_band(ra_pole, dec_pole,
                                                      half_width, resolution,
                                                      clip=clip), 'difference')
        return self

    # --- Region transformations ---

    def complement(self) -> CompoundRegion:
        """Replace the region with its complement (frame minus current)."""
        if self._geom is None or self._geom.is_empty:
            self._geom = self._frame_poly
        else:
            self._geom = self._frame_poly.difference(self._geom)
        return self

    def _binary_op(self, other: CompoundRegion, name: str) -> CompoundRegion:
        """Shared core for the region-region set operations.

        Both regions must be built on the **same frame** — the geometry lives in
        that frame's pixel space, so combining regions from different axes is
        meaningless. Empty (``None``) geometries are handled per operation.
        """
        if not isinstance(other, CompoundRegion):
            raise TypeError(
                f"expected another CompoundRegion, got {type(other).__name__}")
        a, b = self._geom, other._geom
        a_empty = a is None or a.is_empty
        b_empty = b is None or b.is_empty
        if name == 'union':
            g = b if a_empty else a if b_empty else a.union(b)
        elif name == 'intersection':
            g = None if (a_empty or b_empty) else a.intersection(b)
        elif name == 'difference':
            g = None if a_empty else (a if b_empty else a.difference(b))
        else:  # symmetric_difference
            g = b if a_empty else a if b_empty else a.symmetric_difference(b)
        self._geom = g if (g is not None and not g.is_empty) else None
        return self

    def union(self, other: CompoundRegion) -> CompoundRegion:
        """Union in another region (``self ∪ other``), mutating ``self``.

        The region-level counterpart of the shape-specific ``add_*`` methods:
        combine two independently-built regions (e.g. two survey footprints).
        Both must be on the same frame. Returns ``self`` for chaining."""
        return self._binary_op(other, 'union')

    def intersection(self, other: CompoundRegion) -> CompoundRegion:
        """Intersect with another region (``self ∩ other``), mutating ``self``
        (e.g. the sky two surveys have in common). Returns ``self``."""
        return self._binary_op(other, 'intersection')

    def difference(self, other: CompoundRegion) -> CompoundRegion:
        """Subtract another region (``self − other``), mutating ``self`` — punch
        one region's area out of another (e.g. land minus lakes, footprint minus
        an avoidance zone). Returns ``self``."""
        return self._binary_op(other, 'difference')

    def symmetric_difference(self, other: CompoundRegion) -> CompoundRegion:
        """Symmetric difference (``self XOR other`` — area in exactly one of the
        two regions), mutating ``self``. Returns ``self``."""
        return self._binary_op(other, 'symmetric_difference')

    def clip_path(self, complement: bool = False) -> Any:
        """Return a matplotlib ``Path`` (in ``self.ax.transData``) covering this
        region, for use as a clip path. With ``complement=True`` it covers the
        frame *minus* the region. Does not mutate the region.
        """
        from ._frame_geom import _geom_to_clip_path
        return _geom_to_clip_path(self._geom, self._frame_poly,
                                  complement=complement)

    def clip(self, artists: Any, complement: bool = False) -> Any:
        """Clip matplotlib artist(s) to this region (or its complement).

        Masks the given artist(s) so they only render *inside* the region — clip
        a scatter / quiver / image to a survey footprint, a constellation
        boundary, a latitude band, a HEALPix region, or any spherical polygon.
        Works on any FITS-projection frame (all-sky, globe, or planet map).

        Parameters
        ----------
        artists : Artist or list of Artist
            Artist(s) to clip in place (e.g. the return of ``ax.scatter`` /
            ``ax.quiver`` / ``ax.imshow``).
        complement : bool
            If True, clip to *outside* the region instead.

        Returns
        -------
        path : matplotlib.path.Path
            The clip path (also applied to *artists*); apply to more artists
            with ``artist.set_clip_path(path, region.ax.transData)``.
        """
        path = self.clip_path(complement=complement)
        seq = artists if isinstance(artists, (list, tuple)) else [artists]
        for art in seq:
            art.set_clip_path(path, self.ax.transData)
        return path

    def expand(self, angle_deg: Any) -> CompoundRegion:
        """
        Expand the region boundary outward by an angular distance.

        Useful for adding exclusion margins around bright stars, padding
        survey boundaries for dithering, or enlarging avoidance zones.

        The angular distance is converted to an approximate pixel distance
        using the mean pixel scale from the WCS.

        Parameters
        ----------
        angle_deg : float or Quantity
            Expansion distance in degrees.  Positive expands outward.
        """
        angle_deg = _parse_angle(angle_deg)
        if self._geom is None or self._geom.is_empty or angle_deg == 0:
            return self
        px_dist = self._angle_to_pixels(angle_deg)
        self._geom = self._geom.buffer(px_dist)
        self._geom = self._clip_to_frame(self._geom)
        return self

    def contract(self, angle_deg: Any) -> CompoundRegion:
        """
        Contract (erode) the region boundary inward by an angular distance.

        Useful for finding the safe interior of a survey footprint,
        trimming edge effects, or computing conservative coverage.

        Parameters
        ----------
        angle_deg : float or Quantity
            Contraction distance in degrees.  Positive contracts inward.
        """
        angle_deg = _parse_angle(angle_deg)
        if self._geom is None or self._geom.is_empty or angle_deg == 0:
            return self
        px_dist = self._angle_to_pixels(angle_deg)
        self._geom = self._geom.buffer(-px_dist)
        if self._geom is not None and not self._geom.is_empty:
            self._geom = self._clip_to_frame(self._geom)
        else:
            self._geom = None
        return self

    def _angle_to_pixels(self, angle_deg: float) -> float:
        """Convert an angular distance to approximate projected distance.

        The estimate is backend-specific (it depends on the projection's
        scale), so it lives on the projector: the mpl backend uses a
        WCS-native scale, the plotly backend a projected-step heuristic.
        """
        return self.projector.angle_to_pixels(angle_deg)

    # --- Queries ---

    @property
    def area_frac(self) -> float:
        """Fraction of the projection frame covered by the region."""
        if self._geom is None or self._geom.is_empty:
            return 0.0
        return self._geom.area / self._frame_poly.area

    @property
    def solid_angle(self) -> dict[str, float]:
        """
        Approximate solid angle of the region.

        Returns
        -------
        dict
            ``{'sq_deg': float, 'sr': float}`` — the solid angle in
            square degrees and steradians.

        The computation uses the pixel-area fraction times the full-sky
        solid angle.  This is an approximation that assumes uniform pixel
        scale; for most all-sky projections the error is small (<1%).
        """
        frac = self.area_frac
        FULL_SKY_SQ_DEG = 4 * 180**2 / np.pi  # ≈ 41252.96
        FULL_SKY_SR = 4 * np.pi
        return {'sq_deg': frac * FULL_SKY_SQ_DEG,
                'sr': frac * FULL_SKY_SR}

    @property
    def is_empty(self) -> bool:
        """Whether the region is empty."""
        return self._geom is None or self._geom.is_empty

    @property
    def geometry(self) -> Any:
        """The accumulated shapely geometry in pixel space (read-only)."""
        return self._geom

    def contains_point(self, ra: SkyCoord | npt.ArrayLike, dec: Any = None) -> bool:
        """
        Test whether a sky coordinate falls within the region.

        Parameters
        ----------
        ra : float or SkyCoord
            Right ascension in degrees (ICRS), or a SkyCoord.
        dec : float or None
            Declination in degrees (ignored when *ra* is a SkyCoord).

        Returns
        -------
        bool
        """
        if self._geom is None or self._geom.is_empty:
            return False
        ra, dec, _ = _parse_coord(ra, dec, wcs=self._parse_wcs,
                                        frame_name=self._parse_frame)
        # Project through the backend's primitive (mpl pixel coords / plotly
        # canvas coords) so the test runs against ``_geom`` in the same space.
        x, y = self.projector._project_xy(float(ra), float(dec))
        x, y = float(np.ravel(x)[0]), float(np.ravel(y)[0])
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        return self._geom.contains(Point(x, y))

    def contains_points(self, ra: SkyCoord | npt.ArrayLike, dec: Any = None) -> np.ndarray:
        """
        Test whether sky coordinates fall within the region (vectorised).

        Parameters
        ----------
        ra : array-like or SkyCoord
            Right ascension array in degrees (ICRS), or a SkyCoord array.
        dec : array-like or None
            Declination array in degrees (ignored when *ra* is a SkyCoord).

        Returns
        -------
        ndarray of bool
            Same length as input arrays.

        Notes
        -----
        Membership is evaluated in the projected pixel space of the axes the
        region was built on. Global bands (latitude / longitude / frame /
        great-circle) are clipped to the field's world bounds first, so
        membership is correct on a small zoomed field frame as well as on an
        all-sky one. The region object is also self-contained once built:
        ``contains_points`` keeps working after the source figure is closed, so
        it is fine (and handy) to construct a region on a throwaway axes purely
        to reuse it as a membership oracle.
        """
        ra, dec = _parse_coords(ra, dec, wcs=self._parse_wcs,
                                        frame_name=self._parse_frame)
        ra = np.asarray(ra, float)
        dec = np.asarray(dec, float)
        if ra.shape != dec.shape:
            raise ValueError("ra and dec must have the same shape")

        result = np.zeros(ra.shape, dtype=bool)
        if self._geom is None or self._geom.is_empty:
            return result

        x, y = self.projector._project_xy(ra.ravel(), dec.ravel())
        x, y = np.asarray(x, float), np.asarray(y, float)
        valid = np.isfinite(x) & np.isfinite(y)

        # Use shapely prepared geometry for fast batch containment
        from shapely.prepared import prep
        prepared = prep(self._geom)

        for i in np.where(valid)[0]:
            result.ravel()[i] = prepared.contains(Point(float(x[i]), float(y[i])))

        return result

    # --- Render ---

    def render(self, **kwargs: Any) -> list[Any]:
        """
        Render the compound region on the axes.

        When ``edgecolor`` is specified, the fill and boundary are rendered
        separately: fill with no edge (avoiding frame-boundary edge lines),
        and boundary drawn only along the actual shape outline via
        ``render_boundary()``.

        Parameters
        ----------
        **kwargs
            Passed to matplotlib PathPatch (facecolor, edgecolor, alpha, etc.).

        Returns
        -------
        list
            The created artists — the fill ``PathPatch`` objects plus, when an
            ``edgecolor`` is given, the boundary ``Line2D`` objects. Remove them
            all to take the rendered region back off the axes.
        """
        if self._geom is None or self._geom.is_empty:
            return []

        geom = self._geom.intersection(self._frame_poly)
        if geom.is_empty:
            return []

        # Cosmetic cleanup of set-algebra artifacts: a 0.5-pixel
        # morphological close (buffer +eps then -eps) merges
        # near-touching polygon pieces produced when two
        # differently-densified bands cross — the cosmetic seam
        # would otherwise show as a thin line between adjacent
        # filled regions or as a stray boundary segment.
        # ``buffer(0)`` first cleans LineString / point artifacts
        # by collapsing the GeometryCollection to a clean
        # MultiPolygon. This is render-only — does not mutate
        # ``self._geom``.
        geom = _cleanup_for_render(geom)
        if geom.is_empty:
            return []

        # Shared legibility-stroke knob (parity with the shape / band / fill
        # helpers): translate ``stroke_color`` / ``stroke_lw`` into a
        # ``path_effects`` outline on the fill path (visible even though the
        # fill itself carries no edge). An explicit ``path_effects=`` wins.
        stroke_color = kwargs.pop('stroke_color', None)
        stroke_lw = kwargs.pop('stroke_lw', None)
        if stroke_color is not None and 'path_effects' not in kwargs:
            from .._stroke import _stroke_path_effects
            _pe = _stroke_path_effects(
                stroke_color, 3.0 if stroke_lw is None else stroke_lw)
            if _pe is not None:
                kwargs['path_effects'] = _pe

        # Separate edgecolor for boundary-only rendering
        edgecolor = kwargs.pop('edgecolor', kwargs.pop('ec', None))
        linewidth = kwargs.get('linewidth', kwargs.get('lw', 1.0))

        # A non-solid line style belongs on the boundary, not the width-0 fill
        # patches: matplotlib scales the dash pattern by the linewidth, so a
        # dashed style on a linewidth=0 patch raises "At least one value in the
        # dash list must be positive". Pull linestyle/ls/dashes out here and
        # forward them to render_boundary so the outline still dashes.
        boundary_style: dict[str, Any] = {}
        for _sk in ('linestyle', 'ls', 'dashes'):
            if _sk in kwargs:
                boundary_style[_sk] = kwargs.pop(_sk)

        # Render fill (always with no edge to avoid frame-boundary lines)
        _fix_hairline_kwargs(kwargs)
        kwargs['edgecolor'] = 'none'
        kwargs['linewidth'] = 0
        # ``_shapely_to_paths`` defaults to ``min_area=1.0`` (pixel²),
        # which would drop deep-field surveys (e.g. GOODS, ~0.05 px²
        # on default all-sky AIT). The ``_cleanup_for_render`` step
        # above already filtered numerical-noise artifacts, so we
        # accept anything that survived it.
        paths = _shapely_to_paths(geom, min_area=0.0)
        patches = []
        for path in paths:
            patch = PathPatch(path, **kwargs)
            self.ax.add_patch(patch)
            patches.append(patch)

        # Render boundary if edgecolor was specified. Include its Line2D
        # artists in the return so a caller can remove the whole rendered
        # region (fill patches + boundary lines) cleanly.
        if edgecolor is not None and edgecolor != 'none':
            patches.extend(self.render_boundary(
                color=edgecolor, linewidth=linewidth, **boundary_style))

        return patches

    def render_boundary(self, color: Any = 'black', linewidth: float = 1.0,
                        **kwargs: Any) -> list[Any]:
        """
        Render the region boundary on the axes.

        Draws only the actual shape boundary, suppressing edge lines
        that coincide with the projection frame edge.  This is the
        correct rendering for survey footprint outlines overlaid on
        other content.

        Parameters
        ----------
        color : str or tuple
            Line color.
        linewidth : float
            Line width (default 1.0).
        **kwargs
            Additional arguments passed to ``ax.plot()``.

        Returns
        -------
        list of matplotlib Line2D
        """
        # Accept matplotlib aliases (``lw`` for linewidth, ``c`` for color,
        # ``ls`` for linestyle, ...) the same way ``ax.plot`` does. Without this,
        # a caller-supplied ``lw=`` collides with the explicit ``linewidth`` we
        # forward below ("Got both 'linewidth' and 'lw'"). normalize_kwargs
        # canonicalizes each alias; a canonicalized value then cleanly overrides
        # the same-named default when we merge into ``plot_kwargs``.
        from matplotlib.cbook import normalize_kwargs
        from matplotlib.lines import Line2D
        kwargs = normalize_kwargs(kwargs, Line2D)

        if self._geom is None or self._geom.is_empty:
            return []

        geom = self._geom.intersection(self._frame_poly)
        if geom.is_empty:
            return []

        # Cosmetic cleanup — see ``render`` for details. Critical here
        # because Shapely returns ``None`` for ``GeometryCollection
        # .boundary``, and the raw boundary of a fragmented polygon
        # set includes seam edges between near-touching pieces.
        geom = _cleanup_for_render(geom)
        if geom.is_empty:
            return []

        # Extract boundary and remove parts that coincide with the frame edge
        shape_boundary = geom.boundary
        frame_boundary = self._frame_poly.boundary
        # Buffer the frame boundary to catch near-coincident segments — the
        # value is small (sub-pixel) so we only suppress segments that
        # genuinely lie along the projection limb, not segments that merely
        # touch it at a point (a larger buffer eats real boundary where
        # great-circle bands kiss the projection edge).
        tolerance = 0.5  # pixels
        try:
            interior_boundary = shape_boundary.difference(
                frame_boundary.buffer(tolerance))
        except Exception:
            interior_boundary = shape_boundary

        if interior_boundary.is_empty:
            return []

        # Extract coordinate arrays from the boundary geometry
        lines = []
        if interior_boundary.geom_type == 'LineString':
            lines.append(np.array(interior_boundary.coords))
        elif interior_boundary.geom_type == 'MultiLineString':
            for ls in interior_boundary.geoms:
                lines.append(np.array(ls.coords))
        elif interior_boundary.geom_type == 'GeometryCollection':
            for g in interior_boundary.geoms:
                if g.geom_type == 'LineString' and len(g.coords) >= 2:
                    lines.append(np.array(g.coords))
                elif g.geom_type == 'MultiLineString':
                    for ls in g.geoms:
                        lines.append(np.array(ls.coords))

        # Plot each segment
        plot_kwargs = dict(color=color, linewidth=linewidth,
                           transform=self.ax.get_transform('pixel'))
        plot_kwargs.update(kwargs)

        result = []
        for coords in lines:
            if len(coords) >= 2:
                ln, = self.ax.plot(coords[:, 0], coords[:, 1], **plot_kwargs)
                result.append(ln)
        return result
