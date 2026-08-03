"""Backend-agnostic projection adapter for :class:`CompoundRegion`.

The :class:`Projector` protocol decouples the sphere-coord shape
construction (geodesic circles, polygons, latitude bands, frame bands)
from the backend-specific projection + antimeridian + clip pipeline.
Each backend (matplotlib WCSAxes, plotly) provides a concrete
projector and :class:`CompoundRegion` stays one class servicing both.

Two concrete projectors live alongside this module:

* :class:`WCSAxesProjector` (this file) — wraps the existing
  ``_dispatch_projection`` + ``_paths_to_geom`` mpl pipeline.
* :class:`skyplothelper.plotly.projector.SkyplothelperProjector` —
  the plotly-side adapter using ``sph.project()``.

The protocol surface is intentionally narrow: a projector exposes a
``center`` longitude (for antimeridian handling), a ``frame_polygon``
(in the backend's projected coords; used by ``CompoundRegion`` to
clip the accumulated geometry to the visible frame), and a single
``project_polygon(lons, lats, *, clip, ...) -> list[shapely.Polygon]``
method that handles everything from antimeridian splitting to the
backend's own projection math.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def _wrap180(a: Any) -> Any:
    """Wrap degrees to (-180, 180]."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def _unit_vecs(lons: Any, lats: Any) -> npt.NDArray[np.float64]:
    """(lon, lat) degrees → unit vectors, shape (..., 3)."""
    lon = np.radians(np.asarray(lons, dtype=float))
    lat = np.radians(np.asarray(lats, dtype=float))
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def _sph_polygon_contains(lons: Any, lats: Any,
                          plon: float, plat: float) -> bool:
    """Spherical winding-number point-in-polygon.

    Robust where a flat lon/lat test degenerates — a polygon that wraps all
    longitudes (a polar cap, a full latitude band) has no simple planar ring,
    so we sum the signed angles the polygon edges subtend around the test
    point on the sphere: ~±2π ⇒ inside, ~0 ⇒ outside.
    """
    p = _unit_vecs(plon, plat).reshape(3)
    v = _unit_vecs(np.asarray(lons), np.asarray(lats))
    total = 0.0
    for i in range(len(v) - 1):
        na = v[i] - p * float(np.dot(v[i], p))
        nb = v[i + 1] - p * float(np.dot(v[i + 1], p))
        la = float(np.linalg.norm(na))
        lb = float(np.linalg.norm(nb))
        if la < 1e-12 or lb < 1e-12:
            continue
        na = na / la
        nb = nb / lb
        total += float(np.arctan2(np.dot(p, np.cross(na, nb)),
                                  np.dot(na, nb)))
    return abs(total) > np.pi


def _densify_lonlat(coords: npt.NDArray[np.float64],
                    max_step: float = 0.5) -> npt.NDArray[np.float64]:
    """Insert intermediate vertices so each (lon, lat)-degree edge is at most
    ``max_step`` long, so the projected boundary follows the projection's
    curvature rather than chording across it."""
    out = [coords[0]]
    for i in range(1, len(coords)):
        a, b = coords[i - 1], coords[i]
        n = max(1, int(np.ceil(float(np.hypot(*(b - a))) / max_step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out, dtype=float)


def _slerp_unit(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64],
                step_deg: float = 2.0) -> npt.NDArray[np.float64]:
    """Great-circle interpolate between two unit vectors ``a`` and ``b``,
    inclusive, with roughly ``step_deg`` spacing. Used to trace the visible
    limb (both endpoints lie on the limb great circle, so the interpolation
    stays on it)."""
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    ang = np.arccos(dot)
    if ang < 1e-9:
        return np.array([a, b])
    n = max(2, int(np.ceil(np.degrees(ang) / step_deg)) + 1)
    t = np.linspace(0.0, 1.0, n)
    sin_ang = np.sin(ang)
    out = (np.sin((1 - t)[:, None] * ang) * a
           + np.sin(t[:, None] * ang) * b) / sin_ang
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _clip_polygon_to_hemisphere(
    verts: npt.NDArray[np.float64], center: npt.NDArray[np.float64],
    slerp_step_deg: float = 2.0,
) -> npt.NDArray[np.float64] | None:
    """Clip a spherical polygon (unit-vector ring ``verts``) to the visible
    hemisphere ``p·center > 0`` and return the visible boundary as unit
    vectors, with the limb edges traced along the limb great circle.

    Sutherland-Hodgman against the single half-space plane through the center:
    it keys on the *sign* of ``p·center``, so — unlike an azimuthal projection
    — it has no antipode singularity (the far side is simply clipped). A ring
    entirely on the far side (all ``p·center ≤ 0``) yields ``None``; a ring
    that crosses the limb is closed along the limb arc between its exit and
    re-entry crossings. (A far-side ring that *encloses* the whole visible
    hemisphere still yields ``None`` — all its vertices are clipped — which is
    the one documented globe-fill limitation.)
    """
    d = verts @ center
    n = len(verts)
    # (unit vector, is_limb_crossing) in boundary order.
    out: list[tuple[npt.NDArray[np.float64], bool]] = []
    for i in range(n):
        j = (i + 1) % n
        di, dj = float(d[i]), float(d[j])
        if di > 0:
            out.append((verts[i], False))
            if dj <= 0:                       # exit: crossing onto the limb
                t = di / (di - dj)
                cx = verts[i] + t * (verts[j] - verts[i])
                out.append((cx / np.linalg.norm(cx), True))
        elif dj > 0:                          # entry: crossing off the limb
            t = di / (di - dj)
            cx = verts[i] + t * (verts[j] - verts[i])
            out.append((cx / np.linalg.norm(cx), True))
    if len(out) < 3:
        return None
    # Trace every limb edge (both endpoints are crossings) along the limb.
    m = len(out)
    dense: list[npt.NDArray[np.float64]] = []
    for k in range(m):
        va, ca = out[k]
        vb, cb = out[(k + 1) % m]
        dense.append(va)
        if ca and cb:
            arc = _slerp_unit(va, vb, slerp_step_deg)
            dense.extend(arc[1:-1])
    return _despike_ring(np.asarray(dense, dtype=float))


def _despike_ring(pts: npt.NDArray[np.float64],
                  tol: float = 1e-7) -> npt.NDArray[np.float64] | None:
    """Remove degenerate spikes and duplicate points from a closed vertex ring.

    A polygon that *encloses a pole* (e.g. the North American tectonic plate)
    is stored as a ring cut up one meridian to the pole and back down the
    antimeridian. In unit-vector space ``±180°`` are the same point, so that cut
    is a *palindromic retrace* — a zero-width spike that would otherwise be
    drawn as a spurious 180° seam by any edge color. Peeling spikes (a vertex
    whose neighbors coincide) collapses the retrace while leaving the ring still
    enclosing the pole (the boundary just closes across the cut). Also drops
    consecutive duplicates. Returns ``None`` if fewer than 3 vertices survive.
    """
    v = [p for i, p in enumerate(pts)
         if i == 0 or np.linalg.norm(p - pts[i - 1]) > tol]
    changed = True
    while changed and len(v) > 3:
        changed = False
        i = 1
        while i < len(v) - 1:
            if np.linalg.norm(v[i - 1] - v[i + 1]) < tol:   # spike tip
                del v[i]
                if i < len(v) and np.linalg.norm(v[i - 1] - v[i]) < tol:
                    del v[i]
                changed = True
            else:
                i += 1
    return np.asarray(v, dtype=float) if len(v) >= 3 else None


class Projector:
    """Abstract base for projection backends.

    The base owns the shared, vetted projection pipeline:
    :meth:`project_polygon` runs the d3 antimeridian-clip → project →
    stitch → complement-detect sequence against two backend plug-ins,
    :attr:`frame_polygon` and :meth:`_project_xy`, so the matplotlib and
    plotly backends share one implementation instead of two that drift.
    A backend whose projection has cases the shared pipeline can't model
    (matplotlib's pixel-space pole / jump handling) may still override
    :meth:`project_polygon`; the plotly backend inherits it.

    Subclasses implement :attr:`center`, :attr:`frame_polygon`,
    :meth:`_project_xy`, and :meth:`render_region`.
    :class:`CompoundRegion` drives :meth:`project_polygon` then renders
    the resulting geometry.
    """

    @property
    def center(self) -> float:
        """Projection center longitude in degrees (for antimeridian handling)."""
        raise NotImplementedError

    @property
    def wcs_frame(self) -> str:
        """Sky coordinate frame the projected axes are drawn in
        (``'icrs'`` / ``'galactic'`` / ``'ecliptic'`` / ``'supergalactic'``).

        Part of the projector interface: cross-frame paths such as
        :meth:`CompoundRegion.add_lonlat_box` consult it to know how to
        transform user-supplied coordinates into the axes' frame before
        projecting. The base default is ``'icrs'``; the matplotlib
        ``WCSAxesProjector`` derives it from the WCS CTYPE, and the
        plotly / pixel / offset projectors report their own convention.
        """
        return 'icrs'

    @property
    def frame_polygon(self) -> Any:
        """Backend's projection-frame boundary as a shapely Polygon
        in projected coords."""
        raise NotImplementedError

    @property
    def frame_edge_tolerance(self) -> float:
        """Buffer width (in this backend's projected units) used by
        :meth:`CompoundRegion.render_boundary` to suppress boundary
        segments that lie along the projection frame edge.

        The default ``0.5`` is a sub-pixel value tuned for the matplotlib
        FITS pixel frames (whose coordinates span hundreds of pixels). A
        backend whose projected coordinates are on a very different scale
        (the non-FITS custom frames span only a few *radians-scale* units,
        so ``0.5`` would be ~8% of the whole map and eat real interior
        boundary near the limb) overrides this with a frame-relative value.
        """
        return 0.5

    def _project_xy(self, lons: npt.ArrayLike, lats: npt.ArrayLike) -> tuple[Any, Any]:
        """Project sphere ``(lon, lat)`` arrays to this backend's
        projected ``(x, y)`` coords — pixel space for matplotlib, canvas
        space for plotly — with NaN for off-domain points.

        The single projection seam the shared :meth:`project_polygon`
        pipeline delegates to each backend. The plotly
        ``SkyplothelperProjector`` implements it via ``sph.project()``;
        matplotlib's ``WCSAxesProjector`` keeps its own
        :meth:`project_polygon` override (its pixel-space pole / jump
        repair doesn't fit this contract), so it supplies ``_project_xy``
        only when/if it adopts the shared pipeline.
        """
        raise NotImplementedError

    @property
    def _center_lat(self) -> float:
        """Projection center latitude in degrees.

        Backends that track a non-equatorial center (plotly's
        ``lat_center``) override this; the base default suits the
        equator-centered all-sky frames and is consumed only by the
        generic :meth:`angle_to_pixels`.
        """
        return 0.0

    def angle_to_pixels(self, angle_deg: float) -> float:
        """Approximate the projected-coordinate distance of an angular
        separation ``angle_deg`` near the projection center.

        :meth:`CompoundRegion.expand` / :meth:`~CompoundRegion.contract`
        buffer in projected space, so they need a scale to convert a sky
        angle into projected units. The estimate is backend-specific (the
        scale is whatever ``_project_xy`` produces), so it lives on the
        projector rather than on ``CompoundRegion``: this generic default
        projects a small step at the center and measures it; the
        matplotlib backend overrides it with a WCS-native estimate.
        """
        clat = self._center_lat
        x1, _ = self._project_xy(self.center, clat)
        x2, _ = self._project_xy(self.center + angle_deg, clat)
        x1 = float(np.ravel(x1)[0])
        x2 = float(np.ravel(x2)[0])
        if np.isfinite(x1) and np.isfinite(x2):
            return abs(x2 - x1)
        # The center meridian can land on the wrap seam (NaN) for some
        # projections; fall back to a latitudinal step from the center.
        xa, ya = self._project_xy(self.center, clat)
        xb, yb = self._project_xy(self.center, clat + angle_deg)
        return float(np.hypot(float(np.ravel(xb)[0]) - float(np.ravel(xa)[0]),
                              float(np.ravel(yb)[0]) - float(np.ravel(ya)[0])))

    # ------------------------------------------------------------------
    # Open-geometry projection for the standalone overlay helpers
    # (great circles, rulers, vectors, healpix tiles, …). These differ
    # from :meth:`project_polygon` — they return canvas coords ready for
    # a ``go.Scatter`` rather than stitched shapely polygons — and they
    # are where wrap-seam handling lives: a projection with a wrap
    # meridian (the all-sky pseudo-cylindrical frames) splits the
    # geometry at the seam so a polyline / fill breaks cleanly instead of
    # streaking across the canvas, while a seamless frame (a tangent-plane
    # FITS image's pixel or offset coords) has nothing to split. The base
    # defaults are the seamless case; the seam-bearing all-sky projector
    # overrides the two split-aware methods.
    # ------------------------------------------------------------------

    def split_polyline(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return open-polyline sphere-coords ready to project.

        The wrap-seam seam point for open lines. Base (seamless) default:
        the coords unchanged. The all-sky projector overrides this to shift
        into the projection window and insert ``NaN`` breaks at the wrap
        meridian. Exposed (not just folded into :meth:`project_polyline`)
        because callers that build per-vertex hover ``customdata`` need the
        same NaN-aligned coords the projected ``(x, y)`` came from."""
        return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)

    def split_polygon_pieces(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike,
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Return a closed polygon boundary as a list of ``(lons, lats)``
        pieces ready to project, one per visible lobe.

        The wrap-seam seam point for closed fills. Base (seamless) default:
        a single piece (the whole boundary). The all-sky projector overrides
        this to split the boundary at the wrap meridian."""
        return [(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))]

    def project_points(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Project sphere ``(lon, lat)`` to this backend's ``(x, y)`` as
        float arrays — the plain point projection overlay helpers use for
        markers, vector endpoints, and ruler ticks."""
        x, y = self._project_xy(lons, lats)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def _nan_off_frame(
        self, x: npt.ArrayLike, y: npt.ArrayLike, margin_frac: float = 1.0,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Set points falling far outside the frame bbox to ``NaN`` so an
        overlay polyline breaks there instead of streaking across the canvas.

        The all-sky pseudo-cylindrical projectors don't need this — their
        projection is naturally bounded and NaNs its own off-domain points —
        but an unbounded WCS frame (a TAN field's far hemisphere diverges to
        huge pixel values) does. ``margin_frac`` of the frame size is kept
        beyond each edge so overlays that run just off-image still draw to
        the edge; only the blown-up far-side points are dropped."""
        minx, miny, maxx, maxy = self.frame_polygon.bounds
        mx = (maxx - minx) * margin_frac
        my = (maxy - miny) * margin_frac
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        off = ((x < minx - mx) | (x > maxx + mx)
               | (y < miny - my) | (y > maxy + my))
        return np.where(off, np.nan, x), np.where(off, np.nan, y)

    def project_polyline(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Project an open polyline to canvas ``(x, y)``, seam-split first.
        Convenience composing :meth:`split_polyline` + :meth:`project_points`
        for callers that don't also need the split sphere-coords."""
        return self.project_points(*self.split_polyline(lons, lats))

    def project_polygon_pieces(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike,
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Project a closed polygon boundary to a list of canvas ``(x, y)``
        pieces, one ``fill='toself'`` trace each. Convenience composing
        :meth:`split_polygon_pieces` + :meth:`project_points`."""
        return [self.project_points(pl, pb)
                for pl, pb in self.split_polygon_pieces(lons, lats)]

    # ------------------------------------------------------------------
    # Bounded-field (zoomed frame) projection path
    #
    # On a small tangent-plane field, a global band or a large cap has most
    # of its vertices off-frame (NaN through the WCS). The seam-stitch /
    # complement-detect heuristics in ``project_polygon`` then misfire — a
    # latitude band comes out empty OR (via a bad complement flip) covers the
    # whole field, and a subtracted frame-band can wipe everything. The fix is
    # to clip the *spherical* polygon to the field's world box FIRST, so what
    # gets projected is a small unambiguous local patch. This whole path is
    # inert unless ``world_bounds()`` is non-None, which only the bounded
    # ``WCSAxesProjector`` reports — all-sky / globe / plotly frames keep the
    # original ``project_polygon`` behavior byte-for-byte.
    # ------------------------------------------------------------------

    def world_bounds(self) -> tuple[float, float, float, float] | None:
        """``(lon_min, lon_max, lat_min, lat_max)`` (native-frame degrees) of a
        *bounded* frame, or ``None`` when the frame is effectively all-sky.

        Longitudes are absolute (interpret modulo 360 about :attr:`center`;
        ``lon_min`` may exceed ``lon_max`` when the window straddles the
        antimeridian). Only backends whose frame is a zoomed field override
        this; the base default ``None`` disables the bounded-field path."""
        return None

    def project_field_clipped(self, lons: npt.ArrayLike,
                              lats: npt.ArrayLike) -> Any | None:
        """Clip a spherical polygon to the field's world box, then project the
        result directly (no seam / complement heuristics).

        Returns a shapely geometry in projected coords — possibly empty (the
        polygon makes no contribution to this field) — or ``None`` when the
        frame is all-sky, signalling the caller to use :meth:`project_polygon`.
        """
        wb = self.world_bounds()
        if wb is None:
            return None
        from shapely.geometry import Polygon
        clipped = self._clip_world_box(lons, lats, wb)
        if clipped is None or clipped.is_empty:
            return Polygon()
        return self._project_bounded_lonlat(clipped)

    def _clip_world_box(self, lons: npt.ArrayLike, lats: npt.ArrayLike,
                        wb: tuple[float, float, float, float]) -> Any | None:
        """Intersect a spherical polygon with the field's ``(lon, lat)`` box,
        in a center-relative longitude frame. A polygon that wraps all
        longitudes (a polar cap / global band) has no simple planar ring, so
        it's resolved by spherical containment of the box center instead."""
        from shapely.geometry import Polygon, box
        from shapely.validation import make_valid

        lon_min, lon_max, lat_min, lat_max = wb
        c = self.center
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        rl = _wrap180(lons - c)
        box_lo = float(_wrap180(lon_min - c))
        box_hi = float(_wrap180(lon_max - c))
        if box_hi <= box_lo:
            box_hi += 360.0
        field_box = box(box_lo, lat_min, box_hi, lat_max)

        # A ring that winds fully around a pole has an adjacent ±360 jump in
        # relative longitude — it has no simple planar (rl, lat) representation,
        # so resolve it by spherical containment. A full-longitude *latitude
        # band* spans 360° of lon too, but with NO adjacent wrap jump (its
        # relative-lon sweep is monotonic), so it flat-clips as a rectangle.
        winds_pole = bool(np.abs(np.diff(rl)).max() > 180.0)
        if winds_pole:
            cen_lon = c + 0.5 * (box_lo + box_hi)
            cen_lat = 0.5 * (lat_min + lat_max)
            if _sph_polygon_contains(lons, lats, cen_lon, cen_lat):
                return field_box            # field fully inside the polygon
            return None                     # field entirely outside
        poly = Polygon(np.column_stack([rl, lats]))
        if not poly.is_valid:
            poly = make_valid(poly)
        inter = poly.intersection(field_box)
        return None if inter.is_empty else inter

    def _project_bounded_lonlat(self, poly_rel: Any) -> Any:
        """Project a center-relative ``(lon, lat)`` shapely polygon already
        confined to the field: densify + ``_project_xy`` each ring, then clip
        to the frame polygon. No seam/pole handling needed — it's a local
        patch."""
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.ops import unary_union
        from shapely.validation import make_valid

        from ._frame_geom import _safe_intersection

        geoms = (list(poly_rel.geoms)
                 if isinstance(poly_rel, MultiPolygon) else [poly_rel])
        out = []
        for g in geoms:
            shell = self._project_ring(g.exterior)
            if shell is None:
                continue
            holes = [h for h in (self._project_ring(r) for r in g.interiors)
                     if h is not None]
            pp = Polygon(shell, holes)
            if not pp.is_valid:
                pp = make_valid(pp)
            out.append(pp)
        if not out:
            return Polygon()
        return _safe_intersection(unary_union(out), self.frame_polygon)

    def _project_ring(self, ring: Any) -> npt.NDArray[np.float64] | None:
        coords = np.asarray(ring.coords, dtype=float)   # (M, 2) rel-lon, lat
        if len(coords) < 4:
            return None
        dens = _densify_lonlat(coords)
        x, y = self._project_xy(self.center + dens[:, 0], dens[:, 1])
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 3:
            return None
        return np.column_stack([x[m], y[m]])

    def project_polygon(self, lons: npt.ArrayLike, lats: npt.ArrayLike, *,
                        clip: str = 'auto',
                        expected_frac: float | None = None,
                        lat_center: float | None = None,
                        radius_deg: float | None = None,
                        min_piece_area: float | None = None) -> Any | None:
        """Project a closed spherical polygon to shapely Polygons.

        Parameters
        ----------
        lons, lats : array-like
            Closed polygon vertices in ICRS degrees
            (``lons[0] == lons[-1]``, same for lats).
        clip : str
            Projection-seam handling pipeline. Backend-specific
            interpretation, but all backends accept ``'auto'`` /
            ``'d3'`` / ``'simple'`` at minimum.
        expected_frac : float, optional
            Hint to the pipeline about the fraction of the visible
            sphere the polygon covers (used by stitching heuristics
            on the mpl side).
        lat_center, radius_deg : float, optional
            Optional shape-center hints passed through to the
            backend's projection.
        min_piece_area : float, optional
            Backend-specific small-piece filter override.

        Returns
        -------
        shapely.geometry.base.BaseGeometry
            Shapely Polygon / MultiPolygon in projected coords, or
            an empty geometry if nothing projects in. Callers should
            treat empty / ``None`` results as ``"no contribution"``.

        Notes
        -----
        Shared d3 pipeline (the matplotlib backend overrides this with
        its pixel-space variant; plotly inherits it). Two paths:

        * **No-crossing** — the polygon stays on one side of the wrap
          meridian: project the wrap-shifted vertices directly.
        * **Crossing-bearing** — project each
          :func:`~skyplothelper.geometry._antimeridian._antimeridian_clip`
          segment via :meth:`_project_xy`, then stitch consecutive
          segments along the frame silhouette with :meth:`_short_substring`
          (handles both small wrap-straddling shapes and polar shapes
          whose walk passes through the projected pole).

        ``lat_center`` / ``radius_deg`` / ``min_piece_area`` are accepted
        for signature compatibility but unused by this shared path (they
        are hints the matplotlib pixel pipeline consumes).
        """
        from shapely.geometry import Point, Polygon

        from ._antimeridian import _antimeridian_clip

        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        if not (np.isclose(lons[0], lons[-1])
                and np.isclose(lats[0], lats[-1])):
            lons = np.append(lons, lons[0])
            lats = np.append(lats, lats[0])

        # Globe (orthographic all-sky) frames — plotly globes reach the base
        # pipeline here — clip to the visible hemisphere instead of splitting at
        # the antimeridian (which lies on the far side). Flat frames skip this.
        if self._is_globe():
            return self._project_hemisphere_domain_clip(lons, lats,
                                                        expected_frac)

        center = self.center
        segments = _antimeridian_clip(lons, lats, center)
        if not segments:
            return None

        # No-crossing case: project the wrap-shifted polygon directly
        # (shared with the matplotlib backend's d3 fast-path).
        if (len(segments) == 1
                and segments[0]['entry_lat'] is None
                and segments[0]['exit_lat'] is None):
            return self._project_no_crossing(lons, lats, center,
                                             expected_frac)

        # Crossing-bearing case: project each segment, then stitch
        # consecutive segments along the frame silhouette. A single
        # crossing-bearing segment loops back from its exit to its own
        # entry along the silhouette (short walk for wrap-straddling
        # shapes, longer walk through the projected pole for polar ones).
        ext = self.frame_polygon.exterior
        ring_len = ext.length

        px_segs: list[tuple[Any, Any] | None] = []
        for seg in segments:
            xs, ys = self._project_xy(seg['lons'], seg['lats'])
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            finite = np.isfinite(xs) & np.isfinite(ys)
            if finite.sum() < 2:
                px_segs.append(None)
                continue
            px_segs.append((xs[finite], ys[finite]))

        boundary = []
        n_segs = len(px_segs)
        for si in range(n_segs):
            cur = px_segs[si]
            if cur is None:
                continue
            xs_cur, ys_cur = cur
            boundary.extend(list(zip(xs_cur, ys_cur)))
            # Walk the silhouette to the NEXT non-None segment's entry.
            next_si = (si + 1) % n_segs
            steps = 0
            while px_segs[next_si] is None and steps < n_segs:
                next_si = (next_si + 1) % n_segs
                steps += 1
            nxt = px_segs[next_si]
            if nxt is None:
                continue
            xs_nxt, ys_nxt = nxt
            t_exit = ext.project(Point(float(xs_cur[-1]), float(ys_cur[-1])))
            t_entry = ext.project(Point(float(xs_nxt[0]), float(ys_nxt[0])))
            walk = self._short_substring(ext, t_exit, t_entry, ring_len)
            if walk is not None and not walk.is_empty:
                boundary.extend(list(walk.coords))

        if len(boundary) < 3:
            return None
        poly = Polygon(boundary)
        if not poly.is_valid:
            from shapely.validation import make_valid
            poly = make_valid(poly)
        return self._finalize(poly, expected_frac)

    def _finalize(self, poly: Any, expected_frac: float | None) -> Any | None:
        """Clip a stitched polygon to the frame, then complement-flip.

        Shared tail of :meth:`project_polygon`: intersect against the
        frame silhouette and (when ``expected_frac`` is given) flip to
        the complement if that is the more plausible area — the trick
        that orients small wrap-straddling discs and polar caps.
        """
        if poly is None or poly.is_empty:
            return None
        from ._frame_geom import _safe_intersection
        try:
            clipped = _safe_intersection(poly, self.frame_polygon)
        except Exception:
            clipped = poly
        return self._complement_detect(clipped, expected_frac)

    def _project_no_crossing(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike, center: float,
        expected_frac: float | None,
    ) -> Any | None:
        """Project a non-seam-crossing polygon directly — the d3
        fast-path, shared by the base pipeline and the matplotlib
        backend's no-crossing interceptor.

        ``make_valid`` (not ``buffer(0)``) repairs an invalid projected
        ring: it is a no-op for the valid shapes this path actually
        sees (circles, ellipses, boxes, tissot discs), keeps the mpl
        output bit-for-bit, and preserves area for the rare
        self-crossing case rather than dropping a lobe. Closure-agnostic
        — shapely closes the ring, so closed or open input give the same
        polygon.
        """
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        lons = np.asarray(lons, dtype=float)
        shifted = ((lons - center + 180.0) % 360.0 - 180.0 + center)
        xs, ys = self._project_xy(shifted, lats)
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        finite = np.isfinite(xs) & np.isfinite(ys)
        if finite.sum() < 3:
            return None
        poly = Polygon(zip(xs[finite], ys[finite]))
        if not poly.is_valid:
            poly = make_valid(poly)
        return self._finalize(poly, expected_frac)

    # ------------------------------------------------------------------
    # Globe (visible-hemisphere) domain clip — shared by every backend.
    #
    # On an orthographic globe (SIN celestial / planet frame, or a plotly
    # globe) the far hemisphere is not drawable and the visible boundary is
    # the *limb* (the 90°-from-center small circle), NOT the antimeridian. A
    # filled region that spills past the limb can't be closed by projecting-
    # and-dropping the NaN far-side vertices (that chords across the disk) nor
    # by the antimeridian machinery (the center±180 seam is itself on the far
    # side). The robust fix — the one cartopy takes — is to clip the region to
    # the visible-hemisphere DOMAIN as a proper polygon operation.
    #
    # We do it in an azimuthal-equidistant frame centered on the globe center,
    # where the visible hemisphere is a disk of radius 90° and the far side is
    # a finite annulus (90°–180°). A single shapely intersection with the 90°
    # disk handles every case uniformly: a limb-crosser gets a clean limb arc,
    # a cap that encloses the whole hemisphere yields the whole disk, and a
    # far-side-only ring vanishes — no NaN, no chords, no complement guessing.
    # The clipped polygon is mapped back to (lon, lat) and projected with the
    # backend's own ``_project_xy`` (now all within the limb, so all finite).
    # ------------------------------------------------------------------

    def _is_globe(self) -> bool:
        """True when the frame is an orthographic all-sky globe whose region
        fills must be clipped to the visible hemisphere. Backends that can host
        a globe override this (``WCSAxesProjector`` for SIN celestial/planet
        frames, ``SkyplothelperProjector`` for plotly globes); the base default
        is ``False`` (flat frames need no hemisphere clip)."""
        return False

    def _project_hemisphere_domain_clip(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike,
        expected_frac: float | None = None,
    ) -> Any | None:
        """Clip a spherical polygon to the globe's visible hemisphere, then
        project. See the section comment above for the rationale. Returns a
        shapely geometry in the backend's projected coords, or ``None``."""
        from shapely.geometry import Polygon

        from ._frame_geom import _safe_intersection

        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        verts = _unit_vecs(lons, lats)
        center = _unit_vecs(np.array([self.center]),
                            np.array([self._center_lat]))[0]
        clipped = _clip_polygon_to_hemisphere(verts, center)
        if clipped is None or len(clipped) < 3:
            return None
        # unit vectors → (lon, lat); all now on the visible hemisphere, so
        # the backend projection returns finite coords.
        lon_c = np.degrees(np.arctan2(clipped[:, 1], clipped[:, 0]))
        lat_c = np.degrees(np.arcsin(np.clip(clipped[:, 2], -1.0, 1.0)))
        x, y = self._project_xy(lon_c, lat_c)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) < 3:
            return None
        poly = Polygon(zip(x[finite], y[finite]))
        # ``buffer(0)`` cleans self-touches from the projected ring. (A ring
        # that ENCLOSES a visible pole — e.g. the North American plate — carries
        # a meridian cut to the pole that survives as a thin seam under an edge
        # color; proper pole-enclosing handling is a tracked follow-up.)
        poly = poly.buffer(0)
        if poly.is_empty:
            return None
        # Safety clip to the drawn frame silhouette; no complement flip needed —
        # the hemisphere clip already yields the correctly-oriented visible
        # region.
        try:
            poly = _safe_intersection(poly, self.frame_polygon)
        except Exception:
            pass
        return None if poly.is_empty else poly

    def render_region(self, geom: Any, *, complement: bool = False,
                      min_area: float = 1.0, **style: Any) -> list[Any]:
        """Render a projected shapely geometry to backend artists.

        The render half of the projector contract: takes the geometry
        produced by :meth:`project_polygon` (already in the backend's
        projected coords) and emits the backend's native artists,
        adding them to the host figure/axes.

        Parameters
        ----------
        geom : shapely geometry or None
            Region in projected coords. ``None`` / empty renders
            nothing and returns an empty list.
        complement : bool
            If True, render the frame minus *geom* (everything outside
            the region) instead of the region itself.
        min_area : float
            Small-piece filter (px²) forwarded to the backend's
            shapely→artist conversion. Must match the threshold the
            upstream projection used (1.0 for full regions; smaller
            for sub-pixel primitives).
        **style
            Backend-native style kwargs. The matplotlib and plotly
            vocabularies differ (mpl: ``facecolor`` / ``edgecolor`` /
            ``alpha`` …; plotly: ``color`` / ``width`` / ``fillcolor``
            …), so this is intentionally not a unified style dict.

        Returns
        -------
        list
            The created backend artists.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared projection-pipeline core (backend-agnostic).
    #
    # These helpers live on the base because they are pure geometry on
    # ``self.frame_polygon`` — they don't care whether that frame is in
    # matplotlib pixel coords or plotly canvas coords. Keeping the
    # complement-detection heuristic and the frame-silhouette walk here
    # (rather than copied into each backend's stitch code) is what stops
    # the two backends from drifting on them. Backend-specific projection
    # and rendering stay on the subclasses.
    # ------------------------------------------------------------------

    def _complement_detect(self, clipped: Any, expected_frac: float | None) -> Any:
        """Flip an ambiguous stitch result to its complement when that is
        the more plausible area, binding ``self.frame_polygon``. See
        :func:`skyplothelper.geometry._frame_geom._complement_detect` for
        the heuristic itself."""
        from ._frame_geom import _complement_detect
        return _complement_detect(clipped, self.frame_polygon, expected_frac)

    @staticmethod
    def _short_substring(ext: Any, t_from: float, t_to: float,
                         ring_len: float) -> Any:
        """Return the shorter of the two arcs along the closed frame ring
        ``ext`` between linear-referenced positions ``t_from`` and
        ``t_to`` — used to stitch consecutive projected segments along
        the frame silhouette.

        Operates purely on shapely linear referencing, so it is
        coordinate-system-agnostic: the same walk serves the mpl pixel
        frame and the plotly canvas frame.
        """
        from shapely.geometry import LineString
        from shapely.ops import substring
        # Two candidate walks: forward (possibly wrapping past
        # ``ring_len``) and reverse; take whichever is shorter.
        forward = (t_to - t_from) % ring_len
        backward = ring_len - forward
        if forward <= backward:
            if t_to >= t_from:
                return substring(ext, t_from, t_to)
            # Wraps past 0: walk to the ring end, then from 0 to t_to.
            seg1 = substring(ext, t_from, ring_len)
            seg2 = substring(ext, 0.0, t_to)
            return LineString(list(seg1.coords) + list(seg2.coords))
        # backward < forward — walk the other way.
        if t_from >= t_to:
            seg = substring(ext, t_to, t_from)
            return LineString(list(seg.coords)[::-1])
        seg1 = substring(ext, t_to, ring_len)
        seg2 = substring(ext, 0.0, t_from)
        coords = list(seg1.coords) + list(seg2.coords)
        return LineString(coords[::-1])


class WCSAxesProjector(Projector):
    """Projector wrapping matplotlib WCSAxes — the existing mpl path.

    Defers to :func:`skyplothelper.geometry._api._dispatch_projection`
    for the projection-and-clip pipeline (preserving all of the
    ``'d3'`` / ``'simple'`` / ``'project_shape'`` clip modes
    :class:`CompoundRegion` exposed before the refactor), then converts
    the returned mpl Paths to shapely geometry via
    :func:`skyplothelper.geometry._frame_geom._paths_to_geom`. The
    dispatched pipeline already clips to the WCS frame polygon, so no
    separate frame intersection is needed here.

    Parameters
    ----------
    ax : matplotlib WCSAxes
        Host axes carrying the WCS used to project sphere coords.
    """

    def __init__(self, ax: Any) -> None:
        self.ax = ax
        # This class is the FITS-WCS projector: it projects via
        # ``ax.wcs.world_to_pixel_values``. The non-FITS custom-transform frames
        # (Robinson, Eckert, Winkel Tripel, …) have ``ax.wcs is None`` and are
        # served by ``WCSNonFitsProjector`` instead. The ``_projector_for_axes``
        # factory routes each frame to the right class; guard here so a direct
        # ``WCSAxesProjector(non_fits_ax)`` fails clearly rather than with a
        # cryptic ``NoneType has no world_to_pixel_values`` deep in the build.
        if getattr(ax, 'wcs', None) is None:
            raise ValueError(
                "WCSAxesProjector needs a FITS-projection frame (ax.wcs is a "
                "WCS); this axes is a non-FITS custom projection (Robinson, "
                "Eckert, Winkel Tripel, Kavrayskiy, McBryde) with ax.wcs=None. "
                "Use WCSNonFitsProjector, or build via _projector_for_axes(ax) "
                "which routes automatically.")
        # Cache the frame polygon (constructed once, reused per shape).
        from ._frame_geom import _get_frame_polygon, _get_projection_center
        self._frame_polygon = _get_frame_polygon(ax)
        self._center = _get_projection_center(ax)

    @property
    def center(self) -> float:
        return float(self._center)

    @property
    def frame_polygon(self) -> Any:
        return self._frame_polygon

    @property
    def _center_lat(self) -> float:
        """Projection center latitude (crval[1]) — used by the globe limb-fill
        path to test whether a far-side ring encloses the visible hemisphere.
        The base default of 0 would be wrong for a globe centered off the
        equator."""
        try:
            return float(self.ax.wcs.wcs.crval[1])
        except Exception:
            return 0.0

    @property
    def wcs(self) -> Any:
        """The host WCS — exposed so CompoundRegion can resolve
        sexagesimal / hourangle coordinate input via the existing
        ``_parse_coord`` helper."""
        return self.ax.wcs

    @property
    def wcs_frame(self) -> str:
        """The host axes' native sky frame (``'icrs'`` / ``'galactic'`` /
        ``'ecliptic'`` / ``'supergalactic'``). Used by the cross-frame
        ``add_lonlat_box`` path to transform input coords into the
        same frame the projector understands."""
        from ..wcs_frame import _get_wcs_frame_name
        return _get_wcs_frame_name(self.ax)

    def _project_xy(self, lons: npt.ArrayLike, lats: npt.ArrayLike) -> tuple[Any, Any]:
        """Project sphere ``(lon, lat)`` to WCS pixel coords. Used by the
        shared d3 no-crossing fast-path (:meth:`_project_no_crossing`);
        the crossing / pole / multi-segment cases stay on the pixel
        pipeline (see :meth:`project_polygon`)."""
        return self.ax.wcs.world_to_pixel_values(lons, lats)

    def world_bounds(self) -> tuple[float, float, float, float] | None:
        """World ``(lon, lat)`` box of this frame, or ``None`` if it is
        effectively all-sky.

        Sampled by mapping a grid over the frame-polygon bbox back through the
        WCS (off-domain points NaN out and are ignored). Returns ``None`` for a
        frame that spans ~the whole sphere (nothing to clip); for a frame that
        contains a pole, longitude is reported as the full ±180 about the
        center (only latitude clips). A small margin is added so the clip box
        never shaves the visible frame edge. Cached after the first call."""
        cached = getattr(self, '_world_bounds_cache', False)
        if cached is not False:
            return cached  # type: ignore[return-value]
        result = self._compute_world_bounds()
        self._world_bounds_cache = result
        return result

    # A frame whose visible field reaches this far (deg) from its center sees a
    # full hemisphere or more — a globe or an all-sky frame, where the existing
    # project_polygon path already works. Only tighter *field* frames (where a
    # global band mostly lands off-frame) take the bounded-field clip path.
    _FIELD_RADIUS_MAX_DEG = 80.0

    def _compute_world_bounds(self) -> tuple[float, float, float, float] | None:
        minx, miny, maxx, maxy = self._frame_polygon.bounds
        gx = np.linspace(minx, maxx, 40)
        gy = np.linspace(miny, maxy, 40)
        xx, yy = np.meshgrid(gx, gy)
        lon, lat = self.ax.wcs.pixel_to_world_values(xx.ravel(), yy.ravel())
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        good = np.isfinite(lon) & np.isfinite(lat)
        if int(good.sum()) < 4:
            return None
        lon = lon[good]
        lat = lat[good]

        # Gate on angular radius from the projection center (crval): a globe /
        # all-sky frame spans ~a hemisphere and is left on the standard path.
        crval = self.ax.wcs.wcs.crval
        cen = _unit_vecs(float(crval[0]), float(crval[1])).reshape(3)
        dots = np.clip(_unit_vecs(lon, lat) @ cen, -1.0, 1.0)
        if float(np.degrees(np.arccos(dots.min()))) > self._FIELD_RADIUS_MAX_DEG:
            return None

        lat_lo, lat_hi = float(lat.min()), float(lat.max())
        rl = _wrap180(lon - self.center)
        lon_span = float(rl.max() - rl.min())
        lat_span = lat_hi - lat_lo
        # Full-longitude sweep with no small angular radius ⇒ the field wraps a
        # pole; clip latitude only, leave longitude unbounded.
        if lon_span > 350.0:
            lon_lo, lon_hi = self.center - 180.0, self.center + 180.0
        else:
            lon_lo = float(self.center + rl.min())
            lon_hi = float(self.center + rl.max())
        mlat = max(0.05, 0.02 * lat_span)
        mlon = max(0.05, 0.02 * lon_span)
        return (lon_lo - mlon, lon_hi + mlon,
                max(-90.0, lat_lo - mlat), min(90.0, lat_hi + mlat))

    def angle_to_pixels(self, angle_deg: float) -> float:
        """WCS-native pixel-scale estimate for ``CompoundRegion`` buffering.

        Samples the pixel scale at the projection center via the WCS, with
        a CDELT fallback — kept distinct from the base's generic estimate
        because the WCS carries an exact scale the projected-step heuristic
        only approximates."""
        wcs = self.ax.wcs
        try:
            x1, _ = wcs.world_to_pixel_values(wcs.wcs.crval[0],
                                              wcs.wcs.crval[1])
            x2, _ = wcs.world_to_pixel_values(wcs.wcs.crval[0] + angle_deg,
                                              wcs.wcs.crval[1])
            if np.isfinite(x1) and np.isfinite(x2):
                return abs(x2 - x1)
        except Exception:
            pass
        try:
            return abs(angle_deg / wcs.wcs.cdelt[0])
        except Exception:
            return abs(angle_deg)

    def project_polygon(self, lons: npt.ArrayLike, lats: npt.ArrayLike, *,
                        clip: str = 'auto',
                        expected_frac: float | None = None,
                        lat_center: float | None = None,
                        radius_deg: float | None = None,
                        min_piece_area: float | None = None) -> Any | None:
        from ._api import _dispatch_projection
        from ._frame_geom import _paths_to_geom, _shapely_to_paths

        # Preserve the upstream sub-pixel filter: when a caller lowers
        # ``min_piece_area`` (deep-field surveys pass 0), the Path→geom
        # conversion must keep those slivers too, or it would silently
        # drop the very pieces the dispatch pipeline retained.
        geom_min_area = (1.0 if min_piece_area is None
                         else min(1.0, min_piece_area))

        # d3 no-crossing fast-path: this case is pixel-identical to the
        # shared base pipeline, so route it through the base rather than
        # the mpl pixel machinery. Everything below stays backend-specific
        # by design — ``_project_shape`` (single-crossing / pole) and the
        # multi-segment ``use_independent`` stitch repair
        # ``world_to_pixel_values`` pathologies that the plotly backend's
        # ``sph.project()`` simply doesn't have.
        if clip in ('d3', 'auto'):
            from ._antimeridian import _antimeridian_clip
            la = np.asarray(lons, dtype=float)
            lb = np.asarray(lats, dtype=float)

            # Globe (orthographic all-sky SIN) frame: the far hemisphere is not
            # drawable and the ``center±180`` antimeridian is NOT a visible seam
            # — it lies on the far side. Splitting the ring there fragments it
            # and chords across the far side (e.g. Afro-Eurasia). Instead clip
            # the region to the visible hemisphere (the shared azimuthal-
            # equidistant domain clip) and project the visible part. Only the
            # globe takes this branch; flat frames fall through to the
            # antimeridian pipeline unchanged.
            if self._is_globe():
                # Return the hemisphere-clipped geometry directly (already in
                # pixel coords). Do NOT round-trip through
                # _shapely_to_paths/_paths_to_geom: that filter is meant for the
                # pixel-stitch pipeline and drops a pole-enclosing MultiPolygon
                # (a plate wrapping the pole, e.g. the North American plate).
                return self._project_hemisphere_domain_clip(
                    la, lb, expected_frac)

            segments = _antimeridian_clip(la, lb, self.center)
            if (segments and len(segments) == 1
                    and segments[0]['entry_lat'] is None
                    and segments[0]['exit_lat'] is None):
                geom = self._project_no_crossing(la, lb, self.center,
                                                 expected_frac)
                if geom is None:
                    return None
                # Re-apply the exact post-filter the dispatch path used,
                # so the stored geometry is byte-identical to before.
                return _paths_to_geom(
                    _shapely_to_paths(geom, min_area=geom_min_area),
                    min_area=geom_min_area)

        paths = _dispatch_projection(
            self.ax, np.asarray(lons), np.asarray(lats), clip,
            expected_frac=expected_frac,
            lat_center=lat_center,
            radius_deg=radius_deg,
            min_piece_area=min_piece_area,
        )
        if not paths:
            return None
        return _paths_to_geom(paths, min_area=geom_min_area)

    def _is_globe(self) -> bool:
        """True when the axes is a zenithal *all-sky* frame — a SIN/AZP/…
        globe whose far hemisphere is not drawable and whose visible boundary
        is the limb (not the antimeridian).

        Gates the visible-hemisphere domain clip for region fills. A zenithal
        *field* (a small TAN/SIN image) is excluded because it reports a finite
        ``world_bounds`` and is served by the bounded-field clip path instead;
        only the all-sky globe (``world_bounds() is None``) needs the hemisphere
        clip. Cached after the first call."""
        cached = getattr(self, '_globe_cache', None)
        if cached is not None:
            return cached
        result = False
        try:
            from ..projections.project import _ZENITHAL_FITS_CODES
            from ..wcs_frame import _axes_fits_code
            code = _axes_fits_code(self.ax)
            result = (code in _ZENITHAL_FITS_CODES
                      and self.world_bounds() is None)
        except Exception:
            result = False
        self._globe_cache = result
        return result

    def render_region(self, geom: Any, *, complement: bool = False,
                      min_area: float | None = None, **style: Any) -> list[Any]:
        """Render projected shapely geometry to matplotlib PathPatches.

        Mirrors the post-projection render loop the shapes helpers and
        :meth:`CompoundRegion.render` use: convert the geometry to mpl
        Paths via ``_shapely_to_paths`` and add a ``PathPatch`` per
        piece. ``complement=True`` defers to ``_render_complement``,
        which fills the frame minus the shape and draws edges on the
        shape boundary (not the frame).

        ``min_area=None`` selects the backend default (1.0 px² for this
        FITS pixel-space frame), so a caller that means "backend default"
        need not know the coordinate scale — the non-FITS projector reads
        the same ``None`` as a frame-relative threshold.
        """
        if min_area is None:
            min_area = 1.0
        return _render_region_mpl(self.ax, geom, self._frame_polygon,
                                  complement=complement, min_area=min_area,
                                  **style)


def _render_region_mpl(ax: Any, geom: Any, frame_polygon: Any, *,
                       complement: bool = False, min_area: float = 1.0,
                       **style: Any) -> list[Any]:
    """Render a projected shapely geometry to matplotlib PathPatches on ``ax``.

    Shared by both matplotlib projectors: the geometry is already in the
    axes' data coordinates (FITS pixel space for :class:`WCSAxesProjector`,
    the projected plane for :class:`WCSNonFitsProjector`), so the render is
    identical — add a ``PathPatch`` per piece in the default ``ax.transData``.
    Kept as a module function (not a method) so the two projectors can never
    drift on the stroke / complement / hairline handling.
    """
    from matplotlib.patches import PathPatch

    from ._frame_geom import _fix_hairline_kwargs, _shapely_to_paths

    # Shared legibility-stroke knob for every shape helper that funnels
    # through here (add_spherical_polygon / add_geodesic_circle /
    # add_rectangle / add_square / add_ellipse / add_annulus): translate
    # ``stroke_color`` / ``stroke_lw`` into a ``path_effects`` outline. A
    # caller that already passed ``path_effects=`` (e.g. fill_boundaries_
    # globe, which strokes upstream) wins, so this never double-strokes.
    stroke_color = style.pop('stroke_color', None)
    stroke_lw = style.pop('stroke_lw', None)
    if stroke_color is not None and 'path_effects' not in style:
        from .._stroke import _stroke_path_effects
        # Default the width (as the band helpers do) so stroke_color alone
        # is enough to get a stroke.
        _pe = _stroke_path_effects(
            stroke_color, 3.0 if stroke_lw is None else stroke_lw)
        if _pe is not None:
            style['path_effects'] = _pe

    if complement:
        from ._projection import _render_complement
        shape_paths = (_shapely_to_paths(geom, min_area=min_area)
                       if geom is not None and not geom.is_empty
                       else [])
        return _render_complement(ax, shape_paths, frame_polygon, **style)

    if geom is None or geom.is_empty:
        return []
    _fix_hairline_kwargs(style)
    patches = []
    for path in _shapely_to_paths(geom, min_area=min_area):
        patch = PathPatch(path, **style)
        ax.add_patch(patch)
        patches.append(patch)
    return patches


class WCSNonFitsProjector(Projector):
    """Projector for the non-FITS custom-projection matplotlib frames.

    The five skyplothelper-extended projections — Robinson, Eckert IV,
    Winkel Tripel, Kavrayskiy VII, McBryde-Thomas — are drawn by a
    matplotlib ``CurvedTransform`` rather than a FITS WCS, so
    ``ax.wcs is None`` and :class:`WCSAxesProjector` cannot serve them.
    This projector fills that gap by driving the **shared** base
    :meth:`Projector.project_polygon` pipeline (the same one the plotly
    backend uses) against the axes' own world→data transform.

    The projection primitive :meth:`_project_xy` reuses
    ``ax.coords._transform`` — the exact transform the frame and every
    existing line overlay (great circles, coastlines, baselines) already
    render through — so a filled region registers pixel-perfect with the
    lines on the same frame, and any center / direction / oblique aspect
    baked into the frame is honored automatically without re-deriving it.
    (This is why we use the axes transform rather than ``sph.project()``:
    ``project()`` doesn't apply the oblique aspect for these projections,
    and raises outright for some of them — e.g. Eckert IV has no FITS
    projection code.)

    Parameters
    ----------
    ax : matplotlib WCSAxes
        A non-FITS custom-projection frame built by
        :func:`skyplothelper.make_wcs_frame` (``ax.wcs is None``,
        ``ax._sph_proj_key`` set).
    """

    def __init__(self, ax: Any) -> None:
        self.ax = ax
        # This projector is only for the non-FITS frames; a FITS frame must
        # use WCSAxesProjector (its pixel-space pole/jump repair). The factory
        # _projector_for_axes routes correctly; guard here in case of misuse.
        if getattr(ax, 'wcs', None) is not None:
            raise ValueError(
                "WCSNonFitsProjector is for non-FITS custom-projection frames "
                "(ax.wcs is None); this axes has a FITS WCS — use "
                "WCSAxesProjector.")
        # world (lon, lat degrees) → axes data coords, the non-FITS analog of
        # ``wcs.world_to_pixel_values``. ``ax.coords._transform`` is the
        # data→world CurvedTransform the frame was built with; its inverse is
        # what apply_boundary_labels already uses to place tick labels.
        self._world_to_data = ax.coords._transform.inverted()
        self._center = float(getattr(ax, '_sph_center_lon', 0.0))
        self._clat = float(getattr(ax, '_sph_center_lat', 0.0))
        # Frame silhouette in data coords, straight from the drawn spine (no
        # WCS needed); fall back to the axes bbox if the spine isn't built yet.
        from ._frame_geom import _get_visual_frame
        fp = _get_visual_frame(ax)
        if fp is None:
            from shapely.geometry import box
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            fp = box(xlim[0], ylim[0], xlim[1], ylim[1])
        self._frame_polygon = fp

    @property
    def center(self) -> float:
        return self._center

    @property
    def _center_lat(self) -> float:
        return self._clat

    @property
    def wcs_frame(self) -> str:
        """The host axes' native sky frame (``'icrs'`` / ``'galactic'`` / …),
        stamped on the axes as ``_sph_frame`` by ``make_wcs_frame``."""
        return str(getattr(self.ax, '_sph_frame', 'icrs'))

    @property
    def frame_polygon(self) -> Any:
        return self._frame_polygon

    @property
    def frame_edge_tolerance(self) -> float:
        """Frame-relative limb-suppression buffer for
        :meth:`CompoundRegion.render_boundary`.

        The data coords here are radians-scale (frame diagonal ~O(1), not the
        hundreds-of-pixels a FITS frame spans), so the base ``0.5`` default
        would be a large fraction of the map and clip real boundary near the
        edge (the symptom: a compound region's outline goes missing on the
        side nearest the limb). ``diagonal × 1e-3`` reproduces the same
        *relative* sub-pixel suppression the FITS frames get from ``0.5``.
        """
        minx, miny, maxx, maxy = self._frame_polygon.bounds
        return float(np.hypot(maxx - minx, maxy - miny)) * 1e-3

    def _project_xy(self, lons: npt.ArrayLike, lats: npt.ArrayLike) -> tuple[Any, Any]:
        """Project sphere ``(lon, lat)`` degrees to axes data coords via the
        frame's own world→data transform (matches the line overlays exactly)."""
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        pts = self._world_to_data.transform(np.column_stack([lons, lats]))
        return pts[:, 0], pts[:, 1]

    def project_polygon(self, lons: npt.ArrayLike, lats: npt.ArrayLike, *,
                        clip: str = 'auto',
                        expected_frac: float | None = None,
                        lat_center: float | None = None,
                        radius_deg: float | None = None,
                        min_piece_area: float | None = None) -> Any | None:
        """Project a closed spherical polygon via the shared base pipeline.

        Only adds one thing to :meth:`Projector.project_polygon`: when the
        caller supplies no ``expected_frac`` (the standalone shape helpers —
        ``add_geodesic_circle`` etc. — don't), estimate it from the vertices
        so the base complement-detector can orient a wrap-straddling shape
        correctly. The FITS ``clip='d3'`` dispatch computes the same estimate
        the same way; here it lives on the projector because the base pipeline
        is otherwise geometry-only and never sees the raw vertices' span.
        """
        if expected_frac is None:
            from ._frame_geom import _expected_frac_from_vertices
            expected_frac = _expected_frac_from_vertices(lons, lats,
                                                         self._center)
        return super().project_polygon(
            lons, lats, clip=clip, expected_frac=expected_frac,
            lat_center=lat_center, radius_deg=radius_deg,
            min_piece_area=min_piece_area)

    def render_region(self, geom: Any, *, complement: bool = False,
                      min_area: float | None = None, **style: Any) -> list[Any]:
        """Render projected shapely geometry to matplotlib PathPatches (in the
        axes' data coords, via the shared :func:`_render_region_mpl`).

        The default ``min_area`` differs from the FITS projector: these frames'
        data coordinates are the projected plane in *radians-scale* units (the
        whole map spans only a few units, frame area ~O(10)), so the FITS
        pixel-scale default of ``1.0`` would discard every real region. A
        frame-relative sliver threshold (``frame_area × 1e-6``, matching
        :func:`_cleanup_for_render`'s numerical-noise intent) filters stitch
        artifacts while keeping even sub-degree shapes.
        """
        if min_area is None:
            min_area = self._frame_polygon.area * 1e-6
        return _render_region_mpl(self.ax, geom, self._frame_polygon,
                                  complement=complement, min_area=min_area,
                                  **style)


def _projector_for_axes(ax: Any) -> Projector:
    """Return the matplotlib projector matching ``ax``'s frame type.

    A FITS-projection frame (``ax.wcs`` is a WCS) gets
    :class:`WCSAxesProjector` — the pixel-space pole/jump pipeline. A
    non-FITS custom-projection frame (Robinson / Eckert / Winkel Tripel /
    Kavrayskiy / McBryde, ``ax.wcs is None``) gets
    :class:`WCSNonFitsProjector`, which drives the shared base pipeline
    through the frame's own transform. Every region helper and
    :class:`CompoundRegion` builds its projector through this factory so
    both frame families are served identically.
    """
    if getattr(ax, 'wcs', None) is None:
        return WCSNonFitsProjector(ax)
    return WCSAxesProjector(ax)
