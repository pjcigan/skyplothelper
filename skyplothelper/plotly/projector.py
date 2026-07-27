"""Plotly-side :class:`Projector` for :class:`CompoundRegion`.

Lets :class:`skyplothelper.CompoundRegion` drive the same set-algebra
pipeline (union / intersection / difference / complement) without an
mpl axes in the loop, projecting via :func:`sph.project` into the
plotly figure's canvas coordinates so the resulting shapely geometry
can be rendered directly to plotly traces.

The clip → project → stitch → complement-detect pipeline itself lives on
the shared :class:`~skyplothelper.geometry._projector.Projector` base;
this subclass supplies only the plotly-specific plug-ins — the canvas
frame silhouette (:attr:`frame_polygon`) and the projection primitive
(:meth:`_project_xy`).

Use via :func:`skyplothelper.plotly.make_compound_region` (a thin
factory) or by passing the projector to ``CompoundRegion`` directly::

    from skyplothelper.plotly.projector import SkyplothelperProjector
    proj = SkyplothelperProjector.from_figure(fig)
    region = CompoundRegion(proj)
    region.add_circle(30, -10, 20).subtract_circle(40, -5, 8)
    sphpl.add_compound_region(fig, region, fillcolor='rgba(80,200,255,0.4)')
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ..geometry._projector import Projector
from ..projections.project import _ZENITHAL_FITS_CODES
from ..projections.project import project as _project


class SkyplothelperProjector(Projector):
    """Projector driving ``sph.project()`` for plotly figures.

    Parameters
    ----------
    projection : str
        FITS projection code (``'AIT'``, ``'MOL'``, ``'SIN'``, ...) or
        skyplothelper-extended name (``'robinson'``, ``'kavrayskiy'``,
        ...). Same vocabulary as :func:`skyplothelper.project`.
    center : float
        Projection center longitude in degrees.
    lat_center : float
        Projection center latitude in degrees. Default ``0``.
    direction : {'sky', 'geographic'}
        x-axis orientation. Default ``'sky'``.
    frame_n_samples : int
        Number of vertices per side when building the silhouette
        polygon. Default ``361``.
    """

    def __init__(self, projection: str = 'AIT', center: float = 0.0,
                 lat_center: float = 0.0, direction: str = 'sky',
                 frame_n_samples: int = 361) -> None:
        self.projection = projection
        self._center = float(center)
        self.lat_center = float(lat_center)
        self.direction = direction
        self.frame_n_samples = int(frame_n_samples)
        self._frame_polygon: Any = None

    @classmethod
    def from_figure(cls, fig: Any) -> SkyplothelperProjector:
        """Build a projector from the metadata stamped on ``fig`` by
        :func:`skyplothelper.plotly.make_figure`."""
        meta = (getattr(fig, 'layout', None)
                and getattr(fig.layout, 'meta', None)) or {}
        return cls(
            projection=meta.get('sph_projection', 'AIT'),
            center=meta.get('sph_center', 0.0),
            lat_center=meta.get('sph_lat_center', 0.0),
            direction=meta.get('sph_direction', 'sky'),
        )

    @property
    def center(self) -> float:
        return self._center

    @property
    def _center_lat(self) -> float:
        """Projection center latitude — feeds the base ``angle_to_pixels``
        scale estimate used by ``CompoundRegion.expand`` / ``contract``."""
        return self.lat_center

    @property
    def wcs_frame(self) -> str:
        """Plotly figures are rendered in ICRS by convention — the
        cross-frame ``add_lonlat_box`` path uses this to know how to
        transform input coords."""
        return 'icrs'

    @property
    def frame_polygon(self) -> Any:
        """The projection silhouette as a shapely Polygon in canvas
        (x, y) coords. Built lazily and cached on first access."""
        if self._frame_polygon is None:
            self._frame_polygon = self._build_frame_polygon()
        return self._frame_polygon

    def _build_frame_polygon(self) -> Any:
        # A zenithal projection (SIN / TAN / ARC / ZEA / ...) is centered on a
        # point, not wrapped at a meridian: its silhouette is the projected
        # limb about that center, so it needs the hull path below. The wrap-
        # meridian trace here is only valid for cylindrical / pseudo-
        # cylindrical / conic frames, whose boundary IS the ``center ± 180``
        # meridian.
        if self.projection.upper() in _ZENITHAL_FITS_CODES:
            return self._build_limb_polygon()
        from shapely.geometry import Polygon
        eps = 0.05
        lats = np.linspace(-89.95, 89.95, self.frame_n_samples)
        right_lons = np.full_like(lats, self._center + 180.0 - eps)
        left_lons = np.full_like(lats, self._center - 180.0 + eps)
        right_x, right_y = _project(
            right_lons, lats,
            projection=self.projection, center=self._center,
            lat_center=self.lat_center, direction=self.direction)
        left_x, left_y = _project(
            left_lons, lats[::-1],
            projection=self.projection, center=self._center,
            lat_center=self.lat_center, direction=self.direction)
        xs = np.concatenate([right_x, left_x])
        ys = np.concatenate([right_y, left_y])
        # Drop any NaN samples (pole pinch on some projections) so the
        # Polygon is valid.
        finite = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[finite]
        ys = ys[finite]
        poly = Polygon(zip(xs, ys))
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly

    def _build_limb_polygon(self) -> Any:
        """Silhouette of a zenithal projection: the convex hull of the whole
        visible sphere.

        The wrap-meridian trace can't build this — on a globe ``center ± 180``
        is the *far* side of the sphere, which is unprojectable, so that path
        collapses to a degenerate sliver and clips every compound region away
        to nothing. Every zenithal visible region is a convex disk (a limb,
        unlike a conic's re-entrant wedge), so the convex hull of a dense
        whole-sphere sample recovers it exactly — and handles the projections
        whose visible region isn't a *centered* disk (NCP, oblique AZP/SZP),
        which a fixed-radius limb circle would not.

        Points beyond the limb project to NaN (or, near a divergence, to the
        huge magnitudes :func:`project` now masks to NaN) and drop out of the
        hull. Returns an empty ``Polygon`` when essentially nothing is visible
        — the region then renders empty, which is the honest result.
        """
        from shapely.geometry import MultiPoint, Polygon
        n = self.frame_n_samples
        lons = np.linspace(-180.0, 180.0, n) + self._center
        lats = np.linspace(-89.95, 89.95, max(2, n // 2))
        grid_lon, grid_lat = np.meshgrid(lons, lats)
        x, y = self._project_xy(grid_lon.ravel(), grid_lat.ravel())
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) < 4:
            return Polygon()
        hull = MultiPoint(np.column_stack([x[finite], y[finite]])).convex_hull
        return hull if hull.geom_type == 'Polygon' else Polygon()

    def _project_xy(self, lons: npt.ArrayLike,
                    lats: npt.ArrayLike) -> tuple[Any, Any]:
        """Project sphere ``(lon, lat)`` into this figure's canvas coords
        via ``sph.project()``.

        The single backend-specific seam the shared
        :meth:`Projector.project_polygon` pipeline delegates to — all the
        antimeridian clipping, segment stitching, complement-detection
        and frame clipping run on the base against this primitive.
        """
        return _project(
            lons, lats,
            projection=self.projection, center=self._center,
            lat_center=self.lat_center, direction=self.direction)

    # The all-sky pseudo-cylindrical frames have a wrap meridian at
    # ``center ± 180``; these overrides split overlay geometry there so a
    # line / fill breaks cleanly at the seam instead of sweeping across the
    # canvas (the seamless FITS frames inherit the base no-split defaults).
    # Overriding the split seam points (not project_*) keeps the split as
    # the single backend difference; the base composes split + project. The
    # split helpers operate on sphere coords, so they stay backend-side here
    # rather than on the geometry base.

    def split_polyline(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Shift into the projection window and break an open polyline at
        the wrap meridian."""
        from .core import _split_polyline_at_wrap
        seg_lons, seg_lats = _split_polyline_at_wrap(lons, lats, self._center)
        return np.asarray(seg_lons, dtype=float), np.asarray(seg_lats, dtype=float)

    def split_polygon_pieces(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike,
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Split a closed polygon boundary at the wrap meridian into one
        piece per visible lobe."""
        from .core import _split_polygon_at_wrap
        return [(np.asarray(pl, dtype=float), np.asarray(pb, dtype=float))
                for pl, pb in _split_polygon_at_wrap(lons, lats, self._center)]


class WCSPixelProjector(Projector):
    """Projector that maps sky coords to FITS *pixel* coords via an astropy WCS.

    The FITS-image viewer (:func:`skyplothelper.plotly.add_fits_image`) draws
    the image in pixel coordinates as a ``go.Heatmap``; this projector lets
    :class:`~skyplothelper.CompoundRegion` accumulate set-algebra geometry in
    that same pixel space, so circles / polygons / boolean regions overlay on
    the image for free through :func:`~skyplothelper.plotly.add_compound_region`.

    The projection primitive is simply ``wcs.world_to_pixel_values`` (0-based,
    matching the Heatmap grid), so the projector is projection-agnostic — it
    works for SIN / TAN / any FITS-WCS without knowing the projection name.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The image WCS. A WCS with degenerate (Stokes / frequency) axes is
        reduced to its 2-D celestial sub-WCS automatically.
    image_shape : tuple
        ``(ny, nx)`` pixel dimensions of the displayed image.
    """

    def __init__(self, wcs: Any, image_shape: tuple[int, int]) -> None:
        # Reduce degenerate (e.g. 4-D radio-cube) WCS to the 2-D celestial one
        # so world_to_pixel_values takes (lon, lat) and returns (x, y).
        self.wcs = wcs.celestial if getattr(wcs, 'naxis', 2) > 2 else wcs
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self._frame_polygon: Any = None

    @classmethod
    def from_figure(cls, fig: Any) -> WCSPixelProjector:
        """Rebuild a projector from the FITS metadata stamped on ``fig`` by
        :func:`skyplothelper.plotly.add_fits_image` / ``make_fits_figure``."""
        from astropy.io.fits import Header
        from astropy.wcs import WCS
        meta = (getattr(fig, 'layout', None)
                and getattr(fig.layout, 'meta', None)) or {}
        hdr_str = meta.get('sph_wcs_header')
        if not hdr_str:
            raise ValueError(
                "figure carries no FITS WCS metadata (sph_wcs_header) — build "
                "it with skyplothelper.plotly.add_fits_image / make_fits_figure")
        wcs = WCS(Header.fromstring(hdr_str))
        shape = meta.get('sph_image_shape', (0, 0))
        return cls(wcs, (int(shape[0]), int(shape[1])))

    @property
    def center(self) -> float:
        return float(self.wcs.wcs.crval[0])

    @property
    def _center_lat(self) -> float:
        return float(self.wcs.wcs.crval[1])

    @property
    def wcs_frame(self) -> str:
        """Sky frame the WCS coordinates are in (``'icrs'`` by convention for
        the radio / optical FITS images this targets)."""
        return 'icrs'

    @property
    def frame_polygon(self) -> Any:
        """The image footprint as a shapely rectangle in pixel coords
        (cell edges ``[-0.5, nx-0.5] × [-0.5, ny-0.5]``). Used to clip the
        accumulated region geometry to the visible image."""
        if self._frame_polygon is None:
            from shapely.geometry import box
            ny, nx = self.image_shape
            self._frame_polygon = box(-0.5, -0.5, nx - 0.5, ny - 0.5)
        return self._frame_polygon

    def _project_xy(self, lons: npt.ArrayLike,
                    lats: npt.ArrayLike) -> tuple[Any, Any]:
        """Sky ``(lon, lat)`` → FITS pixel ``(x, y)`` (0-based) via the WCS."""
        return self.wcs.world_to_pixel_values(lons, lats)

    def project_points(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Project to pixels, dropping points that fall far off the image so
        an overlay polyline breaks instead of streaking (a TAN field's far
        hemisphere diverges to huge pixel values). Used by the standalone
        overlay helpers; ``CompoundRegion`` projects via ``_project_xy`` and
        clips with its own frame intersection, so it is unaffected."""
        return self._nan_off_frame(*super().project_points(lons, lats))


def _offset_linear_map(wcs2d: Any, ref: Sequence[float],
                       factor: float) -> tuple[float, float, float, float]:
    """Linear pixel→offset map about the reference point.

    Returns ``(sx, sy, cx, cy)`` such that ``offset_x = sx*(px - cx)`` and
    ``offset_y = sy*(py - cy)``, where ``sx`` / ``sy`` are the signed central
    east / north offset per pixel (in the unit whose deg→unit factor is
    ``factor``) and ``(cx, cy)`` is the reference pixel. This is the standard
    constant-scale "relative coordinate" representation used for compact
    fields; the RA flip and cos(dec) are folded into ``sx``.
    """
    cx, cy = wcs2d.world_to_pixel_values(ref[0], ref[1])
    cx, cy = float(np.ravel(cx)[0]), float(np.ravel(cy)[0])
    cosd = np.cos(np.radians(ref[1]))
    ra_xp, _ = wcs2d.pixel_to_world_values(cx + 1.0, cy)
    _, dec_yp = wcs2d.pixel_to_world_values(cx, cy + 1.0)
    dra = ((float(np.ravel(ra_xp)[0]) - ref[0] + 180.0) % 360.0) - 180.0
    sx = dra * cosd * factor
    sy = (float(np.ravel(dec_yp)[0]) - ref[1]) * factor
    return sx, sy, cx, cy


class WCSOffsetProjector(Projector):
    """Projects sky coords to angular-**offset** display coords (mas / arcsec
    from a reference), for offset-mode FITS figures.

    The offset image is shown in a constant-scale relative frame (linear in
    pixel), so plotly's native ticking gives round, zoom-adaptive labels.
    Overlays must land in that same frame: this projector composes the exact
    ``wcs.world_to_pixel`` with the linear pixel→offset map used for the image,
    so circles / regions / markers stay pinned to the right sky positions.
    """

    def __init__(self, wcs: Any, image_shape: tuple[int, int],
                 ref: Sequence[float], factor: float) -> None:
        self.wcs = wcs.celestial if getattr(wcs, 'naxis', 2) > 2 else wcs
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.ref = (float(ref[0]), float(ref[1]))
        self.factor = float(factor)
        self.sx, self.sy, self.cx, self.cy = _offset_linear_map(
            self.wcs, self.ref, self.factor)
        self._frame_polygon: Any = None

    @classmethod
    def from_figure(cls, fig: Any) -> WCSOffsetProjector:
        from astropy.io.fits import Header
        from astropy.wcs import WCS

        from ..ticks import OffsetFormatter
        meta = (getattr(fig, 'layout', None)
                and getattr(fig.layout, 'meta', None)) or {}
        hdr_str = meta.get('sph_wcs_header')
        if not hdr_str:
            raise ValueError(
                "figure carries no FITS WCS metadata — build it with "
                "skyplothelper.plotly.add_fits_image")
        wcs = WCS(Header.fromstring(hdr_str))
        shape = meta.get('sph_image_shape', (0, 0))
        ref = meta.get('sph_ref_coord') or (wcs.celestial.wcs.crval[0],
                                            wcs.celestial.wcs.crval[1])
        unit_key = meta.get('sph_offset_units', 'arcsec')
        factor = OffsetFormatter._UNIT_LABELS.get(
            unit_key, OffsetFormatter._UNIT_LABELS['arcsec'])[1]
        return cls(wcs, (int(shape[0]), int(shape[1])), ref, factor)

    @property
    def center(self) -> float:
        return self.ref[0]

    @property
    def _center_lat(self) -> float:
        return self.ref[1]

    @property
    def wcs_frame(self) -> str:
        return 'icrs'

    @property
    def frame_polygon(self) -> Any:
        if self._frame_polygon is None:
            from shapely.geometry import box
            ny, nx = self.image_shape
            xs = self.sx * (np.array([-0.5, nx - 0.5]) - self.cx)
            ys = self.sy * (np.array([-0.5, ny - 0.5]) - self.cy)
            self._frame_polygon = box(min(xs), min(ys), max(xs), max(ys))
        return self._frame_polygon

    def _project_xy(self, lons: npt.ArrayLike,
                    lats: npt.ArrayLike) -> tuple[Any, Any]:
        px, py = self.wcs.world_to_pixel_values(lons, lats)
        return (self.sx * (np.asarray(px, dtype=float) - self.cx),
                self.sy * (np.asarray(py, dtype=float) - self.cy))

    def project_points(
        self, lons: npt.ArrayLike, lats: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Project to offset coords, dropping far-off-image points (see
        :meth:`WCSPixelProjector.project_points`)."""
        return self._nan_off_frame(*super().project_points(lons, lats))
