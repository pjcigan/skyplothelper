"""Custom matplotlib frames and CurvedTransform classes for non-FITS projections.

Frame classes (BaseFrame subclasses) define the visible boundary envelope on
WCSAxes; Transform classes (CurvedTransform subclasses) wrap the projection
math from ``._math`` so matplotlib can render data through them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from astropy.visualization.wcsaxes import WCSAxes  # noqa: F401
from astropy.visualization.wcsaxes.frame import BaseFrame, EllipticalFrame  # noqa: F401
from matplotlib.path import Path
from matplotlib.transforms import Transform as MplTransform  # noqa: F401

try:
    from astropy.visualization.wcsaxes.frame import RectangularFrame  # noqa: F401
except ImportError:
    RectangularFrame = None  # fallback handled in make_wcs_frame

try:
    from astropy.visualization.wcsaxes.transforms import CurvedTransform
    _HAS_CURVEDTRANSFORM = True
except ImportError:
    _HAS_CURVEDTRANSFORM = False

from ._math import (
    _ROB_LATS,
    _ROB_X,
    _ROB_Y,
    _eckert4_forward,
    _eckert4_inverse,
    _kavrayskiy_forward,
    _kavrayskiy_inverse,
    _mcbryde_forward,
    _mcbryde_inverse,
    _robinson_forward,
    _robinson_inverse,
    _winkel_forward,
    _winkel_inverse,
)

# ===== BaseFrame subclasses (boundary envelopes) =====

class _AllSkyCustomFrame(BaseFrame):
    """
    Base class for custom all-sky frames with proper tick label placement.

    Subclasses parameterize the right half of the boundary curve via two
    methods of a parameter ``t ∈ [-1, 1]`` (with ``t = -1`` at the bottom
    of the boundary and ``t = +1`` at the top):

    - ``_boundary_x(t)`` returns the x-envelope, normalized so the equator
      is 1.0.
    - ``_boundary_y(t)`` returns the y-envelope, normalized so ±1.0 are
      the top/bottom of the boundary. Defaults to ``t`` (linear), which
      is correct for projections whose y at lon=±180 is linear in lat
      (Kavrayskiy, SFL) and for parameter-as-y boundary forms (PAR,
      circular). Pseudo-cylindrical projections with non-linear y(lat)
      at the antimeridian (Robinson, Eckert IV, Winkel, McBryde) override
      this to avoid an inward-bowing frame outline.

    Uses the same three-spine architecture as astropy's EllipticalFrame:
    - 'c' (boundary curve) — drawn, used for clipping
    - 'h' (horizontal) — equator line, used for RA tick/axis label placement
    - 'v' (vertical) — central meridian, used for Dec tick/axis label placement
    """
    spine_names = 'chv'
    _spine_auto_position_order = 'chv'

    def _boundary_x(self, t: npt.ArrayLike) -> np.ndarray:
        """Return x-envelope (0 to 1) along the right boundary parameterized by t ∈ [-1, 1]."""
        raise NotImplementedError

    def _boundary_y(self, t: npt.ArrayLike) -> np.ndarray:
        """Return y-envelope (-1 to 1) along the right boundary parameterized by t ∈ [-1, 1].

        Default is identity, i.e. y is linear in t.
        """
        # np.asarray is a no-op view for the array inputs callers pass;
        # it only formalizes the declared ndarray return for type checking.
        return np.asarray(t)

    def update_spines(self) -> None:
        xmin, xmax = self.parent_axes.get_xlim()
        ymin, ymax = self.parent_axes.get_ylim()
        xcen = 0.5 * (xmin + xmax)
        ycen = 0.5 * (ymin + ymax)
        half_w = 0.5 * (xmax - xmin)
        half_h = 0.5 * (ymax - ymin)

        n = 500
        # Trim the parameter range slightly when the frame has a
        # pointy apex (``env_x(±1) ≈ 0``), e.g. SinusoidalFrame /
        # ParabolicFrame. At the exact apex, all meridians converge
        # to a single (xcen, y_pole) point, which astropy's tick
        # discovery reads as a degenerate vertex with many meridian
        # crossings — producing duplicate ticks and malformed labels
        # (the ``'$'`` rendering bug at SFL/PAR top apex). Stopping
        # the curve at t=±(1-eps) yields ``env_x ≈ small_nonzero`` at
        # the endpoints, which then triggers the "flat segment"
        # branch below (n_flat intermediate vertices over a tiny
        # x-gap), distributing the meridian crossings cleanly across
        # those vertices instead of stacking at a single apex point.
        env_x_top = abs(float(self._boundary_x(np.array([1.0]))[0]))
        env_x_bot = abs(float(self._boundary_x(np.array([-1.0]))[0]))
        APEX_EPS = 1e-3
        t_max = 1.0 - APEX_EPS if env_x_top < APEX_EPS else 1.0
        t_min = -1.0 + APEX_EPS if env_x_bot < APEX_EPS else -1.0
        t = np.linspace(t_min, t_max, n)
        envelope_x = self._boundary_x(t)
        envelope_y = self._boundary_y(t)

        x_right = xcen + half_w * envelope_x
        y_vals = ycen + half_h * envelope_y
        x_left = xcen - half_w * envelope_x

        # For projections with flat poles (env_x[±1] > 0), densify the
        # straight top/bottom connecting segments with intermediate
        # vertices. Astropy computes tick angles from each spine vertex
        # via the inverse-WCS Jacobian, then linearly interpolates angles
        # between vertices. With only the two corner vertices, top-edge
        # ticks get an interpolation of the (slanted) corner Jacobians,
        # which is the wrong direction for meridian-aligned ticks.
        # Inserting intermediate vertices forces astropy to evaluate the
        # gradient at each in-segment position so each top-edge tick
        # picks up the correct local meridian-tangent angle.
        n_flat = 200  # vertices to insert along each flat pole segment
        top_x_r = x_right[-1]
        top_x_l = x_left[-1]
        bot_x_r = x_right[0]
        bot_x_l = x_left[0]
        if not np.isclose(top_x_r, top_x_l):
            top_xs = np.linspace(top_x_r, top_x_l, n_flat + 2)[1:-1]
            top_ys = np.full(n_flat, y_vals[-1])
        else:
            top_xs = np.empty(0)
            top_ys = np.empty(0)
        if not np.isclose(bot_x_r, bot_x_l):
            # left→right (after walking left half down, before closing back to bot_x_r)
            bot_xs = np.linspace(bot_x_l, bot_x_r, n_flat + 2)[1:-1]
            bot_ys = np.full(n_flat, y_vals[0])
        else:
            bot_xs = np.empty(0)
            bot_ys = np.empty(0)

        # 'c' spine: closed boundary path (right edge → top-flat → left edge
        # reversed → bottom-flat → close)
        x_path = np.concatenate(
            [x_right, top_xs, x_left[::-1], bot_xs, [x_right[0]]])
        y_path = np.concatenate(
            [y_vals, top_ys, y_vals[::-1], bot_ys, [y_vals[0]]])
        self['c'].data = np.array([x_path, y_path]).T

        # 'h' spine: horizontal line across equator (for RA tick placement)
        self['h'].data = np.array([
            np.linspace(xmin, xmax, 1000), np.repeat(ycen, 1000)
        ]).T

        # 'v' spine: vertical line at central meridian (for Dec tick placement)
        self['v'].data = np.array([
            np.repeat(xcen, 1000), np.linspace(ymin, ymax, 1000)
        ]).T

        super().update_spines()

    def _update_patch_path(self) -> None:
        """Use only the boundary spine 'c' for the clipping path."""
        self.update_spines()
        vertices = self['c'].data
        # ``_path`` is an inherited BaseFrame attribute with no type info
        # in the astropy stubs, so mypy can't resolve its type here.
        if self._path is None:  # type: ignore[has-type]
            self._path = Path(vertices)
        else:
            self._path.vertices = vertices

    def draw(self, renderer: Any) -> None:
        """Draw only the boundary spine 'c', not the h/v reference lines."""
        from matplotlib.lines import Line2D
        pixel = self['c']._get_pixel()
        line = Line2D(pixel[:, 0], pixel[:, 1],
                      linewidth=self._linewidth, color=self._color, zorder=1000)
        line.draw(renderer)


class CircularFrame(_AllSkyCustomFrame):
    """
    A circular frame for globe/hemisphere projections (SIN, ARC, ZEA, STG).
    """
    def _boundary_x(self, t: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        return np.sqrt(np.maximum(1 - t**2, 0))


class SinusoidalFrame(_AllSkyCustomFrame):
    """
    A sinusoidal frame for the Sanson-Flamsteed (SFL) projection.
    Boundary: x = cos(π/2 · t)
    """
    def _boundary_x(self, t: npt.ArrayLike) -> np.ndarray:
        return np.cos(np.pi / 2 * np.asarray(t))


class ParabolicFrame(_AllSkyCustomFrame):
    """
    A parabolic frame for the PAR projection.
    Boundary: x = 1 − t²
    """
    def _boundary_x(self, t: npt.ArrayLike) -> np.ndarray:
        return 1 - np.asarray(t)**2


# ===== CurvedTransform wrappers (only if astropy provides CurvedTransform) =====

def _wrap_centered_lon(lon: np.ndarray) -> np.ndarray:
    """Fold center-relative longitudes into [-180, 180] for the forward math.

    A plain modulo (``((lon + 180) % 360) - 180``) collapses +180 onto -180,
    which is wrong for an all-sky mesh: lon=-180 and lon=+180 are the same
    meridian but must land on *opposite* edges of the map. Collapsing them
    makes every seam-edge quad span the full panel width, so the seam column
    smears across the whole map for longitude-varying data (latitude-only
    data hides it because the smeared quads share their row's color).

    Keep the closed range [-180, 180] as identity so both seam edges stay
    on their own side, and only fold points genuinely outside it (data given
    in [0, 360], or far-off-center longitudes).
    """
    out_of_range = (lon < -180.0) | (lon > 180.0)
    return np.where(out_of_range, (lon + 180.0) % 360.0 - 180.0, lon)


if _HAS_CURVEDTRANSFORM:

    class RobinsonTransform(CurvedTransform):
        """Forward Robinson projection transform (lon, lat → x, y)."""

        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon

        def transform(self, lonlat: npt.ArrayLike) -> np.ndarray:
            lonlat = np.atleast_2d(lonlat)
            lon = _wrap_centered_lon(lonlat[:, 0] - self.center_lon)
            lat = lonlat[:, 1]
            x, y = _robinson_forward(lon, lat)
            return np.column_stack([x, y])

        transform_non_affine = transform

        def inverted(self) -> InverseRobinsonTransform:
            return InverseRobinsonTransform(self.center_lon)


    class InverseRobinsonTransform(CurvedTransform):
        """Inverse Robinson projection transform (x, y → lon, lat)."""

        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon

        def transform(self, xy: npt.ArrayLike) -> np.ndarray:
            xy = np.atleast_2d(xy)
            x = xy[:, 0]
            y = xy[:, 1]
            lon, lat = _robinson_inverse(x, y)
            lon = lon + self.center_lon
            return np.column_stack([lon, lat])

        transform_non_affine = transform

        def inverted(self) -> RobinsonTransform:
            return RobinsonTransform(self.center_lon)


    class KavrayskiyTransform(CurvedTransform):
        """Forward Kavrayskiy VII transform (lon, lat → x, y)."""
        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon

        def transform(self, lonlat: npt.ArrayLike) -> np.ndarray:
            lonlat = np.atleast_2d(lonlat)
            lon = _wrap_centered_lon(lonlat[:, 0] - self.center_lon)
            x, y = _kavrayskiy_forward(lon, lonlat[:, 1])
            return np.column_stack([x, y])

        transform_non_affine = transform

        def inverted(self) -> InverseKavrayskiyTransform:
            return InverseKavrayskiyTransform(self.center_lon)


    class InverseKavrayskiyTransform(CurvedTransform):
        """Inverse Kavrayskiy VII transform (x, y → lon, lat)."""
        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon

        def transform(self, xy: npt.ArrayLike) -> np.ndarray:
            xy = np.atleast_2d(xy)
            lon, lat = _kavrayskiy_inverse(xy[:, 0], xy[:, 1])
            lon = lon + self.center_lon
            return np.column_stack([lon, lat])

        transform_non_affine = transform

        def inverted(self) -> KavrayskiyTransform:
            return KavrayskiyTransform(self.center_lon)


    class Eckert4Transform(CurvedTransform):
        """Forward Eckert IV transform (lon, lat → x, y)."""
        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon

        def transform(self, lonlat: npt.ArrayLike) -> np.ndarray:
            lonlat = np.atleast_2d(lonlat)
            lon = _wrap_centered_lon(lonlat[:, 0] - self.center_lon)
            x, y = _eckert4_forward(lon, lonlat[:, 1])
            return np.column_stack([x, y])

        transform_non_affine = transform

        def inverted(self) -> InverseEckert4Transform:
            return InverseEckert4Transform(self.center_lon)


    class InverseEckert4Transform(CurvedTransform):
        """Inverse Eckert IV transform (x, y → lon, lat)."""
        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon

        def transform(self, xy: npt.ArrayLike) -> np.ndarray:
            xy = np.atleast_2d(xy)
            lon, lat = _eckert4_inverse(xy[:, 0], xy[:, 1])
            lon = lon + self.center_lon
            return np.column_stack([lon, lat])

        transform_non_affine = transform

        def inverted(self) -> Eckert4Transform:
            return Eckert4Transform(self.center_lon)


    class WinkelTripelTransform(CurvedTransform):
        """Forward Winkel Tripel transform (lon, lat → x, y)."""
        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon

        def transform(self, lonlat: npt.ArrayLike) -> np.ndarray:
            lonlat = np.atleast_2d(lonlat)
            lon = _wrap_centered_lon(lonlat[:, 0] - self.center_lon)
            x, y = _winkel_forward(lon, lonlat[:, 1])
            return np.column_stack([x, y])

        transform_non_affine = transform

        def inverted(self) -> InverseWinkelTripelTransform:
            return InverseWinkelTripelTransform(self.center_lon)


    class InverseWinkelTripelTransform(CurvedTransform):
        """Inverse Winkel Tripel transform (x, y → lon, lat)."""
        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon

        def transform(self, xy: npt.ArrayLike) -> np.ndarray:
            xy = np.atleast_2d(xy)
            lon, lat = _winkel_inverse(xy[:, 0], xy[:, 1])
            lon = lon + self.center_lon
            return np.column_stack([lon, lat])

        transform_non_affine = transform

        def inverted(self) -> WinkelTripelTransform:
            return WinkelTripelTransform(self.center_lon)


    class McBrydeTransform(CurvedTransform):
        """Forward McBryde-Thomas flat-polar quartic transform (lon, lat → x, y)."""
        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon

        def transform(self, lonlat: npt.ArrayLike) -> np.ndarray:
            lonlat = np.atleast_2d(lonlat)
            lon = _wrap_centered_lon(lonlat[:, 0] - self.center_lon)
            x, y = _mcbryde_forward(lon, lonlat[:, 1])
            return np.column_stack([x, y])

        transform_non_affine = transform

        def inverted(self) -> InverseMcBrydeTransform:
            return InverseMcBrydeTransform(self.center_lon)


    class InverseMcBrydeTransform(CurvedTransform):
        """Inverse McBryde-Thomas transform (x, y → lon, lat)."""
        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon

        def transform(self, xy: npt.ArrayLike) -> np.ndarray:
            xy = np.atleast_2d(xy)
            lon, lat = _mcbryde_inverse(xy[:, 0], xy[:, 1])
            lon = lon + self.center_lon
            return np.column_stack([lon, lat])

        transform_non_affine = transform

        def inverted(self) -> McBrydeTransform:
            return McBrydeTransform(self.center_lon)


    class ObliqueAspectTransform(CurvedTransform):
        """Spherical rotation bringing ``(center_lon, center_lat)`` to (0, 0).

        The custom pseudocylindrical CurvedTransforms only center in
        longitude. To give them an oblique aspect (a non-zero ``center_lat``,
        like the FITS path's CRVAL2), compose this rotation BEFORE the
        projection's forward transform and build the projection with
        ``center_lon=0`` — this rotates the sphere so the requested center
        lands at the projection origin, then the equatorial projection runs
        on the rotated graticule. With ``center_lat=0`` it reduces to a pure
        longitude shift, so it is only inserted when an oblique aspect is
        actually requested.
        """

        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0., center_lat: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon
            self.center_lat = center_lat

        def transform(self, lonlat: npt.ArrayLike) -> np.ndarray:
            lonlat = np.atleast_2d(lonlat)
            lam = np.radians(lonlat[:, 0] - self.center_lon)
            phi = np.radians(lonlat[:, 1])
            phi0 = np.radians(self.center_lat)
            cphi, sphi = np.cos(phi), np.sin(phi)
            clam, slam = np.cos(lam), np.sin(lam)
            cphi0, sphi0 = np.cos(phi0), np.sin(phi0)
            # Rotate the (already lon-centered) direction about the y-axis by
            # the center latitude so the center sits on the equator at lon 0.
            xr = cphi * clam * cphi0 + sphi * sphi0
            yr = cphi * slam
            zr = sphi * cphi0 - cphi * clam * sphi0
            lon_out = np.degrees(np.arctan2(yr, xr))
            lat_out = np.degrees(np.arcsin(np.clip(zr, -1.0, 1.0)))
            return np.column_stack([lon_out, lat_out])

        transform_non_affine = transform

        def inverted(self) -> InverseObliqueAspectTransform:
            return InverseObliqueAspectTransform(self.center_lon, self.center_lat)


    class InverseObliqueAspectTransform(CurvedTransform):
        """Inverse of ``ObliqueAspectTransform`` (rotated lon/lat → true lon/lat)."""

        input_dims = 2
        output_dims = 2
        has_inverse = True

        def __init__(self, center_lon: float = 0., center_lat: float = 0.) -> None:
            super().__init__()
            self.center_lon = center_lon
            self.center_lat = center_lat

        def transform(self, lonlat: npt.ArrayLike) -> np.ndarray:
            lonlat = np.atleast_2d(lonlat)
            lam = np.radians(lonlat[:, 0])
            phi = np.radians(lonlat[:, 1])
            phi0 = np.radians(self.center_lat)
            cphi, sphi = np.cos(phi), np.sin(phi)
            clam, slam = np.cos(lam), np.sin(lam)
            cphi0, sphi0 = np.cos(phi0), np.sin(phi0)
            # Undo the y-axis rotation, then add back the center longitude.
            x = cphi * clam * cphi0 - sphi * sphi0
            y = cphi * slam
            z = cphi * clam * sphi0 + sphi * cphi0
            lon_out = np.degrees(np.arctan2(y, x)) + self.center_lon
            lat_out = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
            return np.column_stack([lon_out, lat_out])

        transform_non_affine = transform

        def inverted(self) -> ObliqueAspectTransform:
            return ObliqueAspectTransform(self.center_lon, self.center_lat)


# ===== Projection-specific frames (depend on _math precomputed tables) =====

class RobinsonFrame(_AllSkyCustomFrame):
    """
    Frame boundary for the Robinson projection.

    Robinson's 1974 lookup table is non-linear in lat for both x and y,
    so the boundary curve at lon=±180 must be parameterized by lat with
    explicit ``y(lat)`` rather than the default ``y = t``.
    """
    def _boundary_x(self, t: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        return np.interp(np.abs(t) * 90, _ROB_LATS, _ROB_X)

    def _boundary_y(self, t: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        # _ROB_Y already normalized so _ROB_Y[-1] = 1.0
        return np.interp(np.abs(t) * 90, _ROB_LATS, _ROB_Y) * np.sign(t)


class KavrayskiyFrame(_AllSkyCustomFrame):
    """
    Frame boundary for the Kavrayskiy VII projection.
    Boundary: x(t) = √(1 − 3t²/4).
    Kavrayskiy's y = phi is linear in lat, so default ``_boundary_y``
    (identity) is correct.
    """
    def _boundary_x(self, t: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        return np.sqrt(np.maximum(1. - 0.75 * t**2, 0))


# Precompute Eckert IV boundary table from forward equations
_ECKERT4_BND_LATS = np.linspace(0, 90, 91)
_ECKERT4_BND_X = np.zeros(91)
_ECKERT4_BND_Y = np.zeros(91)
for _i, _lat in enumerate(_ECKERT4_BND_LATS):
    _xb, _yb = _eckert4_forward(180., _lat)
    _ECKERT4_BND_X[_i] = abs(_xb)
    _ECKERT4_BND_Y[_i] = abs(_yb)
_ECKERT4_BND_X /= _ECKERT4_BND_X[0]  # normalize so equator = 1
_ECKERT4_BND_Y /= _ECKERT4_BND_Y[-1]  # normalize so pole = 1


class Eckert4Frame(_AllSkyCustomFrame):
    """
    Frame boundary for the Eckert IV projection.
    Boundary x(t) = (1 + √(1−t²))/2 with t = sin(θ), where θ comes from
    the Eckert IV iteration. The y at lon=±180 is non-linear in lat
    (y ∝ sin θ), so we compute it numerically from the forward equations.
    Has flat poles (nonzero width at ±90°).
    """
    def _boundary_x(self, t: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        return np.interp(np.abs(t) * 90, _ECKERT4_BND_LATS, _ECKERT4_BND_X)

    def _boundary_y(self, t: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        return np.interp(np.abs(t) * 90, _ECKERT4_BND_LATS, _ECKERT4_BND_Y) * np.sign(t)


# Precompute Winkel Tripel boundary table (numerically from forward projection)
_WINKEL_BND_LATS = np.linspace(0, 90, 91)
_WINKEL_BND_X = np.zeros(91)
_WINKEL_BND_Y = np.zeros(91)
for _i, _lat in enumerate(_WINKEL_BND_LATS):
    _xb, _yb = _winkel_forward(180., _lat)
    _WINKEL_BND_X[_i] = abs(_xb)
    _WINKEL_BND_Y[_i] = abs(_yb)
_WINKEL_BND_X /= _WINKEL_BND_X[0]  # normalize so equator = 1
# Winkel y at lon=180, lat=90 is π/2, same as y at lon=0, lat=90 (the pole
# is a single point), so normalizing to the table's max gives a y_norm that
# matches the axis y_ext set by make_wcs_frame from fwd(0, 90).
_WINKEL_BND_Y /= _WINKEL_BND_Y[-1]


class WinkelTripelFrame(_AllSkyCustomFrame):
    """
    Frame boundary for the Winkel Tripel projection.
    Both x and y at the antimeridian are non-linear in lat; computed
    numerically from the projection equations.
    """
    def _boundary_x(self, t: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        return np.interp(np.abs(t) * 90, _WINKEL_BND_LATS, _WINKEL_BND_X)

    def _boundary_y(self, t: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        return np.interp(np.abs(t) * 90, _WINKEL_BND_LATS, _WINKEL_BND_Y) * np.sign(t)


# Precompute McBryde-Thomas boundary table
_MCBRYDE_BND_LATS = np.linspace(0, 90, 91)
_MCBRYDE_BND_X = np.zeros(91)
_MCBRYDE_BND_Y = np.zeros(91)
for _i, _lat in enumerate(_MCBRYDE_BND_LATS):
    _xb, _yb = _mcbryde_forward(180., _lat)
    _MCBRYDE_BND_X[_i] = abs(_xb)
    _MCBRYDE_BND_Y[_i] = abs(_yb)
_MCBRYDE_BND_X /= _MCBRYDE_BND_X[0]  # normalize
_MCBRYDE_BND_Y /= _MCBRYDE_BND_Y[-1]


class McBrydeFrame(_AllSkyCustomFrame):
    """
    Frame boundary for the McBryde-Thomas flat-polar quartic projection.
    Both x and y at the antimeridian are non-linear in lat; computed
    numerically. Has flat poles (nonzero width at ±90°).
    """
    def _boundary_x(self, t: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        return np.interp(np.abs(t) * 90, _MCBRYDE_BND_LATS, _MCBRYDE_BND_X)

    def _boundary_y(self, t: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(t)
        return np.interp(np.abs(t) * 90, _MCBRYDE_BND_LATS, _MCBRYDE_BND_Y) * np.sign(t)
