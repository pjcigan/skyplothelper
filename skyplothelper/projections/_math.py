"""Forward/inverse projection math for non-FITS projections.

Pure numpy. Snyder (1989), "An Album of Map Projections", is authoritative
for the projection equations; do not change the math.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---- Robinson Projection ----
# Reference: Robinson (1974), "A New Map Projection: Its Development and
# Characteristics", in International Yearbook of Cartography.

# Lookup table: (latitude_deg, X_factor, Y_factor) at 5° intervals
_ROBINSON_TABLE = np.array([
    [ 0, 1.0000, 0.0000],
    [ 5, 0.9986, 0.0620],
    [10, 0.9954, 0.1240],
    [15, 0.9900, 0.1860],
    [20, 0.9822, 0.2480],
    [25, 0.9730, 0.3100],
    [30, 0.9600, 0.3720],
    [35, 0.9427, 0.4340],
    [40, 0.9216, 0.4958],
    [45, 0.8962, 0.5571],
    [50, 0.8679, 0.6176],
    [55, 0.8350, 0.6769],
    [60, 0.7986, 0.7346],
    [65, 0.7597, 0.7903],
    [70, 0.7186, 0.8435],
    [75, 0.6732, 0.8936],
    [80, 0.6213, 0.9394],
    [85, 0.5722, 0.9761],
    [90, 0.5322, 1.0000],
])

_ROB_LATS = _ROBINSON_TABLE[:, 0]
_ROB_X = _ROBINSON_TABLE[:, 1]
_ROB_Y = _ROBINSON_TABLE[:, 2]


def _robinson_forward(
        lon: npt.ArrayLike,
        lat: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Robinson projection: (lon, lat) in degrees → (x, y) in projected units."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    abs_lat = np.abs(lat)
    sign_lat = np.sign(lat)

    # Interpolate X and Y factors from the table
    x_factor = np.interp(abs_lat, _ROB_LATS, _ROB_X)
    y_factor = np.interp(abs_lat, _ROB_LATS, _ROB_Y)

    # Robinson projection equations
    # x = 0.8487 * X(lat) * lon  (lon in radians, but we keep degrees and scale)
    # y = 1.3523 * Y(lat) * sign(lat)
    # We normalize so that the map spans [-1, 1] in both axes
    x = 0.8487 * x_factor * np.radians(lon)
    y = 1.3523 * y_factor * sign_lat

    return x, y


def _robinson_inverse(
        x: npt.ArrayLike,
        y: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Robinson inverse: (x, y) → (lon, lat) in degrees.

    For ``|y| > y_pole`` (above/below the flat polar parallel), both lat
    and the x-factor are linearly extrapolated rather than hard-clipped.
    Without the extrapolation, ``∂lon/∂y`` collapses to zero at the
    boundary and astropy's tick-angle Jacobian computation produces
    degenerate (always-vertical) ticks on the top edge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Invert Y to get latitude (with linear extrapolation above ±y_pole)
    sign_y = np.sign(y)
    abs_y_norm = np.abs(y) / 1.3523
    # In-range: table lookup
    abs_lat_in = np.interp(np.clip(abs_y_norm, 0, 1), _ROB_Y, _ROB_LATS)
    # Above range: linear extrap from last two table points
    slope_lat = (_ROB_LATS[-1] - _ROB_LATS[-2]) / (_ROB_Y[-1] - _ROB_Y[-2])
    abs_lat_extrap = 90.0 + (abs_y_norm - 1.0) * slope_lat
    abs_lat = np.where(abs_y_norm <= 1.0, abs_lat_in, abs_lat_extrap)
    lat = abs_lat * sign_y

    # Invert X to get longitude (with x_factor extrapolation for lat > 90)
    x_factor_in = np.interp(np.clip(abs_lat, 0, 90), _ROB_LATS, _ROB_X)
    slope_x = (_ROB_X[-1] - _ROB_X[-2]) / (_ROB_LATS[-1] - _ROB_LATS[-2])
    x_factor_extrap = _ROB_X[-1] + (abs_lat - 90.0) * slope_x
    x_factor = np.where(abs_lat <= 90.0, x_factor_in, x_factor_extrap)
    x_factor = np.where(x_factor > 1e-10, x_factor, 1e-10)
    lon = np.degrees(x / (0.8487 * x_factor))

    return lon, lat


# ---- Kavrayskiy VII projection ----

def _kavrayskiy_forward(
        lon: npt.ArrayLike,
        lat: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Kavrayskiy VII: (lon, lat) degrees → (x, y) projected units."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lam = np.radians(lon)
    phi = np.radians(lat)
    # x = (3λ)/(2π) * √(π²/3 - φ²),  y = φ
    x = 3. / (2. * np.pi) * lam * np.sqrt(np.maximum(np.pi**2 / 3. - phi**2, 0))
    y = phi
    return x, y


def _kavrayskiy_inverse(
        x: npt.ArrayLike,
        y: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Kavrayskiy VII inverse: (x, y) → (lon, lat) degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    phi = y
    lat = np.degrees(phi)
    denom = np.sqrt(np.maximum(np.pi**2 / 3. - phi**2, 1e-20))
    lam = x * 2. * np.pi / (3. * denom)
    lon = np.degrees(lam)
    return lon, lat


# ---- Eckert IV projection ----

def _eckert4_forward(
        lon: npt.ArrayLike,
        lat: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Eckert IV equal-area: (lon, lat) degrees → (x, y) projected units."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lam = np.radians(lon)
    phi = np.radians(lat)

    C = 2. + np.pi / 2.  # = 2 + π/2
    # Newton iteration: solve θ + sin(θ)cos(θ) + 2sin(θ) = C * sin(φ)
    # f(θ) = θ + sin(θ)cos(θ) + 2sin(θ) - C sin(φ)
    # f'(θ) = 1 + cos(2θ) + 2cos(θ) = 2cos²(θ) + 2cos(θ)
    theta = phi.copy()  # initial guess
    target = C * np.sin(phi)
    for _ in range(15):
        st, ct = np.sin(theta), np.cos(theta)
        f = theta + st * ct + 2. * st - target
        fp = 1. + np.cos(2. * theta) + 2. * ct
        fp = np.where(np.abs(fp) > 1e-12, fp, 1e-12)
        theta = theta - f / fp

    Cx = 2. / np.sqrt(np.pi * (4. + np.pi))
    Cy = 2. * np.sqrt(np.pi / (4. + np.pi))
    x = Cx * lam * (1. + np.cos(theta))
    y = Cy * np.sin(theta)
    return x, y


def _eckert4_inverse(
        x: npt.ArrayLike,
        y: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Eckert IV inverse: (x, y) → (lon, lat) degrees.

    For ``|y| > Cy`` (above/below the flat polar parallel), ``theta`` is
    extended past ±π/2 by *mirror reflection* of arcsin: |y_norm|>1 is
    reflected back into [-1, 1] and theta is reflected around ±π/2.
    A naive linear extrapolation has finite slope, but ``arcsin`` has an
    infinite (sqrt-singularity) slope at y_norm=±1; mirror reflection
    preserves the sqrt shape symmetrically across the boundary, so the
    finite-difference Jacobian astropy uses for tick rotation matches on
    both sides. Without this, top-edge ticks rotate at the wrong angles.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    Cx = 2. / np.sqrt(np.pi * (4. + np.pi))
    Cy = 2. * np.sqrt(np.pi / (4. + np.pi))

    y_norm = y / Cy
    abs_y = np.abs(y_norm)
    sign_y = np.sign(y_norm)
    # Reflect |y_norm|>1 into [-1, 1] for arcsin, then reflect theta
    # around ±π/2 so the sqrt-shape slope continues smoothly past the
    # boundary (rather than dropping to a finite linear slope).
    y_for_arcsin = np.where(abs_y <= 1.0, y_norm, 2.0 * sign_y - y_norm)
    theta_lookup = np.arcsin(np.clip(y_for_arcsin, -1.0, 1.0))
    theta = np.where(
        abs_y <= 1.0, theta_lookup, sign_y * np.pi - theta_lookup)

    C = 2. + np.pi / 2.
    phi = np.arcsin(np.clip((theta + np.sin(theta) * np.cos(theta) +
                              2. * np.sin(theta)) / C, -1, 1))

    denom = Cx * (1. + np.cos(theta))
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    lam = x / denom

    lon = np.degrees(lam)
    lat = np.degrees(phi)
    return lon, lat


# ---- Winkel Tripel projection ----

_WINKEL_COSPHI1 = 2. / np.pi  # cos(arccos(2/π)) = 2/π, standard parallel


def _winkel_forward(
        lon: npt.ArrayLike,
        lat: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Winkel Tripel: (lon, lat) degrees → (x, y) projected units."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lam = np.radians(lon)
    phi = np.radians(lat)

    cosphi = np.cos(phi)
    alpha = np.arccos(np.clip(cosphi * np.cos(lam / 2.), -1, 1))
    # sinc(α) = sin(α)/α, with sinc(0)=1. Use np.sinc (= sin(πx)/(πx)) so the
    # α=0 case (lon=lat=0) is handled without a 0/0 "invalid value" warning —
    # np.where would still eagerly evaluate sin(α)/α at α=0 and warn.
    sinc_alpha = np.sinc(alpha / np.pi)

    x = 0.5 * (lam * _WINKEL_COSPHI1 + 2. * cosphi * np.sin(lam / 2.) / sinc_alpha)
    y = 0.5 * (phi + np.sin(phi) / sinc_alpha)
    return x, y


def _winkel_inverse(
        x: npt.ArrayLike,
        y: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Winkel Tripel inverse (Newton 2D iteration): (x, y) → (lon, lat) degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Initial guess from equirectangular
    lam = x / _WINKEL_COSPHI1
    phi = y.copy()

    for _ in range(25):
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        half_lam = lam / 2.
        cos_half = np.cos(half_lam)
        sin_half = np.sin(half_lam)

        alpha = np.arccos(np.clip(cosphi * cos_half, -1, 1))
        # sin(α)/α via np.sinc — warning-free at α=0 (see _winkel_forward).
        sinc_a = np.sinc(alpha / np.pi)

        # f1 = projected_x - x,  f2 = projected_y - y
        f1 = 0.5 * (lam * _WINKEL_COSPHI1 + 2. * cosphi * sin_half / sinc_a) - x
        f2 = 0.5 * (phi + sinphi / sinc_a) - y

        # Jacobian (∂f/∂λ, ∂f/∂φ) — use numerical differences for robustness
        dl = 1e-7
        dp = 1e-7

        x1p, y1p = _winkel_forward(np.degrees(lam + dl), np.degrees(phi))
        x1m, y1m = _winkel_forward(np.degrees(lam - dl), np.degrees(phi))
        x2p, y2p = _winkel_forward(np.degrees(lam), np.degrees(phi + dp))
        x2m, y2m = _winkel_forward(np.degrees(lam), np.degrees(phi - dp))

        df1_dl = (x1p - x1m) / (2. * dl)
        df2_dl = (y1p - y1m) / (2. * dl)
        df1_dp = (x2p - x2m) / (2. * dp)
        df2_dp = (y2p - y2m) / (2. * dp)

        det = df1_dl * df2_dp - df1_dp * df2_dl
        det = np.where(np.abs(det) > 1e-20, det, 1e-20)

        dlam = (f1 * df2_dp - f2 * df1_dp) / det
        dphi = (f2 * df1_dl - f1 * df2_dl) / det

        lam = lam - dlam
        phi = phi - dphi

        if np.all(np.abs(dlam) < 1e-10) and np.all(np.abs(dphi) < 1e-10):
            break

    lon = np.degrees(lam)
    lat = np.degrees(phi)
    return lon, lat


# ---- McBryde-Thomas flat-polar quartic projection ----
# Snyder (1989), "An Album of Map Projections", p. 60

_MCBRYDE_C = 1. + np.sqrt(2.) / 2.  # ≈ 1.7071 (iteration target constant)
_MCBRYDE_6C2 = 6. * (2. + np.sqrt(2.))  # 6(2+√2) ≈ 20.485


def _mcbryde_forward(
        lon: npt.ArrayLike,
        lat: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """McBryde-Thomas flat-polar quartic: (lon, lat) degrees → (x, y)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lam = np.radians(lon)
    phi = np.radians(lat)

    # Newton: solve sin(θ/2) + sin(θ) = (1+√2/2)sin(φ)
    # f(θ) = sin(θ/2) + sin(θ) - C sin(φ)
    # f'(θ) = cos(θ/2)/2 + cos(θ)
    theta = phi.copy()
    target = _MCBRYDE_C * np.sin(phi)
    for _ in range(25):
        half_t = theta / 2.
        f = np.sin(half_t) + np.sin(theta) - target
        fp = np.cos(half_t) / 2. + np.cos(theta)
        fp = np.where(np.abs(fp) > 1e-12, fp, 1e-12)
        delta = f / fp
        theta = theta - delta
        if np.all(np.abs(delta) < 1e-12):
            break

    cos_t = np.cos(theta)
    cos_ht = np.cos(theta / 2.)
    cos_ht = np.where(np.abs(cos_ht) > 1e-12, cos_ht, 1e-12)

    x = lam * (1. + 2. * cos_t / cos_ht) / np.sqrt(_MCBRYDE_6C2)
    y = 2. * np.sqrt(6. / (2. + np.sqrt(2.))) * np.sin(theta / 2.) / 3.
    return x, y


def _mcbryde_inverse(
        x: npt.ArrayLike,
        y: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """McBryde-Thomas inverse: (x, y) → (lon, lat) degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # From y: sin(θ/2) = 3y / (2√(6/(2+√2)))
    scale_y = 2. * np.sqrt(6. / (2. + np.sqrt(2.))) / 3.
    sin_ht = np.clip(y / scale_y, -1, 1)
    half_theta = np.arcsin(sin_ht)
    theta = 2. * half_theta

    cos_t = np.cos(theta)
    cos_ht = np.cos(half_theta)
    cos_ht = np.where(np.abs(cos_ht) > 1e-12, cos_ht, 1e-12)

    denom = (1. + 2. * cos_t / cos_ht) / np.sqrt(_MCBRYDE_6C2)
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    lam = x / denom

    phi = np.arcsin(np.clip((np.sin(half_theta) + np.sin(theta)) / _MCBRYDE_C, -1, 1))

    lon = np.degrees(lam)
    lat = np.degrees(phi)
    return lon, lat

