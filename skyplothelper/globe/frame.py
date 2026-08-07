"""Tilted globe coordinate frame and WCSAxes builders.

``TiltedEarthFrame`` is an astropy frame implementing z-y'-z" proper
Euler rotations on top of ITRS; ``make_globe_frame`` constructs an
orthographic WCSAxes for any rotation state, and ``euler_to_fits_ortho``
maps Euler angles to fits-WCS (CRVAL, LONPOLE) parameters.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import astropy.io.fits as pyfits  # noqa: F401  (used by make_globe_frame)
import astropy.units as u
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
import numpy.typing as npt
from astropy.coordinates import (
    ITRS,
    Angle,
    DynamicMatrixTransform,
    Latitude,
    Longitude,
    QuantityAttribute,
    SphericalRepresentation,
    frame_transform_graph,
)
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix
from astropy.visualization.wcsaxes.frame import EllipticalFrame  # noqa: F401
from astropy.wcs import WCS


class TiltedEarthFrame(ITRS):
    """
    An ITRS-based coordinate frame with three Euler-angle rotation attributes,
    implementing a z-y'-z" proper Euler rotation (Goldstein convention).

    This frame can be used as a coordinate overlay on WCSAxes globe plots
    to display a tilted grid, or for transforming coordinates between the
    standard ITRS frame and a tilted/rotated view.

    Parameters
    ----------
    rotation : Quantity [angle], optional
        Rotation about the z-axis (longitude spin). Default 0 deg.
    obliquity : Quantity [angle], optional
        Rotation about the y'-axis (axial tilt). Default 23.44 deg (Earth).
    perspective : Quantity [angle], optional
        Rotation about the z''-axis (viewing perspective / precession angle).
        Default 45 deg.

    Examples
    --------
    Overlay a tilted grid on a WCS globe plot::

        te = TiltedEarthFrame(rotation=0*u.deg, obliquity=23.44*u.deg,
                              perspective=45*u.deg)
        ax.get_coords_overlay(te).grid(color='r')

    Notes
    -----
    The rotation matrix is constructed as::

        R = I @ Rz(phi) @ Ry(theta) @ Rz(psi)

    where phi=rotation, theta=obliquity, psi=perspective, and I is the 3x3
    identity (earlier broken versions had np.diag([1,1,-1]) here, which caused
    systematic offsets).

    Must use DynamicMatrixTransform (not StaticMatrixTransform) because the
    rotation angles are frame attributes that can change per instance.
    """
    name = 'tilted_earth'

    rotation = QuantityAttribute(default=0. * u.deg, unit='deg')
    obliquity = QuantityAttribute(default=23.44 * u.deg, unit='deg')
    perspective = QuantityAttribute(default=45. * u.deg, unit='deg')

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs['rotation'] = Longitude(u.Quantity(
            kwargs.get('rotation', 0), unit=u.deg))
        kwargs['obliquity'] = Latitude(u.Quantity(
            kwargs.get('obliquity', 0), unit=u.deg))
        kwargs['perspective'] = Longitude(u.Quantity(
            kwargs.get('perspective', 0), unit=u.deg))
        super().__init__(*args, **kwargs)

    default_representation = SphericalRepresentation


def _get_euler_rotation_matrix(
        tilted_frame: TiltedEarthFrame, inverse: bool = False) -> np.ndarray:
    """
    Construct the 3x3 rotation matrix for a TiltedEarthFrame instance.

    Uses proper Euler angles in z-y'-z" convention (Goldstein 1980):
        phi (rotation) -> theta (obliquity) -> psi (perspective)

    Parameters
    ----------
    tilted_frame : TiltedEarthFrame
        Frame instance whose rotation/obliquity/perspective attributes
        define the Euler angles.
    inverse : bool, optional
        If True, return the transpose (inverse) of the rotation matrix.

    Returns
    -------
    rot_matrix : ndarray, shape (3, 3)
    """
    phi = Angle(tilted_frame.rotation.to('deg')).wrap_at(360 * u.deg).value
    theta = Angle(tilted_frame.obliquity.to('deg')).wrap_at(90 * u.deg).value
    psi = Angle(tilted_frame.perspective.to('deg')).wrap_at(360 * u.deg).value

    rot_matrix = (
        np.diag([1., 1., 1.])
        @ rotation_matrix(phi, "z")
        @ rotation_matrix(theta, "y")
        @ rotation_matrix(psi, "z")
    )

    if inverse:
        return matrix_transpose(rot_matrix)
    return rot_matrix


# Register the frame transforms on the astropy transform graph.
# DynamicMatrixTransform is required because the Euler angles are per-instance.

@frame_transform_graph.transform(DynamicMatrixTransform, ITRS, TiltedEarthFrame)
def _itrs_to_tilted(
        itrs_frame: ITRS, tilted_frame: TiltedEarthFrame) -> np.ndarray:
    """ITRS -> TiltedEarthFrame rotation matrix."""
    return _get_euler_rotation_matrix(tilted_frame)


@frame_transform_graph.transform(DynamicMatrixTransform, TiltedEarthFrame, ITRS)
def _tilted_to_itrs(
        tilted_frame: TiltedEarthFrame, itrs_frame: ITRS) -> np.ndarray:
    """TiltedEarthFrame -> ITRS rotation matrix (transpose)."""
    return _get_euler_rotation_matrix(tilted_frame, inverse=True)


# =============================================================================
# Euler Angle Utilities
# =============================================================================

def euler_to_fits_ortho(
        rotation: Any, obliquity: Any, perspective: Any,
        units: str = 'deg') -> tuple[Any, Any, Any]:
    """
    Convert proper Euler angles (z-x'-z") to fits SIN orthographic projection
    parameters.

    The input [rotation, obliquity, perspective] angles are the intuitive
    physical rotation state of a planetary body. The output [center_lon,
    center_lat, lonpole] values can be passed directly to make_globe_frame().

    Rotation conventions:

    * ``rotation`` — Longitudinal spin. Positive = CCW (like Earth's day cycle).
    * ``obliquity`` — Axial tilt. Positive = north pole toward viewer
      (Earth = 23.44 deg).
    * ``perspective`` — Viewing angle from ecliptic orbit. Increasing values
      give CW precession (like Earth's ~25,700 yr cycle).

    .. note::
       **Gimbal lock at zero tilt.** ``rotation`` and ``perspective`` are both
       rotations about the (untilted) polar axis, so at ``obliquity = 0`` they
       collapse to a single degree of freedom — the result depends only on
       ``rotation + perspective`` and ``perspective`` produces no *separate*
       visible effect (``euler_to_fits_ortho(30, 0, 40)`` ==
       ``euler_to_fits_ortho(70, 0, 0)``). This is fundamental, not a numerical
       artifact: ``perspective`` (precession) is only meaningful once the pole
       is tilted, and its visible effect scales with the tilt. To demonstrate
       perspective, set a nonzero ``obliquity`` first (e.g. 23.44°). When you
       already have an orientation as a quaternion, use
       :func:`quaternion_to_fits_ortho` instead, which sidesteps Euler-angle
       gimbal lock entirely.

    Parameters
    ----------
    rotation : float, array-like, or Angle
        Euler z-axis rotation angle(s).
    obliquity : float, array-like, or Angle
        Euler x'-axis rotation angle(s).
    perspective : float, array-like, or Angle
        Euler z''-axis rotation angle(s).
    units : str, optional
        Angular units for input/output: 'deg' or 'rad'. Default 'deg'.

    Returns
    -------
    center_lons : ndarray
        Center longitudes for plot frame (use with center_LONdeg).
    center_lats : ndarray
        Center latitudes for plot frame (use with center_LATdeg).
    lonpoles : ndarray
        Native longitude of celestial pole (use with lonpole).

    Examples
    --------
    Simple Earth rotation, fixed tilt::

        lons, lats, poles = euler_to_fits_ortho(
            np.linspace(0, 360, 24, endpoint=False),
            np.full(24, 24.),
            np.full(24, 30.))

    Precession cycle (keep same face toward viewer)::

        angles = np.linspace(0, 360, 24, endpoint=False)
        lons, lats, poles = euler_to_fits_ortho(-angles, 24, angles)
    """
    # Parse inputs to Angle objects
    if not isinstance(rotation, Angle):
        rotation = Angle(rotation, units).wrap_at(Angle(180, 'deg'))
    if not isinstance(obliquity, Angle):
        obliquity = Angle(obliquity, units).wrap_at(Angle(90, 'deg'))
        obliquity.wrap_angle = 90 * u.deg
    if not isinstance(perspective, Angle):
        perspective = Angle(perspective, units).wrap_at(Angle(180, 'deg'))

    # Build rotation matrices.
    # astropy's rotation_matrix handles scalar and array Angles, but can fail
    # for very large arrays (>100 elements), so we fall back to a loop.
    try:
        rot_matrices = (
            np.diag([1., 1., 1.])
            @ rotation_matrix(rotation, "z")
            @ rotation_matrix(obliquity, "x")
            @ rotation_matrix(perspective, "z")
        )
    except Exception:
        rot_matrices = np.array([
            (np.diag([1., 1., 1.])
             @ rotation_matrix(phi, "z")
             @ rotation_matrix(theta, "x")
             @ rotation_matrix(psi, "z"))
            for phi, theta, psi in zip(rotation, obliquity, perspective)
        ])

    return _rotmat_to_fits_ortho(rot_matrices, units)


def _rotmat_to_fits_ortho(rot_matrices: Any, units: str = 'deg',
                          ) -> tuple[Any, Any, Any]:
    """Extract FITS SIN ortho parameters from a body-orientation rotation matrix.

    The actual euler→ortho / quaternion→ortho mapping. Given the 3×3 rotation
    matrix (or a stack ``(..., 3, 3)``) that describes the body's orientation,
    read off ``(center_lon, center_lat, lonpole)`` from specific matrix
    elements. The ``...`` indexing makes this work element-wise for a single
    matrix (scalar outputs) or an array of matrices (array outputs).

    Shared by :func:`euler_to_fits_ortho` and :func:`quaternion_to_fits_ortho`
    so the two entry points are guaranteed to agree for the same orientation.
    The sign flips account for the geographic longitude direction convention.
    """
    center_lons = Angle(
        np.arctan2(-rot_matrices[..., 0, 1], rot_matrices[..., 1, 1]), 'rad')
    center_lats = -1 * Angle(
        np.arcsin(rot_matrices[..., 2, 1]), 'rad')
    lonpoles = -1 * Angle(
        np.arctan2(-rot_matrices[..., 2, 0], rot_matrices[..., 2, 2]), 'rad')

    return (center_lons.to(units).value,
            center_lats.to(units).value,
            lonpoles.to(units).value)


def _quaternion_to_rotmat(quat: npt.NDArray[np.float64],
                          ) -> npt.NDArray[np.float64]:
    """Rotation matrix (or stack) from a unit quaternion in ``(w, x, y, z)``
    (scalar-first) order.

    The standard quaternion→matrix formula, vectorized: ``quat`` of shape
    ``(4,)`` gives a ``(3, 3)`` matrix; ``(N, 4)`` gives ``(N, 3, 3)``. The
    quaternion is normalized first, so it need not be unit on input. No SciPy
    dependency — just numpy.
    """
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    m[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[..., 0, 1] = 2.0 * (x * y - z * w)
    m[..., 0, 2] = 2.0 * (x * z + y * w)
    m[..., 1, 0] = 2.0 * (x * y + z * w)
    m[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[..., 1, 2] = 2.0 * (y * z - x * w)
    m[..., 2, 0] = 2.0 * (x * z - y * w)
    m[..., 2, 1] = 2.0 * (y * z + x * w)
    m[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return m


def quaternion_to_fits_ortho(quaternion: Any, scalar_first: bool = True,
                             units: str = 'deg') -> tuple[Any, Any, Any]:
    """Convert a body-orientation quaternion to FITS SIN orthographic params.

    The quaternion counterpart of :func:`euler_to_fits_ortho`: instead of
    proper Euler angles ``[rotation, obliquity, perspective]``, you give the
    body's orientation as a single rotation quaternion (avoids gimbal lock,
    interpolates smoothly via slerp for animations, and is the natural form
    when orientation comes from attitude tracking). The output
    ``[center_lon, center_lat, lonpole]`` can be passed straight to
    :func:`make_globe_frame` / :func:`make_planet_frame`.

    The quaternion encodes the body's orientation directly — the SAME rotation
    that the Euler angles build — so the two functions agree:
    ``quaternion_to_fits_ortho(q)`` equals
    ``euler_to_fits_ortho(rotation, obliquity, perspective)`` whenever ``q`` is
    the quaternion for that orientation. (The identity quaternion → the
    untilted equatorial view, ``(0, 0, 0)``.)

    Parameters
    ----------
    quaternion : array-like
        A rotation quaternion, shape ``(4,)`` — or ``(N, 4)`` for a batch (e.g.
        an animation), which returns three length-``N`` arrays. Need not be
        normalized (it is normalized internally). The component ORDER is set by
        ``scalar_first``.
    scalar_first : bool, optional
        Component ORDER of ``quaternion`` (this only sets which slot is the
        scalar part; it does not change the rotation the quaternion encodes).
        ``True`` (default) reads it as ``(w, x, y, z)`` — the scalar-first /
        Hamilton convention used in most textbooks and when writing a
        quaternion as ``w + xi + yj + zk``. ``False`` reads it as
        ``(x, y, z, w)`` — the scalar-LAST convention used by SciPy's
        :meth:`Rotation.as_quat`, ROS, and Eigen; pass ``False`` to feed one of
        those directly.
    units : str, optional
        Angular units of the returned values: ``'deg'`` (default) or ``'rad'``.

    Returns
    -------
    center_lons : ndarray or float
        Sub-observer longitude(s) for the plot frame (use with
        ``center_LONdeg``).
    center_lats : ndarray or float
        Sub-observer latitude(s) (use with ``center_LATdeg``).
    lonpoles : ndarray or float
        Native longitude(s) of the celestial pole (use with ``lonpole``).

    Notes
    -----
    **Orientation convention.** The quaternion encodes the body's orientation
    as the SAME rotation that :func:`euler_to_fits_ortho` builds internally —
    they share the matrix→ortho extraction — so the two agree:
    ``quaternion_to_fits_ortho(q)`` equals
    ``euler_to_fits_ortho(rotation, obliquity, perspective)`` exactly when ``q``
    is the quaternion of that orientation. Note ``euler_to_fits_ortho`` composes
    its rotation from astropy's *passive* (frame-rotation) matrices, so if you
    construct a cross-check quaternion from those same angles via SciPy's
    *active* convention you must negate them:
    ``Rotation.from_euler('ZXZ', [-rotation, -obliquity, -perspective],
    degrees=True)``. A quaternion that comes straight from an attitude / tracking
    system already encodes an orientation, so just pass it (with the right
    ``scalar_first``) — no angle juggling needed.

    Examples
    --------
    The identity quaternion is the untilted equatorial view::

        clon, clat, lonpole = quaternion_to_fits_ortho([1, 0, 0, 0])
        # -> (0.0, 0.0, 0.0)

    Drive a globe frame from a SciPy orientation quaternion (scalar-last)::

        from scipy.spatial.transform import Rotation
        q = Rotation.from_rotvec([23.44, 0, 0], degrees=True).as_quat()  # x,y,z,w
        clon, clat, lonpole = quaternion_to_fits_ortho(q, scalar_first=False)
        ax = make_planet_frame(111, center_LONdeg=clon, center_LATdeg=clat,
                               lonpole=lonpole)

    Animate by feeding a batch of slerp-interpolated quaternions ``(N, 4)`` —
    one row per frame — and reading the three returned arrays.

    See Also
    --------
    euler_to_fits_ortho : the Euler-angle counterpart (same orientation → same
        result); both share the matrix→ortho extraction.
    """
    q = np.asarray(quaternion, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError(
            "quaternion must have 4 components (shape (4,) or (N, 4)); got "
            f"trailing dimension {q.shape[-1]}")
    if not scalar_first:
        # Reorder scalar-last (x, y, z, w) → the internal scalar-first
        # (w, x, y, z) that _quaternion_to_rotmat expects.
        q = np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)
    return _rotmat_to_fits_ortho(_quaternion_to_rotmat(q), units)


def make_globe_angles(
        start_angles_deg: Sequence[float], n_steps: int,
        spin_rate: float = 0., nut_func: Callable[..., Any] = np.sin,
        nut_rate: float = 0., nut_amp: float = 5., prec_rate: float = 0.,
        spin_total: float | None = None, nut_cycles: float | None = None,
        prec_total: float | None = None,
        out_format: str = 'deg',
        out_system: str = 'plot') -> tuple[Any, Any, Any]:
    """
    Generate sequences of rotation angles for globe animation.

    Given starting Euler angles and rates for spin, nutation, and precession,
    produce arrays of angle sets that can drive a frame-by-frame animation.

    Parameters
    ----------
    start_angles_deg : list of 3 floats
        Starting [rotation, obliquity, perspective] in degrees.
    n_steps : int
        Number of time steps (frames).
    spin_rate : float, optional
        Spin rate in degrees per step. Positive = CCW (like Earth).
    nut_func : callable, optional
        Function for nutation oscillation, e.g. np.sin (default) or np.cos.
    nut_rate : float, optional
        Nutation phase rate in degrees per step.
    nut_amp : float, optional
        Nutation amplitude in degrees. Default 5.
    prec_rate : float, optional
        Precession rate in degrees per step. Positive = CW (like Earth).
    spin_total, prec_total : float, optional
        Whole-animation convenience: the *total* spin / precession sweep in
        degrees over all ``n_steps`` frames, i.e. the corresponding
        ``*_rate = total / n_steps``. Use ``spin_total=360`` for one clean full
        rotation over the animation instead of computing the per-step rate by
        hand. The sweep is endpoint-exclusive (frame 0 = start, frame
        ``n_steps`` would be start + total) so a full-``360`` sweep loops
        seamlessly. Pass the ``*_rate`` **or** the ``*_total``, not both.
    nut_cycles : float, optional
        Whole-animation convenience for nutation: the number of full oscillation
        cycles of ``nut_func`` over the animation, i.e.
        ``nut_rate = 360 * nut_cycles / n_steps``. Pass ``nut_rate`` **or**
        ``nut_cycles``, not both.
    out_format : str, optional
        'deg' (default), 'rad', or 'angle' (returns Angle objects).
    out_system : str, optional
        'plot' or 'zxy' for fits parameters [center_lon, center_lat, lonpole].
        'euler' or 'zxz' for raw Euler angles [rotation, obliquity, perspective].

    Returns
    -------
    alphas, betas, gammas : ndarray or float
        Angle arrays. Meaning depends on out_system. If n_steps=1, returns
        scalar floats instead of arrays.

    Notes
    -----
    Precession here drives the **perspective** (third Euler / ``psi``) angle.
    Because every skyplothelper globe frame is re-aimed per step, incrementing
    that angle reads on screen as the pole *precessing* around the sky. The same
    angle series fed to a **fixed-camera 3-D engine** instead spins the body
    about its already-tilted pole (the pole itself does not move) — worth
    keeping in mind when porting these angles to an external 3-D animation.

    See Also
    --------
    make_globe_frame : Build the orthographic ('globe') axes each angle set aims.
    make_planet_frame : Earth / planet (geographic) globe variant.

    Examples
    --------
    Earth spinning 1 full day over 24 frames, fixed tilt::

        lons, lats, poles = make_globe_angles(
            [60, 24, 45], 24, spin_rate=15.)

    One clean full rotation over the animation, no rate arithmetic (loops
    seamlessly)::

        lons, lats, poles = make_globe_angles(
            [0, 24, 45], 120, spin_total=360.)

    Nutation + precession demo::

        lons, lats, poles = make_globe_angles(
            [0, 24, -45], 120, spin_rate=2., nut_rate=30.,
            nut_amp=5., prec_rate=2.)
    """
    phi0, theta0, psi0 = start_angles_deg

    # Whole-animation conveniences: *_total is the total sweep in degrees over
    # all n_steps (rate = total / n_steps); nut_cycles is the number of full
    # oscillations. Each overrides its per-step rate; passing both is ambiguous.
    if spin_total is not None:
        if spin_rate:
            raise ValueError('pass spin_rate or spin_total, not both')
        spin_rate = spin_total / n_steps
    if prec_total is not None:
        if prec_rate:
            raise ValueError('pass prec_rate or prec_total, not both')
        prec_rate = prec_total / n_steps
    if nut_cycles is not None:
        if nut_rate:
            raise ValueError('pass nut_rate or nut_cycles, not both')
        nut_rate = 360. * nut_cycles / n_steps

    # Build angle arrays from rates
    if spin_rate == 0.:
        phis = np.zeros(n_steps) + phi0
    else:
        phis = np.arange(phi0, phi0 + spin_rate * n_steps, spin_rate)[:n_steps]

    if nut_rate == 0.:
        thetas = np.zeros(n_steps) + theta0
    else:
        phase = np.arange(0, np.radians(nut_rate * n_steps),
                          np.radians(nut_rate))[:n_steps]
        thetas = theta0 + nut_amp * nut_func(phase)

    if prec_rate == 0.:
        psis = np.zeros(n_steps) + psi0
    else:
        psis = np.arange(psi0, psi0 + prec_rate * n_steps, prec_rate)[:n_steps]

    # Convert to requested output system
    if out_system.lower() in ('plot', 'zxy'):
        alphas, betas, gammas = euler_to_fits_ortho(phis, thetas, psis,
                                                     units='deg')
    elif out_system.lower() in ('euler', 'zxz'):
        alphas, betas, gammas = phis, thetas, psis
    else:
        raise ValueError(
            f'out_system="{out_system}" not recognized. '
            'Choose from ["plot", "euler"].')

    # Format output
    if out_format == 'rad':
        alphas, betas, gammas = (np.radians(alphas), np.radians(betas),
                                  np.radians(gammas))
    elif out_format.lower() == 'angle':
        alphas = Angle(alphas, 'deg')
        betas = Angle(betas, 'deg')
        gammas = Angle(gammas, 'deg')

    # Return scalars for single-step case
    if n_steps == 1:
        alphas, betas, gammas = alphas[0], betas[0], gammas[0]

    return alphas, betas, gammas


# =============================================================================
# WCS Globe Frame Creation
# =============================================================================

def _resolve_globe_alias(new_val: Any, old_val: Any, new_name: str,
                         old_name: str, default: Any) -> Any:
    """Resolve a canonical make_globe_frame kwarg against its deprecated alias.

    Several make_globe_frame kwargs were renamed to match
    :func:`~skyplothelper.wcs_frame.make_wcs_frame` (so one name works on
    :func:`make_planet_frame` regardless of projection): ``lon_deg_spacing`` /
    ``lat_deg_spacing`` → ``lon_spacing`` / ``lat_spacing``, and ``Naxispix`` →
    ``npix``. The old names still work but warn; when neither is passed the
    behavior is unchanged (*default*).
    """
    if old_val is not None:
        warnings.warn(
            f"make_globe_frame: '{old_name}' is deprecated; use '{new_name}' "
            f"instead (same meaning).", DeprecationWarning, stacklevel=3)
        if new_val is None:
            return old_val
    return new_val if new_val is not None else default


def make_globe_frame(subplot_number: Any = 111, center_LONdeg: float = 0,
                     center_LATdeg: float = 0.,
                     radesys: str = 'ICRS', direction: str = 'sky',
                     lon_units: str = 'auto', lon_west: bool = False,
                     projection: str = 'SIN', equinox: float = 2000.0,
                     lonpole: float = 0., latpole: float = 0.,
                     obstime: Any = None,
                     Naxispix: int | None = None, npix: int | None = None,
                     lon_deg_spacing: float | None = None,
                     lat_deg_spacing: float | None = None,
                     lon_spacing: float | None = None,
                     lat_spacing: float | None = None,
                     grid: bool = True, gridcolor: Any = '0.8',
                     gridalpha: float = 0.3,
                     gridlw: float | None = None,
                     gridls: str | None = None,
                     aspect: Any = 1, return_header: bool = False,
                     extra_cards: dict[str, Any] | None = None,
                     tick_style: str = 'in_frame',
                     tick_rotation: Any = 'tangent',
                     auto_fontsize: bool = True, fig: Any = None) -> Any:
    """
    Create a WCSAxes plot frame in orthographic ('globe') projection.

    Builds a dummy fits header defining the SIN projection, creates a WCS
    from it, and sets up a matplotlib subplot with EllipticalFrame clipping.

    Parameters
    ----------
    subplot_number : int, tuple, or SubplotSpec
        Matplotlib subplot position: an int (e.g. ``111``), a
        ``(nrows, ncols, index)`` tuple (for grids beyond 9 panels), or a
        ``SubplotSpec``. Default ``111``.
    center_LONdeg : float
        Center longitude for the globe, in degrees.
    center_LATdeg : float
        Center latitude for the globe, in degrees.
    radesys : str
        Reference frame. 'ICRS', 'GALACTIC', 'SUPERGALACTIC',
        'ECLIPTIC', 'HELIOECLIPTIC', 'ITRS', etc.
    direction : str, optional
        Longitude orientation. ``'sky'`` (default) shows the celestial
        looking-out view with longitude / RA increasing to the *left*
        (east-left, north-up) — consistent with every other skyplothelper
        frame. Pass ``'geographic'`` for an Earth / planet globe viewed from
        outside, with longitude increasing to the *right* (east-right): this
        is what the Earth helpers (:func:`~skyplothelper.globe.plot_baselines`,
        :func:`~skyplothelper.globe.make_nightshade_blend`) and
        ``radesys='ITRS'`` globes want. Accepts aliases (``'astro'``, ``'geo'``,
        ``'earth'``, …).
    lon_units : {'auto', 'hours', 'degrees'}
        Longitude tick units. ``'auto'`` (default): hour-angle for a sky ICRS
        globe, degrees for a geographic or body-fixed (ITRS) globe. ``'hours'``
        / ``'degrees'`` force the unit. Aliases ``hms`` / ``h``, ``deg`` /
        ``d``.
    lon_west : bool
        Label longitude **westward** (default ``False``): west-longitude with a
        ``W`` / ``E`` hemisphere suffix (east −71° → ``71°W``), forcing degrees.
        Only the labels change; the data stays east-longitude. For planetary /
        observing conventions that count longitude in °W. See
        :func:`~skyplothelper.wcs_frame.make_wcs_frame` for the full note.
    projection : str
        Fits projection code. Default 'SIN' (orthographic).
    equinox : float, optional
        Equinox year for FK5/FK4 frames.
    lonpole : float, optional
        Native longitude of celestial pole (phi_p). Controls the rotation
        of the globe about the line of sight. This is the key parameter
        for achieving obliquity tilt in the SIN projection.
    latpole : float, optional
        LATPOLE, needed for some projection types.
    obstime : str, astropy Time, datetime, or None, optional
        Observation time in any format parsable by astropy.Time.
    npix : int, optional
        Pixel dimension for the projection grid (NAXIS1=NAXIS2). Controls the
        resolution of projected images. Default 360; use 720-1440 for
        large/high-res plots. The **canonical** name, shared with
        :func:`~skyplothelper.wcs_frame.make_wcs_frame` and
        :func:`~skyplothelper.make_planet_frame`.
    Naxispix : int, optional
        **Deprecated** alias for ``npix`` (same meaning). Kept for backward
        compatibility; passing it emits a ``DeprecationWarning``.
    lon_spacing, lat_spacing : float, optional
        Spacing between longitude / latitude grid lines, in degrees. Default
        30. These are the **canonical** names, shared with
        :func:`~skyplothelper.wcs_frame.make_wcs_frame` and
        :func:`~skyplothelper.make_planet_frame`.
    lon_deg_spacing, lat_deg_spacing : float, optional
        **Deprecated** aliases for ``lon_spacing`` / ``lat_spacing`` (same
        meaning). Kept for backward compatibility; passing them emits a
        ``DeprecationWarning``.
    grid : bool, optional
        Whether to draw a coordinate grid. Default True.
    gridcolor : str or color, optional
        Grid color. Default '0.8'.
    gridlw : float, optional
        Grid line width. ``None`` (default) keeps this frame's historical
        0.5; ``gridcolor`` and ``gridalpha`` were exposed but this was not.
    gridls : str, optional
        Grid line style. ``None`` (default) inherits
        ``rcParams['grid.linestyle']``.
    gridalpha : float, optional
        Grid alpha. Default 0.3.
    aspect : float or str, optional
        Subplot aspect ratio. Default 1.
    return_header : bool, optional
        If True, also return the fits header used to build the WCS.
    extra_cards : dict or None, optional
        Additional fits header cards to include (e.g. {'PV2_1': 0.1}).
    tick_style : {'in_frame', 'boundary', 'native'}, optional
        Where tick labels are drawn. Default ``'in_frame'`` —
        labels along the central parallel (equator) + central
        meridian, tangent-rotated. ``'boundary'`` puts them on the
        circular spine (via
        :func:`~skyplothelper.coord_overlay.add_overlay_ticks` in
        boundary mode — tangent-rotated and bypassing astropy's
        spurious-tick bugs). ``'native'`` keeps astropy's default
        boundary labels. See the matching ``tick_style`` kwarg on
        :func:`~skyplothelper.wcs_frame.make_wcs_frame` for the
        full description.
    tick_rotation : {'tangent', 'tangent_upright', 'horizontal'}, float, or callable, optional
        Rotation of each tick label. Default ``'tangent'`` (aliased
        ``'tangent_noflip'``) follows the gridline tangent continuously,
        keeping labels upright for the current view without flipping
        between neighbors; ``'tangent_upright'`` clamps each label upright
        (flipping 180° where the tangent crosses ±90°). Ignored when
        ``tick_style='native'``.
    auto_fontsize : bool, optional
        Auto-shrink tick label fontsize to fit the available axes
        width. Default ``True``. See the matching kwarg on
        :func:`~skyplothelper.wcs_frame.make_wcs_frame` for the full
        description.
    fig : matplotlib Figure, optional
        Figure to draw into. Defaults to the current figure (``plt.gcf()``).
        Pass an explicit figure to place the globe into one cell of a specific
        figure's grid without disturbing the others — e.g. alongside flat
        frames from :func:`~skyplothelper.wcs_frame.make_wcs_frame`. (Parity
        with ``make_wcs_frame``; previously the globe always used the current
        figure, which overplotted an existing grid.) ``subplot_number`` may
        also be a pre-existing Axes (from ``plt.subplots`` / ``GridSpec``),
        which is swapped in place for the globe.

    Returns
    -------
    ax : WCSAxes
        The created plot axis.
    hdr : Header
        Only returned if return_header=True.

    Examples
    --------
    Sky SIN globe (default — east increases to the left)::

        ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)

    Tilted Earth view — pass ``direction='geo'`` so longitude increases
    east-to-right, the natural terrestrial orientation::

        ax, hdr = make_globe_frame(111, center_LONdeg=30,
            center_LATdeg=15, radesys='ITRS', lonpole=-23.44,
            direction='geo', return_header=True)
    """
    # Canonical names shared with make_wcs_frame (npix, lon_spacing,
    # lat_spacing); the Naxispix / lon_deg_spacing / lat_deg_spacing aliases
    # still work but warn. Reassign so the rest of the body reads the resolved
    # value (defaults unchanged: 360 px, 30° spacing).
    Naxispix = _resolve_globe_alias(npix, Naxispix, "npix", "Naxispix", 360)
    lon_deg_spacing = _resolve_globe_alias(
        lon_spacing, lon_deg_spacing, "lon_spacing", "lon_deg_spacing", 30.0)
    lat_deg_spacing = _resolve_globe_alias(
        lat_spacing, lat_deg_spacing, "lat_spacing", "lat_deg_spacing", 30.0)
    # Map radesys to fits coordinate type codes
    rs = radesys.lower()
    if 'super' in rs:
        ctt1, ctt2 = 'SLON', 'SLAT'
        radesys = 'ICRS'
    elif 'gal' in rs:
        ctt1, ctt2 = 'GLON', 'GLAT'
        radesys = 'ICRS'
    elif 'hel' in rs:
        ctt1, ctt2 = 'HLON', 'HLAT'
        radesys = 'ICRS'
    elif 'ecl' in rs:
        ctt1, ctt2 = 'ELON', 'ELAT'
        radesys = 'ICRS'
    elif rs == 'gappt':
        ctt1, ctt2 = 'ELON', 'ELAT'
    elif rs == 'itrs':
        ctt1, ctt2 = 'TLON', 'TLAT'
    else:
        ctt1, ctt2 = 'RA--', 'DEC-'

    N = int(Naxispix)
    # Longitude orientation from ``direction`` (default 'sky' = east-LEFT, the
    # astronomical looking-out view, consistent with the other frames;
    # 'geographic' = east-RIGHT for Earth / planet globes). With CDELT2
    # negative (north up) the orthographic handedness makes CDELT1 negative =
    # east-right, so 'geographic' uses the negative sign and 'sky' flips it.
    # CDELT2 is unchanged (latitude orientation untouched).
    from ..projections.project import resolve_direction
    _geographic = resolve_direction(direction) == 'geographic'
    _cd1mag = 2 / np.pi / (N / 180)
    _cdelt1 = -_cd1mag if _geographic else _cd1mag
    hdr = pyfits.Header({
        'NAXIS': 2, 'NAXIS1': N, 'NAXIS2': N,
        'CRPIX1': N / 2 + 0.5, 'CRPIX2': N / 2 + 0.5,
        'CRVAL1': center_LONdeg, 'CRVAL2': center_LATdeg,
        'CDELT1': _cdelt1,
        'CDELT2': -2 / np.pi / (N / 180),
        'CUNIT1': 'deg', 'CUNIT2': 'deg',
        'CTYPE1': f'{ctt1}-{projection}',
        'CTYPE2': f'{ctt2}-{projection}',
        'RADESYS': radesys,
        'LONPOLE': lonpole,
        'LATPOLE': latpole,
    })
    if radesys in ('FK5', 'FK4', 'FK4-NO-E'):
        hdr['EQUINOX'] = equinox
    if obstime is not None:
        # .utc is load-bearing: FITS defines DATE-OBS as UTC, so a TT/TDB Time
        # has to be converted rather than formatted as-is (~69 s for TT).
        from .._timeinput import to_time
        hdr['DATE-OBS'] = to_time(obstime, _caller='make_globe_frame').utc.isot
    if extra_cards is not None:
        for key, val in extra_cards.items():
            hdr[key] = val

    wcs = WCS(hdr)

    # Create the WCSAxes with an elliptical (circular) boundary, using
    # ``fig.add_subplot`` on an EXPLICIT figure rather than the stateful
    # ``plt.subplot`` (which always targets ``plt.gcf()``). That parity with
    # make_wcs_frame is what lets a globe drop cleanly into one cell of an
    # existing grid instead of overplotting the whole current figure.
    # ``subplot_number`` accepts an int (111), a 3-tuple ((2, 3, 1)), a
    # SubplotSpec, OR a pre-existing Axes (swapped in place — the common
    # "give me a globe in this subplot" spelling for plt.subplots / GridSpec
    # workflows). A WCS ``projection`` makes add_subplot return a WCSAxes.
    from matplotlib.axes import Axes
    if isinstance(subplot_number, Axes):
        existing_ax = subplot_number
        if fig is None:
            fig = existing_ax.figure
        elif fig is not existing_ax.figure:
            raise ValueError(
                "make_globe_frame() received both an explicit ``fig=`` and a "
                "``subplot_number=`` Axes from a different figure — pass at "
                "most one figure context.")
        spec = existing_ax.get_subplotspec()
        if spec is None:
            raise ValueError(
                "Pre-existing Axes passed via ``subplot_number=`` has no "
                "SubplotSpec (probably from ``fig.add_axes(rect)``). Pass a "
                "SubplotSpec, an int subplot number, or an Axes in a subplot "
                "grid (``plt.subplots`` / ``GridSpec``).")
        existing_ax.remove()
        subplot_number = spec
    if fig is None:
        fig = plt.gcf()
    _spec = (subplot_number if isinstance(subplot_number, tuple)
             else (subplot_number,))
    ax: Any = fig.add_subplot(*_spec, projection=wcs,
                              frame_class=EllipticalFrame, aspect=aspect)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N + 0.5)
    for i in (0, 1):
        ax.coords[i].set_ticklabel(exclude_overlapping=True)
        ax.coords[i].set_ticks_visible(False)
    ax.coords[0].set_ticks(spacing=lon_deg_spacing * u.deg)
    ax.coords[1].set_ticks(spacing=lat_deg_spacing * u.deg)
    if grid:
        # gridlw / gridls default to this path's historical values rather
        # than to rcParams: the package's three grid-drawing paths each
        # forced a different property (this one lw, the all-sky one ls, the
        # coords one neither), so inheriting would have changed renders in
        # three different directions. They are at least overridable now.
        ax.grid(color=gridcolor, alpha=gridalpha,
                lw=(0.5 if gridlw is None else gridlw),
                **({} if gridls is None else {'ls': gridls}))

    # Auto-shrink tick-label fontsize to the available axes width
    # before _apply_tick_style runs. Same flow as make_wcs_frame:
    # canvas.draw() materializes labels, the helper introspects + sets
    # the fontsize on both coords, then _apply_tick_style forwards it
    # into overlay-mode label_kwargs. See make_wcs_frame's
    # ``auto_fontsize`` kwarg docstring for the full rationale. The
    # call is try/excepted: auto-fontsize is a convenience, never a
    # reason for make_globe_frame to fail.
    auto_fs = None
    if auto_fontsize:
        from ..autosize import auto_size_ticklabels
        try:
            ax.figure.canvas.draw()
        except Exception:
            pass
        try:
            auto_fs = auto_size_ticklabels(ax)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"make_globe_frame: auto_fontsize failed "
                f"({type(exc).__name__}: {exc}); falling back to "
                f"rcParams default. Pass auto_fontsize=False to suppress.",
                UserWarning, stacklevel=2)

    # Delegate to the shared helper so make_wcs_frame and
    # make_globe_frame share the same tick-style resolution. Imported
    # lazily so the globe package doesn't pin a circular import on
    # wcs_frame at load time. ``frame_shape='circular'`` is hard-coded
    # because make_globe_frame is SIN-only.
    # Longitude units. A geographic globe reads longitude in degrees, not RA
    # hours (sky globes keep hours). The visible globe labels are drawn by the
    # coord-overlay machinery, so the unit is passed to it as a label format;
    # set_format_unit on ax.coords is also applied for consistency.
    from ..projections.project import resolve_lon_units
    from ..wcs_frame import (
        _apply_lon_units,
        _apply_tick_style,
        _install_west_longitude_labels,
        _overlay_lon_fmt,
    )
    _lon_u = resolve_lon_units(lon_units)
    # West-longitude labeling (route b): 'west' overlay fmt for the visible
    # in-frame labels + the native formatter for the frame-edge tick style.
    _lon_fmt = 'west' if lon_west else _overlay_lon_fmt(_lon_u, _geographic)
    _apply_tick_style(ax, 'circular', tick_style, tick_rotation,
                      label_fontsize=auto_fs,
                      lon_label_fmt=_lon_fmt,
                      lon_spacing=lon_deg_spacing, lat_spacing=lat_deg_spacing)
    _apply_lon_units(ax, _lon_u, _geographic)
    if lon_west:
        _install_west_longitude_labels(ax)

    if return_header:
        return ax, hdr
    return ax


def make_planet_frame(subplot_number: Any = 111, *, body: str = 'earth',
                      center_LONdeg: float = 0.,
                      center_LATdeg: float = 0., projection: str = 'SIN',
                      radesys: str | None = None, lon_west: bool = False,
                      lon_spacing: float | None = None,
                      lat_spacing: float | None = None,
                      npix: int | None = None, grid: bool = True,
                      **kwargs: Any) -> Any:
    """Globe frame for a planetary body, viewed from outside.

    Thin convenience over :func:`make_globe_frame` for surface globes of a
    body (Earth, Moon, Mars, …). It flips the two defaults that the generic
    sky-oriented :func:`make_globe_frame` gets "wrong" for a planet:

    * ``direction='geographic'`` — longitude increases east-to-**right**, the
      natural way to view a body's surface from outside (the sky default is
      east-left).
    * ``radesys='ITRS'`` — a body-fixed longitude/latitude frame (``TLON`` /
      ``TLAT``) rather than celestial RA/Dec.

    Everything else is inherited from :func:`make_globe_frame` — including its
    ``tick_style='in_frame'`` default, so the two globe builders place tick
    labels the same way out of the box (pass ``tick_style='native'`` for
    frame-edge labels, e.g. when they read more cleanly over an opaque planet
    raster).

    Use it for the Earth helpers (:func:`~skyplothelper.globe.plot_baselines`
    VLBI networks, :func:`~skyplothelper.globe.make_nightshade_blend`) and any
    body-surface map.

    Parameters
    ----------
    subplot_number : int, tuple, or SubplotSpec
        Matplotlib subplot position (int / ``(nrows, ncols, index)`` tuple /
        ``SubplotSpec``), as in :func:`make_globe_frame`. Default ``111``.
    body : str
        Planetary body name. Default ``'earth'``. Earth uses the ITRS
        body-fixed frame, so obstime-based features (Sun position /
        nightshade) work. Other bodies have **no standard FITS-WCS frame
        code yet**, so they fall back to the same generic body-fixed
        lon/lat — pass coordinates in that body's own longitude/latitude
        system (no celestial transform is applied). ``body`` is descriptive
        here; the orientation and body-fixed frame are what it sets up.
    center_LONdeg, center_LATdeg : float
        Sub-observer longitude / latitude (degrees).
    projection : str
        FITS or friendly projection code. Default ``'SIN'`` — a true
        orthographic **globe** (one hemisphere, viewed from outside); its
        behavior is unchanged from earlier releases and keeps the dedicated
        globe builder. Any **other** projection makes a **flat** planet map
        instead of a globe — ``'CAR'`` / ``'plate_carree'``, ``'MOL'``,
        ``'robinson'``, etc. — routed through :func:`make_wcs_frame` so the full
        projection registry (including non-FITS projections like Robinson) is
        available for body-surface maps. Use a flat projection for regional or
        whole-world station / baseline maps that want the sph machinery (lon/lat
        coordinate input, regions, overlays) rather than plain matplotlib axes.
    radesys : str, optional
        Override the coordinate frame. Defaults to ``'ITRS'`` (body-fixed
        lon/lat). Rarely needed.
    lon_west : bool
        Label longitude **westward** (default ``False``) — for bodies /
        catalogs that count longitude in °W (classical Mars / lunar frames,
        station catalogs). The ticks read west-longitude with a ``W`` / ``E``
        hemisphere suffix (east −71° → ``71°W``); **only the labels change**,
        the data stays east-longitude and the map keeps its normal (unmirrored)
        planet orientation. Works on the SIN globe and the flat planet
        projections alike. Feed °W input coordinates through
        :func:`~skyplothelper.lon_west_to_east` once at the door. See
        :func:`~skyplothelper.wcs_frame.make_wcs_frame`.
    lon_spacing, lat_spacing : float, optional
        Grid-line spacing in degrees. **One name works for every projection** —
        it forwards to the right builder underneath. Default ``None`` keeps each
        builder's own default (30° on the ``SIN`` globe; automatic, ~8 lines
        across the field, on the flat projections). This is the same name used
        by :func:`~skyplothelper.wcs_frame.make_wcs_frame`.
    npix : int, optional
        Pixel-grid size (NAXIS1=NAXIS2) for the surface raster. One name for
        every projection (forwards to ``Naxispix`` on the ``SIN`` globe, ``npix``
        on the flat builders). Default ``None`` keeps each builder's own default.
    grid : bool
        Draw the coordinate grid (default ``True``).
    **kwargs
        Forwarded to the underlying builder (``lonpole`` / ``latpole`` for
        tilt, ``obstime``, ``direction`` to override the geographic default,
        etc.) — see **Other Parameters**. Note the globe-only features
        (nightshade, back-hemisphere culling) apply to the ``SIN`` globe; on a
        flat projection the whole surface is shown.

    Other Parameters
    ----------------
    gridcolor, gridalpha, gridlw, gridls : optional
        Grid line color / alpha / width / style (forwarded).
    lonpole, latpole : float, optional
        Pole orientation for a tilted view (e.g. ``lonpole=-23.44`` for Earth's
        obliquity).
    obstime : optional
        Observation time (enables nightshade / Sun-position features on Earth).
    tick_style, tick_rotation, auto_fontsize, fig, return_header : optional
        As on :func:`make_globe_frame` (``SIN``) /
        :func:`~skyplothelper.wcs_frame.make_wcs_frame` (flat) — whichever this
        call routes to by ``projection``.

    Returns
    -------
    ax : WCSAxes
        The created plot axis (or ``(ax, hdr)`` if ``return_header=True`` is
        forwarded).

    Examples
    --------
    Tilted Earth globe with a VLBI baseline network::

        from skyplothelper.globe import make_planet_frame, plot_baselines
        ax = make_planet_frame(111, center_LONdeg=-90, center_LATdeg=30,
                               lonpole=-23.44)
        plot_baselines(ax, vlbi_sites, back_hemisphere_linestyle=':')

    Mars surface globe (generic body-fixed lon/lat)::

        ax = make_planet_frame(111, body='mars', center_LONdeg=0)

    Flat whole-world VLBI baseline map on a Robinson projection::

        ax = make_planet_frame(111, projection='robinson', center_LONdeg=-100)
        plot_baselines(ax, vlbi_sites)
    """
    kwargs.setdefault('direction', 'geographic')
    if radesys is None:
        radesys = 'ITRS'

    # SIN is a true orthographic globe (one hemisphere visible from outside) and
    # keeps the dedicated, battle-tested globe builder — the default, behavior
    # unchanged. Any OTHER projection is a flat world map, so route it through
    # make_wcs_frame, which is registry-driven and can express the non-FITS
    # projections (Robinson & co.) that make_globe_frame's hand-rolled SIN
    # header cannot. The body-fixed frame (radesys/ITRS) and geographic
    # longitude direction carry over identically; only the projection geometry
    # and frame shape differ.
    from ..projections.registry import _resolve_projection
    proj_key, _ = _resolve_projection(projection)
    if proj_key == 'sin':
        # Globe branch: None → make_globe_frame's own defaults (30°, 360 px).
        return make_globe_frame(subplot_number, center_LONdeg=center_LONdeg,
                                center_LATdeg=center_LATdeg, radesys=radesys,
                                projection=projection, lon_west=lon_west,
                                lon_spacing=lon_spacing, lat_spacing=lat_spacing,
                                npix=npix, grid=grid, **kwargs)

    from ..wcs_frame import make_wcs_frame
    # The two builders diverge on a couple of kwarg names (make_globe_frame
    # predates make_wcs_frame); translate the one that commonly gets forwarded.
    if 'return_header' in kwargs:
        kwargs['return_hdr'] = kwargs.pop('return_header')
    # Flat branch: None → make_wcs_frame's 'auto' default (its spacing sentinel);
    # npix=None → make_wcs_frame's own npix default.
    return make_wcs_frame(subplot_number, projection=projection,
                          center_lon=center_LONdeg, center_lat=center_LATdeg,
                          frame=radesys, lon_west=lon_west, grid=grid, npix=npix,
                          lon_spacing='auto' if lon_spacing is None else lon_spacing,
                          lat_spacing='auto' if lat_spacing is None else lat_spacing,
                          **kwargs)


# =============================================================================
# Image Projection
# =============================================================================
