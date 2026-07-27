"""Procedural instrument markers — antenna, telescope, dome.

These are vector site-marker primitives for the typical use cases:
plotting radio antennas (with dish elevation), refractor telescopes
(with tube elevation), and observatory domes (with slit azimuth)
on top of WCSAxes / globe / plain mpl axes.

All three follow the same architecture as :func:`~.add_reticle`:
display-point geometry inside a
:class:`~matplotlib.offsetbox.DrawingArea` wrapped in an
:class:`~matplotlib.offsetbox.AnchoredOffsetbox`, so the markers
are pixel-stable across figure resize / pan / zoom and inherit the
reticle's coord-resolution helper (``coord_type='pixel' | 'world'``,
optional ``frame=`` for celestial transforms). They also share its
optional pixel-stable text label (``label=`` / ``label_side=``), exposed
on the returned box as ``.label_artist``.

Vector-only by design (no bundled PNG icons) — that keeps the
package lean and lets the markers scale to any size without
resolution loss.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Circle, PathPatch, Polygon
from matplotlib.path import Path

from .._stroke import _stroke_path_effects
from .reticle import (
    _LABEL_DIRECTIONS,
    _VALID_LABEL_SIDES,
    _place_offset_label,
    _resolve_anchor,
    _resolve_auto_label_side,
)

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

# Conservative sprite half-extent as a fraction of ``size`` (display
# points), used to offset an optional label just past the marker. Covers
# the worst case (an antenna whose dish/focus swings out to ~0.55*size)
# with a little margin; the per-marker geometry never reaches ``size``.
_LABEL_CLEARANCE_FRAC = 0.6


def _pivot_offset(rotation_deg: float, height_pts: float) -> tuple[float, float]:
    """Display-point offset of a marker's pivot from its base, up the pier.

    The pivot sits *height_pts* up the local vertical, then the whole sprite
    is rotated by *rotation_deg* (CCW). Matching the sprite geometry
    (``local @ R.T`` with ``R = _rotation_matrix(rotation_deg)``), a local
    ``(0, h)`` maps to ``h * (-sin, cos)`` of the rotation angle.
    """
    r = np.deg2rad(float(rotation_deg))
    return (-height_pts * float(np.sin(r)), height_pts * float(np.cos(r)))


class MarkerAnchors:
    """The base and pivot of an instrument marker, resolvable to live coords.

    Instrument-marker sprites are **pixel-stable** (their geometry is in
    display points), so the base (ground foot) and pivot (the antenna's
    elevation hinge / the telescope's tube mount / the dome's slit hinge) sit
    at fixed display-point offsets from the placed anchor — but their *data*
    coordinates depend on the current view. This object holds the stable
    offsets and resolves them to current axes-data coords on access:

    * :attr:`base` / :attr:`pivot` — ``(x, y)`` in the axes' data (pixel)
      space, recomputed each access (correct after pan / zoom / resize).
    * :attr:`base_offset` / :attr:`pivot_offset` — the stable ``(dx, dy)``
      offsets in **display points** from the anchor (what sight-line math
      wants; independent of the view).
    * :attr:`anchor` — the coordinate the marker was placed at.
    * :meth:`sight_line_origin` — a point just inside the dish/tube along an
      aim direction, for drawing sight-lines that read as leaving the optics.

    Exposed as ``.anchors`` on the box returned by the marker helpers.
    """

    def __init__(self, ax: Any, anchor_xy: tuple[float, float],
                 anchor_transform: Any, size: float,
                 base_offset: tuple[float, float],
                 pivot_offset: tuple[float, float]) -> None:
        self._ax = ax
        self._anchor_xy = (float(anchor_xy[0]), float(anchor_xy[1]))
        self._transform = anchor_transform
        self.size = float(size)
        self.base_offset = (float(base_offset[0]), float(base_offset[1]))
        self.pivot_offset = (float(pivot_offset[0]), float(pivot_offset[1]))

    @property
    def anchor(self) -> tuple[float, float]:
        """The ``(x, y)`` the marker was placed at (in its own coord space)."""
        return self._anchor_xy

    def _to_display(self, offset_pts: tuple[float, float]) -> npt.NDArray[np.float64]:
        anchor_disp = np.asarray(
            self._transform.transform(self._anchor_xy), dtype=float)
        scale = self._ax.figure.dpi / 72.0
        return anchor_disp + np.asarray(offset_pts, dtype=float) * scale

    def _to_data(self, offset_pts: tuple[float, float]) -> tuple[float, float]:
        disp = self._to_display(offset_pts)
        x, y = self._ax.transData.inverted().transform(disp)
        return (float(x), float(y))

    @property
    def base(self) -> tuple[float, float]:
        """Ground-foot ``(x, y)`` in axes data coords (live)."""
        return self._to_data(self.base_offset)

    @property
    def pivot(self) -> tuple[float, float]:
        """Pivot (elevation hinge / tube mount / slit hinge) in data coords."""
        return self._to_data(self.pivot_offset)

    def base_display(self) -> tuple[float, float]:
        """Ground-foot ``(x, y)`` in display pixels (live)."""
        d = self._to_display(self.base_offset)
        return (float(d[0]), float(d[1]))

    def pivot_display(self) -> tuple[float, float]:
        """Pivot ``(x, y)`` in display pixels (live)."""
        d = self._to_display(self.pivot_offset)
        return (float(d[0]), float(d[1]))

    def sight_line_origin(self, aim_angle_deg: float, into_bowl: float = 0.55,
                          coords: str = 'data') -> tuple[float, float]:
        """A sight-line start point just inside the optics.

        The pivot, nudged ``into_bowl * size`` display points along
        *aim_angle_deg* (screen degrees, CCW from +x) so a ray reads as
        leaving the dish/tube, not the pier foot — the recipe the VLBI
        long-track reference used. ``coords='data'`` (default) or
        ``'display'``.
        """
        a = np.deg2rad(float(aim_angle_deg))
        off = (self.pivot_offset[0] + into_bowl * self.size * float(np.cos(a)),
               self.pivot_offset[1] + into_bowl * self.size * float(np.sin(a)))
        if coords == 'display':
            d = self._to_display(off)
            return (float(d[0]), float(d[1]))
        return self._to_data(off)


def _attach_label_removal(anchor: Any, label_artist: Any) -> None:
    """Make ``anchor.remove()`` also take *label_artist* off the axes.

    The label is a sibling axes artist (an ``annotate`` Text), not a child
    of the offset box, so the box's own ``remove`` wouldn't clear it. We
    wrap the bound ``remove`` so a labeled marker comes off as one unit.
    """
    box_remove = anchor.remove

    def _remove() -> None:
        if getattr(label_artist, 'axes', None) is not None:
            label_artist.remove()
        box_remove()

    setattr(anchor, 'remove', _remove)


def _add_anchored_patches(ax: Any, coord: SkyCoord | tuple[float, float], coord_type: str,
                          frame: str | None, patches: list[Any],
                          total_size: float, zorder: int, *,
                          label: Any = None, label_side: str = 'auto',
                          label_offset: float = 3.0, label_color: Any = None,
                          label_fontsize: Any = None,
                          label_path_effects: Any = None,
                          label_kwargs: dict[str, Any] | None = None,
                          marker_size: float | None = None,
                          base_offset: tuple[float, float] = (0.0, 0.0),
                          pivot_offset: tuple[float, float] = (0.0, 0.0),
                          ) -> AnchoredOffsetbox:
    """Wrap *patches* (each centered on (total_size/2, total_size/2)
    in display points) in an AnchoredOffsetbox at *coord*.

    Returns the anchor box so callers can attach further state. When
    *label* is given, a pixel-stable text label is drawn just past the
    sprite on the *label_side* compass direction (``'auto'`` picks the
    emptiest quadrant) using the same offset-points machinery as
    :class:`~skyplothelper.overlays.reticle.Reticle`. The label is exposed
    as ``box.label_artist`` (``None`` when unlabeled), and the box's
    ``.remove()`` then takes the label off too.
    """
    anchor_x, anchor_y, anchor_transform = _resolve_anchor(
        coord, ax, coord_type, frame)
    da = DrawingArea(total_size, total_size, 0, 0)
    for p in patches:
        da.add_artist(p)
    anchor = AnchoredOffsetbox(
        loc='center', child=da, pad=0.0,
        frameon=False,
        bbox_to_anchor=(anchor_x, anchor_y),
        bbox_transform=anchor_transform,
        borderpad=0,
    )
    anchor.set_zorder(zorder)
    ax.add_artist(anchor)

    label_artist = None
    if label is not None:
        side = label_side
        if side == 'auto':
            side = _resolve_auto_label_side(
                ax, anchor_x, anchor_y, anchor_transform)
        elif side not in _LABEL_DIRECTIONS:
            raise ValueError(
                f"label_side must be one of {_VALID_LABEL_SIDES!r}, "
                f"got {label_side!r}")
        # Sit the label at zorder+3 so it clears the sprite's stacked
        # sub-artists (patches at zorder, struts +1, focal dot +2).
        label_artist = _place_offset_label(
            ax, anchor_x, anchor_y, anchor_transform,
            label=str(label), side=side,
            outer_extent=total_size * _LABEL_CLEARANCE_FRAC,
            label_offset=label_offset, color=label_color,
            fontsize=label_fontsize, zorder=zorder + 3,
            path_effects=label_path_effects,
            label_kwargs=label_kwargs)
        _attach_label_removal(anchor, label_artist)
    # Always expose the attribute (None when unlabeled) for a stable API.
    setattr(anchor, 'label_artist', label_artist)
    # Expose the base (foot) + pivot geometry for sight-lines / ground-line
    # placement (see MarkerAnchors). marker_size is None only for callers that
    # don't supply the geometry.
    if marker_size is not None:
        setattr(anchor, 'anchors', MarkerAnchors(
            ax, (anchor_x, anchor_y), anchor_transform, marker_size,
            base_offset, pivot_offset))
    else:
        setattr(anchor, 'anchors', None)
    return anchor


def _rotation_matrix(deg: float) -> npt.NDArray[np.float64]:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------
# Marker aiming — solve (elevation, rotation) so the *collecting* element
# (the dish bowl / the telescope tube's objective end) points at a target.
# ---------------------------------------------------------------------------
#
# The two procedural markers do NOT share a rotation convention (this is the
# subtle bit that makes hand-rolling the math error-prone):
#
#   * antenna : the dish bowl's on-screen opening direction is
#       ``dish_elev + 2*rotation``  — the outer ``rotation`` is applied to the
#       already-``(dish_elev+rotation)``-rotated dish, so it counts twice.
#   * telescope : the tube's on-screen direction is ``tube_elev + rotation``
#       — a single application.
#
# Both bases point "up" (ground-normal) at ``90 + rotation``. We therefore
# solve, per marker, for the two knobs that put the base-up at a desired
# screen angle U and the collecting element at a desired screen angle B.
#
# All screen angles are measured CCW from +x in *display* coordinates (pixels
# after ``ax.transData`` / the anchor transform) — NOT data coords — so a
# non-square axes doesn't skew the result.

_VALID_AIM_MODES = ('aimed', 'planted')
_VALID_TARGET_COORDS = ('display', 'data', 'axes', 'figure', 'world')


def _wrap180(angle: float) -> float:
    """Wrap *angle* (degrees) to the half-open range (-180, 180]."""
    a = (float(angle) + 180.0) % 360.0 - 180.0
    return a + 360.0 if a <= -180.0 else a


def _anchor_display(ax: Any, coord: SkyCoord | tuple[float, float], coord_type: str,
                    frame: str | None) -> npt.NDArray[np.float64]:
    """Resolve *coord* to a display-pixel ``(x, y)`` point.

    Reuses :func:`~.reticle._resolve_anchor` so pixel / world / SkyCoord
    inputs all resolve exactly as the markers themselves anchor.
    """
    x, y, transform = _resolve_anchor(coord, ax, coord_type, frame)
    return np.asarray(transform.transform((x, y)), dtype=float)


def _target_display(ax: Any, target: SkyCoord | tuple[float, float] | None, target_coords: str,
                    frame: str | None) -> npt.NDArray[np.float64]:
    """Resolve an aim *target* to a display-pixel ``(x, y)`` point.

    A :class:`~astropy.coordinates.SkyCoord` always resolves as a world
    position (``target_coords`` ignored); otherwise ``target_coords`` picks
    the transform: ``'display'`` (raw pixels), ``'data'`` (:attr:`ax.transData`),
    ``'axes'`` (:attr:`ax.transAxes` fraction), ``'figure'``
    (:attr:`fig.transFigure` fraction), or ``'world'`` (lon/lat via the WCS).
    """
    from astropy.coordinates import SkyCoord

    if isinstance(target, SkyCoord):
        return _anchor_display(ax, target, 'world', frame)
    if target_coords not in _VALID_TARGET_COORDS:
        raise ValueError(
            f"target_coords must be one of {_VALID_TARGET_COORDS!r}, "
            f"got {target_coords!r}")
    if target_coords == 'world':
        return _anchor_display(ax, target, 'world', frame)
    try:
        # Bound to a local because the union admits None, which the except
        # below deliberately turns into a clear message rather than letting
        # it surface as a bare unpacking TypeError.
        pair: Any = target
        x, y = pair
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"target must be an (x, y) pair (or SkyCoord), got {target!r}"
        ) from exc
    if target_coords == 'display':
        return np.array([float(x), float(y)])
    transform = {
        'data': ax.transData,
        'axes': ax.transAxes,
        'figure': ax.figure.transFigure,
    }[target_coords]
    return np.asarray(transform.transform((float(x), float(y))), dtype=float)


def _screen_angle(from_xy: npt.ArrayLike, to_xy: npt.ArrayLike) -> float:
    """Angle (degrees CCW from +x, wrapped to ``[0, 360)``) between two
    *display-pixel* points. The single definition of "which way is the
    target", shared by every aiming path."""
    d = np.asarray(to_xy, dtype=float) - np.asarray(from_xy, dtype=float)
    return float(np.degrees(np.arctan2(d[1], d[0]))) % 360.0


def _aim_angle(ax: Any, coord: SkyCoord | tuple[float, float], target: SkyCoord | tuple[float, float] | None, *, coord_type: str = 'pixel',
               frame: str | None = None,
               target_coords: str = 'display') -> float:
    """Screen angle from *coord* to *target*, both resolved to display pixels.

    The shared primitive behind the vector markers' ``aim_at=`` solve
    (:func:`aim_angles`) and the raster stamps' ``aim_at=``
    (:func:`~skyplothelper.imscatter_rotated`). Resolving to display pixels
    first is what keeps a non-square axes from skewing the angle.
    """
    return _screen_angle(_anchor_display(ax, coord, coord_type, frame),
                         _target_display(ax, target, target_coords, frame))


def _aim_solve(ax: Any, coord: SkyCoord | tuple[float, float], target: SkyCoord | tuple[float, float] | None, *, marker: str, mode: str,
               globe_center: SkyCoord | tuple[float, float] | None, coord_type: str, frame: str | None,
               target_coords: str, max_tilt: float, flip: Any = 'auto',
               rest_elev: float = 90.0
               ) -> tuple[float, float, float, float | None]:
    """Core solver → ``(rotation, elev, aim_angle, radial_angle)``.

    ``elev`` is ``dish_elev`` (antenna) or ``tube_elev`` (telescope). See the
    module comment above for the geometry. ``radial_angle`` is ``None`` in
    ``'aimed'`` mode (no globe center is used).
    """
    if mode not in _VALID_AIM_MODES:
        raise ValueError(
            f"aim_mode must be one of {_VALID_AIM_MODES!r}, got {mode!r}")
    if flip not in (True, False, 'auto'):
        raise ValueError(
            f"flip must be True, False, or 'auto', got {flip!r}")

    m_xy = _anchor_display(ax, coord, coord_type, frame)
    t_xy = _target_display(ax, target, target_coords, frame)
    phi = _screen_angle(m_xy, t_xy)  # marker -> target

    rho: float | None = None
    if mode == 'aimed':
        # The whole sprite points at the target: base-up and the collecting
        # element both aim along phi.
        base_up = phi
        aim = phi
    else:  # 'planted' — pier stays along the local vertical; only dish tilts.
        if globe_center is None:
            raise ValueError(
                "aim_mode='planted' needs globe_center=(lon, lat) (or the "
                "disk-center coord) to know which way is locally 'up'.")
        g_xy = _anchor_display(ax, globe_center, coord_type, frame)
        r = m_xy - g_xy
        rho = float(np.degrees(np.arctan2(r[1], r[0]))) % 360.0  # outward normal
        # When the target sits behind the local horizon (>90 deg from the
        # outward normal) a real ground mount can't reach it. For this
        # schematic we 'flip' the base 180 deg about the local vertical (a
        # 180-deg azimuth swing): the pier stays planted along the local
        # normal, but the dish/tube can rise "upward" in the marker's own
        # sky toward the target instead of tangling across the mount.
        behind = abs(_wrap180(phi - rho)) > 90.0
        do_flip = (flip is True) or (flip == 'auto' and behind)
        base_up = (rho + 180.0) if do_flip else rho
        # Clamp the residual tilt away from the (possibly flipped) local up
        # so an extreme target doesn't over-rotate the dish.
        tilt = _wrap180(phi - base_up)
        if abs(tilt) > max_tilt:
            tilt = float(np.sign(tilt) * max_tilt)
        aim = base_up + tilt

    # The rest pose of the instrument. 90 means it points at the zenith,
    # which was assumed outright before this became a parameter.
    # Generalizing here covers BOTH markers and BOTH aim modes: the
    # downstream formulas are unchanged and fall out as
    # dish_elev = 2*rest_elev - aim (antenna) and tube_elev = rest_elev
    # (telescope), keeping the collecting element on the target either way.
    rotation = base_up - rest_elev
    if marker == 'antenna':
        elev = aim - 2.0 * rotation          # bowl = dish_elev + 2*rotation
    elif marker == 'telescope':
        elev = aim - rotation                # tube = tube_elev + rotation
    else:
        raise ValueError(
            f"marker must be 'antenna' or 'telescope', got {marker!r}")

    return _wrap180(rotation), _wrap180(elev), phi, rho


def aim_angles(ax: Any, coord: SkyCoord | tuple[float, float], target: SkyCoord | tuple[float, float] | None, *, marker: str = 'antenna',
               mode: str = 'aimed', globe_center: SkyCoord | tuple[float, float] | None = None,
               coord_type: str = 'pixel', frame: str | None = None,
               target_coords: str = 'display',
               max_tilt: float = 180.0,
               flip: Any = 'auto',
               rest_elev: float = 90.0) -> dict[str, float | None]:
    """Solve the marker angles that point its collecting element at *target*.

    This is the vetted geometry kernel behind the ``aim_at=`` shortcut on
    :func:`add_antenna_marker` / :func:`add_telescope_marker`; call it
    directly when you want the numbers (e.g. to drive a raster icon, or to
    reuse one solve across several markers).

    The *collecting* element is what should face the source — the parabolic
    **dish bowl** for an antenna, the **objective end of the tube** for a
    telescope (the photon-collecting surface), NOT the mount / pier.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes the marker lives on. Its figure must already be drawn
        (call ``fig.canvas.draw()`` once) so display transforms are valid.
    coord : (x, y) or SkyCoord
        The marker's own anchor (same meaning as the markers' ``coord``).
    target : (x, y) or SkyCoord
        Where the instrument should point. Interpreted per ``target_coords``
        (a ``SkyCoord`` is always a world position).
    marker : {'antenna', 'telescope'}
        Which marker's rotation convention to solve for. Default ``'antenna'``.
    mode : {'aimed', 'planted'}
        ``'aimed'`` (default) points the whole sprite at the target — best for
        "this array is observing that source" figures. ``'planted'`` keeps the
        pier along the local vertical on a globe (needs ``globe_center``) and
        tilts only the dish/tube — like a ground alt-az mount.
    globe_center : (x, y) or SkyCoord, optional
        The disk / projection center, used by ``'planted'`` to find local
        "up". Resolved with the same ``coord_type`` / ``frame`` as ``coord``.
    coord_type : {'pixel', 'world'}
        How ``coord`` (and ``globe_center``) are interpreted. Default
        ``'pixel'`` (data coords).
    frame : str or None
        Celestial frame for world coords with numeric tuples (see
        :func:`add_reticle`).
    target_coords : {'display', 'data', 'axes', 'figure', 'world'}
        How a numeric ``target`` tuple is interpreted. Default ``'display'``.
    max_tilt : float
        ``'planted'`` only — the largest angle (degrees) the dish/tube may
        tilt away from the (possibly flipped) local up. Default ``180`` (no
        clamp); set e.g. ``90`` to stop the dish pointing below the local
        horizon. Rarely needed together with ``flip``.
    flip : {'auto', True, False}
        ``'planted'`` only. When the target sits behind the marker's local
        horizon (more than 90 deg from the outward normal), a ground mount
        can't reach it. ``'auto'`` (default) then swings the base 180 deg
        about the local vertical (a 180-deg azimuth flip) so the pier stays
        planted but the dish/tube rises toward the target instead of tangling
        across the mount. ``True`` always flips; ``False`` never does (the
        base always points outward, and a behind-horizon target relies on
        ``max_tilt`` clamping).

    rest_elev : float, optional
        Screen elevation in degrees that the collecting element points at when
        ``rotation=0`` -- the rest pose of the instrument. Default ``90``
        (straight up), which is what the solver assumed before this became a
        parameter, so existing calls are unchanged. Lower it to aim an
        instrument that sits at a working elevation rather than at the zenith;
        the solve then yields ``dish_elev = 2*rest_elev - aim_angle`` for an
        antenna, or ``tube_elev = rest_elev`` for a telescope, keeping the
        collecting element on the target in either aim mode.
    Returns
    -------
    dict
        ``{'rotation': ..., 'dish_elev'|'tube_elev': ..., 'aim_angle': phi,
        'radial_angle': rho}`` — splat the first two into the marker call:
        ``add_antenna_marker(ax, coord, **{k: v for k, v in r.items()
        if k in ('rotation', 'dish_elev')})``. ``radial_angle`` is ``None``
        in ``'aimed'`` mode. For a raster icon with native pointing direction
        ``rest_angle``, use ``rotation = r['aim_angle'] - rest_angle``.

    Examples
    --------
    >>> fig.canvas.draw()
    >>> r = aim_angles(ax, (250, 250), (400, 300), marker='antenna')
    >>> add_antenna_marker(ax, (250, 250), dish_elev=r['dish_elev'],
    ...                    rotation=r['rotation'])
    """
    rotation, elev, phi, rho = _aim_solve(
        ax, coord, target, marker=marker, mode=mode, globe_center=globe_center,
        coord_type=coord_type, frame=frame, target_coords=target_coords,
        max_tilt=max_tilt, flip=flip, rest_elev=rest_elev)
    elev_key = 'dish_elev' if marker == 'antenna' else 'tube_elev'
    return {'rotation': rotation, elev_key: elev,
            'aim_angle': phi, 'radial_angle': rho}


def _apply_stroke(patch: Any, stroke_color: Any, stroke_lw: float,
                  edge_lw: float) -> None:
    if stroke_color is None:
        return
    stroke_pad = max(0.0, float(stroke_lw) - float(edge_lw))
    if stroke_pad <= 0:
        return
    # The appended ``Normal()`` this used to carry redrew the sprite edge a
    # second time, darkening it at the antialiasing level -- ``withStroke``
    # already draws the core on top, so it was redundant. See _stroke.py.
    effects = _stroke_path_effects(stroke_color, stroke_lw)
    if effects:
        patch.set_path_effects(effects)


def add_antenna_marker(ax: Any, coord: SkyCoord | tuple[float, float], *, dish_elev: float = 45.0,
                        rotation: float = 0.0,
                        aim_at: SkyCoord | tuple[float, float] | None = None, aim_mode: str = 'aimed',
                        globe_center: SkyCoord | tuple[float, float] | None = None,
                        target_coords: str = 'display',
                        max_tilt: float = 180.0, flip: Any = 'auto',
                        rest_elev: float = 90.0,
                        size: float = 22.0, coord_type: str = 'pixel',
                        frame: str | None = None,
                        face_color: Any = 'white', edge_color: Any = 'black',
                        edge_lw: float = 0.8, alpha: float = 1.0,
                        stroke_color: Any = None, stroke_lw: float = 2.0,
                        label: Any = None, label_side: str = 'auto',
                        label_offset: float = 3.0, label_color: Any = None,
                        label_fontsize: Any = None,
                        label_kwargs: dict[str, Any] | None = None,
                        zorder: int = 10) -> AnchoredOffsetbox:
    """Procedurally draw a radio-antenna site marker.

    Shape: a small trapezoidal **base**, a vertical **mount pole**,
    and a **parabolic dish** rendered as a circular-arc sector that
    opens in the dish-pointing direction. The dish points at
    ``dish_elev`` degrees above the local horizon (``0`` = horizontal,
    ``90`` = zenith); an additional ``rotation`` rotates the whole
    sprite CCW in the figure plane (useful for surface-tangent
    orientation on globe views).

    Parameters
    ----------
    ax : matplotlib Axes (WCSAxes / globe / plain mpl)
    coord : (x, y) or SkyCoord
        Anchor position. ``coord_type='pixel'`` interprets as data
        coords; ``coord_type='world'`` accepts (lon, lat) in degrees
        or a :class:`~astropy.coordinates.SkyCoord`.
    dish_elev : float
        Dish-pointing elevation above the local horizon (degrees).
        ``0`` = pointing sideways, ``45`` = halfway up, ``90`` = at
        zenith. Default ``45``.
    rotation : float
        Additional CCW rotation of the entire sprite (degrees) — set
        this to the surface-tangent angle on globe views to keep the
        antenna's base "flat on the ground". Default ``0``.
    aim_at : (x, y) or SkyCoord, optional
        Point the **dish bowl** (the photon-collecting surface) at this
        target. When given, ``dish_elev`` and ``rotation`` are solved
        automatically (via :func:`aim_angles`) and any values passed for
        them are ignored. Interpreted per ``target_coords``.
    aim_mode : {'aimed', 'planted'}
        Aiming style when ``aim_at`` is set. ``'aimed'`` (default) points the
        whole sprite at the target; ``'planted'`` keeps the pier upright on a
        globe (needs ``globe_center``) and tilts only the dish. See
        :func:`aim_angles`.
    globe_center : (x, y) or SkyCoord, optional
        Disk / projection center, required by ``aim_mode='planted'``.
    target_coords : {'display', 'data', 'axes', 'figure', 'world'}
        How a numeric ``aim_at`` tuple is interpreted. Default ``'display'``.
    max_tilt : float
        ``aim_mode='planted'`` only — max dish tilt from local up (degrees).
        Default ``180`` (no clamp).
    flip : {'auto', True, False}
        ``aim_mode='planted'`` only — flip the base 180 deg about the local
        vertical when the target is behind the local horizon so the dish can
        still rise toward it. The pier *foot* stays glued to ``coord`` (the
        sprite pivots about its base); only the dish swings to the other
        side. See :func:`aim_angles`. Default ``'auto'``.
    rest_elev : float, optional
        Screen elevation in degrees that the collecting element points at when
        ``rotation=0`` -- the rest pose of the instrument. Default ``90``
        (straight up), which is what the solver assumed before this became a
        parameter, so existing calls are unchanged. Lower it to aim an
        instrument that sits at a working elevation rather than at the zenith;
        the solve then yields ``dish_elev = 2*rest_elev - aim_angle`` for an
        antenna, or ``tube_elev = rest_elev`` for a telescope, keeping the
        collecting element on the target in either aim mode.
    size : float
        Overall sprite size in display points (matches the
        ``size`` convention of :func:`~.add_reticle`). Default ``22``.
    coord_type : {'pixel', 'world'}
    frame : str or None
        Coordinate frame for ``coord_type='world'`` with a numeric
        tuple. See :func:`~.add_reticle` for details.
    face_color, edge_color : matplotlib colors
        Fill / outline colors. Default white-on-black for icon
        legibility on either light or dark backdrops.
    edge_lw : float
        Outline width in points.
    alpha : float
    stroke_color : str or None
        Optional stroke color drawn underneath the outline and the label
        (for legibility on noisy backgrounds). Default ``None``.
    stroke_lw : float
        Stroke width in points (only used when ``stroke_color`` is set).
    label : str or None
        Optional text label drawn just past the marker on the
        ``label_side`` compass direction (pixel-stable, via
        :func:`~matplotlib.axes.Axes.annotate` with offset points).
        Default ``None`` (no label).
    label_side : str
        Which side the label sits on: ``'auto'`` (default) picks the
        emptiest quadrant at draw time, or pin any of the eight compass
        points ``'N'``, ``'NE'``, ..., ``'NW'``.
    label_offset : float
        Extra gap (display points) between the marker's bounding box and
        the label. Default ``3``.
    label_color : matplotlib color or None
        Label text color. Default ``None`` → follow ``edge_color``.
    label_fontsize : float or str or None
        Label font size (any matplotlib size spec). Default ``None`` →
        the rcParams default.
    label_kwargs : dict or None
        Extra keyword arguments forwarded to
        :func:`~matplotlib.axes.Axes.annotate` (e.g. to override
        ``ha`` / ``va`` / ``color`` / ``fontweight``). Default ``None``.
    zorder : int

    Returns
    -------
    anchor : matplotlib.offsetbox.AnchoredOffsetbox
        The offset box wrapping the marker. Use ``anchor.remove()``
        to take it back off the axes. When ``label`` is set, the label
        :class:`~matplotlib.text.Annotation` is exposed as
        ``anchor.label_artist`` (``None`` otherwise) and is removed
        together with the marker.

        ``anchor.anchors`` is a :class:`MarkerAnchors` exposing the marker's
        base (ground foot) and pivot (elevation hinge / tube mount / slit
        hinge) as live axes-data coords (``.base`` / ``.pivot``) and stable
        display-point offsets (``.base_offset`` / ``.pivot_offset``), plus
        ``.sight_line_origin(aim_angle_deg)`` for a ray that leaves the optics.
        All three markers are **base-anchored**: the foot sits at the placed
        coord, so a row of markers plants their feet on one ground line.

    Examples
    --------
    >>> add_antenna_marker(ax, (180, -30), coord_type='world',
    ...                    dish_elev=60, size=24)
    >>> add_antenna_marker(ax, (250, 250), label='DSS-14', label_side='NE')
    >>> add_antenna_marker(ax, (250, 250), dish_elev=0,
    ...                    rotation=20, face_color='lightyellow')
    >>> # Point the dish at an off-frame source:
    >>> add_antenna_marker(ax, (250, 250), aim_at=(500, 380))
    """
    if aim_at is not None:
        rotation, dish_elev, _phi, _rho = _aim_solve(
            ax, coord, aim_at, marker='antenna', mode=aim_mode,
            globe_center=globe_center, coord_type=coord_type, frame=frame,
            target_coords=target_coords, max_tilt=max_tilt, flip=flip,
            rest_elev=rest_elev)
    R = _rotation_matrix(rotation)
    # Pivot about the *base* (ground contact), not the sprite midpoint.
    # A planted ``flip`` is a 180° azimuth swing about the local vertical:
    # the pier foot must stay glued to ``coord`` while the dish swings to
    # the other side. Centering the DrawingArea on mid-sprite made the
    # base orbit the dish instead. Pad the area so the dish never clips
    # when the whole assembly rotates about the foot.
    area = 2.0 * size
    cx = cy = area / 2.0

    # --- Base trapezoid: bottom-center at the local origin (= anchor)
    base_h = size * 0.16
    base_w_top = size * 0.30
    base_w_bot = size * 0.42
    y0 = 0.0
    base_pts = np.array([
        (-base_w_bot / 2, y0),
        (+base_w_bot / 2, y0),
        (+base_w_top / 2, y0 + base_h),
        (-base_w_top / 2, y0 + base_h),
    ])
    base_pts = base_pts @ R.T + [cx, cy]
    base = Polygon(base_pts, closed=True, facecolor=face_color,
                    edgecolor=edge_color, linewidth=edge_lw,
                    alpha=alpha, zorder=zorder)

    # --- Mount pole (thin vertical rectangle above the base, ~30% tall)
    pole_w = size * 0.08
    pole_h = size * 0.30
    py0 = y0 + base_h
    pole_pts = np.array([
        (-pole_w / 2, py0),
        (+pole_w / 2, py0),
        (+pole_w / 2, py0 + pole_h),
        (-pole_w / 2, py0 + pole_h),
    ])
    pole_pts = pole_pts @ R.T + [cx, cy]
    pole = Polygon(pole_pts, closed=True, facecolor=face_color,
                    edgecolor=edge_color, linewidth=edge_lw,
                    alpha=alpha, zorder=zorder)

    # --- Dish: a CONCAVE parabolic curve mounted on top of the pole.
    # The rotation pivot sits just *inside* the back of the bowl —
    # roughly 15% of the visible depth from the back-most curve point
    # forward toward the rim. This mimics the elevation axis of a
    # real alt-az dish mount, which passes through the structural
    # backbone behind the bowl. From a side-view, the effect as
    # ``dish_elev`` slews from horizon to zenith is that the bowl
    # pivots around a point near its back, with the back of the bowl
    # barely shifting and the rim swinging — what a real antenna
    # actually does — and the pier-top remains visually inside the
    # bowl, partially obscured by the dish fill.
    #
    # Because the bezier control (the "vertex") is a magnet rather
    # than a point the curve passes through, we have to back-solve
    # for it: a quadratic Bezier from (rim_x, ±r) with control
    # (vertex_x, 0) puts the curve midpoint at x = (rim_x + vertex_x)/2.
    # Setting the midpoint just behind the pivot (at -back_offset)
    # and choosing how far the rim sits forward gives vertex_x.
    dish_r = size * 0.32
    dish_visible_depth = dish_r * 0.75
    pivot_back_frac = 0.15  # pivot sits 15% from back, 85% from rim
    back_offset = pivot_back_frac * dish_visible_depth
    rim_x = (1.0 - pivot_back_frac) * dish_visible_depth
    vertex_x = -2.0 * back_offset - rim_x  # back-solved bezier control
    focal_len = dish_r * 1.05
    mount_top_local = np.array([0.0, py0 + pole_h])
    vertex_l = np.array([vertex_x, 0.0])
    rim_top_l = np.array([rim_x, +dish_r])
    rim_bot_l = np.array([rim_x, -dish_r])
    focus_l = np.array([rim_x + focal_len, 0.0])

    Rdish = _rotation_matrix(float(dish_elev) + float(rotation))

    def _to_world(pt_local: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Rotate the dish-local point about the mount top in the
        # pre-rotation frame, then apply the outer ``rotation`` and
        # offset by the base-anchored sprite center.
        return ((pt_local @ Rdish.T + mount_top_local) @ R.T
                 + np.array([cx, cy]))

    rim_top = _to_world(rim_top_l)
    rim_bot = _to_world(rim_bot_l)
    vertex = _to_world(vertex_l)
    focus = _to_world(focus_l)

    # Dish path: rim_top → (Bezier control = vertex) → rim_bot → close
    # to rim_top with a chord. The chord is the rim plane viewed
    # edge-on — geometrically correct, and reads as the dish's front
    # opening at low elevations.
    dish_path = Path(
        [rim_top, vertex, rim_bot, rim_top],
        [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY],
    )
    dish = PathPatch(dish_path, facecolor=face_color,
                      edgecolor=edge_color, linewidth=edge_lw,
                      alpha=alpha, zorder=zorder)

    # Two struts from rim to focus + a small focal-point dot.
    strut_top = PathPatch(
        Path([rim_top, focus], [Path.MOVETO, Path.LINETO]),
        facecolor='none', edgecolor=edge_color,
        linewidth=max(0.5, edge_lw * 0.8), alpha=alpha,
        zorder=zorder + 1)
    strut_bot = PathPatch(
        Path([rim_bot, focus], [Path.MOVETO, Path.LINETO]),
        facecolor='none', edgecolor=edge_color,
        linewidth=max(0.5, edge_lw * 0.8), alpha=alpha,
        zorder=zorder + 1)
    focal_dot = Circle(tuple(focus), max(0.6, size * 0.04),
                        facecolor=edge_color, edgecolor=edge_color,
                        linewidth=0, alpha=alpha,
                        zorder=zorder + 2)

    patches = [base, pole, dish, strut_top, strut_bot, focal_dot]
    for p in patches:
        _apply_stroke(p, stroke_color, stroke_lw, edge_lw)

    _lbl_color = label_color if label_color is not None else edge_color
    return _add_anchored_patches(
        ax, coord, coord_type, frame, patches, area, zorder,
        label=label, label_side=label_side, label_offset=label_offset,
        label_color=_lbl_color, label_fontsize=label_fontsize,
        label_path_effects=(_stroke_path_effects(stroke_color, stroke_lw)
                            if stroke_color is not None else None),
        label_kwargs=label_kwargs,
        marker_size=size,
        pivot_offset=_pivot_offset(rotation, py0 + pole_h))


def add_telescope_marker(ax: Any, coord: SkyCoord | tuple[float, float], *, tube_elev: float = 30.0,
                          rotation: float = 0.0,
                          aim_at: SkyCoord | tuple[float, float] | None = None, aim_mode: str = 'aimed',
                          globe_center: SkyCoord | tuple[float, float] | None = None,
                          target_coords: str = 'display',
                          max_tilt: float = 180.0, flip: Any = 'auto',
                          rest_elev: float = 90.0,
                          size: float = 22.0, coord_type: str = 'pixel',
                          frame: str | None = None,
                          face_color: Any = 'white', edge_color: Any = 'black',
                          edge_lw: float = 0.8, alpha: float = 1.0,
                          stroke_color: Any = None, stroke_lw: float = 2.0,
                          label: Any = None, label_side: str = 'auto',
                          label_offset: float = 3.0, label_color: Any = None,
                          label_fontsize: Any = None,
                          label_kwargs: dict[str, Any] | None = None,
                          zorder: int = 10) -> AnchoredOffsetbox:
    """Procedurally draw a refractor-telescope site marker.

    Shape: a tripod **mount** at the base, a long cylindrical
    **tube** pointing at ``tube_elev`` degrees above the local
    horizon, and a small **eyepiece** at the back end of the tube.
    The whole sprite can be additionally rotated by ``rotation``
    CCW (for surface-tangent orientation on globe views).

    Same anchoring / coord-resolution conventions as
    :func:`add_antenna_marker`.

    Parameters
    ----------
    ax : matplotlib Axes
    coord : (x, y) or SkyCoord
    tube_elev : float
        Tube-pointing elevation above the local horizon (degrees).
        Default ``30``.
    rotation : float
        Extra CCW rotation of the whole sprite (degrees).
    aim_at : (x, y) or SkyCoord, optional
        Point the **tube's objective end** (the photon-collecting front)
        at this target; solves ``tube_elev`` and ``rotation`` automatically
        via :func:`aim_angles` (ignoring any passed values). Interpreted per
        ``target_coords``.
    aim_mode : {'aimed', 'planted'}
        Aiming style (see :func:`aim_angles`). Default ``'aimed'``.
    globe_center : (x, y) or SkyCoord, optional
        Disk / projection center, required by ``aim_mode='planted'``.
    target_coords : {'display', 'data', 'axes', 'figure', 'world'}
        How a numeric ``aim_at`` tuple is interpreted. Default ``'display'``.
    max_tilt : float
        ``aim_mode='planted'`` only — max tube tilt from local up (degrees).
        Default ``180`` (no clamp).
    flip : {'auto', True, False}
        ``aim_mode='planted'`` only — flip the base 180 deg about the local
        vertical when the target is behind the local horizon so the tube can
        still rise toward it. See :func:`aim_angles`. Default ``'auto'``.
    rest_elev : float, optional
        Screen elevation in degrees that the collecting element points at when
        ``rotation=0`` -- the rest pose of the instrument. Default ``90``
        (straight up), which is what the solver assumed before this became a
        parameter, so existing calls are unchanged. Lower it to aim an
        instrument that sits at a working elevation rather than at the zenith;
        the solve then yields ``dish_elev = 2*rest_elev - aim_angle`` for an
        antenna, or ``tube_elev = rest_elev`` for a telescope, keeping the
        collecting element on the target in either aim mode.
    size : float
        Overall sprite size in display points.
    coord_type, frame, face_color, edge_color, edge_lw, alpha, stroke_color, stroke_lw, label, label_side, label_offset, label_color, label_fontsize, label_kwargs, zorder :
        See :func:`add_antenna_marker`.

    Returns
    -------
    anchor : matplotlib.offsetbox.AnchoredOffsetbox
        See :func:`add_antenna_marker` for the ``label_artist`` and
        ``anchors`` attributes (base-anchored; ``.anchors`` gives base/pivot).

    Examples
    --------
    >>> add_telescope_marker(ax, (120, 40), coord_type='world',
    ...                      tube_elev=45)
    >>> add_telescope_marker(ax, (250, 250), aim_at=(450, 500))
    """
    if aim_at is not None:
        rotation, tube_elev, _phi, _rho = _aim_solve(
            ax, coord, aim_at, marker='telescope', mode=aim_mode,
            globe_center=globe_center, coord_type=coord_type, frame=frame,
            target_coords=target_coords, max_tilt=max_tilt, flip=flip,
            rest_elev=rest_elev)
    R = _rotation_matrix(rotation)
    # Base-anchored: the foot sits at the rotation origin (= the placed coord),
    # like the antenna, so `for xy in sites: marker(ax, xy)` plants all three
    # markers' feet on one ground line. Pad the area (2*size) so the tube never
    # clips as the assembly rotates about the foot.
    area = 2.0 * size
    cx = cy = area / 2.0
    base_y = 0.0

    # --- Tripod: three short lines spreading from a central peak just
    # above the bottom. Drawn as a thin triangle outline for the
    # outer two legs, plus a center leg.
    tripod_peak = base_y + size * 0.30
    leg_dx = size * 0.22
    tripod_pts = np.array([
        (-leg_dx, base_y),
        (0.0, tripod_peak),
        (+leg_dx, base_y),
    ])
    tripod_pts = tripod_pts @ R.T + [cx, cy]
    tripod = PathPatch(Path(tripod_pts, [Path.MOVETO, Path.LINETO,
                                          Path.LINETO]),
                        facecolor='none', edgecolor=edge_color,
                        linewidth=edge_lw, alpha=alpha, zorder=zorder)

    # Center leg (drop a short vertical from the peak to the ground)
    center_leg_pts = np.array([
        (0.0, base_y),
        (0.0, tripod_peak),
    ])
    center_leg_pts = center_leg_pts @ R.T + [cx, cy]
    center_leg = PathPatch(Path(center_leg_pts,
                                  [Path.MOVETO, Path.LINETO]),
                            facecolor='none', edgecolor=edge_color,
                            linewidth=edge_lw, alpha=alpha,
                            zorder=zorder)

    # --- Tube: a long thin rectangle centered on the tripod peak,
    # pointing at tube_elev + rotation. Tube length 62% of size,
    # width 16%.
    tube_len = size * 0.62
    tube_w = size * 0.16
    # Build tube in its own frame (long axis along +x), then rotate
    # by (tube_elev + rotation - 0) so that tube_elev=0 → horizontal
    # right.
    tube_local = np.array([
        (-tube_len * 0.18, -tube_w / 2),  # eyepiece end (a bit past peak)
        (+tube_len * 0.82, -tube_w / 2),  # objective end (far end)
        (+tube_len * 0.82, +tube_w / 2),
        (-tube_len * 0.18, +tube_w / 2),
    ])
    Rtube = _rotation_matrix(float(tube_elev) + float(rotation))
    tube_pts = tube_local @ Rtube.T + (
        np.array([0.0, tripod_peak]) @ R.T) + [cx, cy]
    tube = Polygon(tube_pts, closed=True, facecolor=face_color,
                    edgecolor=edge_color, linewidth=edge_lw,
                    alpha=alpha, zorder=zorder)

    # --- Eyepiece: a small bump at the back-end (-x in tube frame).
    eye_local = np.array([
        (-tube_len * 0.30, -tube_w * 0.4),
        (-tube_len * 0.18, -tube_w * 0.4),
        (-tube_len * 0.18, +tube_w * 0.4),
        (-tube_len * 0.30, +tube_w * 0.4),
    ])
    eye_pts = eye_local @ Rtube.T + (
        np.array([0.0, tripod_peak]) @ R.T) + [cx, cy]
    eyepiece = Polygon(eye_pts, closed=True, facecolor=face_color,
                        edgecolor=edge_color, linewidth=edge_lw,
                        alpha=alpha, zorder=zorder)

    patches = [tripod, center_leg, tube, eyepiece]
    for p in patches:
        _apply_stroke(p, stroke_color, stroke_lw, edge_lw)

    _lbl_color = label_color if label_color is not None else edge_color
    return _add_anchored_patches(
        ax, coord, coord_type, frame, patches, area, zorder,
        label=label, label_side=label_side, label_offset=label_offset,
        label_color=_lbl_color, label_fontsize=label_fontsize,
        label_path_effects=(_stroke_path_effects(stroke_color, stroke_lw)
                            if stroke_color is not None else None),
        label_kwargs=label_kwargs,
        marker_size=size,
        pivot_offset=_pivot_offset(rotation, tripod_peak))


def add_dome_marker(ax: Any, coord: SkyCoord | tuple[float, float], *, slit_azim: float = 0.0,
                     rotation: float = 0.0,
                     size: float = 22.0, coord_type: str = 'pixel',
                     frame: str | None = None,
                     face_color: Any = 'white', edge_color: Any = 'black',
                     edge_lw: float = 0.8, alpha: float = 1.0,
                     stroke_color: Any = None, stroke_lw: float = 2.0,
                     label: Any = None, label_side: str = 'auto',
                     label_offset: float = 3.0, label_color: Any = None,
                     label_fontsize: Any = None,
                     label_kwargs: dict[str, Any] | None = None,
                     zorder: int = 10) -> AnchoredOffsetbox:
    """Procedurally draw an observatory-dome site marker.

    Shape: a rectangular **building** base, a semicircular **dome**
    on top, and a **slit** at the top of the dome. The slit is
    rendered as a vertical rectangle whose visible width is the
    cos-projection of ``slit_azim`` onto the viewing plane — a
    side-view interpretation of the dome's azimuthal rotation:

    * ``slit_azim=0`` → slit faces the viewer head-on, full width.
    * ``slit_azim=±45°`` → slit appears ~70% as wide.
    * ``slit_azim=±90°`` → slit perpendicular to viewer, invisible.
    * ``|slit_azim| > 90°`` → slit is on the back of the dome,
      hidden; only the dome shell is drawn.

    ``rotation`` rotates the entire sprite CCW (surface-tangent
    orientation on globe views) and is independent of ``slit_azim``.

    Same anchoring / coord-resolution conventions as
    :func:`add_antenna_marker`.

    Parameters
    ----------
    ax : matplotlib Axes
    coord : (x, y) or SkyCoord
    slit_azim : float
        Rotation of the slit from the vertical (degrees, CCW).
        Default ``0``.
    rotation : float
        Extra CCW rotation of the whole sprite (degrees).
    size : float
        Overall sprite size in display points.
    coord_type, frame, face_color, edge_color, edge_lw, alpha, stroke_color, stroke_lw, label, label_side, label_offset, label_color, label_fontsize, label_kwargs, zorder :
        See :func:`add_antenna_marker`.

    Returns
    -------
    anchor : matplotlib.offsetbox.AnchoredOffsetbox
        See :func:`add_antenna_marker` for the ``label_artist`` and
        ``anchors`` attributes (base-anchored; ``.anchors`` gives base/pivot).

    Examples
    --------
    >>> add_dome_marker(ax, (155, 19), coord_type='world',
    ...                 slit_azim=30, face_color='whitesmoke')
    >>> add_dome_marker(ax, (155, 19), coord_type='world',
    ...                 slit_azim=30, label='UKIRT')
    """
    R = _rotation_matrix(rotation)
    # Base-anchored: foot at the rotation origin (= placed coord), like the
    # antenna, so a row of markers plant their feet on one ground line.
    area = 2.0 * size
    cx = cy = area / 2.0

    # --- Building base (rectangle at the bottom, ~30% tall, 70% wide)
    base_h = size * 0.28
    base_w = size * 0.68
    by0 = 0.0
    base_pts = np.array([
        (-base_w / 2, by0),
        (+base_w / 2, by0),
        (+base_w / 2, by0 + base_h),
        (-base_w / 2, by0 + base_h),
    ])
    base_pts = base_pts @ R.T + [cx, cy]
    base = Polygon(base_pts, closed=True, facecolor=face_color,
                    edgecolor=edge_color, linewidth=edge_lw,
                    alpha=alpha, zorder=zorder)

    # --- Dome: a clean filled semicircle on top of the base,
    # rendered as a single PathPatch with a curved upper arc and a
    # straight chord at the bottom (which is hidden against the
    # building base, so no visible horizontal line through the dome).
    # Built from a quartic-Bezier approximation of a half circle plus
    # a closing chord.
    dome_r = size * 0.34
    dome_cx, dome_cy = 0.0, by0 + base_h
    center_arr = np.array([dome_cx, dome_cy])

    # Approximate the half-circle with 8 cubic Bezier segments via
    # matplotlib's Path.arc helper; the result is a closed semicircle
    # path. ``Path.arc(theta1, theta2)`` returns the arc as a Path on
    # the unit circle centered at the origin.
    arc_unit = Path.arc(0.0, 180.0)
    # Scale by dome_r, translate to (dome_cx, dome_cy), then apply
    # outer rotation + sprite-center offset.
    arc_verts = np.asarray(arc_unit.vertices) * dome_r + center_arr
    arc_verts = arc_verts @ R.T + [cx, cy]
    dome_verts = np.vstack([arc_verts, arc_verts[0]])
    dome_codes = np.concatenate([arc_unit.codes, [Path.CLOSEPOLY]])
    dome = PathPatch(Path(dome_verts, dome_codes),
                      facecolor=face_color, edgecolor=edge_color,
                      linewidth=edge_lw, alpha=alpha, zorder=zorder)

    # --- Slit: a side-view dome opens by rotating azimuthally around
    # its vertical axis. A slit at azimuth θ projects to:
    #   * base point at x = R·sin θ on the dome's equator (i.e. the
    #     bottom of the dome on the side view), shifted away from the
    #     center as θ grows;
    #   * apex still at the dome's top (0, R) — all meridians
    #     converge at the apex regardless of azimuth;
    #   * base width = slit_w_full·cos θ, tapering to zero at the
    #     apex (foreshortening of constant-angular-width slit).
    # The slit's left/right edges follow quarter-ellipse meridian
    # projections, rendered as quadratic Bezier curves through a
    # bounding-box-corner control point. For |θ| > 90° the slit is on
    # the back of the dome (cos < 0) and isn't drawn.
    slit_w_full = size * 0.10
    azim_rad = np.deg2rad(float(slit_azim))
    cos_az = float(np.cos(azim_rad))
    sin_az = float(np.sin(azim_rad))
    patches = [base, dome]
    if cos_az > 0.01:  # slit on visible (front) hemisphere
        slit_w_base = slit_w_full * cos_az
        slit_x_base = dome_r * sin_az
        # Base-left and base-right anchored on the dome's equator
        # (the chord at y = 0 in the dome's local frame).
        base_left = np.array([slit_x_base - slit_w_base / 2, 0.0])
        base_right = np.array([slit_x_base + slit_w_base / 2, 0.0])
        apex_pt = np.array([0.0, dome_r])
        # Quadratic-Bezier control points biased to the bounding-box
        # corner — this approximates the meridian-projection
        # quarter-ellipse and makes the slit appear to follow the
        # dome's curvature as θ → 90°.
        left_ctrl = np.array([slit_x_base - slit_w_base / 2, dome_r])
        right_ctrl = np.array([slit_x_base + slit_w_base / 2, dome_r])
        slit_local = np.stack([
            base_left, left_ctrl, apex_pt,
            right_ctrl, base_right, base_left,
        ])
        slit_pts = slit_local + center_arr
        slit_pts = slit_pts @ R.T + [cx, cy]
        slit_codes = [
            Path.MOVETO,
            Path.CURVE3, Path.CURVE3,
            Path.CURVE3, Path.CURVE3,
            Path.CLOSEPOLY,
        ]
        slit = PathPatch(Path(slit_pts, slit_codes),
                          facecolor=edge_color, edgecolor=edge_color,
                          linewidth=edge_lw, alpha=alpha,
                          zorder=zorder + 1)
        patches.append(slit)

    for p in patches:
        _apply_stroke(p, stroke_color, stroke_lw, edge_lw)

    _lbl_color = label_color if label_color is not None else edge_color
    return _add_anchored_patches(
        ax, coord, coord_type, frame, patches, area, zorder,
        label=label, label_side=label_side, label_offset=label_offset,
        label_color=_lbl_color, label_fontsize=label_fontsize,
        label_path_effects=(_stroke_path_effects(stroke_color, stroke_lw)
                            if stroke_color is not None else None),
        label_kwargs=label_kwargs,
        marker_size=size,
        pivot_offset=_pivot_offset(rotation, dome_cy))
