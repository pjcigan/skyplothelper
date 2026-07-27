"""Label flipping, padding, and placement for cone wedge plots.

The pad-recomputer classes (``_RayPadRecomputer`` etc.) are matplotlib
draw-event callbacks that update tick-label padding to track the wedge
geometry as the figure is resized. ``flip_label`` and the
``set_label_pad`` / ``get_label_pad`` accessors are the public surface.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt  # noqa: F401  (used in some helpers)
import matplotlib.transforms as mtransforms  # noqa: F401
import numpy as np


def _normalize_rot_for_readability(rot_deg: float) -> tuple[float, bool]:
    """
    Reduce a rotation angle to the (-90°, 90°] range.

    Text rotated outside this range reads bottom-to-top or upside-down,
    which is generally less readable. By default we flip such angles by
    180° so the text reads in the conventional direction along the
    same line. The user can pass ``..._flip=True`` to invert this and
    keep the original angle.

    Returns the (possibly flipped) rotation and a bool ``did_flip``.
    """
    rot = ((rot_deg + 180) % 360) - 180  # → (-180, 180]
    if rot > 90 or rot <= -90:
        return rot + 180 if rot <= -90 else rot - 180, True
    return rot, False


# ---------------------------------------------------------------------------
# Pad recomputers — small helper objects that re-place a label artist
# when the user changes its pad value via :func:`set_label_pad`.
#
# Each label artist gets a `_cone_pad_recompute` attribute pointing to
# one of these objects, plus a `_cone_pad` float (the current pad). The
# recomputer's `__call__(text, new_pad)` re-positions the artist to
# match `new_pad` while preserving everything else (rotation, text,
# style).
# ---------------------------------------------------------------------------

class _RayPadRecomputer:
    """For the radial axis label in 'ray' alignment mode."""
    def __init__(self, ax: Any, *, edge_theta: float, mid_r: float,
                  effective_side: str, rorigin: float, r_max: float) -> None:
        self.ax = ax
        self.edge_theta = edge_theta
        self.mid_r = mid_r
        self.effective_side = effective_side
        self.rorigin = rorigin
        self.r_max = r_max

    def __call__(self, text: Any, new_pad: float) -> None:
        ax = self.ax
        ax.figure.canvas.draw()
        mid_disp = ax.transData.transform(
            [[self.edge_theta, self.mid_r]])[0]
        outer_offset = 0.05 * (self.r_max - self.rorigin)
        outer_disp = ax.transData.transform(
            [[self.edge_theta, self.mid_r + outer_offset]])[0]
        edge_vec = outer_disp - mid_disp
        edge_norm = np.hypot(edge_vec[0], edge_vec[1])
        if edge_norm > 1e-12:
            edge_unit = edge_vec / edge_norm
            if self.effective_side == 'left':
                outward = np.array([-edge_unit[1], edge_unit[0]])
            else:
                outward = np.array([edge_unit[1], -edge_unit[0]])
        else:
            outward = (np.array([-1.0, 0.0])
                       if self.effective_side == 'left'
                       else np.array([1.0, 0.0]))
        # Wedge radius in display px.
        arc_disp = ax.transData.transform([[0.0, self.r_max]])[0]
        apex_disp = ax.transData.transform([[0.0, self.rorigin]])[0]
        radius_px = np.hypot(arc_disp[0] - apex_disp[0],
                              arc_disp[1] - apex_disp[1])
        offset_px = new_pad * radius_px
        text_disp = mid_disp + outward * offset_px
        text_axes = ax.transAxes.inverted().transform([text_disp])[0]
        text.set_position((text_axes[0], text_axes[1]))


class _AnglePadRecomputer:
    """For the angular axis label in 'tangent' alignment mode."""
    def __init__(self, ax: Any, *, anchor_polar: tuple[float, float],
                  outward_polar: tuple[float, float]) -> None:
        self.ax = ax
        self.anchor_polar = anchor_polar      # (theta, r) of arc midpoint
        self.outward_polar = outward_polar    # (theta, r) of apex (for outward dir)

    def __call__(self, text: Any, new_pad: float) -> None:
        ax = self.ax
        ax.figure.canvas.draw()
        arc_disp = ax.transData.transform([self.anchor_polar])[0]
        apex_disp = ax.transData.transform([self.outward_polar])[0]
        radius_px = np.hypot(arc_disp[0] - apex_disp[0],
                              arc_disp[1] - apex_disp[1])
        outward_vec = arc_disp - apex_disp
        outward_norm = np.hypot(outward_vec[0], outward_vec[1])
        if outward_norm > 1e-12:
            outward = outward_vec / outward_norm
        else:
            outward = np.array([0.0, 1.0])
        offset_px = new_pad * radius_px
        text_disp = arc_disp + outward * offset_px
        text_axes = ax.transAxes.inverted().transform([text_disp])[0]
        text.set_position((text_axes[0], text_axes[1]))


class _VerticalPadRecomputer:
    """For the radial axis label in 'vertical' alignment mode (axes-fraction)."""
    def __init__(self, radial_axis_side: str) -> None:
        self.side = radial_axis_side

    def __call__(self, text: Any, new_pad: float) -> None:
        if self.side == 'left':
            text.set_position((-new_pad, 0.5))
        else:
            text.set_position((1.0 + new_pad, 0.5))


class _HorizPadRecomputer:
    """For the angular label in 'horizontal' mode (xaxis_text1_transform)."""
    def __init__(self, ax: Any, *, kind: str) -> None:
        self.ax = ax
        self.kind = kind  # 'angle'

    def __call__(self, text: Any, new_pad: float) -> None:
        text.set_position((0, -new_pad))


class _EWHorizPadRecomputer:
    """For the radial label on E/W frames in 'vertical' mode (above/below)."""
    def __init__(self, radial_axis_side: str) -> None:
        self.side = radial_axis_side

    def __call__(self, text: Any, new_pad: float) -> None:
        if self.side == 'left':
            text.set_position((0.5, 1.0 + new_pad))
        else:
            text.set_position((0.5, 0.0 - new_pad))


def set_label_pad(text: Any, pad: float) -> None:
    """
    Adjust the pad (offset distance) of a cone-frame axis label.

    This is the cone-frame analogue of matplotlib's
    :attr:`~matplotlib.axis.Axis.labelpad`. It re-places the label
    artist at the given pad while preserving its rotation, text, and
    style.

    The pad is interpreted as a fraction of the wedge radius (in
    display pixels) for ``'ray'`` and ``'tangent'`` alignment modes,
    and as a fraction of axes width for the legacy
    ``'vertical'``/``'horizontal'`` modes — i.e. the same units as
    the original ``r_label_offset`` / ``angle_label_outside`` the
    label was created with. Default values are 0.20 (radial) and 0.18
    (angular); larger values push the label further from the tick
    labels, smaller bring it closer.

    Parameters
    ----------
    text : matplotlib.text.Text or None
        A label artist created by :func:`make_cone_frame` or
        :func:`make_twinr`, e.g. ``ax._cone_r_label_text`` or
        ``ax._cone_angle_label_text``. Pass ``None`` for a no-op.
    pad : float
        The new pad value.

    Examples
    --------
    Increase a bowtie's radial label pad to clear longer tick labels::

        top, bot = make_bowtie_frame(angle_half_width=40, r_max=0.15)
        for half in (top, bot):
            set_label_pad(half._cone_r_label_text, 0.28)
            set_label_pad(half._cone_angle_label_text, 0.22)

    Tighten a single-cone frame's labels::

        ax = make_cone_frame(angle_half_width=40, r_max=0.15)
        set_label_pad(ax._cone_r_label_text, 0.12)
        set_label_pad(ax._cone_angle_label_text, 0.10)

    See Also
    --------
    flip_label : toggle a label's reading direction by 180°.
    make_cone_frame : ``r_label_offset`` and ``angle_label_outside``
        set the initial pad at construction time.
    """
    if text is None:
        return
    recomputer = getattr(text, '_cone_pad_recompute', None)
    if recomputer is None:
        # Label wasn't created by cone_frame, or the cache attr is
        # missing — fall back to no-op rather than crashing.
        return
    recomputer(text, float(pad))
    text._cone_pad = float(pad)


def get_label_pad(text: Any) -> float | None:
    """
    Return the current pad value of a cone-frame axis label.

    Parameters
    ----------
    text : matplotlib.text.Text or None

    Returns
    -------
    float or None
        The current pad, or ``None`` if no pad info is cached.
    """
    if text is None:
        return None
    return getattr(text, '_cone_pad', None)


def _place_axis_labels(ax: Any, r_label: str | None, angle_label: str | None,
                        fontsize: int,
                        half_width_deg: float, zero_location: str,
                        radial_axis_side: str = 'left',
                        r_label_offset: float = 0.20,
                        r_label_align: str = 'ray',
                        angle_label_outside: float = 0.18,
                        r_label_color: Any = None,
                        r_label_flip: bool = False,
                        angle_label_align: str = 'tangent',
                        angle_label_flip: bool = False) -> tuple[Any, Any]:
    """Add text labels for the radial and angular axes outside the wedge.

    Both labels by default rotate to follow the wedge's frame edge: the
    radial label runs parallel to the slanted radial edge at the edge
    midpoint, and the angular label runs tangent to the outer arc at
    the arc's midpoint (theta=0). This matches the cartesian-axis
    convention where labels follow the axis direction.

    Parameters
    ----------
    r_label_align : {'ray', 'vertical'}
        ``'ray'`` (default) puts the label parallel to the slanted
        radial edge, centered on the edge midpoint. ``'vertical'``
        keeps the older behavior (rotation ±90°, at axes y=0.5).
    r_label_flip : bool
        Default ``False``: the label's rotation is normalized to be
        readable (auto-flipped 180° if it would otherwise read
        upside-down). Set ``True`` to invert this — useful when the
        auto-readable direction isn't the one you want.
    angle_label_align : {'tangent', 'horizontal'}
        ``'tangent'`` (default) makes the angular label parallel to
        the outer arc tangent at theta=0, so the label rotates with
        the frame. ``'horizontal'`` keeps the label flat (the older
        behavior).
    angle_label_flip : bool
        Same flip semantics as ``r_label_flip``.
    r_label_offset, angle_label_outside : float
        Distance of each label from the corresponding tick labels.
        Both are interpreted as fractions of the axes' width. Default
        ``0.20`` for r and ``0.18`` for angle work well across typical
        wedge geometries; pass larger values to push labels further
        out, smaller to bring them closer.

    Returns
    -------
    (r_label_text, angle_label_text)
        References to the matplotlib ``Text`` artists for each label
        (or ``None`` if the corresponding label was suppressed).
        Stored on ``ax`` as ``ax._cone_r_label_text`` and
        ``ax._cone_angle_label_text`` for later manipulation.
    """
    fig = ax.figure
    # We need transData to be calibrated for both the angular and the
    # radial label placement, since both are positioned via display
    # coords. One draw covers both.
    fig.canvas.draw()

    r_label_text = None
    angle_label_text = None

    # --- Angular axis label -------------------------------------------
    if angle_label:
        if angle_label_align == 'tangent':
            # Position: just outside the arc at theta=0 (which after
            # any zero_offset rotation is the "top center" of the arc).
            # Tangent direction: difference vector along the arc near
            # theta=0. Outward direction: from apex to arc point.
            r_min = ax.get_rmin()
            r_max = ax.get_rmax()
            try:
                rorigin = ax.get_rorigin()
            except AttributeError:
                rorigin = r_min
            arc_disp = ax.transData.transform([[0.0, r_max]])[0]
            # Apex point in display coords.
            apex_disp = ax.transData.transform([[0.0, rorigin]])[0]
            # Wedge radius in display pixels — natural scale for the
            # offset, robust to subplot sizing.
            radius_px = np.hypot(arc_disp[0] - apex_disp[0],
                                  arc_disp[1] - apex_disp[1])
            # Tangent: difference of two points along the arc near theta=0.
            tangent_offset = np.radians(0.5)
            arc_offset_disp = ax.transData.transform(
                [[tangent_offset, r_max]])[0]
            tan_vec = arc_offset_disp - arc_disp
            tan_norm = np.hypot(tan_vec[0], tan_vec[1])
            if tan_norm > 1e-12:
                tan_unit = tan_vec / tan_norm
            else:
                tan_unit = np.array([1.0, 0.0])
            # Outward radial direction: from apex to arc center.
            outward_vec = arc_disp - apex_disp
            outward_norm = np.hypot(outward_vec[0], outward_vec[1])
            if outward_norm > 1e-12:
                outward = outward_vec / outward_norm
            else:
                outward = np.array([0.0, 1.0])
            # Place text outward from arc, rotated to tangent direction.
            # Offset is interpreted as fraction of wedge radius —
            # naturally scales with the wedge size and is robust to
            # nested subplot layouts.
            offset_px = angle_label_outside * radius_px
            text_disp = arc_disp + outward * offset_px
            text_axes = ax.transAxes.inverted().transform([text_disp])[0]
            rot = np.degrees(np.arctan2(tan_unit[1], tan_unit[0]))
            rot_final, did_flip = _normalize_rot_for_readability(rot)
            if angle_label_flip:
                rot_final = (rot_final + 180) % 360 - 180
            angle_label_text = ax.text(
                text_axes[0], text_axes[1], angle_label,
                transform=ax.transAxes, ha='center', va='center',
                rotation=rot_final, rotation_mode='anchor',
                fontsize=fontsize)
            # Cache info needed to recompute the label position when
            # the user calls `set_label_pad` later. We store it on the
            # Text artist itself rather than on `ax` because there
            # might be multiple labels per axes (primary + twin) and
            # this keeps each label self-contained.
            angle_label_text._cone_pad = float(angle_label_outside)
            angle_label_text._cone_pad_recompute = _AnglePadRecomputer(
                ax, anchor_polar=(0.0, r_max),
                outward_polar=(0.0, rorigin))
        else:  # 'horizontal' — keep flat
            trans, _, _ = ax.get_xaxis_text1_transform(0)
            angle_label_text = ax.text(
                0, -angle_label_outside, angle_label,
                transform=trans, ha='center', va='top',
                fontsize=fontsize)
            # 'horizontal' uses a different transform; the pad is the
            # raw axes-fraction offset, applied directly via y.
            angle_label_text._cone_pad = float(angle_label_outside)
            angle_label_text._cone_pad_recompute = _HorizPadRecomputer(
                ax, kind='angle')

    # --- Radial axis label --------------------------------------------
    r_text_kwargs = dict(ha='center', va='center', fontsize=fontsize)
    if r_label_color is not None:
        r_text_kwargs['color'] = r_label_color
    if r_label:
        if zero_location in ('N', 'S') or r_label_align == 'ray':
            # 'ray' mode works for any zero_location since it uses
            # transData directly. For E/W with 'vertical' we still fall
            # back to the old E/W code path below.
            if r_label_align == 'ray':
                half_width_rad = np.radians(half_width_deg)
                r_min = ax.get_rmin()
                r_max = ax.get_rmax()
                try:
                    rorigin = ax.get_rorigin()
                except AttributeError:
                    rorigin = r_min
                mid_r = 0.5 * (r_min + r_max)

                # Inversion: same logic as the tick-side detection in
                # make_cone_frame. For zero='S' direction=-1, the
                # matplotlib internal theta meaning of left/right is
                # flipped relative to the physical screen, so we flip
                # the edge selection accordingly.
                inverted = ((zero_location == 'S' and
                             ax.get_theta_direction() == -1) or
                            (zero_location == 'N' and
                             ax.get_theta_direction() == +1))
                effective_side = (
                    ('right' if radial_axis_side == 'left' else 'left')
                    if inverted else radial_axis_side)
                edge_theta = (-half_width_rad
                              if effective_side == 'left'
                              else +half_width_rad)

                mid_disp = ax.transData.transform(
                    [[edge_theta, mid_r]])[0]
                outer_offset = 0.05 * (r_max - r_min)
                outer_disp = ax.transData.transform(
                    [[edge_theta, mid_r + outer_offset]])[0]

                # Wedge radius in display px — natural scale for offset.
                arc_disp = ax.transData.transform([[0.0, r_max]])[0]
                apex_disp = ax.transData.transform([[0.0, rorigin]])[0]
                radius_px = np.hypot(arc_disp[0] - apex_disp[0],
                                      arc_disp[1] - apex_disp[1])

                edge_vec = outer_disp - mid_disp
                edge_norm = np.hypot(edge_vec[0], edge_vec[1])
                if edge_norm > 1e-12:
                    edge_unit = edge_vec / edge_norm
                    if effective_side == 'left':
                        outward = np.array([-edge_unit[1], edge_unit[0]])
                    else:
                        outward = np.array([edge_unit[1], -edge_unit[0]])
                    rot = np.degrees(np.arctan2(edge_unit[1],
                                                  edge_unit[0]))
                else:
                    outward = (np.array([-1.0, 0.0])
                               if effective_side == 'left'
                               else np.array([1.0, 0.0]))
                    rot = (90.0 + half_width_deg
                           if effective_side == 'left'
                           else 90.0 - half_width_deg)

                # Offset perpendicular to the edge, scaled by wedge
                # radius (in display pixels).
                offset_px = r_label_offset * radius_px
                text_disp = mid_disp + outward * offset_px
                text_axes = ax.transAxes.inverted().transform(
                    [text_disp])[0]

                rot_final, did_flip = _normalize_rot_for_readability(rot)
                if r_label_flip:
                    rot_final = (rot_final + 180) % 360 - 180

                r_label_text = ax.text(
                    text_axes[0], text_axes[1], r_label,
                    transform=ax.transAxes,
                    rotation=rot_final, rotation_mode='anchor',
                    **r_text_kwargs)
                r_label_text._cone_pad = float(r_label_offset)
                r_label_text._cone_pad_recompute = _RayPadRecomputer(
                    ax, edge_theta=edge_theta, mid_r=mid_r,
                    effective_side=effective_side, rorigin=rorigin,
                    r_max=r_max)
            else:  # 'vertical' — old behavior at axes y=0.5
                rot = 90.0 if radial_axis_side == 'left' else -90.0
                if r_label_flip:
                    rot = (rot + 180) % 360 - 180
                if radial_axis_side == 'left':
                    x = -r_label_offset
                else:
                    x = 1.0 + r_label_offset
                r_label_text = ax.text(
                    x, 0.5, r_label, transform=ax.transAxes,
                    rotation=rot, rotation_mode='anchor',
                    **r_text_kwargs)
                r_label_text._cone_pad = float(r_label_offset)
                r_label_text._cone_pad_recompute = _VerticalPadRecomputer(
                    radial_axis_side)
        else:
            # E/W orientations with 'vertical' alignment: horizontal
            # label above or below
            if radial_axis_side == 'left':
                y, va = (1.0 + r_label_offset, 'bottom')
            else:
                y, va = (0.0 - r_label_offset, 'top')
            r_label_text = ax.text(
                0.5, y, r_label, transform=ax.transAxes,
                rotation=0, **{**r_text_kwargs, 'va': va})
            r_label_text._cone_pad = float(r_label_offset)
            r_label_text._cone_pad_recompute = _EWHorizPadRecomputer(
                radial_axis_side)

    # Stash on ax for later manipulation by the user.
    ax._cone_r_label_text = r_label_text
    ax._cone_angle_label_text = angle_label_text
    return r_label_text, angle_label_text


def flip_label(text: Any) -> None:
    """
    Toggle a label's rotation by 180° to flip its reading direction.

    Useful when the auto-readability flip in :func:`make_cone_frame`
    or :func:`make_twinr` doesn't pick the orientation you want, e.g.
    on heavily-rotated frames where the "natural" direction looks
    awkward.

    Parameters
    ----------
    text : matplotlib.text.Text or None
        A label artist (e.g. ``ax._cone_r_label_text``,
        ``ax._cone_angle_label_text``). Pass ``None`` is a no-op.

    Examples
    --------
    ::

        ax = make_cone_frame(111, ..., zero_offset=-60)
        flip_label(ax._cone_r_label_text)        # flip Redshift
        flip_label(ax._cone_angle_label_text)    # flip R.A.
    """
    if text is None:
        return
    text.set_rotation((text.get_rotation() + 180) % 360)


# ---------------------------------------------------------------------------
# Data plotting helpers
# ---------------------------------------------------------------------------
