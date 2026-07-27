"""Twin radial-axis support for cone wedge plots.

``make_twinr`` adds a secondary radial axis with a user-supplied
forward/inverse conversion (e.g. redshift <-> comoving distance) on the
opposite wedge edge.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib as mpl
import numpy as np

from .labels import (
    _normalize_rot_for_readability,
    _RayPadRecomputer,
    _VerticalPadRecomputer,
)


def make_twinr(ax: Any, convert: Callable[..., Any],
               inverse: Callable[..., Any] | None = None, *,
               radial_axis_side: str = 'right', r_label: str | None = None,
               r_label_align: str = 'ray', r_label_flip: bool = False,
               r_label_offset: float = 0.20,
               r_tick_spacing: float | None = None,
               label_fontsize: int = 11, tick_fontsize: int = 9,
               color: Any = None,
               restrict_parent_ticks: bool = True) -> Any:
    """
    Create a twin radial axis that shares the wedge geometry but shows a
    different radial scale on the opposite edge.

    This is the polar analogue of :meth:`matplotlib.axes.Axes.twinx` /
    :meth:`~matplotlib.axes.Axes.secondary_yaxis`. Matplotlib's own
    ``secondary_yaxis`` does technically work on polar axes, but it
    renders the secondary ticks along the **cartesian right edge of the
    subplot bbox** rather than along the wedge ray — not what astronomy
    users want. This helper overlays a second polar Axes at the same
    figure rectangle and puts its radial ticks on the chosen side of the
    wedge instead.

    Parameters
    ----------
    ax : PolarAxes
        The primary axes, typically returned by :func:`make_cone_frame`.
    convert : callable
        Forward conversion ``primary_r → twin_r``. For the cosmology
        case, something like
        ``lambda z: Planck18.comoving_distance(z).to(u.Mpc).value``.
    inverse : callable or None
        Inverse conversion ``twin_r → primary_r``. Optional; only needed
        if you want zoom/pan on the twin axes to propagate back to the
        primary (rare for non-interactive plots). For astropy cosmology
        use ``astropy.cosmology.z_at_value`` wrapped in a lambda. If
        ``None``, zoom propagation is one-way (primary → twin only).
    radial_axis_side : {'left', 'right'}
        Side of the wedge on which to display the twin's radial ticks
        and axis label. Default ``'right'``, opposite the typical
        primary side of ``'left'``.
    r_label : str or None
        Axis label for the twin's radial axis, e.g. ``'Comoving distance
        [Mpc]'``. No auto-generation here — caller supplies whatever
        the conversion represents.
    r_label_align : {'vertical', 'ray'}
        Same meaning as in :func:`make_cone_frame`.
    r_label_offset : float
        Distance of the axis label outside the axes box. Default 0.20;
        may need increasing for long conversions whose tick labels have
        many digits. In ``'ray'`` alignment it is a fraction of the
        wedge radius; in ``'vertical'`` alignment, an axes fraction.
    r_tick_spacing : float or None
        Explicit tick spacing in the twin's units (e.g. ``100`` for
        every 100 Mpc). ``None`` uses matplotlib defaults.
    label_fontsize, tick_fontsize : int
        Font sizes.

    Returns
    -------
    twin : matplotlib.projections.polar.PolarAxes
        The overlay axes. Plot data on this for the secondary scale, or
        (more commonly) ignore it after creation — the ticks and label
        are its sole purpose.

        Attached attributes:

        * ``twin._cone_twin_parent`` — the primary axes.
        * ``twin._cone_convert`` — the forward conversion.
        * ``twin._cone_inverse`` — the inverse (or None).

    Notes
    -----
    Zoom / pan handling. A callback is registered on the primary's
    ``'ylim_changed'`` so that when ``ax.set_rmin`` / ``ax.set_rmax`` is
    called, the twin's limits update via ``convert``. If ``inverse`` is
    provided, the reverse callback is registered too; a re-entrance
    guard prevents infinite recursion.

    Layout caveats.

    * The overlay axes is added at ``ax.get_position()`` and stays in
      sync under normal figure resizing. If you call
      ``ax.set_position`` manually afterwards, also call
      ``twin.set_position(ax.get_position())``.
    * Do not call ``tight_layout`` *after* creating the twin —
      matplotlib may reposition the overlay inconsistently. Call it
      before, or use ``constrained_layout`` from the start.
    * For best results keep ``radial_axis_side`` on the primary and the
      twin on *different* sides; putting both on the same side stacks
      two sets of tick labels.

    Examples
    --------
    Redshift primary, comoving-distance twin::

        from astropy.cosmology import Planck18
        import astropy.units as u

        ax = make_cone_frame(111, angle_center=180, angle_half_width=40,
                             r_min=0, r_max=0.2, radial_axis_side='left')
        cone_scatter(ax, ra, z, s=3)

        twin = make_twinr(
            ax,
            convert=lambda z: Planck18.comoving_distance(z).to(u.Mpc).value,
            radial_axis_side='right',
            r_label='Comoving distance [Mpc]',
            r_tick_spacing=100,
        )
    """
    if radial_axis_side not in ('left', 'right'):
        raise ValueError(f"radial_axis_side={radial_axis_side!r} must be "
                          "'left' or 'right'.")

    fig = ax.figure
    # Force layout so ax.get_position() returns the rendered position
    # (not the original requested position from subplot_spec). After
    # subplots_adjust runs, those can differ.
    fig.canvas.draw()

    # Overlay axes at the SAME rectangle. frame_on=False suppresses the
    # duplicated wedge outline; patch.set_visible(False) prevents a
    # second opaque background covering the primary.
    twin = fig.add_axes(ax.get_position(),
                         projection='polar',
                         zorder=ax.get_zorder() + 0.5,
                         frame_on=False)
    twin.patch.set_visible(False)

    # Make the twin's position automatically track the parent's rendered
    # position — matplotlib resolves the locator on every layout pass,
    # which is exactly what we want for subplots_adjust / tight_layout /
    # interactive resizing.
    def _twin_locator(twin_ax: Any, renderer: Any) -> Any:
        return ax.get_position()
    twin.set_axes_locator(_twin_locator)

    # Copy the primary's angular geometry so the wedge outlines align.
    twin.set_thetamin(ax.get_thetamin())
    twin.set_thetamax(ax.get_thetamax())
    twin.set_theta_direction(ax.get_theta_direction())
    twin.set_theta_offset(ax.get_theta_offset())

    # Copy r_origin if present. `get_rorigin` exists on modern matplotlib;
    # guard for older versions just in case.
    try:
        primary_rorigin = ax.get_rorigin()
        twin.set_rorigin(convert(primary_rorigin)
                         if primary_rorigin is not None else 0.0)
    except AttributeError:
        pass

    # Set twin's r range via the conversion.
    r0, r1 = ax.get_rmin(), ax.get_rmax()
    twin.set_rmin(float(convert(r0)))
    twin.set_rmax(float(convert(r1)))

    # Suppress decorations already provided by the primary.
    twin.grid(False)                 # no double grid
    twin.set_xticks([])              # no double angular ticks
    twin.set_xticklabels([])         # and no labels if default ticks exist

    # Radial ticks on the requested side of the wedge.
    # Set rlabel_position so the tick labels sit along the chosen edge
    # of the wedge (same logic as the primary; small inset keeps them
    # inside the outline).
    half_width_deg = 0.5 * (twin.get_thetamax() - twin.get_thetamin())
    inset = min(1.0, half_width_deg * 0.05)

    # Detect if the twin's polar internal coords are physically inverted
    # (zero='S' direction=-1, etc.). Same logic as in make_cone_frame —
    # the twin inherits the parent's geometry. effective_side flips the
    # user's requested physical side to the matplotlib internal side.
    twin_dir = twin.get_theta_direction()
    twin_offset_rad = twin.get_theta_offset()
    # Convert offset to a cardinal-style boolean: is theta=0 pointing
    # south (offset roughly -pi/2 or +3pi/2)?
    south = abs(np.cos(twin_offset_rad) - 0) < 0.5 and \
            np.sin(twin_offset_rad) < 0
    north = abs(np.cos(twin_offset_rad) - 0) < 0.5 and \
            np.sin(twin_offset_rad) > 0
    inverted = ((south and twin_dir == -1) or
                (north and twin_dir == +1))
    effective_side = (
        ('right' if radial_axis_side == 'left' else 'left')
        if inverted else radial_axis_side)

    # Match the primary's radial ticks: geometry follows the active base
    # style (see make_cone_frame), so a 'structural' base gives the twin
    # inward ticks too rather than a lone outward axis beside the primary.
    twin_kwargs = dict(axis='y', labelsize=tick_fontsize,
                       length=mpl.rcParams['ytick.major.size'],
                       width=mpl.rcParams['ytick.major.width'],
                       direction=mpl.rcParams['ytick.direction'])
    if color is not None:
        twin_kwargs['color'] = color
        twin_kwargs['labelcolor'] = color
    if effective_side == 'right':
        twin.set_rlabel_position(+(half_width_deg - inset))
        twin_kwargs.update(labelleft=False, labelright=True,
                            left=False, right=True)
    else:
        twin.set_rlabel_position(-(half_width_deg - inset))
        twin_kwargs.update(labelleft=True, labelright=False,
                            left=True, right=False)
    twin.tick_params(**twin_kwargs)

    # Restrict the PARENT's tick marks to its own side, so primary and
    # twin don't both render ticks at the same edge. Only labels are
    # already side-restricted in the parent; we additionally hide the
    # parent's tick MARKS on the side now occupied by the twin.
    # Use parent's effective_side via _cone_radial_axis_side, but apply
    # the same inversion check for the parent.
    if restrict_parent_ticks:
        parent_side = getattr(ax, '_cone_radial_axis_side', 'left')
        # Determine parent's effective_side. The parent's geometry is
        # the same as twin's, so the same `inverted` flag applies.
        parent_effective = (
            ('right' if parent_side == 'left' else 'left')
            if inverted else parent_side)
        ax.tick_params(axis='y',
                       left=(parent_effective == 'left'),
                       right=(parent_effective == 'right'))

    if r_tick_spacing is not None:
        tmin, tmax = twin.get_rmin(), twin.get_rmax()
        twin.set_yticks(np.arange(tmin,
                                    tmax + 0.5 * r_tick_spacing,
                                    r_tick_spacing))

    # Radial axis label. For 'ray' alignment, place at the midpoint of
    # the wedge edge (matching _place_axis_labels). For 'vertical',
    # keep at axes y=0.5 (current behavior, which is already at the
    # bbox center which approximately equals the wedge midpoint for
    # vertical labels). Store the resulting Text artist on the twin
    # axes as `_cone_r_label_text` so the user can call
    # :func:`flip_label` on it later, or manipulate position/rotation
    # directly via the standard matplotlib Text API.
    half_width = 0.5 * (twin.get_thetamax() - twin.get_thetamin())
    twin_label_text = None
    if r_label:
        twin_text_kwargs = dict(ha='center', va='center',
                                  rotation_mode='anchor',
                                  fontsize=label_fontsize)
        if color is not None:
            twin_text_kwargs['color'] = color

        if r_label_align == 'ray':
            fig.canvas.draw()  # ensure transData is calibrated
            half_width_rad = np.radians(half_width)
            r_min_t = twin.get_rmin()
            r_max_t = twin.get_rmax()
            mid_r = 0.5 * (r_min_t + r_max_t)

            # Same inversion logic as in _place_axis_labels: detect
            # zero='S' direction=-1 (or zero='N' direction=+1) which
            # flips the meaning of physical-left/right relative to
            # matplotlib's internal theta. We infer the inversion from
            # twin's theta_offset and theta_direction since
            # zero_location isn't passed in directly.
            twin_dir = twin.get_theta_direction()
            twin_offset_rad = twin.get_theta_offset()
            south = (abs(np.cos(twin_offset_rad)) < 0.5 and
                     np.sin(twin_offset_rad) < 0)
            north = (abs(np.cos(twin_offset_rad)) < 0.5 and
                     np.sin(twin_offset_rad) > 0)
            inverted = ((south and twin_dir == -1) or
                        (north and twin_dir == +1))
            effective_side = (
                ('right' if radial_axis_side == 'left' else 'left')
                if inverted else radial_axis_side)
            edge_theta = (-half_width_rad if effective_side == 'left'
                          else +half_width_rad)
            mid_disp = twin.transData.transform([[edge_theta, mid_r]])[0]
            outer_offset = 0.05 * (r_max_t - r_min_t)
            outer_disp = twin.transData.transform(
                [[edge_theta, mid_r + outer_offset]])[0]
            edge_vec = outer_disp - mid_disp
            edge_norm = np.hypot(edge_vec[0], edge_vec[1])
            if edge_norm > 1e-12:
                edge_unit = edge_vec / edge_norm
                if effective_side == 'left':
                    outward = np.array([-edge_unit[1], edge_unit[0]])
                else:
                    outward = np.array([edge_unit[1], -edge_unit[0]])
                rot = np.degrees(np.arctan2(edge_unit[1], edge_unit[0]))
            else:
                outward = (np.array([-1.0, 0.0])
                           if effective_side == 'left'
                           else np.array([1.0, 0.0]))
                rot = (90.0 + half_width if effective_side == 'left'
                       else 90.0 - half_width)

            # Use the wedge radius in display pixels as the natural
            # scale for the offset (consistent with _place_axis_labels).
            try:
                rorigin_t = twin.get_rorigin()
            except AttributeError:
                rorigin_t = r_min_t
            arc_disp_t = twin.transData.transform([[0.0, r_max_t]])[0]
            apex_disp_t = twin.transData.transform([[0.0, rorigin_t]])[0]
            radius_px = np.hypot(arc_disp_t[0] - apex_disp_t[0],
                                  arc_disp_t[1] - apex_disp_t[1])
            offset_px = r_label_offset * radius_px
            text_disp = mid_disp + outward * offset_px
            text_axes = twin.transAxes.inverted().transform([text_disp])[0]

            rot_final, did_flip = _normalize_rot_for_readability(rot)
            if r_label_flip:
                rot_final = (rot_final + 180) % 360 - 180
            twin_label_text = twin.text(
                text_axes[0], text_axes[1], r_label,
                transform=twin.transAxes, rotation=rot_final,
                **twin_text_kwargs)
            twin_label_text._cone_pad = float(r_label_offset)
            twin_label_text._cone_pad_recompute = _RayPadRecomputer(
                twin, edge_theta=edge_theta, mid_r=mid_r,
                effective_side=effective_side, rorigin=rorigin_t,
                r_max=r_max_t)
        else:  # 'vertical' — keep older behavior
            rot = 90.0 if radial_axis_side == 'left' else -90.0
            if r_label_flip:
                rot = (rot + 180) % 360 - 180
            if radial_axis_side == 'left':
                x = -r_label_offset
            else:
                x = 1.0 + r_label_offset
            twin_label_text = twin.text(
                x, 0.5, r_label, transform=twin.transAxes,
                rotation=rot, **twin_text_kwargs)
            twin_label_text._cone_pad = float(r_label_offset)
            twin_label_text._cone_pad_recompute = _VerticalPadRecomputer(
                radial_axis_side)
    twin._cone_r_label_text = twin_label_text

    # Synchronization setup.
    #
    # matplotlib's polar `set_rmax`/`set_rmin` bypass the `ylim_changed`
    # callback system, so we monkey-patch the methods on the ax INSTANCE
    # (not the class) to sync the twin whenever they're called. We also
    # connect a traditional ylim_changed callback for `set_ylim` usage.
    # A re-entrance guard prevents the primary-to-twin-to-primary infinite
    # loop when `inverse` is provided.
    _guard = {'syncing': False}

    def _forward_sync(r0: float, r1: float) -> None:
        """Update twin r limits from primary r limits."""
        if _guard['syncing']:
            return
        _guard['syncing'] = True
        try:
            twin.set_rmin(float(convert(r0)))
            twin.set_rmax(float(convert(r1)))
            # Also call set_ylim in case tick rebuild depends on it:
            twin.set_ylim(float(convert(r0)), float(convert(r1)))
        finally:
            _guard['syncing'] = False

    def _reverse_sync(t0: float, t1: float) -> None:
        """Update primary r limits from twin r limits (if inverse given)."""
        if _guard['syncing'] or inverse is None:
            return
        _guard['syncing'] = True
        try:
            ax.set_rmin(float(inverse(t0)))
            ax.set_rmax(float(inverse(t1)))
            ax.set_ylim(float(inverse(t0)), float(inverse(t1)))
        finally:
            _guard['syncing'] = False

    # Method wrappers on the primary instance.
    _orig_primary_set_rmax = ax.set_rmax
    _orig_primary_set_rmin = ax.set_rmin

    def _wrapped_primary_set_rmax(value: float, *a: Any, **kw: Any) -> Any:
        result = _orig_primary_set_rmax(value, *a, **kw)
        _forward_sync(ax.get_rmin(), value)
        return result

    def _wrapped_primary_set_rmin(value: float, *a: Any, **kw: Any) -> Any:
        result = _orig_primary_set_rmin(value, *a, **kw)
        _forward_sync(value, ax.get_rmax())
        return result

    ax.set_rmax = _wrapped_primary_set_rmax
    ax.set_rmin = _wrapped_primary_set_rmin

    # The ylim_changed callback still fires for `ax.set_ylim(rmin, rmax)`.
    ax.callbacks.connect(
        'ylim_changed',
        lambda a: _forward_sync(a.get_rmin(), a.get_rmax()))

    # Reverse direction — only if inverse provided.
    if inverse is not None:
        _orig_twin_set_rmax = twin.set_rmax
        _orig_twin_set_rmin = twin.set_rmin

        def _wrapped_twin_set_rmax(value: float, *a: Any, **kw: Any) -> Any:
            result = _orig_twin_set_rmax(value, *a, **kw)
            _reverse_sync(twin.get_rmin(), value)
            return result

        def _wrapped_twin_set_rmin(value: float, *a: Any, **kw: Any) -> Any:
            result = _orig_twin_set_rmin(value, *a, **kw)
            _reverse_sync(value, twin.get_rmax())
            return result

        twin.set_rmax = _wrapped_twin_set_rmax
        twin.set_rmin = _wrapped_twin_set_rmin

        twin.callbacks.connect(
            'ylim_changed',
            lambda a: _reverse_sync(a.get_rmin(), a.get_rmax()))

    # Metadata.
    twin._cone_twin_parent = ax
    twin._cone_convert = convert
    twin._cone_inverse = inverse

    return twin

