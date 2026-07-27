"""Cone (z-RA wedge) and bowtie frame builders.

``make_cone_frame`` builds a single-wedge polar Axes for redshift-cone
or angular-cone plots; ``make_bowtie_frame`` builds a back-to-back pair
sharing a common origin. Internal helpers (``_default_r_label``,
``_format_hour``, ``_relabel_angular_ticks_absolute``,
``_angle_to_theta``) are kept private.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms  # noqa: F401  (used for label placement)
import numpy as np
import numpy.typing as npt

from .cosmology import _HAVE_ASTROPY, _R_VARIABLES, redshift_to_r  # noqa: F401
from .labels import _place_axis_labels, set_label_pad  # noqa: F401


def _set_only(**kwargs: Any) -> dict[str, Any]:
    """Drop the keys the caller left at ``None``.

    Lets a wrapper forward "what was actually asked for" instead of
    restating the wrapped function's defaults, which would make those
    defaults unreachable through the wrapper.
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def make_cone_frame(subplot_spec: Any = 111, *,
                    angle_center: float = 0., angle_half_width: float = 45.,
                    r_min: float = 0., r_max: float = 0.2,
                    r_origin: float | None = None,
                    r_variable: str = 'redshift', r_unit: str = 'Mpc',
                    cosmology: Any = None,
                    angle_direction: int = -1, zero_location: str = 'N',
                    zero_offset: float = 0.0,
                    angle_unit: str = 'deg',
                    angle_tick_spacing: float | None = None,
                    r_tick_spacing: float | None = None,
                    r_label: str | None = None, angle_label: str = 'R.A.',
                    label_fontsize: int = 11, tick_fontsize: int = 9,
                    radial_axis_side: str = 'left',
                    rlabel_position: str | float = 'auto',
                    r_label_offset: float = 0.20,
                    r_label_align: str = 'ray', r_label_flip: bool = False,
                    angle_label_align: str = 'tangent',
                    angle_label_flip: bool = False,
                    angle_label_outside: float = 0.18,
                    radial_axis_color: Any = None,
                    grid: bool = True, gridcolor: Any = '0.8',
                    gridalpha: float = 0.5,
                    gridlw: float | None = None,
                    gridls: str | None = None,
                    fig: Any = None) -> Any:
    """
    Build a cosmology-style cone (pie-slice / wedge) polar Axes.

    Creates a matplotlib polar subplot, configures the wedge limits,
    orientation, tick positions, and labels, and returns the Axes. The
    returned Axes object has an attached attribute ``_cone_angle_center``
    (set to ``angle_center``) so that companion functions like
    :func:`cone_scatter` can shift angular inputs into the wedge's
    internal theta coordinate without the caller re-passing it.

    Parameters
    ----------
    subplot_spec : int or matplotlib.gridspec.SubplotSpec
        Where to place the axes. Same meaning as the first argument to
        :func:`matplotlib.figure.Figure.add_subplot`.
    angle_center : float
        Center of the wedge, in the unit given by ``angle_unit`` (default
        degrees). Accepts any spherical longitude — equatorial R.A.,
        galactic *l*, ecliptic longitude, etc. For ``angle_unit='hour'``
        (equatorial R.A. convention), pass in decimal hours
        (e.g. ``12.5`` for ``12h30m``); internally converted to degrees.
    angle_half_width : float
        Half-width of the wedge in the same unit as ``angle_center``. The
        wedge spans ``angle_center ± angle_half_width``.
    r_min, r_max : float
        Inner and outer radial limits. Units match ``r_variable``:

        * ``'redshift'``: dimensionless z.
        * ``'comoving_distance'``: units of ``r_unit`` (default Mpc).
        * ``'lookback_time'``: units of ``r_unit`` (typically 'Gyr').
    r_origin : float or None
        Radial coordinate of the cone tip (wedge vertex). Default
        ``None`` means the tip sits at ``r_min`` — a truncated wedge.
        Set explicitly (typically ``r_origin=0``) to decouple the tip
        from ``r_min``: the wedge becomes annular, with its inner edge
        at ``r_min`` and its geometric apex still at ``r_origin``.
        Useful for plots like "survey volume from z=0.05 to 0.2 with
        the observer position shown at the tip."
        Passes through to :meth:`matplotlib.projections.polar.PolarAxes.set_rorigin`.
    r_variable : {'redshift', 'comoving_distance', 'lookback_time'}
        What the radial axis represents. This affects the default
        ``r_label`` and the radial tick formatting.
    r_unit : str or astropy.units.Unit
        Display unit for distance or time axes. Ignored for redshift.
    cosmology : astropy.cosmology instance or None
        Attached as ``ax._cone_cosmology`` for optional use by
        :func:`cone_scatter_z` and other helpers. Not required for plain
        redshift plots.
    angle_direction : int
        Direction of increasing angle. ``-1`` (default) is clockwise as
        seen on screen — the astronomical convention where R.A.
        increases to the left when looking at the sky. ``+1`` is
        counterclockwise, which is more natural for galactic longitude
        (which increases in the same rotational sense as ecliptic
        longitude, *i.e.* eastward).
    zero_location : {'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'}
        Where ``angle_center`` appears in the frame. ``'N'`` (default)
        puts the center at the top, so the wedge opens upward. ``'W'``
        tips the cone so the tip is on the left (wedge opens rightward)
        — often a better fit than ``'N'`` for landscape paper figures.
        The eight compass positions cover 45° increments; for arbitrary
        orientations combine with ``zero_offset`` below.
    zero_offset : float
        Additional rotation in **degrees** beyond the ``zero_location``
        direction, passed through to
        :meth:`matplotlib.projections.polar.PolarAxes.set_theta_zero_location`.
        Default ``0``. Use for non-cardinal orientations, e.g.
        ``zero_location='N', zero_offset=30`` tips the cone 30° clockwise
        from straight up. Note that the radial-axis label placement is
        tuned for the cardinal ``zero_location`` values; for large
        ``zero_offset`` you may want to suppress the auto label
        (``r_label=''``) and add it yourself via ``ax.text``.
    angle_unit : {'deg', 'hour'}
        Input / display unit for the angular coordinate. Use ``'hour'``
        only for equatorial R.A., where ``angle_center`` and
        ``angle_half_width`` are in decimal hours and angular tick
        labels are formatted as ``HHh MMm``. Use ``'deg'`` for galactic,
        ecliptic, or any other longitude.
    angle_tick_spacing : float or None
        Spacing of angular ticks along the wedge arc, in the same unit as
        ``angle_unit``. ``None`` → matplotlib default.
    r_tick_spacing : float or None
        Spacing of radial tick marks in the units of ``r_variable``.
        ``None`` → matplotlib default.
    r_label, angle_label : str or None
        Axis labels. ``r_label=None`` auto-generates from ``r_variable``
        (e.g. "Redshift", "Comoving distance [Mpc]"). ``angle_label``
        defaults to ``'R.A.'`` (the most common use case, equatorial
        redshift surveys); for galactic coordinates pass something like
        ``angle_label='Galactic longitude $\\ell$'``, and for ecliptic
        ``angle_label='Ecliptic longitude $\\lambda$'``. Set to ``''``
        (empty) or ``None`` to suppress the label entirely.
    label_fontsize, tick_fontsize : int
        Font sizes for axis labels and tick labels respectively.
    radial_axis_side : {'left', 'right'}
        Which edge of the wedge gets the radial tick labels and the
        radial axis label. Default ``'left'``.
    rlabel_position : 'auto' or float
        Angle (in degrees) at which matplotlib places the radial tick
        labels. ``'auto'`` picks an appropriate value based on
        ``radial_axis_side`` (just inside the selected edge).
        Pass a numeric value only to override for advanced use.
    r_label_offset : float
        Horizontal offset of the radial axis label from the axes edge
        (in axes-fraction units). Increase if the tick labels are long
        and overlap the axis label.
    r_label_align : {'vertical', 'ray'}
        Orientation of the radial axis label text:

        * ``'vertical'`` (default): always written vertically
          (``rotation=±90°``). Consistent across panels with different
          ``angle_half_width``.
        * ``'ray'``: written parallel to the wedge edge
          (``rotation=90°±half_width``). Visually "ties" the label to
          the wedge geometry. Works cleanly for half-widths in the
          20°–60° range; becomes nearly horizontal at very wide wedges.
    angle_label_outside : float
        How far outside the outer arc to place the angular axis label
        (e.g. "R.A."). In units of the polar axis transform (roughly the
        fraction of ``r_max`` span beyond the arc). Default 0.18.
    grid : bool
        Draw radial + angular grid lines.
    gridcolor, gridalpha : matplotlib color, float
        Grid styling.
    gridlw : float, optional
        Grid line width. ``None`` (default) keeps this frame's historical
        0.5; ``gridcolor`` and ``gridalpha`` were exposed but this was not.
    gridls : str, optional
        Grid line style. ``None`` (default) inherits
        ``rcParams['grid.linestyle']``.
    fig : matplotlib.figure.Figure or None
        Figure to add the subplot to. If ``None``, uses the current figure
        (creates one if none exists).

    Returns
    -------
    ax : matplotlib.projections.polar.PolarAxes
        The configured polar Axes. Attached attributes:

        * ``ax._cone_angle_center`` — stored ``angle_center`` in degrees.
        * ``ax._cone_angle_unit`` — the ``angle_unit`` setting.
        * ``ax._cone_r_variable`` — the ``r_variable`` setting.
        * ``ax._cone_r_unit`` — the ``r_unit`` setting.
        * ``ax._cone_cosmology`` — cosmology object or None.
        * ``ax._cone_radial_axis_side`` — ``'left'`` or ``'right'``.

    Examples
    --------
    Simple redshift wedge::

        ax = make_cone_frame(111, angle_center=180, angle_half_width=30,
                             r_min=0, r_max=0.15)

    Distance wedge with Planck18 cosmology, radial axis on the right::

        from astropy.cosmology import Planck18
        ax = make_cone_frame(111, angle_center=180, angle_half_width=30,
                             r_max=600, r_variable='comoving_distance',
                             r_unit='Mpc', cosmology=Planck18,
                             radial_axis_side='right')

    RA in hours, with explicit angular tick spacing::

        ax = make_cone_frame(111, angle_center=12, angle_half_width=3,
                             r_max=0.15, angle_unit='hour',
                             angle_tick_spacing=1)  # one tick per hour

    Annular wedge (survey volume from z=0.05 to z=0.2 with observer
    position visible at the tip)::

        ax = make_cone_frame(111, angle_center=180, angle_half_width=40,
                             r_min=0.05, r_max=0.2, r_origin=0)

    Tipped cone, tip on the left (cone opens rightward) — fits better
    in landscape paper figures::

        ax = make_cone_frame(111, angle_center=180, angle_half_width=40,
                             r_max=0.2, zero_location='W')

    Notes
    -----
    Matplotlib also offers :meth:`~matplotlib.projections.polar.PolarAxes.set_rgrids`
    as a convenience that sets radial gridline values, labels, and the
    rlabel_position angle in one call. This module deliberately uses
    separate ``set_yticks`` / ``tick_params`` / ``set_rlabel_position``
    calls instead, so that tick styling and radial-axis styling can be
    controlled independently (e.g. coloring radial grid lines differently
    from angular ones, or changing tick fontsize without re-specifying
    locations). Call ``set_rgrids`` yourself on the returned axes if
    you want the combined form.
    """
    if r_variable not in _R_VARIABLES:
        raise ValueError(
            f"r_variable={r_variable!r} must be one of {_R_VARIABLES}.")
    if radial_axis_side not in ('left', 'right'):
        raise ValueError(
            f"radial_axis_side={radial_axis_side!r} must be 'left' or 'right'.")

    if fig is None:
        fig = plt.gcf()

    if angle_unit == 'hour':
        angle_center_deg = float(angle_center) * 15.0
        angle_half_width_deg = float(angle_half_width) * 15.0
    else:
        angle_center_deg = float(angle_center)
        angle_half_width_deg = float(angle_half_width)

    ax = fig.add_subplot(subplot_spec, projection='polar')

    ax.set_thetamin(-angle_half_width_deg)
    ax.set_thetamax(+angle_half_width_deg)
    ax.set_theta_zero_location(zero_location, offset=float(zero_offset))
    ax.set_theta_direction(angle_direction)
    ax.set_rmin(r_min)
    ax.set_rmax(r_max)
    if r_origin is not None:
        ax.set_rorigin(float(r_origin))

    if angle_tick_spacing is not None:
        step_deg = (angle_tick_spacing * 15.0 if angle_unit == 'hour'
                    else float(angle_tick_spacing))
        n_each_side = int(np.floor(angle_half_width_deg / step_deg))
        tick_degs = np.arange(-n_each_side, n_each_side + 1) * step_deg
        ax.set_xticks(np.radians(tick_degs))
        if angle_unit == 'hour':
            labels = [_format_hour((angle_center_deg + t) / 15.0 % 24.0)
                      for t in tick_degs]
        else:
            labels = [f'{(angle_center_deg + t) % 360:g}°' for t in tick_degs]
        ax.set_xticklabels(labels, fontsize=tick_fontsize)
    else:
        _relabel_angular_ticks_absolute(ax, angle_center_deg, angle_unit,
                                         tick_fontsize)

    if r_tick_spacing is not None:
        r_ticks = np.arange(r_min, r_max + 0.5 * r_tick_spacing, r_tick_spacing)
        ax.set_yticks(r_ticks)

    # Configure radial tick marks and labels.
    # - Tick marks default to BOTH sides (matplotlib convention for
    #   single-axis plots). When :func:`make_twinr` is later called,
    #   it overrides this to put primary ticks only on the primary
    #   side, leaving the twin side clean for the twin's ticks.
    # - Tick LABELS go only on `radial_axis_side`.
    # - Optional `radial_axis_color` colors ticks, tick labels, and
    #   the axis label, useful for highlighting one of two twin axes.
    #
    # IMPORTANT — orientation inversion. matplotlib polar's
    # `labelleft`/`labelright` and `set_rlabel_position` are interpreted
    # in matplotlib's INTERNAL theta coordinate system, which gets
    # flipped when zero_location='S' direction=-1 (or 'N' direction=+1):
    # the thetamin edge that is normally physical-left ends up at
    # physical-right. We detect this and flip both `labelleft/right`
    # and the rlabel_position sign so the user's `radial_axis_side`
    # always means the **physical** side they expect.
    inverted = ((zero_location == 'S' and angle_direction == -1) or
                (zero_location == 'N' and angle_direction == +1))
    effective_side = (
        ('right' if radial_axis_side == 'left' else 'left')
        if inverted else radial_axis_side)

    # Radial tick GEOMETRY (direction/size/width) follows the active base
    # style, exactly as the angular theta ticks already do — e.g. a
    # 'structural'/'journal' base sets ytick.direction='in'. Hardcoding these
    # would make the radial axis alone ignore the style, leaving inward angular
    # ticks beside outward radial ones. (The WCS frame reads the rc the same
    # way; see wcs_frame._overlay_tick_direction_from_rc.) The remaining
    # kwargs — which side gets marks/labels, and the twin-axis color — are
    # cone-specific and stay explicit.
    tick_kwargs = dict(axis='y', labelsize=tick_fontsize,
                       length=plt.rcParams['ytick.major.size'],
                       width=plt.rcParams['ytick.major.width'],
                       direction=plt.rcParams['ytick.direction'],
                       left=True, right=True)
    if radial_axis_color is not None:
        tick_kwargs['color'] = radial_axis_color
        tick_kwargs['labelcolor'] = radial_axis_color
    if effective_side == 'left':
        tick_kwargs['labelleft'] = True
        tick_kwargs['labelright'] = False
    else:
        tick_kwargs['labelleft'] = False
        tick_kwargs['labelright'] = True
    ax.tick_params(**tick_kwargs)
    # rlabel_position uses the same effective_side so it agrees with
    # the labelleft/labelright placement.
    if rlabel_position == 'auto':
        inset = min(1.0, angle_half_width_deg * 0.05)
        if effective_side == 'left':
            rlabel_position = -(angle_half_width_deg - inset)
        else:
            rlabel_position = +(angle_half_width_deg - inset)
    ax.set_rlabel_position(float(rlabel_position))

    if grid:
        # Same shape as make_globe_frame: gridcolor and gridalpha were
        # exposed while the width was forced, so that one property was
        # unreachable. Defaults preserve this path's historical 0.5 rather
        # than inheriting rcParams, which would move existing renders.
        ax.grid(True, color=gridcolor, alpha=gridalpha,
                linewidth=(0.5 if gridlw is None else gridlw),
                **({} if gridls is None else {'linestyle': gridls}))
    else:
        ax.grid(False)

    if r_label is None:
        r_label = _default_r_label(r_variable, r_unit)
    _place_axis_labels(ax, r_label, angle_label, label_fontsize,
                        angle_half_width_deg, zero_location,
                        radial_axis_side, r_label_offset,
                        r_label_align, angle_label_outside,
                        r_label_color=radial_axis_color,
                        r_label_flip=r_label_flip,
                        angle_label_align=angle_label_align,
                        angle_label_flip=angle_label_flip)

    ax._cone_angle_center = angle_center_deg
    ax._cone_angle_unit = angle_unit
    ax._cone_r_variable = r_variable
    ax._cone_r_unit = r_unit
    ax._cone_cosmology = cosmology
    ax._cone_radial_axis_side = radial_axis_side

    return ax


def _default_r_label(r_variable: str, r_unit: str) -> str:
    if r_variable == 'redshift':
        return 'Redshift'
    if r_variable == 'comoving_distance':
        return f'Comoving distance [{r_unit}]'
    if r_variable == 'lookback_time':
        return f'Lookback time [{r_unit}]'
    return r_variable


def _format_hour(hour: float) -> str:
    """Format a decimal hour as HHhMMm (minutes rounded to nearest integer)."""
    h = int(hour)
    m = int(round((hour - h) * 60))
    if m == 60:
        h += 1
        m = 0
    return f'{h}h{m:02d}m' if m else f'{h}h'


def _relabel_angular_ticks_absolute(ax: Any, angle_center_deg: float,
                                     angle_unit: str,
                                     tick_fontsize: int) -> None:
    """
    Replace the default polar theta tick labels (which show ±degrees
    relative to the internal theta=0) with absolute RA values.
    """
    ticks_rad = ax.get_xticks()
    tick_degs = np.degrees(ticks_rad)
    if angle_unit == 'hour':
        labels = [_format_hour((angle_center_deg + t) / 15.0 % 24.0)
                  for t in tick_degs]
    else:
        labels = [f'{(angle_center_deg + t) % 360:g}°' for t in tick_degs]
    # Pin the current tick locations (FixedLocator) before relabeling. The
    # labels are computed for exactly these positions, so fixing them is
    # correct — and it silences matplotlib's "set_ticklabels() should only be
    # used with a fixed number of ticks" UserWarning (the default theta locator
    # is dynamic).
    ax.set_xticks(ticks_rad)
    ax.set_xticklabels(labels, fontsize=tick_fontsize)



# ===== Internal: angle -> theta conversion =====

def _angle_to_theta(angle: npt.ArrayLike, angle_center_deg: float,
                    angle_unit: str) -> npt.NDArray[np.float64]:
    """Convert user angular input to internal polar theta (radians)."""
    angle = np.asarray(angle, dtype=float)
    if angle_unit == 'hour':
        angle_deg = angle * 15.0
    else:
        angle_deg = angle
    # Relative angle in (-180, 180] so that angle_center maps to 0 and
    # wrap is clean.
    rel = ((angle_deg - angle_center_deg + 180.0) % 360.0) - 180.0
    return np.radians(rel)



# ===== Bowtie (back-to-back wedge) =====

def make_bowtie_frame(*,
                      angle_center: float = 0., angle_half_width: float = 45.,
                      r_min: float = 0., r_max: float = 0.2,
                      r_origin: float | None = None,
                      r_variable: str = 'redshift', r_unit: str = 'Mpc',
                      cosmology: Any = None,
                      angle_direction: int = -1, angle_unit: str = 'deg',
                      angle_tick_spacing: float | None = None,
                      r_tick_spacing: float | None = None,
                      r_label: str | None = None, angle_label: str = 'R.A.',
                      r_label_align: str = 'ray', r_label_flip: bool = False,
                      angle_label_align: str = 'tangent',
                      angle_label_flip: bool = False,
                      r_label_offset: float = 0.20,
                      angle_label_outside: float = 0.18,
                      orientation: str = 'vertical',
                      hspace: float = 0.0, wspace: float = 0.0,
                      label_fontsize: int | None = None,
                      tick_fontsize: int | None = None,
                      grid: bool = True, gridcolor: Any = None,
                      gridalpha: float | None = None,
                      gridlw: float | None = None,
                      gridls: str | None = None,
                      suppress_apex_label: bool = True,
                      both_angle_labels: bool = True,
                      radial_axis_color: Any = None,
                      top_kwargs: dict[str, Any] | None = None,
                      bot_kwargs: dict[str, Any] | None = None,
                      left_kwargs: dict[str, Any] | None = None,
                      right_kwargs: dict[str, Any] | None = None,
                      fig: Any = None,
                      subplot_spec: Any = None) -> tuple[Any, Any]:
    """
    Build a "bowtie" dual-sided cone frame: two cones sharing a common apex.

    Used for redshift-survey plots that span both galactic hemispheres
    (the canonical CfA Redshift Survey "Stick Man" layout) or any other
    pair of opposite sky regions whose data is naturally compared at
    a common observer position. Each half is a regular cone frame, so
    all helpers (:func:`cone_scatter`, :func:`cone_hexbin`,
    :func:`add_minor_rticks`, :func:`make_twinr`, etc.) work on each
    half independently.

    All cone-geometry parameters (``angle_center``, ``angle_half_width``,
    ``r_min``, ``r_max``, etc.) have the same meaning as in
    :func:`make_cone_frame`. They are applied identically to both halves.

    The styling knobs shared with :func:`make_cone_frame` — ``gridcolor``,
    ``gridalpha``, ``gridlw``, ``gridls``, ``label_fontsize``,
    ``tick_fontsize`` — default to ``None`` here, meaning "whatever
    :func:`make_cone_frame` defaults to". Only values you set are forwarded,
    so there is a single source of truth for each default.

    Parameters
    ----------
    orientation : {'vertical', 'horizontal'}
        Layout of the bowtie:

        * ``'vertical'`` (default): top wedge opens upward, bottom wedge
          opens downward. Shared apex is horizontal (left-to-right
          centerline).
        * ``'horizontal'``: left wedge opens leftward, right wedge opens
          rightward. Shared apex is vertical.
    hspace, wspace : float
        GridSpec spacing parameter for the internal 2-cell layout. The
        default of ``0`` means the cells touch (no gap between
        bboxes); the apex auto-alignment (see Notes) handles the actual
        meeting of the two wedges' apex points regardless of this
        value. Pass a positive value (e.g. ``0.1``) to introduce a
        visible gap between halves; pass a negative value to overlap
        the bboxes (which trims the outer arcs of each half).
        ``hspace`` is used for vertical orientation; ``wspace`` for
        horizontal.
    suppress_apex_label : bool
        If True (default), the radial tick label at ``r_min`` (e.g. "0.00")
        is hidden on both halves so the shared apex looks clean.
    both_angle_labels : bool
        If True (default), both halves get the angular axis label
        (``angle_label``). If False, only the *outer* edge gets it
        (top in vertical, right in horizontal); useful for shorter
        figures or when the second label would crowd a colorbar.
    radial_axis_color : matplotlib color or None
        Optional color applied to both halves' radial axis (ticks,
        tick labels, axis label). Default ``None`` uses matplotlib's
        defaults. Pass a color to highlight the bowtie radial axis as
        a unit (e.g., ``radial_axis_color='steelblue'``).
    top_kwargs, bot_kwargs : dict or None
        (``orientation='vertical'`` only) Per-half override dicts.
        Any kwarg accepted by :func:`make_cone_frame` may be
        overridden for one half — e.g.
        ``top_kwargs={'gridcolor': 'steelblue', 'angle_label': 'NGP'}``
        and
        ``bot_kwargs={'gridcolor': 'crimson', 'angle_label': 'SGP'}``.
        Use this for the common "two halves want different colors /
        labels" pattern without manually re-styling the returned
        axes after construction.

        .. warning::
           Avoid overriding *geometry* kwargs (``angle_center``,
           ``angle_half_width``, ``r_min``, ``r_max``,
           ``r_origin``) per half — the apex auto-alignment assumes
           both halves share the same wedge geometry, and mismatched
           values will produce a misaligned apex. The intended use
           is for *cosmetic* overrides (color, labels, tick
           formatting).
    left_kwargs, right_kwargs : dict or None
        (``orientation='horizontal'`` only) Same as
        ``top_kwargs``/``bot_kwargs`` but for the horizontal layout.
    subplot_spec : SubplotSpec or None
        Where to place the bowtie. ``None`` (default) creates a 1x1
        outer GridSpec on the figure. Pass a
        :class:`matplotlib.gridspec.SubplotSpec` to embed the bowtie
        inside a larger figure layout (e.g. as one panel of a
        multi-panel plot).

    Returns
    -------
    top, bot : (PolarAxes, PolarAxes) for ``orientation='vertical'``
    left, right : (PolarAxes, PolarAxes) for ``orientation='horizontal'``
        Each axes is a fully-configured cone frame with attached
        ``_cone_*`` metadata. Each also has ``_cone_bowtie_role``
        (``'top'``, ``'bot'``, ``'left'``, or ``'right'``) and
        ``_cone_bowtie_partner`` (a reference to the other half) for
        introspection.

    Notes
    -----
    Apex alignment. matplotlib's polar axes auto-position the wedge
    inside the bbox with internal padding (~10–13% of the bbox height
    from the bbox edge to the apex). The exact padding depends on the
    wedge's bounding-box aspect ratio, which depends on
    ``angle_half_width``. Rather than picking a fixed ``hspace`` that
    works only for one specific half-width,
    :func:`make_bowtie_frame` measures each half's apex position after
    a probe render and installs an :func:`~matplotlib.axes.Axes.set_axes_locator`
    on each axes that maintains the apex alignment on every layout
    pass. This makes the alignment robust to:

    * any wedge ``angle_half_width``
    * any figure size or aspect ratio
    * outer ``GridSpec`` nesting (via ``subplot_spec``)
    * ``subplots_adjust`` / ``tight_layout`` calls afterwards

    Per-half customization. Because each returned axes is a regular
    cone frame, you can customize them individually after creation:

    .. code-block:: python

        top, bot = make_bowtie_frame(angle_half_width=45,
                                       r_max=0.2)
        cone_scatter(top, north_galaxy_ras, north_galaxy_zs,
                     s=3, color='steelblue')
        cone_scatter(bot, south_galaxy_ras, south_galaxy_zs,
                     s=3, color='crimson')

    Examples
    --------
    Classic CfA-style vertical bowtie::

        from skyplothelper.cone import make_bowtie_frame, cone_scatter

        fig = plt.figure(figsize=(9, 12))
        top, bot = make_bowtie_frame(angle_center=180,
                                       angle_half_width=40,
                                       r_min=0, r_max=0.15,
                                       angle_tick_spacing=10,
                                       r_tick_spacing=0.03)
        cone_scatter(top, north_ras, north_zs, s=3, color='0.3')
        cone_scatter(bot, south_ras, south_zs, s=3, color='0.3')

    Horizontal bowtie (apex on vertical centerline)::

        left, right = make_bowtie_frame(orientation='horizontal',
                                          angle_center=180,
                                          angle_half_width=30,
                                          r_max=0.2)
    """
    if fig is None:
        fig = plt.gcf()
    if orientation not in ('vertical', 'horizontal'):
        raise ValueError(
            f"orientation={orientation!r} must be 'vertical' or "
            "'horizontal'.")

    # Determine the container subplot spec.
    if subplot_spec is None:
        from matplotlib.gridspec import GridSpec
        outer = GridSpec(1, 1, figure=fig)
        outer_spec = outer[0]
    else:
        outer_spec = subplot_spec

    # Common kwargs for both halves.
    common = dict(
        angle_center=angle_center, angle_half_width=angle_half_width,
        r_min=r_min, r_max=r_max, r_origin=r_origin,
        r_variable=r_variable, r_unit=r_unit, cosmology=cosmology,
        angle_direction=angle_direction, angle_unit=angle_unit,
        angle_tick_spacing=angle_tick_spacing,
        r_tick_spacing=r_tick_spacing,
        r_label=r_label,
        r_label_align=r_label_align, r_label_flip=r_label_flip,
        angle_label_align=angle_label_align,
        angle_label_flip=angle_label_flip,
        r_label_offset=r_label_offset,
        angle_label_outside=angle_label_outside,
        grid=grid,
        radial_axis_color=radial_axis_color,
        fig=fig,
        # Forward only what the caller actually set. Restating
        # make_cone_frame's defaults here would shadow them: every value
        # would arrive explicitly, so changing a default in one place and
        # not the other would silently do nothing for twin-panel callers.
        **_set_only(label_fontsize=label_fontsize,
                    tick_fontsize=tick_fontsize, gridcolor=gridcolor,
                    gridalpha=gridalpha, gridlw=gridlw, gridls=gridls),
    )

    # Resolve per-half overrides; reject the orientation-incompatible
    # pair early so users get a clear error rather than silently
    # ignored kwargs.
    top_kwargs = top_kwargs or {}
    bot_kwargs = bot_kwargs or {}
    left_kwargs = left_kwargs or {}
    right_kwargs = right_kwargs or {}
    if orientation == 'vertical' and (left_kwargs or right_kwargs):
        raise TypeError(
            "left_kwargs / right_kwargs only apply to "
            "orientation='horizontal'. Use top_kwargs / bot_kwargs "
            "for orientation='vertical'."
        )
    if orientation == 'horizontal' and (top_kwargs or bot_kwargs):
        raise TypeError(
            "top_kwargs / bot_kwargs only apply to "
            "orientation='vertical'. Use left_kwargs / right_kwargs "
            "for orientation='horizontal'."
        )

    if orientation == 'vertical':
        sub = outer_spec.subgridspec(2, 1, hspace=hspace)
        # Top half: zero='N' (apex at bottom of its wedge, opens up).
        # Per-half overrides (top_kwargs) win over common kwargs and
        # over the explicit angle_label below.
        top_args = {**common, 'angle_label': angle_label, **top_kwargs}
        top = make_cone_frame(sub[0], zero_location='N', **top_args)
        # Bottom half: zero='S' (apex at top, opens down)
        bot_label = angle_label if both_angle_labels else ''
        bot_args = {**common, 'angle_label': bot_label, **bot_kwargs}
        bot = make_cone_frame(sub[1], zero_location='S', **bot_args)
        first, second = top, bot
        first._cone_bowtie_role = 'top'
        second._cone_bowtie_role = 'bot'
    else:  # horizontal
        sub = outer_spec.subgridspec(1, 2, wspace=wspace)
        # Left half: zero='W' (apex at right of wedge, opens left)
        left_args = {**common, 'angle_label': angle_label, **left_kwargs}
        left = make_cone_frame(sub[0], zero_location='W', **left_args)
        # Right half: zero='E' (apex at left, opens right)
        right_label = angle_label if both_angle_labels else ''
        right_args = {**common, 'angle_label': right_label, **right_kwargs}
        right = make_cone_frame(sub[1], zero_location='E', **right_args)
        first, second = left, right
        first._cone_bowtie_role = 'left'
        second._cone_bowtie_role = 'right'

    # Suppress the apex tick label (typically "0.00") so the shared
    # apex looks clean. Each half retains all OTHER radial tick labels.
    if suppress_apex_label:
        for half in (first, second):
            ticks = np.asarray(half.get_yticks(), dtype=float)
            mask = ticks > r_min + 1e-9
            half.set_yticks(ticks[mask])

    # ------------------------------------------------------------------
    # Apex auto-alignment via set_axes_locator.
    #
    # The optimal `hspace` (or `wspace`) for apex-touching depends on the
    # wedge geometry in a non-trivial way (matplotlib polar's internal
    # margin scales with the wedge's bounding-box aspect ratio). Rather
    # than picking a fixed default that's only right for one specific
    # wedge angle, we measure each apex's actual figure-coord position
    # after a probe render, compute the shift needed to bring them to a
    # common point, and install a locator on each axes that maintains
    # that shift on every layout pass.
    #
    # This makes the apex alignment robust to:
    #   * any wedge half_width
    #   * any figure size or aspect ratio
    #   * outer GridSpec nesting
    #   * `subplots_adjust` / `tight_layout` afterwards
    # ------------------------------------------------------------------
    fig.canvas.draw()  # calibrate transData
    fig_w, fig_h = fig.bbox.width, fig.bbox.height

    # Measure each apex's natural figure-coord position.
    if orientation == 'vertical':
        first_apex_y = first.transData.transform([[0, 0]])[0][1] / fig_h
        second_apex_y = second.transData.transform([[0, 0]])[0][1] / fig_h
        target_y = 0.5 * (first_apex_y + second_apex_y)
        first_shift_y = target_y - first_apex_y
        second_shift_y = target_y - second_apex_y
        first_shift_x = second_shift_x = 0.0
    else:  # horizontal
        first_apex_x = first.transData.transform([[0, 0]])[0][0] / fig_w
        second_apex_x = second.transData.transform([[0, 0]])[0][0] / fig_w
        target_x = 0.5 * (first_apex_x + second_apex_x)
        first_shift_x = target_x - first_apex_x
        second_shift_x = target_x - second_apex_x
        first_shift_y = second_shift_y = 0.0

    # Install locators that track the SubplotSpec's natural position
    # but apply the measured shift each layout pass.
    from matplotlib.transforms import Bbox
    first_spec = sub[0]
    second_spec = sub[1]

    def _make_shift_locator(spec: Any, dx: float, dy: float) -> Any:
        def locator(ax_: Any, renderer: Any) -> Any:
            pos = spec.get_position(fig)
            return Bbox.from_extents(pos.x0 + dx, pos.y0 + dy,
                                      pos.x1 + dx, pos.y1 + dy)
        return locator

    first.set_axes_locator(_make_shift_locator(first_spec,
                                                 first_shift_x,
                                                 first_shift_y))
    second.set_axes_locator(_make_shift_locator(second_spec,
                                                  second_shift_x,
                                                  second_shift_y))

    # Cross-reference the two halves for introspection.
    first._cone_bowtie_partner = second
    second._cone_bowtie_partner = first

    return first, second


# ---------------------------------------------------------------------------
# Self-contained demo
# ---------------------------------------------------------------------------
