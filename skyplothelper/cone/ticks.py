"""Cone radial-tick helpers.

``add_minor_rticks`` draws minor ticks at user-specified ``r`` spacing;
``log_r`` switches the radial axis to logarithmic with sensible defaults.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np


def add_minor_rticks(ax: Any, step: float, length: float = 2.5,
                     color: Any = None, linewidth: float = 0.7) -> list[Any]:
    """
    Draw minor radial tick marks along the wedge's tick-labeled edge.

    matplotlib polar axes don't natively support minor radial ticks on
    wedges (they render as full concentric arcs, not short edge marks).
    This helper draws short line segments at each minor-tick position,
    just outside the wedge's radial-label edge.

    Parameters
    ----------
    ax : PolarAxes
        The frame (primary or twin).
    step : float
        Minor-tick spacing in the axes' r units.
    length : float
        Tick mark length in **points** (display units), so all minor
        ticks are visually the same size regardless of their r value
        — matching matplotlib's standard tick behavior. Default 2.5,
        chosen to be visibly shorter than the major tick length of 4
        used by :func:`make_cone_frame`, so the visual hierarchy reads
        correctly (major > minor).
    color : matplotlib color or None
        Tick color. Defaults to the same color matplotlib uses for
        major ticks. To match a colored twin axis, pass the same
        ``color`` value used in :func:`make_twinr`.
    linewidth : float
        Line width of the minor tick marks. Default 0.7 (slightly
        thinner than the major-tick default of 0.8).

    Returns
    -------
    lines : list of Line2D
        One per minor tick drawn.

    Notes
    -----
    Ticks point outward from the wedge interior, matching the
    ``direction='out'`` orientation that
    :func:`make_cone_frame` uses for its major ticks.

    The marks DO NOT auto-update if limits change; call
    :func:`add_minor_rticks` again after ``ax.set_rmin`` /
    ``ax.set_rmax``.

    Examples
    --------
    ::

        ax = make_cone_frame(111, angle_center=180, angle_half_width=40,
                             r_min=0, r_max=0.2, r_tick_spacing=0.05)
        add_minor_rticks(ax, step=0.01)   # minor every 0.01 in z
    """
    import matplotlib
    from matplotlib.lines import Line2D

    fig = ax.figure
    fig.canvas.draw()  # ensure transData is calibrated

    r0, r1 = ax.get_rmin(), ax.get_rmax()
    # Build minor tick values — start at the first multiple of step >= r0.
    # Use a strict upper bound to avoid an FP-induced extra tick at or
    # beyond r1 (e.g. an arange that drifts to 0.20000000000000007 with
    # r1=0.20).
    start = np.ceil(r0 / step) * step
    minor_r = np.arange(start, r1 - 1e-9, step)
    minor_r = minor_r[(minor_r >= r0 - 1e-9) & (minor_r <= r1 - 1e-9)]

    # Skip ones coinciding with major ticks (within 1e-9).
    major_r = np.asarray(ax.get_yticks(), dtype=float)
    mask = np.ones(len(minor_r), dtype=bool)
    for mr in major_r:
        mask &= np.abs(minor_r - mr) > 1e-9 * max(1.0, abs(mr))
    minor_r = minor_r[mask]

    # Tick edge: actual wedge edge (not rlabel_position which is offset
    # inward by a small inset). The sign of rlabel_position tells us
    # which edge the user wants; we use the corresponding wedge boundary.
    rlabel_deg = ax.get_rlabel_position()
    if rlabel_deg >= 0:
        edge_deg = ax.get_thetamax()    # right edge
    else:
        edge_deg = ax.get_thetamin()    # left edge
    edge_rad = np.radians(edge_deg)

    # length is in points; convert to pixels (display units)
    dpi = fig.dpi
    length_px = length * (dpi / 72.0)

    # Compute the OUTWARD direction at each tick in display coords.
    # We do this by transforming the tick point and a slightly-offset-
    # in-theta point, then taking the perpendicular direction.
    if color is None:
        # Radial ticks are the y-axis in a wedge frame, matching the
        # ``ytick.*`` params the rest of ``cone/`` reads for radial styling.
        color = matplotlib.rcParams.get('ytick.color', 'black')

    lines = []
    for r in minor_r:
        # Tick anchor on the wedge edge, in polar data coords.
        anchor_data = np.array([[edge_rad, r]])
        anchor_display = ax.transData.transform(anchor_data)[0]

        # A nearby point slightly toward the wedge interior, used to
        # determine the inward → outward direction in display coords.
        # Move 1 degree inward in theta (toward 0).
        sign_inward = -1.0 if edge_deg >= 0 else +1.0
        inward_theta = edge_rad + sign_inward * np.radians(1.0)
        inward_data = np.array([[inward_theta, r]])
        inward_display = ax.transData.transform(inward_data)[0]

        # Outward unit vector in display coords.
        inward_vec = inward_display - anchor_display
        norm = np.hypot(inward_vec[0], inward_vec[1])
        if norm < 1e-12:
            continue
        outward_vec = -inward_vec / norm  # unit length, points outward

        # Tick endpoints in display coords.
        end_display = anchor_display + outward_vec * length_px

        # Convert both endpoints back to data coords for ax.plot.
        inv = ax.transData.inverted()
        seg_data = inv.transform([anchor_display, end_display])

        ln = Line2D(seg_data[:, 0], seg_data[:, 1],
                     color=color, linewidth=linewidth,
                     solid_capstyle='butt', clip_on=False,
                     transform=ax.transData)
        ax.add_line(ln)
        lines.append(ln)

    return lines


def log_r(ax: Any, minor_ticks: bool = True) -> Any:
    """
    Switch the radial axis to a logarithmic scale.

    A thin wrapper around ``ax.set_rscale('log')`` that also applies a
    readable tick formatter. matplotlib's default log-scale formatter
    on polar axes produces overlapping scientific-notation labels on
    wedge frames; this helper replaces the formatter with a plain
    ``ScalarFormatter``.

    Parameters
    ----------
    ax : PolarAxes
    minor_ticks : bool
        If True (default), also enable minor ticks at the usual
        log-scale decade subdivisions (2, 3, ..., 9 × 10^n).

    Returns
    -------
    ax : PolarAxes
        The same axes, for chaining.

    Notes
    -----
    Log scale on polar axes has some well-known quirks in matplotlib:

    * ``r_min=0`` is obviously invalid; set ``r_min > 0`` before
      switching.
    * The visual spacing of decades is linear in the projected radius,
      not logarithmic — but the tick *values* are log-spaced. This is
      usually what you want for a cosmology cone plot with both
      nearby and distant objects.
    * Gridlines at decades are not automatic; use
      ``ax.yaxis.set_major_locator(LogLocator())`` if you want them.

    Examples
    --------
    ::

        ax = make_cone_frame(111, angle_center=180, angle_half_width=40,
                             r_min=0.001, r_max=3.0)
        log_r(ax)
    """
    from matplotlib.ticker import LogLocator, ScalarFormatter
    if ax.get_rmin() <= 0:
        raise ValueError(
            f"log_r: rmin must be > 0 for log scale (currently {ax.get_rmin()}). "
            "Note that matplotlib's scatter/plot auto-expands view limits to "
            "include all data; if you called ax.scatter with small values "
            "before log_r, re-apply ax.set_rmin(your_min) after plotting and "
            "before log_r. Alternatively, call log_r before plotting data.")
    ax.set_rscale('log')
    # Use ScalarFormatter for readable "0.01, 0.1, 1, 10" instead of
    # "10^-2, 10^-1, ..." clutter.
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    if minor_ticks:
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=list(np.arange(2, 10) * 0.1)))
    return ax

