"""Plotting helpers for cone wedge axes.

``cone_scatter``, ``cone_plot``, ``cone_scatter_z`` are angle/r wrappers
around the underlying polar axes; ``cone_hexbin`` and ``cone_pcolormesh``
use a Cartesian overlay (``_make_cartesian_overlay``) so that hexbin /
pcolormesh work in cone coordinates.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
import numpy.typing as npt

from .cosmology import redshift_to_r
from .frame import _angle_to_theta


def cone_scatter(ax: Any, angle: npt.ArrayLike, r: npt.ArrayLike,
                 **kwargs: Any) -> Any:
    """
    Scatter data on a cone frame.

    Parameters
    ----------
    ax : PolarAxes
        An axes returned by :func:`make_cone_frame`.
    angle : array_like
        Angular coordinate (R.A., galactic longitude, ecliptic
        longitude, ...) in the frame's ``angle_unit`` (default
        degrees). Wrap-around across 360° / 0° is handled automatically
        so narrow wedges spanning the 0° line work transparently.
    r : array_like
        Radial values in the frame's ``r_variable`` units. If you have
        redshifts but the frame is in distance, either convert first
        (via :func:`redshift_to_r`) or use :func:`cone_scatter_z`.
    **kwargs :
        Passed to :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    collection : matplotlib.collections.PathCollection

    Examples
    --------
    >>> import skyplothelper as sph
    >>> ax = sph.make_cone_frame(111, angle_center=180, angle_half_width=60,
    ...                          r_min=0, r_max=0.1)
    >>> sph.cone_scatter(ax, ra, redshift, s=8)      # angle=RA (deg), r=z
    """
    angle_center = getattr(ax, '_cone_angle_center', 0.0)
    angle_unit = getattr(ax, '_cone_angle_unit', 'deg')
    theta = _angle_to_theta(angle, angle_center, angle_unit)
    return ax.scatter(theta, np.asarray(r, dtype=float), **kwargs)


def cone_plot(ax: Any, angle: npt.ArrayLike, r: npt.ArrayLike,
              **kwargs: Any) -> Any:
    """Line version of :func:`cone_scatter`. Returns a list of Line2D."""
    angle_center = getattr(ax, '_cone_angle_center', 0.0)
    angle_unit = getattr(ax, '_cone_angle_unit', 'deg')
    theta = _angle_to_theta(angle, angle_center, angle_unit)
    return ax.plot(theta, np.asarray(r, dtype=float), **kwargs)


def cone_scatter_z(ax: Any, angle: npt.ArrayLike, z: npt.ArrayLike,
                   cosmology: Any = None, r_unit: str | None = None,
                   **kwargs: Any) -> Any:
    """
    Scatter data given as ``(angle, z)``, converting redshift to whatever
    radial variable the frame is using.

    Parameters
    ----------
    ax : PolarAxes
        An axes returned by :func:`make_cone_frame`.
    angle : array_like
        Angular coordinate in the frame's angular unit.
    z : array_like
        Redshift values.
    cosmology : astropy.cosmology instance or None
        Used for z ↔ distance / time conversion. If ``None``, uses
        ``ax._cone_cosmology`` from the frame (if set).
    r_unit : str or None
        Override the frame's ``_cone_r_unit`` for this call.
    **kwargs :
        Passed to :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    collection : matplotlib.collections.PathCollection
    """
    r_variable = getattr(ax, '_cone_r_variable', 'redshift')
    if r_unit is None:
        r_unit = getattr(ax, '_cone_r_unit', 'Mpc')
    if cosmology is None:
        cosmology = getattr(ax, '_cone_cosmology', None)

    r = redshift_to_r(z, r_variable=r_variable, cosmology=cosmology,
                     r_unit=r_unit)
    return cone_scatter(ax, angle, r, **kwargs)


# ---------------------------------------------------------------------------
# Advanced helpers: twin radial axis, minor ticks, log scale, hexbin
# ---------------------------------------------------------------------------


# ===== Cartesian overlay (for hexbin / pcolormesh) =====

def _cone_to_cartesian(
    theta_internal: npt.ArrayLike, r: npt.ArrayLike, ax: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Convert polar (theta_internal, r) to Cartesian screen-projected
    (x, y) for the given cone axes.

    The conversion accounts for ``theta_zero_location`` (via theta_offset)
    and ``theta_direction`` (CW or CCW). Output coords are what
    matplotlib would render the data at on the polar canvas.

    Parameters
    ----------
    theta_internal : array_like
        Theta in radians, already shifted relative to ``angle_center``
        (i.e. the output of :func:`_angle_to_theta`).
    r : array_like
        Radial coordinate in the frame's r-units.
    ax : PolarAxes
        The cone frame, used for ``get_theta_offset`` and
        ``get_theta_direction``.

    Returns
    -------
    x, y : ndarray
        Cartesian screen-projected coordinates. Origin is the wedge tip
        (or rorigin if set), x is horizontal, y is vertical.
    """
    theta_offset = ax.get_theta_offset()  # radians; matplotlib internal
    theta_direction = ax.get_theta_direction()  # +1 or -1
    # matplotlib polar internal: screen angle = theta_offset + direction * theta
    screen_theta = theta_offset + theta_direction * np.asarray(theta_internal)
    r = np.asarray(r, dtype=float)
    x = r * np.cos(screen_theta)
    y = r * np.sin(screen_theta)
    return x, y


def _make_cartesian_overlay(ax: Any, zorder_offset: float = 0.5,
                            clip_to_wedge: bool = True) -> Any:
    """
    Build a Cartesian overlay axes aligned with the wedge's rendered area.

    Used by :func:`cone_hexbin` and :func:`cone_pcolormesh` to render
    binned data with correctly-shaped cells in the user's view-space,
    rather than the (theta, r) data-space where polar hexbin / pcolormesh
    misbehave on wedges.

    The overlay is positioned to match where the wedge actually renders
    on the figure (which is generally NOT the full axes bounding box —
    a 40°-half-width wedge occupies roughly the upper-middle 60% of its
    axes box). We compute that rendering region by mapping the polar
    wedge's bounds (apex, outer arc corners) through ``ax.transData`` to
    figure coordinates and use the result as the overlay rectangle.

    Side effect: sets ``ax.patch.set_facecolor('none')`` so the parent's
    background doesn't occlude the overlay. The wedge spine, ticks, and
    labels render normally on top.

    Parameters
    ----------
    ax : PolarAxes
        The cone frame to align with.
    zorder_offset : float
        How far below the parent to place the overlay (default 0.5).
        Lower means the parent definitely renders on top.
    clip_to_wedge : bool
        If True (default), set a clip path on the overlay so all rendered
        content is masked to the wedge interior. Hexbin cells outside
        the wedge are hidden cleanly.

    Returns
    -------
    overlay : matplotlib.axes.Axes
        The Cartesian overlay. Add a colorbar via
        ``fig.colorbar(mappable, ax=overlay, ...)``.

    Notes
    -----
    The overlay is anchored to the **rendered position at the time of
    creation**. Pan/zoom on ``ax`` does NOT propagate to the overlay
    — for static plots this is fine, for interactive use call this
    fresh after limit changes.
    """
    fig = ax.figure
    fig.canvas.draw()  # force layout so transData is calibrated

    # Make parent transparent so the overlay below shows through.
    ax.patch.set_facecolor('none')

    # Compute Cartesian extent of the wedge in data coords.
    half_width_rad = np.radians(
        0.5 * (ax.get_thetamax() - ax.get_thetamin()))
    rmin = ax.get_rmin()
    rmax = ax.get_rmax()
    try:
        rorigin = ax.get_rorigin()
    except AttributeError:
        rorigin = rmin

    theta_sample = np.linspace(-half_width_rad, +half_width_rad, 60)
    x_outer, y_outer = _cone_to_cartesian(theta_sample,
                                            np.full_like(theta_sample, rmax),
                                            ax)
    x_inner, y_inner = _cone_to_cartesian(theta_sample,
                                            np.full_like(theta_sample, rmin),
                                            ax)
    x_apex, y_apex = _cone_to_cartesian(np.array([0.0]),
                                          np.array([rorigin]), ax)
    xs = np.concatenate([x_outer, x_inner, x_apex])
    ys = np.concatenate([y_outer, y_inner, y_apex])
    x_lo, x_hi = xs.min(), xs.max()
    y_lo, y_hi = ys.min(), ys.max()
    # NOTE: do NOT pad the xlim/ylim. The overlay rect's figure-coord
    # extent (computed below from the same boundary points) tracks these
    # exact x_lo/x_hi/y_lo/y_hi. If we pad here, data values at the
    # wedge boundary would render INSIDE the rect, producing the visible
    # "shrunk inward" effect. Hexbin and pcolormesh both happily render
    # cells exactly to the data-range edge; clip-to-wedge handles any
    # overshoot from the algorithms' grid-extension behavior.

    # Project wedge bounds into pixel/display coords by sampling many
    # points along the wedge boundary and transforming via ax.transData.
    # This is more robust than the corner-Cartesian-to-polar inversion
    # used previously, which compounded geometric errors.
    boundary_theta_internal = np.concatenate([
        theta_sample,                       # outer arc
        theta_sample[::-1],                 # inner arc (or apex)
        np.array([theta_sample[0]]),        # close
    ])
    boundary_r = np.concatenate([
        np.full_like(theta_sample, rmax),
        np.full_like(theta_sample, rmin if rmin > 0 else rorigin),
        np.array([rmax]),
    ])
    boundary_data = np.column_stack([boundary_theta_internal, boundary_r])
    boundary_display = ax.transData.transform(boundary_data)

    # Convert to figure-fraction coords using the figure's bbox.
    fig_inv = fig.transFigure.inverted()
    boundary_fig = fig_inv.transform(boundary_display)
    fx_lo = boundary_fig[:, 0].min()
    fx_hi = boundary_fig[:, 0].max()
    fy_lo = boundary_fig[:, 1].min()
    fy_hi = boundary_fig[:, 1].max()

    overlay = fig.add_axes([fx_lo, fy_lo, fx_hi - fx_lo, fy_hi - fy_lo],
                            zorder=ax.get_zorder() - zorder_offset,
                            frame_on=False)
    overlay.patch.set_visible(False)
    overlay.set_xticks([])
    overlay.set_yticks([])
    for spine in overlay.spines.values():
        spine.set_visible(False)

    overlay.set_xlim(x_lo, x_hi)
    overlay.set_ylim(y_lo, y_hi)
    overlay.set_aspect('equal')

    # Compute the wedge bbox as a fixed fraction OF the parent's bbox at
    # creation time. This fraction depends only on the wedge geometry
    # (thetamin/thetamax/zero_offset/zero_location) and is invariant to
    # later position changes of the parent. The locator then computes
    # the overlay's actual figure-coords by multiplying the parent's
    # current bbox by this fraction — this works regardless of when
    # subplots_adjust runs.
    parent_bbox = ax.get_position()
    rel_x_lo = (fx_lo - parent_bbox.x0) / parent_bbox.width
    rel_x_hi = (fx_hi - parent_bbox.x0) / parent_bbox.width
    rel_y_lo = (fy_lo - parent_bbox.y0) / parent_bbox.height
    rel_y_hi = (fy_hi - parent_bbox.y0) / parent_bbox.height

    def _overlay_locator(overlay_ax: Any, renderer: Any) -> Any:
        from matplotlib.transforms import Bbox
        p = ax.get_position()
        new_fx_lo = p.x0 + rel_x_lo * p.width
        new_fx_hi = p.x0 + rel_x_hi * p.width
        new_fy_lo = p.y0 + rel_y_lo * p.height
        new_fy_hi = p.y0 + rel_y_hi * p.height
        return Bbox.from_extents(new_fx_lo, new_fy_lo,
                                  new_fx_hi, new_fy_hi)

    overlay.set_axes_locator(_overlay_locator)

    if clip_to_wedge:
        # Build the wedge polygon in overlay's data coords for clipping.
        # Walk: apex → outer arc (theta_min → theta_max) → back to apex.
        if rmin > 0 and rorigin != rmin:
            # Annular wedge: include the inner arc as well.
            x_inner_rev = x_inner[::-1]
            y_inner_rev = y_inner[::-1]
            poly_x = np.concatenate([x_outer, x_inner_rev,
                                      [x_outer[0]]])
            poly_y = np.concatenate([y_outer, y_inner_rev,
                                      [y_outer[0]]])
        else:
            # Simple cone: apex + outer arc + back to apex.
            poly_x = np.concatenate([[x_apex[0]], x_outer, [x_apex[0]]])
            poly_y = np.concatenate([[y_apex[0]], y_outer, [y_apex[0]]])
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        path = Path(np.column_stack([poly_x, poly_y]))
        clip_patch = PathPatch(path, transform=overlay.transData,
                                facecolor='none', edgecolor='none')
        overlay.add_patch(clip_patch)
        overlay._cone_clip_patch = clip_patch  # store for downstream use

    overlay._cone_overlay_parent = ax
    return overlay



# ===== hexbin / pcolormesh =====

def cone_hexbin(ax: Any, angle: npt.ArrayLike, r: npt.ArrayLike,
                **kwargs: Any) -> Any:
    """
    Hexagonal-binning version of :func:`cone_scatter` with correctly
    shaped hexagon cells.

    Renders into a Cartesian overlay axes aligned with the wedge — this
    avoids the well-known matplotlib issue where ``ax.hexbin`` on a
    polar axes operates in (theta_radians, r_units) data space and
    produces hexagons that look like dots because the two coordinates
    have wildly different scales (theta in radians ~1, r in physical
    units ~0.2). After this conversion, hexagon cells are
    constant-area-on-screen, which is what users typically want for
    "show me galaxy density on the wedge."

    Parameters
    ----------
    ax : PolarAxes
        Axes returned by :func:`make_cone_frame`.
    angle : array_like
        Angular coordinate in the frame's ``angle_unit``.
    r : array_like
        Radial values in the frame's ``r_variable`` units.
    **kwargs :
        Passed to :meth:`matplotlib.axes.Axes.hexbin`. Common overrides:
        ``gridsize=40``, ``cmap='viridis'``, ``mincnt=1`` (hide empty
        hexes), ``bins='log'`` (log color scaling for dynamic range).

    Returns
    -------
    image : matplotlib.collections.PolyCollection
        The hexbin collection. Suitable for ``plt.colorbar(image, ax=ax)``.

    Notes
    -----
    Cells are uniform-area on the rendered figure. This is not the same
    as uniform-area-on-sky: at small r (near the cone tip) a screen-area
    hex covers more solid angle than at large r. For uniform-on-sky
    binning, use :func:`cone_pcolormesh` with a proper (theta, r) grid
    where the cell sizes scale with r — that's the "uniform solid angle"
    convention more natural for projected density.

    Range of cells matters too: matplotlib computes hex extents from the
    data range by default. To force cells to fill the whole wedge,
    pass ``extent=(xmin, xmax, ymin, ymax)`` matching the projected
    wedge bbox.

    A faint outline of the wedge is preserved by drawing on top of the
    primary axes — use the colorbar's ``ax`` argument to attach it to
    the overlay so it doesn't crowd the cone frame.

    Examples
    --------
    ::

        ax = make_cone_frame(111, angle_center=180, angle_half_width=40,
                             r_min=0, r_max=0.2)
        hb = cone_hexbin(ax, galaxy_ras, galaxy_zs,
                         gridsize=40, cmap='viridis', mincnt=1)
        plt.colorbar(hb, ax=ax, label='N galaxies', pad=0.15)
    """
    angle_center = getattr(ax, '_cone_angle_center', 0.0)
    angle_unit = getattr(ax, '_cone_angle_unit', 'deg')
    theta_internal = _angle_to_theta(angle, angle_center, angle_unit)
    x, y = _cone_to_cartesian(theta_internal, r, ax)

    overlay = _make_cartesian_overlay(ax)
    image = overlay.hexbin(x, y, **kwargs)
    # Clip hexagons to the wedge polygon to suppress the spillover.
    if hasattr(overlay, '_cone_clip_patch'):
        image.set_clip_path(overlay._cone_clip_patch)
    return image


def cone_pcolormesh(ax: Any, angle_edges: npt.ArrayLike,
                    r_edges: npt.ArrayLike, C: npt.ArrayLike,
                    **kwargs: Any) -> Any:
    """
    pcolormesh-style binned rendering on a cone wedge.

    Unlike :func:`cone_hexbin`, this draws cells in **native (theta, r)
    coordinates** — the cells follow the polar grid, so cell area scales
    with ``r`` (cells are smaller near the apex, larger at the outer
    arc). This is the natural choice for representing **uniform
    solid-angle bins** as seen from Earth: each cell covers
    ``Δtheta · r · Δr`` square units, and at fixed ``Δtheta·Δr`` the
    cells subtend uniform solid angle in spherical-projection sense.

    Parameters
    ----------
    ax : PolarAxes
        Axes returned by :func:`make_cone_frame`.
    angle_edges : array_like, shape (Nangle+1,)
        Angle bin EDGES in the frame's ``angle_unit``. Use
        ``np.linspace`` or similar.
    r_edges : array_like, shape (Nr+1,)
        Radial bin EDGES in the frame's ``r_variable`` units.
    C : array_like, shape (Nr, Nangle)
        Cell values. Note the orientation: rows are radial, columns
        are angular — same convention as ``np.histogram2d`` returns
        when called as ``H, _, _ = np.histogram2d(r, angle, bins=...)``.
    **kwargs :
        Passed to :meth:`matplotlib.axes.Axes.pcolormesh`. Common:
        ``cmap='viridis'``, ``shading='flat'``, ``norm=LogNorm()``.

    Returns
    -------
    quadmesh : matplotlib.collections.QuadMesh
        Suitable for ``plt.colorbar``.

    Notes
    -----
    Pre-binning is the user's responsibility (e.g. via
    ``np.histogram2d``). This is intentional: pcolormesh-style
    representation is meaningful only when the bin definition matches
    the science (uniform-z bins? uniform-comoving-distance bins? both?
    log-spaced?). Forcing a default would obscure that choice.

    Cell-edge alignment with the wedge boundary is automatic — the
    angle_edges array spans the wedge, and matplotlib draws the
    quadrilateral cells in (theta, r) data space which renders correctly
    on polar.

    For a histogram of many points use::

        H, ang_e, r_e = np.histogram2d(angles, redshifts,
                                        bins=[40, 30],
                                        range=[[140, 220], [0, 0.2]])
        # histogram2d returns H[ang_idx, r_idx] — transpose for pcolormesh
        cone_pcolormesh(ax, ang_e, r_e, H.T, cmap='viridis')

    Examples
    --------
    ::

        # Galaxy density on a uniform (theta, r) grid
        ang_edges = np.linspace(140, 220, 41)        # 40 angle bins
        r_edges   = np.linspace(0, 0.2, 21)          # 20 radial bins
        H, _, _ = np.histogram2d(galaxy_zs, galaxy_ras,
                                  bins=[r_edges, ang_edges])
        pc = cone_pcolormesh(ax, ang_edges, r_edges, H,
                              cmap='viridis', shading='flat')
        plt.colorbar(pc, ax=ax, label='N galaxies', pad=0.15)
    """
    angle_center = getattr(ax, '_cone_angle_center', 0.0)
    angle_unit = getattr(ax, '_cone_angle_unit', 'deg')
    theta_edges = _angle_to_theta(angle_edges, angle_center, angle_unit)
    r_edges = np.asarray(r_edges, dtype=float)
    # Build 2D meshgrid for pcolormesh on polar axes
    THETA, R = np.meshgrid(theta_edges, r_edges)
    return ax.pcolormesh(THETA, R, np.asarray(C, dtype=float), **kwargs)


# ---------------------------------------------------------------------------
# Bowtie (dual-sided cone) frame
# ---------------------------------------------------------------------------
