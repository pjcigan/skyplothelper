"""Coordinate grid overlay helpers.

``add_second_grid`` draws a secondary frame's grid (e.g. galactic over an
ICRS plot); ``style_grid`` and the ``highlight_gridline(s)`` family are for
emphasizing specific gridlines.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ._stroke import _stroke_path_effects


def add_second_grid(ax: Any, overlay_frame: str = 'galactic',
                    color: str = 'gray', alpha: float = 0.3,
                    linestyle: str = ':', linewidth: float = 0.5,
                    grid: bool = True, ticks: bool = False,
                    tick_labels: bool = False, **kwargs: Any) -> Any:
    """
    Overlay a second coordinate system's grid on a WCSAxes.

    Convenience wrapper around :func:`~skyplothelper.coord_overlay.add_coord_overlay`
    (gridlines) and :func:`~skyplothelper.coord_overlay.add_overlay_ticks`
    (tick marks / labels along the projection's natural boundary).

    Parameters
    ----------
    ax : WCSAxes
    overlay_frame : str
        Frame for the overlay grid: 'galactic', 'fk5', 'icrs',
        'geocentrictrueecliptic', 'supergalactic', etc.
    color : str
    alpha : float
    linestyle : str
    linewidth : float
    grid : bool
        Show grid lines.
    ticks : bool
        Show tick marks for the overlay frame along the projection's
        natural boundary curve (MOL ellipse, SIN circle, etc.).
    tick_labels : bool
        Show tick labels for the overlay frame along the boundary curve.
    **kwargs
        Additional style kwargs forwarded to
        :func:`~skyplothelper.coord_overlay.add_coord_overlay`
        (e.g. ``zorder``).

    Returns
    -------
    overlay : CoordinateOverlay
        The skyplothelper overlay object (for further customization
        via ``.render_ticks``, ``.render_labels``, etc.). Note: this
        is a behavior change from the prior astropy-backed
        implementation, which returned a ``CoordinatesMap``.

    Examples
    --------
    >>> # ICRS plot with galactic grid overlay
    >>> fig, ax = sph.allsky_figure('AIT', frame='ICRS')
    >>> sph.add_second_grid(ax, 'galactic', color='blue', alpha=0.2)

    >>> # Galactic plot with ecliptic overlay (ticks on MOL boundary)
    >>> fig, ax = sph.allsky_figure('MOL', frame='Galactic')
    >>> ov = sph.add_second_grid(ax, 'geocentrictrueecliptic',
    ...                          color='gold', ticks=True, tick_labels=True)
    """
    from .coord_overlay import add_coord_overlay, add_overlay_ticks

    overlay = None
    if grid:
        overlay = add_coord_overlay(ax, frame=overlay_frame,
                                    color=color, alpha=alpha,
                                    ls=linestyle, lw=linewidth, **kwargs)

    if ticks or tick_labels:
        tick_overlay = add_overlay_ticks(
            ax, frame=overlay_frame,
            tick_kwargs={'color': color},
            label_kwargs={'color': color},
            show_ticks=ticks, show_labels=tick_labels)
        if overlay is None:
            overlay = tick_overlay

    return overlay


def style_grid(ax: Any, stroke_lw: float | None = None,
               stroke_color: str | None = None,
               path_effects: list[Any] | None = None,
               color: str | None = None, alpha: float | None = None,
               lw: float | None = None, ls: str | None = None,
               **kwargs: Any) -> None:
    """
    Restyle the coordinate grid on a WCSAxes after creation.

    Applies styling retroactively by updating the internal grid kwargs
    on each coordinate axis. Call ``fig.canvas.draw()`` afterward to
    see changes.

    The most common use is adding a stroke (outline) for readability
    on busy backgrounds:

        ``sph.style_grid(ax, stroke_lw=3, stroke_color='k')``

    Parameters
    ----------
    ax : WCSAxes
    stroke_lw : float, optional
        Stroke (outline) linewidth. If set, creates a ``withStroke``
        path effect.
    stroke_color : str, optional
        Stroke color. Default 'k' if stroke_lw is set.
    path_effects : list, optional
        Explicit list of path effects. Overrides stroke_lw/stroke_color.
    color : str, optional
        Grid line color (overrides existing).
    alpha : float, optional
        Grid transparency (overrides existing).
    lw : float, optional
        Grid linewidth (overrides existing).
    ls : str, optional
        Grid linestyle (overrides existing).
    **kwargs
        Additional kwargs merged into the grid line properties.

    Examples
    --------
    >>> ax = sph.make_wcs_frame(111, 'AIT', center=180)
    >>> sph.style_grid(ax, stroke_lw=3, stroke_color='k')

    >>> sph.style_grid(ax, color='w', alpha=0.6, lw=0.5,
    ...                stroke_lw=2, stroke_color='0.3')

    Notes
    -----
    **Implementation:** Works by re-calling ``ax.coords.grid()`` with
    the new kwargs. Astropy merges these into the existing internal
    ``_grid_lines_kwargs`` dict, so previously-set properties are
    preserved unless explicitly overridden.

    **Alternative:** You can also pass ``path_effects`` directly when
    first creating the grid::

        pe = [PathEffects.withStroke(linewidth=3, foreground='k')]
        ax.coords.grid(color='w', path_effects=pe)

    Both approaches produce identical results. ``style_grid()`` is
    convenient when you want to restyle a grid that was already created
    by ``make_wcs_frame(grid=True)``.
    """
    # Build path effects from stroke shorthand
    # NOTE the local contract: here ``stroke_lw`` is what enables the stroke
    # and ``stroke_color=None`` means "default to black" -- unlike the rest of
    # the package, where a ``None`` color disables it. The default is resolved
    # first so the shared helper can be used without changing that meaning.
    if path_effects is None and stroke_lw is not None:
        path_effects = _stroke_path_effects(stroke_color or 'k', stroke_lw)

    # Build kwargs for ax.coords.grid() — re-calling it merges into
    # the existing _grid_lines_kwargs on each coordinate.
    grid_kw: dict[str, Any] = {}
    if color is not None:
        grid_kw['color'] = color
    if alpha is not None:
        grid_kw['alpha'] = alpha
    if lw is not None:
        grid_kw['lw'] = lw
    if ls is not None:
        grid_kw['ls'] = ls
    if path_effects is not None:
        grid_kw['path_effects'] = path_effects
    grid_kw.update(kwargs)

    ax.coords.grid(**grid_kw)


def highlight_gridline(ax: Any, value: float, coord: str = 'lon',
                       color: Any = 'red', lw: float = 2,
                       alpha: float = 1.0, ls: str = '-',
                       n_samples: int = 1000,
                       stroke_lw: float | None = None,
                       stroke_color: str | None = None, zorder: int = 5,
                       label: str | None = None,
                       **kwargs: Any) -> list[Any]:
    """
    Draw a highlighted meridian or parallel on a WCSAxes.

    Draws a coordinate line at a specific longitude or latitude value
    with custom styling, on top of (or instead of) the regular grid.
    Useful for marking the Galactic center meridian, the ecliptic equator,
    specific survey boundaries, or for visual tracking in animations.

    Parameters
    ----------
    ax : WCSAxes
    value : float
        Coordinate value in degrees. For lon: the longitude of the
        meridian. For lat: the latitude of the parallel.
    coord : str
        Which coordinate: 'lon'/'meridian'/'ra'/'l' for a meridian,
        'lat'/'parallel'/'dec'/'b' for a parallel.
    color : str
        Line color.
    lw : float
        Linewidth.
    alpha : float
    ls : str
        Linestyle ('-', '--', ':', etc.)
    n_samples : int
        Number of points along the line. Higher values give smoother
        curves on projections with strong distortion.
    stroke_lw : float, optional
        Outline stroke linewidth.
    stroke_color : str, optional
        Outline stroke color.
    zorder : int
        Drawing order (default 5, above grid but below data).
    label : str, optional
        Legend label for the line.
    **kwargs
        Additional kwargs passed to ``ax.plot()``.

    Returns
    -------
    lines : list of Line2D
        The plotted line artist(s).

    Examples
    --------
    >>> # Highlight the Galactic center meridian
    >>> sph.highlight_gridline(ax, 0, 'lon', color='red', lw=2)

    >>> # Mark the ecliptic equator with a dashed gold line
    >>> sph.highlight_gridline(ax, 0, 'lat', color='gold', lw=2, ls='--')

    >>> # Multiple highlighted meridians for an animation
    >>> for lon in [0, 90, 180, 270]:
    ...     sph.highlight_gridline(ax, lon, 'lon', color=f'C{lon//90}')

    >>> # With stroke for visibility on dark backgrounds
    >>> sph.highlight_gridline(ax, 30, 'lat', color='w', lw=1.5,
    ...                        stroke_lw=3, stroke_color='k')
    """
    coord_lower = coord.lower()
    is_lon = coord_lower in ('lon', 'longitude', 'meridian', 'ra', 'l',
                              'glon', 'slon', 'elon', '0', 'x')
    is_lat = coord_lower in ('lat', 'latitude', 'parallel', 'dec', 'b',
                              'glat', 'slat', 'elat', '1', 'y')
    if not (is_lon or is_lat):
        raise ValueError(
            f"coord must be 'lon'/'meridian' or 'lat'/'parallel', got '{coord}'")

    # Build the coordinate line
    if is_lon:
        lats = np.linspace(-90, 90, n_samples)
        lons = np.full_like(lats, float(value))
    else:
        lons = np.linspace(0, 360, n_samples)
        lats = np.full_like(lons, float(value))

    # Build plot kwargs
    plot_kw = dict(color=color, lw=lw, alpha=alpha, ls=ls,
                   zorder=zorder, **kwargs)
    if label is not None:
        plot_kw['label'] = label

    # Path effects
    pe: list[Any] = []
    if stroke_lw is not None:
        # Same local contract as above: None color -> black, not "disabled".
        pe.extend(_stroke_path_effects(stroke_color or 'k', stroke_lw) or [])
    if pe:
        plot_kw['path_effects'] = pe

    # Determine the transform
    # For WCSAxes, use get_transform('world')
    # For cartopy GeoAxes, use PlateCarree
    if getattr(ax, '_sph_is_cartopy', False):
        transform = ax._sph_data_crs
    elif hasattr(ax, 'get_transform'):
        try:
            transform = ax.get_transform('world')
        except Exception:
            transform = ax.transData
    else:
        transform = ax.transData

    plot_kw['transform'] = transform

    # On a WCS frame a parallel spans every longitude and crosses the
    # projection's wrap (antimeridian) seam; sampled monotonically in lon it
    # would otherwise be drawn as one straight segment streaking across the
    # canvas. Wrap each longitude into the projection window and NaN-break the
    # seam jump so the line draws off one frame edge and onto the other.
    # (Constant-lon meridians are unaffected — all samples wrap alike.)
    # Cartopy GeoAxes handle geographic wrapping themselves, so skip them.
    if not getattr(ax, '_sph_is_cartopy', False) and hasattr(ax, 'wcs'):
        from .data_plots import _wrap_break_lonlat
        from .wcs_frame import _get_wcs_center_lon
        try:
            center = _get_wcs_center_lon(ax)
        except Exception:
            center = 0.0
        lons, lats = _wrap_break_lonlat(lons, lats, center)

    lines = ax.plot(lons, lats, **plot_kw)
    return lines


def highlight_gridlines(ax: Any, lon_values: Sequence[float] | None = None,
                        lat_values: Sequence[float] | None = None,
                        lon_colors: list[str] | None = None,
                        lat_colors: list[str] | None = None,
                        lon_cmap: Any = None, lat_cmap: Any = None,
                        color: str = 'red', lw: float = 2,
                        alpha: float = 1.0, ls: str = '-',
                        stroke_lw: float | None = None,
                        stroke_color: str | None = None,
                        zorder: int = 5, **kwargs: Any) -> list[Any]:
    """
    Highlight multiple meridians and/or parallels in one call.

    Colors can be specified per-line via explicit lists, or sampled
    from a colormap across the set of values.

    Parameters
    ----------
    ax : WCSAxes
    lon_values : list of float, optional
        Longitudes of meridians to highlight (degrees).
    lat_values : list of float, optional
        Latitudes of parallels to highlight (degrees).
    lon_colors : list of str, optional
        Per-meridian colors. If shorter than lon_values, cycles.
        If None and lon_cmap is None, uses ``color`` for all.
    lat_colors : list of str, optional
        Per-parallel colors. Same cycling behavior.
    lon_cmap : str or Colormap, optional
        Colormap to sample meridian colors from, evenly spaced across
        [0, 1]. Overrides lon_colors. Any valid matplotlib colormap
        name or Colormap instance.
    lat_cmap : str or Colormap, optional
        Colormap to sample parallel colors from. Overrides lat_colors.
    color : str
        Default color when neither per-line colors nor cmap specified.
    lw, alpha, ls, stroke_lw, stroke_color, zorder :
        Passed to ``highlight_gridline()``.
    **kwargs
        Additional kwargs passed through.

    Returns
    -------
    artists : list
        All line artists created.

    Examples
    --------
    >>> # Color-coded meridians from a colormap
    >>> sph.highlight_gridlines(ax,
    ...     lon_values=np.arange(0, 360, 30), lon_cmap='hsv')

    >>> # Parallels with a diverging colormap (blue at poles, red at equator)
    >>> sph.highlight_gridlines(ax,
    ...     lat_values=np.arange(-60, 61, 15), lat_cmap='RdYlBu_r')

    >>> # Mix: cmap for meridians, explicit colors for parallels
    >>> sph.highlight_gridlines(ax,
    ...     lon_values=np.arange(0, 360, 30), lon_cmap='twilight',
    ...     lat_values=[0], lat_colors=['gold'], lw=2)

    >>> # Explicit color list (cycles if shorter than values)
    >>> sph.highlight_gridlines(ax,
    ...     lon_values=[0, 90, 180, 270],
    ...     lon_colors=['red', 'green', 'blue', 'orange'])
    """
    artists: list[Any] = []

    if lon_values is not None:
        lon_vals_list = list(lon_values)
        n_lon = len(lon_vals_list)

        # Resolve colors: cmap > explicit list > default
        lon_colors_resolved: list[Any] | None
        if lon_cmap is not None:
            cmap = plt.get_cmap(lon_cmap)
            lon_colors_resolved = [cmap(i / max(n_lon - 1, 1))
                                   for i in range(n_lon)]
        elif lon_colors is not None:
            lon_colors_resolved = lon_colors
        else:
            lon_colors_resolved = None

        for i, lon in enumerate(lon_vals_list):
            if lon_colors_resolved is not None:
                c = lon_colors_resolved[i % len(lon_colors_resolved)]
            else:
                c = color
            lines = highlight_gridline(ax, lon, 'lon', color=c, lw=lw,
                                       alpha=alpha, ls=ls, stroke_lw=stroke_lw,
                                       stroke_color=stroke_color, zorder=zorder,
                                       **kwargs)
            artists.extend(lines)

    if lat_values is not None:
        lat_vals_list = list(lat_values)
        n_lat = len(lat_vals_list)

        # Resolve colors: cmap > explicit list > default
        lat_colors_resolved: list[Any] | None
        if lat_cmap is not None:
            cmap = plt.get_cmap(lat_cmap)
            lat_colors_resolved = [cmap(i / max(n_lat - 1, 1))
                                   for i in range(n_lat)]
        elif lat_colors is not None:
            lat_colors_resolved = lat_colors
        else:
            lat_colors_resolved = None

        for i, lat in enumerate(lat_vals_list):
            if lat_colors_resolved is not None:
                c = lat_colors_resolved[i % len(lat_colors_resolved)]
            else:
                c = color
            lines = highlight_gridline(ax, lat, 'lat', color=c, lw=lw,
                                       alpha=alpha, ls=ls, stroke_lw=stroke_lw,
                                       stroke_color=stroke_color, zorder=zorder,
                                       **kwargs)
            artists.extend(lines)

    return artists
