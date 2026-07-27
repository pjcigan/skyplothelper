"""Cartopy-backed alternative frame builders.

Optional dependency: ``cartopy``. If not installed, calling these will
raise an informative ImportError. ``make_cartopy_frame`` parallels
``make_wcs_frame`` but uses cartopy's projection set (richer non-FITS
options + Earth feature overlays).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    from cartopy.feature.nightshade import Nightshade
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False
    Nightshade = None


def _require_cartopy() -> None:
    if not _HAS_CARTOPY:
        raise ImportError(
            "This functionality requires cartopy. "
            "Install with: pip install cartopy"
        )


_CARTOPY_PROJECTIONS = {
    # Pseudocylindrical
    'mollweide':      ('Mollweide',       lambda c: {'central_longitude': c}),
    'mol':            ('Mollweide',       lambda c: {'central_longitude': c}),
    'robinson':       ('Robinson',        lambda c: {'central_longitude': c}),
    'sinusoidal':     ('Sinusoidal',      lambda c: {'central_longitude': c}),
    'sfl':            ('Sinusoidal',      lambda c: {'central_longitude': c}),
    'eckert_iv':      ('EckertIV',        lambda c: {'central_longitude': c}),
    'eckert_vi':      ('EckertVI',        lambda c: {'central_longitude': c}),
    # Cylindrical
    'plate_carree':   ('PlateCarree',     lambda c: {'central_longitude': c}),
    'platecarree':    ('PlateCarree',     lambda c: {'central_longitude': c}),
    'car':            ('PlateCarree',     lambda c: {'central_longitude': c}),
    'mercator':       ('Mercator',        lambda c: {'central_longitude': c}),
    'mer':            ('Mercator',        lambda c: {'central_longitude': c}),
    # Pseudoazimuthal
    'aitoff':         ('AzimuthalEquidistant', lambda c: {'central_longitude': c}),
    'hammer':         ('LambertAzimuthalEqualArea', lambda c: {'central_longitude': c}),
    # Azimuthal
    'orthographic':   ('Orthographic',    lambda c: {'central_longitude': c,
                                                      'central_latitude': 0}),
    'sin':            ('Orthographic',    lambda c: {'central_longitude': c,
                                                      'central_latitude': 0}),
    'lambert_azimuthal': ('LambertAzimuthalEqualArea',
                                          lambda c: {'central_longitude': c}),
    'zea':            ('LambertAzimuthalEqualArea',
                                          lambda c: {'central_longitude': c}),
    'stereographic':  ('Stereographic',   lambda c: {'central_longitude': c,
                                                      'central_latitude': 0}),
    'stg':            ('Stereographic',   lambda c: {'central_longitude': c,
                                                      'central_latitude': 0}),
    # Conic
    'lambert_conformal': ('LambertConformal', lambda c: {'central_longitude': c}),
    'albers':         ('AlbersEqualArea', lambda c: {'central_longitude': c}),
    # Interrupted / specialty
    'interrupted_goode': ('InterruptedGoodeHomolosine',
                                          lambda c: {'central_longitude': c}),
    'goode':          ('InterruptedGoodeHomolosine',
                                          lambda c: {'central_longitude': c}),
}


def _resolve_cartopy_crs(projection: Any, center_lon: float = 0.,
                         center_lat: float = 0.) -> Any:
    """
    Resolve a projection name to a cartopy CRS object.

    Parameters
    ----------
    projection : str or cartopy CRS
        Projection name (case-insensitive, underscores/hyphens interchangeable)
        or a pre-built cartopy CRS instance.
    center_lon : float
        Central longitude in degrees
    center_lat : float
        Central latitude in degrees (used for azimuthal projections)

    Returns
    -------
    crs : cartopy.crs.Projection
    """
    if not _HAS_CARTOPY:
        raise ImportError(
            "cartopy is required for make_cartopy_frame(). "
            "Install with: pip install cartopy"
        )

    # If already a CRS object, return as-is
    if hasattr(projection, 'proj4_params'):
        return projection

    norm = projection.lower().strip().replace('-', '_').replace(' ', '_')

    if norm in _CARTOPY_PROJECTIONS:
        cls_name, kwargs_fn = _CARTOPY_PROJECTIONS[norm]
        kwargs = kwargs_fn(center_lon)
        # Override central_latitude for projections that accept it
        if center_lat != 0. and 'central_latitude' in kwargs:
            kwargs['central_latitude'] = center_lat
        elif center_lat != 0. and cls_name in ('Orthographic', 'Stereographic',
                                                 'LambertAzimuthalEqualArea'):
            kwargs['central_latitude'] = center_lat
        crs_cls = getattr(ccrs, cls_name)
        return crs_cls(**kwargs)

    # Try direct attribute lookup on ccrs (e.g., 'NearsidePerspective')
    norm_camel = norm.replace('_', '')
    for attr in dir(ccrs):
        if attr.lower() == norm_camel:
            crs_cls = getattr(ccrs, attr)
            try:
                return crs_cls(central_longitude=center_lon)
            except TypeError:
                return crs_cls()

    available = sorted(set(_CARTOPY_PROJECTIONS.keys()))
    raise ValueError(
        f"Unknown cartopy projection '{projection}'. "
        f"Available: {', '.join(available)}"
    )


def _cartopy_label_color(ax: Any = None, light: str = '0.3') -> str:
    """Gridline / axis-label color for a cartopy frame.

    These labels were a deliberately muted ``'0.3'`` — softer than the
    primary ink, which suits a busy geographic map. So they are *not*
    resolved to a tick rcParam: that would promote them to full strength and
    change every existing light-theme render. Only the dark case was broken,
    and :func:`muted_ink` fixes exactly that while leaving light untouched.

    Cartopy is also not WCSAxes: it drives Gridliner label text from its own
    style dicts and inherits no tick rcParams, so the value has to be
    resolved here and handed over explicitly.
    """
    from .style import muted_ink
    ref = ax if ax is not None else rcParams['axes.facecolor']
    return muted_ink(ref, light=light)


def make_cartopy_frame(subplotnumber: Any = 111, projection: Any = 'mollweide',
                       center: SkyCoord | float | tuple[float, float] = 180.,
                       frame: str = 'ICRS', grid: bool = True,
                       gridcolor: str = '0.6', gridalpha: float = 0.5,
                       gridlw: float | None = None,
                       gridls: str | None = None,
                       lon_spacing: float = 30, lat_spacing: float = 30,
                       coastlines: bool = False, coastline_color: str = '0.3',
                       coastline_lw: float = 0.5, land: bool = False,
                       ocean: bool = False,
                       land_color: str = '0.85', ocean_color: str = 'lightblue',
                       nightshade: Any = None, nightshade_alpha: float = 0.3,
                       invert_lon: bool | None = None,
                       global_extent: bool = True,
                       fig: Any = None, auto_fontsize: bool = True,
                       label_color: Any = None,
                       **kwargs: Any) -> Any:
    """
    Create a cartopy GeoAxes with skyplothelper-like API.

    Convenience wrapper around cartopy projections, providing access to
    cartopy features (interrupted projections, coastlines, land/ocean
    shading, nightshade) that aren't available in astropy WCSAxes.

    For astronomical data plotting, use ``transform=ccrs.PlateCarree()``
    (or the returned ``data_crs``) when calling scatter/plot, just as
    you would use ``transform=ax.get_transform('world')`` on WCSAxes.

    Parameters
    ----------
    subplotnumber : int
        Subplot specification (e.g., 111, 121)
    projection : str or cartopy CRS
        Projection name (case-insensitive). Common choices:
        'mollweide', 'robinson', 'plate_carree', 'orthographic',
        'sinusoidal', 'eckert_iv', 'lambert_azimuthal',
        'interrupted_goode', 'hammer', 'stereographic'.
        Can also pass a pre-built cartopy CRS instance.
    center : float, tuple, or SkyCoord
        A scalar :class:`~astropy.coordinates.SkyCoord` is converted into the
        frame being built, matching :func:`~skyplothelper.make_wcs_frame`.
        Center longitude (float) or (lon, lat) tuple in degrees.
    frame : str
        Coordinate frame for axis labels: 'ICRS', 'Galactic', etc.
        Note: cartopy always works in geographic (lon/lat) coordinates.
        For Galactic frame, data must be pre-converted to Galactic
        coordinates before plotting.
    grid : bool
        Draw coordinate grid (gridlines)
    gridcolor, gridalpha : str, float
        Grid styling
    gridlw : float, optional
        Grid line width. ``None`` (default) leaves cartopy's own.
    gridls : str, optional
        Grid line style. ``None`` (default) keeps this backend's historical
        dotted grid.
    lon_spacing, lat_spacing : float
        Grid line spacing in degrees
    coastlines : bool
        Draw Earth coastlines
    coastline_color : str
        Coastline color
    coastline_lw : float
        Coastline linewidth
    land, ocean : bool
        Shade land masses / ocean areas
    land_color, ocean_color : str
        Fill colors for land / ocean
    label_color : color, optional
        Color of the gridline / axis labels. ``None`` (default) picks a
        muted tone that reads against the axes background — the familiar
        ``'0.3'`` on a light theme, a light gray on a dark one. Cartopy
        styles these labels from its own Gridliner dicts and inherits no
        tick rcParams, so this is the only way to set them.
    nightshade : datetime, astropy Time, or str, optional
        If provided, add nightshade overlay for this time. Anything
        :func:`~skyplothelper.to_time` accepts works here.
    nightshade_alpha : float
        Nightshade transparency
    invert_lon : bool or None
        Controls x-axis orientation. ``True`` flips the axis so
        longitude increases to the left (astronomical / RA convention);
        ``False`` keeps the standard cartographic west-to-east
        orientation. Default ``None`` auto-detects: if any Earth-feature
        flag (``coastlines``, ``land``, ``ocean``, ``nightshade``) is
        set the axes are treated as Earth-style and *not* inverted;
        otherwise the axes default to the sky-style inverted
        orientation. Pass an explicit bool to override.
    global_extent : bool
        If True (default), set global extent. Set False for regional views.
    figsize : tuple, optional
        Figure size. Defaults to (14, 7) for all-sky, (8, 8) for globe.
    fig : Figure, optional
        Existing figure. If None, creates one (only if figsize given or
        subplotnumber is 111).
    **kwargs
        Additional kwargs passed to ``fig.add_subplot()``.

    Returns
    -------
    ax : cartopy GeoAxes
        The cartopy axes. Plot data with:
        ``ax.plot(lon, lat, transform=ccrs.PlateCarree())``

    Notes
    -----
    Unlike ``make_wcs_frame()``, cartopy axes do not use WCS or FITS
    headers. Coordinate transforms use cartopy's PROJ-based system.
    Data should be plotted using ``transform=ccrs.PlateCarree()`` for
    lon/lat data in degrees.

    **Coastlines / land / ocean features** require cartopy to download
    Natural Earth shapefiles on first use. This needs an internet
    connection; offline environments will raise a download error. The
    files are cached after first download.

    **Gridline labels:** Cartopy's ``draw_labels=True`` works on most
    projections (Mollweide, Robinson, PlateCarree, etc.) but may fail
    silently on some (Interrupted Goode). When gridline labels succeed,
    the manual axis labels (``RA (°)``, etc.) are suppressed to avoid
    overlap. When they fail, the manual labels are shown as a fallback.

    The returned axes has convenience attributes:
    - ``ax._sph_data_crs`` : the PlateCarree CRS for data transforms
    - ``ax._sph_frame`` : the coordinate frame string
    - ``ax._sph_center_lon`` : the center longitude

    Examples
    --------
    >>> ax = sph.make_cartopy_frame(111, 'mollweide', center=180)
    >>> ax.scatter(ra, dec, transform=ccrs.PlateCarree(), s=1)

    >>> # Earth map with coastlines (auto: invert_lon=False)
    >>> ax = sph.make_cartopy_frame(111, 'robinson', center=0,
    ...     coastlines=True, land=True)

    >>> # Interrupted Goode homolosine (cartopy-only projection)
    >>> ax = sph.make_cartopy_frame(111, 'interrupted_goode', center=0)

    >>> # Globe view with nightshade (auto: invert_lon=False)
    >>> from datetime import datetime
    >>> ax = sph.make_cartopy_frame(111, 'orthographic', center=(-80, 40),
    ...     coastlines=True, nightshade=datetime.now())
    """
    if not _HAS_CARTOPY:
        raise ImportError(
            "cartopy is required for make_cartopy_frame(). "
            "Install with: pip install cartopy"
        )

    # Parse center. A SkyCoord is converted into the frame being built.
    if hasattr(center, 'transform_to'):
        from .geometry._parsing import _coords_to_frame_deg
        center = _coords_to_frame_deg(center, frame)
    if isinstance(center, (list, tuple)):
        center_lon, center_lat = float(center[0]), float(center[1])
    else:
        center_lon = float(center)
        center_lat = 0.

    # Resolve projection
    proj_crs = _resolve_cartopy_crs(projection, center_lon, center_lat)

    # ``figsize=`` is a figure-level setting — reject it with a
    # pointer to the right place (this is an axis-builder).
    if 'figsize' in kwargs:
        raise TypeError(
            "make_cartopy_frame() does not accept ``figsize=`` — "
            "it is an axis-builder. Create the figure first via "
            "``plt.figure(figsize=...)`` and pass it as ``fig=fig``."
        )

    # ``subplotnumber`` may also be a pre-existing matplotlib Axes
    # — swap it for a cartopy axes at the same SubplotSpec position.
    # Mirrors the same pattern in ``make_wcs_frame``.
    from matplotlib.axes import Axes
    if isinstance(subplotnumber, Axes):
        existing_ax = subplotnumber
        existing_fig = existing_ax.figure
        if fig is None:
            fig = existing_fig
        elif fig is not existing_fig:
            raise ValueError(
                "make_cartopy_frame() received both an explicit "
                "``fig=`` and a ``subplotnumber=`` Axes from a "
                "different figure — pass at most one figure context."
            )
        spec = existing_ax.get_subplotspec()
        if spec is None:
            raise ValueError(
                "Pre-existing Axes passed via ``subplotnumber=`` has "
                "no SubplotSpec (probably created via "
                "``fig.add_axes(rect)``). Pass a SubplotSpec, an int "
                "subplot number, or an Axes that lives in a subplot "
                "grid."
            )
        existing_ax.remove()
        subplotnumber = spec

    # Determine if globe-like (azimuthal with center_lat != 0)
    is_globe = isinstance(proj_crs, (ccrs.Orthographic, ccrs.Stereographic))  # noqa: F841

    # Use existing figure or fall back to plt.gcf().
    if fig is None:
        fig = plt.gcf()

    # Create axes
    ax = fig.add_subplot(subplotnumber, projection=proj_crs, **kwargs)

    # Extent
    if global_extent:
        try:
            ax.set_global()
        except Exception:
            pass  # Some projections don't support set_global

    # Features
    if coastlines:
        ax.coastlines(color=coastline_color, linewidth=coastline_lw)
    if land:
        ax.add_feature(cf.LAND, facecolor=land_color, edgecolor='none')
    if ocean:
        ax.add_feature(cf.OCEAN, facecolor=ocean_color, edgecolor='none')

    # Nightshade
    if nightshade is not None:
        # cartopy duck-types the date, so everything but a datetime has to be
        # converted. Going through _to_datetime also fixes the scale: the old
        # Time.datetime kept TT/TDB as-is, where cartopy wants UTC.
        from ._timeinput import _to_datetime
        ax.add_feature(Nightshade(
            _to_datetime(nightshade, caller='make_cartopy_map'),
            alpha=nightshade_alpha))

    # Gridlines
    _has_gridline_labels = False
    if grid:
        # Try with labels first; fall back to no labels if projection
        # doesn't support them (most non-PlateCarree/Mercator projections)
        try:
            # cartopy's Gridliner takes its styling at construction, so the
            # sentinels are resolved here rather than set afterwards.
            gl = ax.gridlines(draw_labels=True, color=gridcolor,
                              alpha=gridalpha,
                              linestyle=(':' if gridls is None else gridls),
                              x_inline=False, y_inline=False,
                              **({} if gridlw is None
                                 else {'linewidth': gridlw}))
            gl.top_labels = False
            gl.right_labels = False
            # Size already came from rcParams; the color did not, so a dark
            # theme got near-black gridline labels on a dark canvas.
            #
            # These are cartopy Gridliner labels, NOT matplotlib tick labels
            # and NOT WCSAxes coord labels — cartopy styles them from these
            # dicts and inherits no tick rcParams on its own, so the value
            # has to be resolved and handed over explicitly. ``*.labelcolor``
            # defaults to the sentinel ``'inherit'``, meaning "use
            # ``*.color``", so it cannot be passed through raw.
            _lbl_color = (label_color if label_color is not None
                          else _cartopy_label_color(ax))
            gl.xlabel_style = {'size': rcParams.get('xtick.labelsize', 10),
                               'color': _lbl_color}
            gl.ylabel_style = {'size': rcParams.get('ytick.labelsize', 10),
                               'color': _lbl_color}
            _has_gridline_labels = True
        except Exception:
            gl = ax.gridlines(draw_labels=False, color=gridcolor,
                              alpha=gridalpha,
                              linestyle=(':' if gridls is None else gridls),
                              **({} if gridlw is None
                                 else {'linewidth': gridlw}))

        # Set grid spacing
        lon_ticks = np.arange(-180, 181, lon_spacing)
        lat_ticks = np.arange(-90, 91, lat_spacing)
        gl.xlocator = plt.FixedLocator(lon_ticks.tolist())
        gl.ylocator = plt.FixedLocator(lat_ticks.tolist())

    # Resolve invert_lon default: when unset, treat axes that have
    # any Earth feature requested (coastlines / land / ocean /
    # nightshade) as cartographic — don't invert the longitude axis.
    # Sky-only views (no Earth features) default to inverted, matching
    # the RA convention used by make_wcs_frame.
    if invert_lon is None:
        invert_lon = not (coastlines or land or ocean
                          or (nightshade is not None))

    # Invert longitude for astronomical convention
    if invert_lon and not is_globe:
        try:
            ax.invert_xaxis()
        except Exception:
            pass

    # Store metadata for compatibility with other skyplothelper functions
    data_crs = ccrs.PlateCarree()
    ax._sph_data_crs = data_crs
    ax._sph_frame = frame.lower() if frame.lower() in (
        'galactic', 'supergalactic', 'ecliptic') else 'icrs'
    ax._sph_center_lon = center_lon
    ax._sph_is_cartopy = True

    # Frame-appropriate axis labels — only add manual text labels when
    # cartopy's gridline labels are NOT active (to avoid overlap)
    if not _has_gridline_labels:
        from .constants import frame_short_labels
        lon_label, lat_label = frame_short_labels(
            getattr(ax, '_sph_frame', None), default=('Lon', 'Lat'))
        label_fs = rcParams.get('axes.labelsize', 12)
        label_color = (label_color if label_color is not None
                       else _cartopy_label_color(ax))
        ax.text(0.5, -0.03, f'{lon_label} (°)', transform=ax.transAxes,
                ha='center', va='top', fontsize=label_fs, color=label_color)
        ax.text(-0.03, 0.5, f'{lat_label} (°)', transform=ax.transAxes,
                ha='right', va='center', rotation=90, fontsize=label_fs,
                color=label_color)

    # Auto-shrink gridliner label fontsize to the available axes width.
    # A canvas.draw() is needed so the gridliner has actually rendered
    # its label artists for char-count introspection. The whole block
    # is try/excepted: auto-fontsize is a convenience, never a reason
    # for make_cartopy_frame to fail.
    if auto_fontsize:
        from .autosize import auto_size_ticklabels
        try:
            ax.figure.canvas.draw()
        except Exception:
            pass
        try:
            auto_size_ticklabels(ax)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"make_cartopy_frame: auto_fontsize failed "
                f"({type(exc).__name__}: {exc}); falling back to "
                f"rcParams default. Pass auto_fontsize=False to suppress.",
                UserWarning, stacklevel=2)

    return ax


def cartopy_figure(projection: Any = 'mollweide',
                   center: SkyCoord | float | tuple[float, float] = 180., frame: str = 'ICRS',
                   figsize: tuple[float, float] | None = None,
                   coastlines: bool = False, land: bool = False,
                   invert_lon: bool | None = None, grid: bool = True,
                   **kwargs: Any) -> tuple[Any, Any]:
    """
    Create a cartopy figure with one call, returning (fig, ax).

    Thin wrapper around ``make_cartopy_frame()`` that also creates the
    figure. Analogous to ``allsky_figure()`` for WCSAxes.

    Parameters
    ----------
    projection : str or cartopy CRS
    center : float, tuple, or SkyCoord
        A scalar :class:`~astropy.coordinates.SkyCoord` is converted into the
        frame being built, matching :func:`~skyplothelper.make_wcs_frame`.
    frame : str
    figsize : tuple, optional
    coastlines : bool
    land : bool
    invert_lon : bool or None
        Default ``None`` auto-detects from Earth-feature flags; see
        :func:`make_cartopy_frame` for details.
    grid : bool
    **kwargs
        Passed to make_cartopy_frame()

    Returns
    -------
    fig : Figure
    ax : cartopy GeoAxes

    Examples
    --------
    >>> fig, ax = sph.cartopy_figure('robinson', center=180, frame='Galactic')
    >>> ax.scatter(glon, glat, transform=ccrs.PlateCarree(), s=1)

    >>> fig, ax = sph.cartopy_figure('interrupted_goode', coastlines=True)
    """
    fig = plt.figure(figsize=figsize or (14, 7))
    ax = make_cartopy_frame(111, projection=projection, center=center,
                            frame=frame, grid=grid, coastlines=coastlines,
                            land=land, invert_lon=invert_lon, fig=fig,
                            **kwargs)
    return fig, ax


def list_cartopy_projections() -> None:
    """
    List available cartopy projection names for make_cartopy_frame().

    Returns a sorted list of accepted projection name strings.
    """
    # Deduplicate by CRS class name
    seen: dict[str, list[str]] = {}
    for name, (cls_name, _) in sorted(_CARTOPY_PROJECTIONS.items()):
        if cls_name not in seen:
            seen[cls_name] = []
        seen[cls_name].append(name)

    print("Available cartopy projections for make_cartopy_frame():")
    print(f"{'Cartopy class':<35} {'Accepted names'}")
    print(f"{'─' * 35} {'─' * 40}")
    for cls_name, names in sorted(seen.items()):
        print(f"  {cls_name:<33} {', '.join(names)}")
    print("\nAny cartopy CRS class name also works (case-insensitive).")
    print("Or pass a pre-built cartopy CRS instance directly.")

