"""Earth-baseline (great-circle) plotting.

``plot_baselines`` draws great-circle baselines between named radio
telescope sites or other Earth coordinates. Independent of
:func:`skyplothelper.overlays.planes.add_great_circle`, which traces
*full* circles defined by a pole or coordinate-frame equator and is
WCSAxes-only — the two functions share the spherical-arc math
(both consume :func:`skyplothelper.globe.spherical.great_circle_arc`)
but differ in user-facing scope: this helper renders ARC SEGMENTS
between named endpoints plus markers / labels / length annotations,
on plain mpl / WCSAxes / globe axes alike.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from astropy.coordinates import SkyCoord  # noqa: F401
from matplotlib import rcParams

from ..constants import planet_radii
from .plotting import _is_globe_axes, _wrap_fix_lons, plot_line_globe
from .spherical import (
    great_circle_arc,
    great_circle_distance,
    midpoint,
    orthographic_visibility,
)


def _mirror_to_near_side(lons: Any, lats: Any, center_lon: float,
                         center_lat: float) -> tuple[Any, Any]:
    """Reflect far-side ``(lon, lat)`` through the sky plane to their near-side
    mirror images, which share the same orthographic disk position.

    A back-hemisphere point projects to the same spot on the globe disk as its
    reflection through the plane that passes through the globe center
    perpendicular to the view axis. The mirror point is on the near side, so it
    projects through the WCS normally — exact for any ``CDELT`` / ``LONPOLE`` /
    ``PC`` combination. (The globe frame builders emit ``LONPOLE=0`` — the
    native system rotated 180° vs the ``LONPOLE=180`` default, compensated by
    the ``CDELT`` signs — so a hand-rolled Snyder-ortho + ``CRPIX``/``CDELT``
    shortcut lands every far-side point reflected through the disk center.)
    ``NaN`` inputs pass through as ``NaN``.
    """
    def _vec(lon: Any, lat: Any) -> Any:
        lonr = np.radians(np.asarray(lon, dtype=float))
        latr = np.radians(np.asarray(lat, dtype=float))
        cl = np.cos(latr)
        return np.stack([cl * np.cos(lonr), cl * np.sin(lonr),
                         np.sin(latr)], axis=-1)

    n = _vec(center_lon, center_lat)
    v = _vec(lons, lats)
    v = v - 2.0 * np.sum(v * n, axis=-1)[..., None] * n
    mir_lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    mir_lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    return mir_lon, mir_lat


def _normalize_sites(sites: Any) -> list[tuple[str, float, float]]:
    """
    Normalize a ``sites`` argument (dict, list-of-tuples, or sequence of
    astropy EarthLocation / SkyCoord) to a list of ``(name, lon_deg,
    lat_deg)`` tuples.
    """
    # Try astropy types (optional).
    try:
        import astropy.units as u
        from astropy.coordinates import EarthLocation, SkyCoord
    except ImportError:
        u = None
        EarthLocation = None
        SkyCoord = None

    def _extract_lonlat(obj: Any) -> tuple[float, float]:
        """Return (lon_deg, lat_deg) from a point-like object."""
        if EarthLocation is not None and isinstance(obj, EarthLocation):
            return float(obj.lon.to('deg').value), float(obj.lat.to('deg').value)
        if SkyCoord is not None and isinstance(obj, SkyCoord):
            return (float(obj.spherical.lon.to('deg').value),
                    float(obj.spherical.lat.to('deg').value))
        if u is not None and isinstance(obj, tuple) and len(obj) == 2 \
                and isinstance(obj[0], u.Quantity):
            return (float(obj[0].to('deg').value),
                    float(obj[1].to('deg').value))
        # Assume numeric (lon, lat)
        return float(obj[0]), float(obj[1])

    out = []
    if isinstance(sites, dict):
        for name, pt in sites.items():
            lon, lat = _extract_lonlat(pt)
            out.append((str(name), lon, lat))
    else:
        # Assume iterable of (name, lon, lat) or (name, point_object).
        for entry in sites:
            if len(entry) == 3:
                name, lo, la = entry
                out.append((str(name), float(lo), float(la)))
            elif len(entry) == 2:
                name, pt = entry
                lo, la = _extract_lonlat(pt)
                out.append((str(name), lo, la))
            else:
                raise ValueError(
                    f"Invalid site entry {entry!r}; expected "
                    "(name, lon, lat) or (name, point).")
    return out



def _format_baseline_length(
    dist_km: float, unit: Any = 'km', body: str = 'earth',
) -> str:
    """Format a physical or angular distance for display next to a baseline."""
    try:
        import astropy.units as u
        if isinstance(unit, u.UnitBase):
            # Interpret astropy unit.
            if unit.physical_type == 'length':
                val = (dist_km * u.km).to(unit).value
                return f'{val:,.0f} {unit.to_string("generic")}'
            elif unit.physical_type == 'angle':
                R = planet_radii.get(body.lower(), 6371.)
                ang_rad = dist_km / R
                val = (ang_rad * u.rad).to(unit).value
                return f'{val:,.1f} {unit.to_string("generic")}'
    except ImportError:
        pass
    s = str(unit).lower()
    if s == 'km':
        return f'{dist_km:,.0f} km' if dist_km >= 10 else f'{dist_km:.1f} km'
    if s in ('mi', 'miles'):
        return f'{dist_km * 0.621371:,.0f} mi'
    if s == 'm':
        return f'{dist_km * 1000:,.0f} m'
    if s == 'deg':
        R = planet_radii.get(body.lower(), 6371.)
        return f'{np.degrees(dist_km / R):,.1f}°'
    if s == 'rad':
        R = planet_radii.get(body.lower(), 6371.)
        return f'{dist_km / R:.3f} rad'
    return f'{dist_km:,.0f} km'


def plot_baselines(ax: Any, sites: Any, pairs: Any = 'all',
                   color: Any = 'steelblue', linewidth: float = 0.8,
                   linestyle: str = '-',
                   alpha: float = 0.9, zorder: int = 3, n_pts: int = 100,
                   show_markers: bool = True, marker: str = 'o',
                   marker_size: float = 30,
                   marker_color: Any = None, marker_edgecolor: Any = None,
                   marker_edgewidth: float = 0.6, marker_zorder: int = 5,
                   show_site_labels: bool = True, site_label_fontsize: float = 8,
                   site_label_color: Any = None,
                   site_label_offset: tuple[float, float] = (5, 5),
                   show_lengths: bool = False, length_unit: Any = 'km',
                   length_fontsize: float = 7, length_color: Any = None,
                   length_bbox: Any = None,
                   body: str = 'earth', hemisphere_only: bool | None = None,
                   wrap_fix: str = 'auto',
                   back_hemisphere_linestyle: str | None = None,
                   back_hemisphere_markers: bool = False,
                   back_hemisphere_alpha: float = 0.35,
                   **line_kwargs: Any) -> dict[str, Any]:
    """
    Plot interferometric baselines as great-circle arcs between sites.

    Designed for drawing networks of radio-telescope (or other) baselines
    on any map type: plate-carrée / CAR WCSAxes, orthographic SIN globes,
    or plain matplotlib lat-lon axes. Handles antimeridian wraparound on
    flat maps and back-hemisphere clipping on globes automatically.

    Parameters
    ----------
    ax : matplotlib Axes
        Plain axes with data in degrees, or WCSAxes with any sky/planetary
        projection (CAR / AIT / MOL / SIN / TAN / ZEA / ...).
    sites : dict or iterable
        Telescope sites. Accepted forms:

        * ``{'VLA': (lon, lat), 'GBT': (lon, lat), ...}`` — dict with
          scalar (lon, lat) tuples in degrees.
        * ``[('VLA', lon, lat), ('GBT', lon, lat), ...]`` — iterable of
          ``(name, lon, lat)`` tuples.
        * ``{'VLA': EarthLocation.of_site('VLA'), ...}`` — astropy
          EarthLocations or SkyCoords (longitude and latitude extracted
          automatically in degrees).
    pairs : 'all' or list of (name1, name2)
        Which baselines to draw. ``'all'`` (default) draws every unique
        unordered pair. Otherwise pass a list of name-tuples —
        e.g. ``[('VLA', 'GBT'), ('GBT', 'Arecibo')]``.
    color, linewidth, linestyle, alpha, zorder :
        Styling for the arc lines. Any matplotlib line kwargs.
    n_pts : int
        Points per great-circle arc. Higher values → smoother curves
        under strong projection warping.
    show_markers : bool
        Whether to scatter the site positions.
    marker, marker_size, marker_color, marker_edgecolor, marker_edgewidth, marker_zorder :
        Styling for site markers.
    show_site_labels : bool
        Whether to annotate each site with its name.
    site_label_fontsize, site_label_color, site_label_offset :
        Styling for site-name annotations. ``site_label_offset`` is in
        display points, `(dx, dy)`.
    show_lengths : bool
        If True, label each baseline at its midpoint with its length.
    length_unit : str or astropy.units.Unit
        Unit for baseline length labels: ``'km'`` (default), ``'m'``,
        ``'mi'``, ``'deg'``, ``'rad'``, or an astropy
        :class:`~astropy.units.UnitBase` (e.g. ``u.km``, ``u.au``,
        ``u.arcmin`` — astropy angular units work via the body radius).
    length_fontsize, length_color :
        Styling for baseline-length labels. If ``length_color`` is None,
        defaults to ``color``.
    length_bbox : dict or False, optional
        Plaque drawn behind each length label, so it stays readable where it
        crosses a baseline. ``None`` (default) fills with the axes background
        at 70% opacity, which follows the theme. Pass a dict of
        ``Text(bbox=...)`` properties to restyle it, or ``False`` to drop it.
    body : str
        Planet body for distance computation — looked up in
        :data:`planet_radii`. Default ``'earth'``.
    hemisphere_only : bool or None
        On globe axes, mask baselines that cross to the far hemisphere.
        ``None`` (default) auto-detects: True for SIN/ZEA-like globes,
        False otherwise.
    wrap_fix : {'auto', 'on', 'off'}
        On flat maps, insert NaN breaks where arcs cross the antimeridian
        so they render as two pieces at the map edges rather than a line
        across the whole plot. Ignored in globe mode.
    back_hemisphere_linestyle : str or None
        On globe axes: if set (e.g. ``':'`` or ``'--'``), also draw the
        back-hemisphere portion of each arc in this linestyle (same color,
        reduced alpha). Gives a "hidden line" effect useful for showing
        the full orbit of a baseline around the body.
    back_hemisphere_markers : bool
        On globe axes: also draw markers (and, if ``show_site_labels``,
        labels) for sites on the *far* hemisphere, at their true
        (hidden-line) disk position and faded by ``back_hemisphere_alpha``.
        Default ``False``.
    back_hemisphere_alpha : float
        Opacity for the far-side arcs' markers / labels. Default ``0.35``.
        (The far-side arc *lines* themselves render at half the main
        ``alpha``.)
    **line_kwargs :
        Additional kwargs forwarded to the underlying ``ax.plot``.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``'arcs'`` — list of ``Line2D`` (or lists-of-Line2D on globe
          axes) for each baseline arc drawn.
        * ``'back_arcs'`` — list of back-hemisphere arcs if
          ``back_hemisphere_linestyle`` was set; empty otherwise.
        * ``'markers'`` — the scatter artist for site positions, or None.
        * ``'back_markers'`` — the scatter artist for far-side site
          positions when ``back_hemisphere_markers`` is set, else None.
        * ``'site_labels'`` — list of annotation artists for site names.
        * ``'back_site_labels'`` — list of far-side site-name annotations
          when ``back_hemisphere_markers`` is set; empty otherwise.
        * ``'length_labels'`` — list of annotation artists for baseline
          lengths.

    Examples
    --------
    Simple VLBI-style network on a flat USA map::

        sites = {
            'VLA':       (-107.62, 34.08),
            'Greenbank': (-79.84,  38.43),
            'Arecibo':   (-66.75,  18.34),
            'Owens V':   (-118.28, 37.23),
        }
        plot_baselines(ax, sites, show_lengths=True)

    Full network on an Earth globe with hidden-line back-hemisphere. Use
    :func:`~skyplothelper.globe.make_planet_frame`, which builds the globe with
    the geographic (east-right) orientation terrestrial sites want::

        from skyplothelper.globe import make_planet_frame, plot_baselines
        globe_ax = make_planet_frame(111, center_LONdeg=-90, center_LATdeg=30)
        plot_baselines(globe_ax, sites,
                       color='crimson', linewidth=1.2,
                       back_hemisphere_linestyle=':')

    Only a subset of pairs::

        plot_baselines(ax, sites,
                       pairs=[('VLA', 'Arecibo'), ('VLA', 'Owens V')],
                       show_lengths=True, length_unit='mi')

    Notes
    -----
    Telescope sites are terrestrial, so on a globe you almost always want the
    geographic (east-right) orientation. Build the frame with
    :func:`~skyplothelper.globe.make_planet_frame` (geographic by default), or
    :func:`~skyplothelper.globe.make_globe_frame` with ``direction='geo'``.
    Flat CAR / lat-lon axes already increase longitude to the right, so no
    flip is needed there.
    """
    sites_list = _normalize_sites(sites)
    name_to_idx = {name: i for i, (name, _, _) in enumerate(sites_list)}

    # Resolve pairs to index pairs.
    if pairs == 'all' or pairs is None:
        import itertools
        idx_pairs = list(itertools.combinations(range(len(sites_list)), 2))
    else:
        idx_pairs = []
        for n1, n2 in pairs:
            if n1 not in name_to_idx or n2 not in name_to_idx:
                raise KeyError(f"Unknown site in pair ({n1!r}, {n2!r}); "
                               f"known sites: {list(name_to_idx)}")
            idx_pairs.append((name_to_idx[n1], name_to_idx[n2]))

    # Auto-detect hemisphere mode.
    if hemisphere_only is None:
        hemisphere_only = _is_globe_axes(ax)

    # Does the axes have a WCS / world-transform? (Plain matplotlib has no
    # get_transform; we plot in data coords directly.)
    has_world = hasattr(ax, 'get_transform') and hasattr(ax, 'wcs')

    if length_color is None:
        length_color = color

    arc_artists = []
    back_arc_artists = []
    length_label_artists = []

    for i, j in idx_pairs:
        n1, lo1, la1 = sites_list[i]
        n2, lo2, la2 = sites_list[j]

        lons_arc, lats_arc = great_circle_arc(lo1, la1, lo2, la2, n_pts=n_pts)

        if hemisphere_only and has_world:
            # Front-hemisphere clip via plot_line_globe.
            lines = plot_line_globe(
                ax, lons_arc, lats_arc,
                hemisphere_only=True, densify=False,
                color=color, linewidth=linewidth, linestyle=linestyle,
                alpha=alpha, zorder=zorder, **line_kwargs)
            arc_artists.append(lines)

            # Optional back-hemisphere hidden-line rendering. Astropy's SIN
            # transform maps back-hemisphere points to NaN, so we mirror them
            # to their near-side images (same disk position) and project those
            # through the WCS — exact for the globe's LONPOLE=0 frame, where a
            # hand-rolled ortho + CRPIX/CDELT shortcut reflects every point
            # through the disk center. See _mirror_to_near_side.
            if back_hemisphere_linestyle is not None:
                wcs = ax.wcs
                center_lon = float(wcs.wcs.crval[0])
                center_lat = float(wcs.wcs.crval[1])
                vis = orthographic_visibility(
                    lons_arc, lats_arc, center_lon, center_lat)
                back_lons = np.where(~vis, lons_arc, np.nan)
                back_lats = np.where(~vis, lats_arc, np.nan)
                mir_lon, mir_lat = _mirror_to_near_side(
                    back_lons, back_lats, center_lon, center_lat)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pix_x, pix_y = wcs.world_to_pixel_values(mir_lon, mir_lat)
                back_lines = ax.plot(
                    pix_x, pix_y,
                    color=color, linewidth=linewidth,
                    linestyle=back_hemisphere_linestyle,
                    alpha=alpha * 0.5, zorder=zorder, **line_kwargs)
                back_arc_artists.append(back_lines)
        else:
            # Flat map: apply antimeridian wrap fix if requested.
            if wrap_fix != 'off':
                lons_plot, lats_plot = _wrap_fix_lons(lons_arc, lats_arc)
            else:
                lons_plot, lats_plot = lons_arc, lats_arc

            if has_world:
                plot_kw = dict(transform=ax.get_transform('world'))
            else:
                plot_kw = {}
            lines = ax.plot(
                lons_plot, lats_plot,
                color=color, linewidth=linewidth, linestyle=linestyle,
                alpha=alpha, zorder=zorder, **plot_kw, **line_kwargs)
            arc_artists.append(lines)

        # Length label at arc midpoint.
        if show_lengths:
            mid_lon, mid_lat = midpoint(lo1, la1, lo2, la2)
            dist_km = great_circle_distance(lo1, la1, lo2, la2, body=body)
            label = _format_baseline_length(dist_km, unit=length_unit,
                                             body=body)
            # The label sits on a small plaque so it stays readable over the
            # baseline beneath it. The plaque follows the canvas rather than
            # a literal white, which stayed a white box on a dark theme no
            # matter what every color knob was set to.
            if length_bbox is None:
                _bbox = dict(facecolor=rcParams['axes.facecolor'],
                             edgecolor='none', alpha=0.7, pad=1)
            elif length_bbox is False:
                _bbox = None
            else:
                _bbox = dict(length_bbox)
            text_kw = dict(fontsize=length_fontsize, color=length_color,
                           ha='center', va='center', zorder=zorder + 1,
                           bbox=_bbox)
            if has_world:
                text_kw['transform'] = ax.get_transform('world')
            t = ax.text(mid_lon, mid_lat, label, **text_kw)
            length_label_artists.append(t)

    # Site markers.
    marker_artist = None
    if show_markers and sites_list:
        mk_lons = [lo for _, lo, _ in sites_list]
        mk_lats = [la for _, _, la in sites_list]
        # ``scatter(c=None)`` does NOT inherit a theme color — it takes the
        # next property-cycle entry — so the ink has to be resolved here.
        # ``site_label_color`` a few lines down is the opposite case: it
        # reaches ``ax.annotate``, where ``None`` correctly inherits
        # ``rcParams['text.color']``, so it is left alone.
        if marker_color is None:
            marker_color = rcParams['text.color']
        if marker_edgecolor is None:
            # The paired inverse: an outline that separates the marker from
            # the map behind it, so it follows the background, not the ink.
            marker_edgecolor = rcParams['axes.facecolor']
        scatter_kw = dict(s=marker_size, c=marker_color,
                          marker=marker, edgecolors=marker_edgecolor,
                          linewidths=marker_edgewidth, zorder=marker_zorder)
        if has_world:
            scatter_kw['transform'] = ax.get_transform('world')
        marker_artist = ax.scatter(mk_lons, mk_lats, **scatter_kw)

    # Ghosted far-side markers + labels on a globe: the WCS culls
    # back-hemisphere sites to NaN, so mirror them to their near-side images
    # (same disk position — see _mirror_to_near_side) and draw them faded.
    back_marker_artist = None
    back_site_label_artists: list[Any] = []
    if (back_hemisphere_markers and sites_list and has_world
            and hemisphere_only):
        wcs = ax.wcs
        clon, clat = float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])
        all_lons = np.array([lo for _, lo, _ in sites_list], dtype=float)
        all_lats = np.array([la for _, _, la in sites_list], dtype=float)
        back = ~orthographic_visibility(all_lons, all_lats, clon, clat)
        if np.any(back):
            mlon, mlat = _mirror_to_near_side(
                all_lons[back], all_lats[back], clon, clat)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                bx, by = wcs.world_to_pixel_values(mlon, mlat)
            if show_markers:
                back_marker_artist = ax.scatter(
                    bx, by, s=marker_size, c=marker_color, marker=marker,
                    edgecolors=marker_edgecolor, linewidths=marker_edgewidth,
                    alpha=back_hemisphere_alpha, zorder=marker_zorder)
            if show_site_labels:
                dx_pts, dy_pts = site_label_offset
                back_names = [nm for (nm, _, _), b
                              in zip(sites_list, back) if b]
                for nm, xb, yb in zip(back_names, np.atleast_1d(bx),
                                      np.atleast_1d(by)):
                    if not (np.isfinite(xb) and np.isfinite(yb)):
                        continue
                    txt = ax.annotate(
                        nm, xy=(float(xb), float(yb)),
                        xytext=(dx_pts, dy_pts), textcoords='offset points',
                        fontsize=site_label_fontsize, color=site_label_color,
                        ha='left', va='bottom', alpha=back_hemisphere_alpha,
                        zorder=marker_zorder + 1)
                    back_site_label_artists.append(txt)

    # Site-name labels.
    site_label_artists = []
    if show_site_labels and sites_list:
        dx_pts, dy_pts = site_label_offset
        # A back-hemisphere (or otherwise off-projection) site projects to a
        # non-finite position; the markers/arcs are already culled there, but
        # an annotation text artist placed at NaN makes matplotlib log
        # "posx and posy should be finite values". Skip those labels — same
        # culling the markers get. (Plain non-WCS axes are always finite.)
        if has_world:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                px, py = ax.wcs.world_to_pixel_values(
                    [lo for _, lo, _ in sites_list],
                    [la for _, _, la in sites_list])
            label_ok = np.isfinite(px) & np.isfinite(py)
        else:
            label_ok = np.ones(len(sites_list), dtype=bool)
        for k, (name, lo, la) in enumerate(sites_list):
            if not label_ok[k]:
                continue
            text_kw = dict(xytext=(dx_pts, dy_pts),
                           textcoords='offset points',
                           fontsize=site_label_fontsize,
                           color=site_label_color,
                           ha='left', va='bottom',
                           zorder=marker_zorder + 1)
            if has_world:
                text_kw['xycoords'] = ax.get_transform('world')
            txt = ax.annotate(name, xy=(lo, la), **text_kw)
            site_label_artists.append(txt)

    return {
        'arcs': arc_artists,
        'back_arcs': back_arc_artists,
        'markers': marker_artist,
        'back_markers': back_marker_artist,
        'site_labels': site_label_artists,
        'back_site_labels': back_site_label_artists,
        'length_labels': length_label_artists,
    }


# =============================================================================
# Boundary Data: Coastlines, Tectonic Plates, and Time Zones
# =============================================================================
#
# Data sources and licenses for datasets downloaded by prepare_earth_data():
#
# -----------------------------------------------------------------------------
# Natural Earth (coastlines, time zones)                        PUBLIC DOMAIN
# -----------------------------------------------------------------------------
#   Website:  https://www.naturalearthdata.com/
#   License:  Public domain. No permission is needed to use Natural Earth.
#             Crediting the authors is unnecessary. However, if you wish to
#             cite the map data, simply use: "Made with Natural Earth."
#   Authors:  Tom Patterson and Nathaniel Vaughn Kelso (primary editors),
#             with contributions from the wider Natural Earth community.
#   Terms:    https://www.naturalearthdata.com/about/terms-of-use/
#
#   Specific datasets used here:
#     * Coastline (physical): 110m, 50m, 10m resolutions
#     * Time zones (cultural): 10m resolution only
#       Donated to Natural Earth by International Mapping Associates, Inc.
#       Attribution to International Mapping Associates or the CIA World
#       Factbook (original source of some boundary data) is optional.
#
# -----------------------------------------------------------------------------
# Tectonic plate boundaries — Peter Bird (2003)                  ACADEMIC CITATION
# -----------------------------------------------------------------------------
#   Primary citation (required if used in publications):
#     Bird, P. (2003), An updated digital model of plate boundaries,
#       Geochemistry, Geophysics, Geosystems, 4(3), 1027,
#       doi:10.1029/2001GC000252.
#
#   Re-distribution source (GeoJSON conversion):
#     https://github.com/fraxen/tectonicplates
#       Maintained by Hugo Ahlenius / Nordpil / fraxen. Packaged the original
#       Bird 2003 boundaries into convenient GeoJSON for web use.
#
# -----------------------------------------------------------------------------
# Attribution in plots
# -----------------------------------------------------------------------------
# When publishing figures made with these data, the suggested credits are:
#
#   "Coastlines and time zones: Natural Earth (public domain)."
#   "Plate boundaries: Bird (2003), doi:10.1029/2001GC000252."
#
# The prepare_earth_data() function prints these notices at download time.
# =============================================================================

# Default search paths for .npz data files (relative to this module)
