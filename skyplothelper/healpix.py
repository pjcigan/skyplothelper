"""HEALPix utilities — binning, plotting, queries, smoothing.

Optional dependency: ``healpy``. If not installed, calling these will
raise an informative ImportError. The plotters delegate to
``make_wcs_frame`` for axes creation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import astropy.units as u  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

from astropy.coordinates import SkyCoord  # noqa: F401
from astropy.wcs import WCS  # noqa: F401

try:
    import healpy as hp
    _HAS_HEALPY = True
except ImportError:
    _HAS_HEALPY = False

# Internal cross-module deps (lazy at function level would also work; module-
# level here keeps things readable since they're always needed in plotting paths).
from .geometry._parsing import _coords_or_arrays_deg
from .overlays.annotations import add_colorbar  # noqa: F401
from .overlays.planes import add_plane_overlay  # noqa: F401
from .ticks import format_ticklabels  # noqa: F401
from .wcs_frame import (  # noqa: F401
    _get_wcs_frame_name,
    clip_to_projection_boundary,
    make_wcs_frame,
)


def _require_healpy(funcname: str = '') -> None:
    if not _HAS_HEALPY:
        raise ImportError(f'healpy required for {funcname}')


def healpix_to_celestial(healpix_array: npt.ArrayLike,
                         lonlatlims: str | Sequence[Sequence[float]] = 'allsky',
                         center_deg: float = 180.,
                         xyres_pix: tuple[int, int] = (2000, 1000),
                         blank_value: float = 0.,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a HEALPix array to lon/lat meshgrid for plotting with pcolormesh.

    Parameters
    ----------
    healpix_array : ndarray
        Full HEALPix map (length npix)
    lonlatlims : str or list
        'allsky' or [[lon_min, lon_max], [lat_min, lat_max]]
    center_deg : float
    xyres_pix : tuple
        Output resolution [x, y] in pixels
    blank_value : float

    Returns
    -------
    plotloncenters, plotlatcenters, plotvalues : ndarrays
    """
    _require_healpy('healpix_to_celestial')
    healpix_array = np.asarray(healpix_array)
    nside = hp.npix2nside(len(healpix_array))

    if isinstance(lonlatlims, str) and lonlatlims == 'allsky':
        limits: Sequence[Sequence[float]] = [
            [center_deg - 180, center_deg + 180], [-90, 90]]
    else:
        limits = lonlatlims  # type: ignore[assignment]  # narrowed: non-'allsky' input is the Sequence branch

    (lon_min, lon_max), (lat_min, lat_max) = limits[0], limits[1]
    plotlon = np.linspace(lon_min, lon_max, xyres_pix[0] + 1)
    plotlat = np.linspace(lat_min, lat_max, xyres_pix[1] + 1)
    plotlons, plotlats = np.meshgrid(plotlon, plotlat)
    plotloncenters = plotlons[:-1, :-1] + np.diff(plotlon) / 2
    plotlatcenters = plotlats[:-1, :-1] + np.diff(plotlats, axis=0)[:, :-1] / 2

    plothppixels = hp.ang2pix(nside, plotloncenters, plotlatcenters, lonlat=True)
    plotvalues = healpix_array[plothppixels]

    return plotloncenters, plotlatcenters, plotvalues


def mask_seam_crossing_quads(
        ax: Any,
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        values: npt.ArrayLike,
        *,
        span_fraction: float = 0.1) -> np.ndarray:
    """Blank ``pcolormesh`` cells that bridge a projection discontinuity.

    On interrupted / wrap-around projections (HEALPix HPX/XPH, the quadcube
    faces TSC/CSC/QSC) a regular ``(lon, lat)`` mesh has cells that straddle
    the antimeridian or a face boundary. ``pcolormesh`` draws each such cell
    as one quad bridging the gap, smearing color across regions that should
    be blank — most visibly the HPX polar caps filling solid. This returns a
    float copy of ``values`` with the offending cells set to ``NaN`` so they
    render blank.

    Two kinds of bad cell are detected from the mesh projected into the axes'
    data coordinates:

    - **stretched** — a cell edge longer than ``span_fraction`` of the panel
      span (a quad reaching across a seam), and
    - **folded** — a quad whose signed area flips orientation relative to the
      mesh majority (a cube-face fold-over).

    It is a no-op on continuous projections (AIT, MOL, …), whose cells are
    all small and same-handed, so it is safe to apply unconditionally.

    Note: this does NOT fix the polyconic (PCO) all-sky drape, whose cells
    are normal-sized and same-handed but genuinely overlap (the projection is
    multivalued past ~90° from the central meridian); PCO is best shown
    graticule-only or as a
    field view for data.

    Parameters
    ----------
    ax : WCSAxes
        Target axes (provides ``get_transform('world')`` and ``transData``).
    lon, lat : 2D array_like
        Cell-center longitude / latitude meshes (degrees), matching
        ``values`` in shape — i.e. the arrays handed to ``pcolormesh``.
    values : 2D array_like
        Data values per cell.
    span_fraction : float
        Edge-length threshold as a fraction of the larger panel span. A cell
        whose projected edge exceeds this is treated as seam-bridging.

    Returns
    -------
    masked : ndarray (float)
        Copy of ``values`` with seam-crossing / folded cells set to ``NaN``.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    out = np.array(values, dtype=float)
    if lon.ndim != 2 or lon.shape != out.shape or lat.shape != out.shape:
        return out  # only handle 2D center meshes matching values

    disp = ax.get_transform('world').transform(
        np.column_stack([lon.ravel(), lat.ravel()]))
    data = ax.transData.inverted().transform(disp)
    x = data[:, 0].reshape(lon.shape)
    y = data[:, 1].reshape(lon.shape)
    if np.isfinite(x).sum() < 4:
        return out

    xspan = float(np.nanmax(x) - np.nanmin(x))
    yspan = float(np.nanmax(y) - np.nanmin(y))
    thr = span_fraction * max(xspan, yspan)
    bad = np.zeros(lon.shape, dtype=bool)

    # (a) stretched cells: a center-to-neighbor edge spanning > thr is a quad
    # reaching across a seam — flag both cells sharing that edge.
    eh = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    he = np.isfinite(eh) & (eh > thr)
    bad[:, :-1] |= he
    bad[:, 1:] |= he
    ev = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    ve = np.isfinite(ev) & (ev > thr)
    bad[:-1, :] |= ve
    bad[1:, :] |= ve

    # (b) folded cells: signed (shoelace) area whose sign opposes the mesh
    # majority — a fold-over (cube faces). Flag the quad's four corner cells.
    x00, y00 = x[:-1, :-1], y[:-1, :-1]
    x01, y01 = x[:-1, 1:], y[:-1, 1:]
    x11, y11 = x[1:, 1:], y[1:, 1:]
    x10, y10 = x[1:, :-1], y[1:, :-1]
    area = 0.5 * ((x00 * y01 - x01 * y00) + (x01 * y11 - x11 * y01)
                  + (x11 * y10 - x10 * y11) + (x10 * y00 - x00 * y10))
    finite_area = area[np.isfinite(area)]
    if finite_area.size:
        majority = np.sign(np.median(finite_area))
        flip = np.isfinite(area) & (np.sign(area) != majority) & (area != 0)
        bad[:-1, :-1] |= flip
        bad[:-1, 1:] |= flip
        bad[1:, :-1] |= flip
        bad[1:, 1:] |= flip

    out[bad] = np.nan

    # XPH antimeridian dedup: on the HEALPix butterfly the antimeridian folds
    # onto a single corner facet, so the mesh's two seam-edge columns
    # (lon ≈ center ± 180) both project there and overlap (a curved double-image
    # in the corner). The overlap quads are neither stretched nor orientation-
    # flipped, so the checks above miss them. Blank a thin strip hugging the
    # antimeridian; it removes the overlap and leaves only a thin seam,
    # consistent with the other facet seams. XPH-specific — for the other
    # projections the antimeridian maps to opposite edges, where this strip
    # would just punch an unwanted gap.
    try:
        code = str(ax.wcs.wcs.ctype[0]).split('-')[-1].strip().upper()
        center_lon = float(ax.wcs.wcs.crval[0])
    except Exception:
        code, center_lon = None, None
    if code == 'XPH' and center_lon is not None:
        rel = ((lon - center_lon + 180.0) % 360.0) - 180.0
        out[np.abs(rel) > 178.0] = np.nan  # within 2° of the antimeridian

    return out


def plot_healpix_map(healpix_array: npt.ArrayLike, ax: Any = None,
                     lonlatlims: str | Sequence[Sequence[float]] = 'allsky',
                     center_deg: float = 180.,
                     xyres_pix: tuple[int, int] = (2000, 1000),
                     blank_value: float = 0., mask_seams: bool = True,
                     **plot_kwargs: Any) -> Any:
    """Plot a HEALPix array on a WCSAxes or plain matplotlib axis.

    On a WCSAxes, ``mask_seams`` (default True) blanks ``pcolormesh`` cells
    that bridge a projection seam (HPX/XPH/quadcube), preventing the
    smear-across-boundary artifact; see ``mask_seam_crossing_quads``. It is a
    no-op on continuous projections. Pass ``mask_seams=False`` to disable.
    """
    _require_healpy('plot_healpix_map')
    plonc, platc, pvals = healpix_to_celestial(
        healpix_array, lonlatlims, center_deg, xyres_pix, blank_value)
    if ax is None:
        return plt.pcolormesh(plonc, platc, pvals, **plot_kwargs)
    if mask_seams:
        pvals = mask_seam_crossing_quads(ax, plonc, platc, pvals)
    qm = ax.pcolormesh(plonc, platc, pvals,
                       transform=ax.get_transform('world'), **plot_kwargs)
    if mask_seams:
        # Also clip to the projection's visible outline so interrupted
        # projections (HPX/BON/PCO/conics) don't bleed data into the empty
        # bbox regions (no-op for continuous projections).
        clip_to_projection_boundary(ax, qm)
    return qm


def sources_to_healpix_bins(lons: SkyCoord | npt.ArrayLike, lats: npt.ArrayLike | None = None,
                            nside: int | None = None,
                            lonlatlims: str | Sequence[Sequence[float]] = 'allsky',
                            center_deg: float = 180.,
                            xyres_pix: tuple[int, int] = (2000, 1000),
                            blank_value: float = 0.,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin source positions into HEALPix count map and return plot-ready grids.

    Parameters
    ----------
    lons, lats : array-like, or SkyCoord in ``lons``
        Source coordinates in degrees. A ``SkyCoord`` array may be passed as
        ``lons`` instead, replacing both; following arguments must then be
        keywords (``nside=64``, not a bare ``64``). A HEALPix map carries no
        frame of its own, so a SkyCoord KEEPS its frame rather than being
        converted.
    nside : int

    Returns
    -------
    plotloncenters, plotlatcenters, plotvalues : ndarrays

    Examples
    --------
    >>> lon, lat, counts = sph.sources_to_healpix_bins(ra, dec, nside=64)
    >>> lon, lat, counts = sph.sources_to_healpix_bins(catalog, nside=64)
    """
    _require_healpy('sources_to_healpix_bins')
    # frame_name=None: HEALPix maps carry no frame, so a SkyCoord
    # keeps its own (matching the bare vertex builders).
    lons, lats = _coords_or_arrays_deg(lons, lats, None, 'sources_to_healpix_bins')
    if nside is None:
        raise TypeError('sources_to_healpix_bins: nside is required.')
    coords_hp = hp.ang2pix(nside, lons, lats, lonlat=True)
    npix = hp.nside2npix(nside)
    pix_i, counts = np.unique(coords_hp, return_counts=True)
    hpxmap = np.zeros(npix) + blank_value
    hpxmap[pix_i] = counts
    return healpix_to_celestial(hpxmap, lonlatlims, center_deg, xyres_pix, blank_value)


def sources_to_healpix_plot(lons: SkyCoord | npt.ArrayLike, lats: npt.ArrayLike | None = None,
                            nside: int | None = None, ax: Any = None,
                            lonlatlims: str | Sequence[Sequence[float]] = 'allsky',
                            center_deg: float = 180.,
                            xyres_pix: tuple[int, int] = (2000, 1000),
                            blank_value: float = 0., mask_seams: bool = True,
                            **plot_kwargs: Any) -> Any:
    """Bin source coords into HEALPix and plot with pcolormesh.

    On a WCSAxes, ``mask_seams`` (default True) blanks seam-bridging cells on
    interrupted projections (see ``mask_seam_crossing_quads``); no-op on
    continuous projections.

    Parameters
    ----------
    lons, lats : array-like, or SkyCoord in ``lons``
        Source coordinates in degrees. A ``SkyCoord`` array may be passed as
        ``lons``, replacing both; following arguments must then be keywords
        (``nside=64``, not a bare ``64``). A SkyCoord keeps its own frame —
        a HEALPix map has none of its own.
    nside : int
        HEALPix resolution parameter.
    ax : WCSAxes, optional
        Target axes. Uses ``plt.pcolormesh`` when omitted.

    Examples
    --------
    >>> sph.sources_to_healpix_plot(ra, dec, nside=64, ax=ax)
    >>> sph.sources_to_healpix_plot(catalog_skycoord, nside=64, ax=ax)
    """
    plonc, platc, pvals = sources_to_healpix_bins(
        lons, lats, nside, lonlatlims, center_deg, xyres_pix, blank_value)
    if ax is None:
        return plt.pcolormesh(plonc, platc, pvals, **plot_kwargs)
    if mask_seams:
        pvals = mask_seam_crossing_quads(ax, plonc, platc, pvals)
    qm = ax.pcolormesh(plonc, platc, pvals,
                       transform=ax.get_transform('world'), **plot_kwargs)
    if mask_seams:
        # Also clip to the projection's visible outline so interrupted
        # projections (HPX/BON/PCO/conics) don't bleed data into the empty
        # bbox regions (no-op for continuous projections).
        clip_to_projection_boundary(ax, qm)
    return qm


def bin_data_as_healpix(lons: SkyCoord | npt.ArrayLike, lats: npt.ArrayLike | None = None,
                        data: npt.ArrayLike | None = None,
                        nside: int | None = None,
                        statistic: str | Callable[[np.ndarray], float] = 'mean',
                        lonlatlims: str | Sequence[Sequence[float]] = 'allsky',
                        center_deg: float = 180.,
                        xyres_pix: tuple[int, int] = (2000, 1000),
                        blank_value: float = np.nan,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin arbitrary data values into HEALPix pixels with aggregation statistics.

    Parameters
    ----------
    lons, lats : array-like, or SkyCoord in ``lons``
        Source coordinates in degrees. A ``SkyCoord`` array may be passed as
        ``lons`` instead, replacing both; following arguments must then be
        keywords. A SkyCoord keeps its own frame (a HEALPix map has none).
    data : array-like
        Data values per source
    nside : int
    statistic : str or callable
        ``'mean'``, ``'median'``, ``'sum'``, ``'min'``, ``'max'``, ``'std'``,
        ``'count'``, or a callable that takes the 1-D array of (finite) values
        in a cell and returns a scalar — e.g.
        ``lambda v: np.percentile(v, 90)``.

    Returns
    -------
    hpxmap, plotloncenters, plotlatcenters, plotvalues

    Examples
    --------
    >>> lon, lat, vals = sph.bin_data_as_healpix(ra, dec, flux, nside=64)
    >>> lon, lat, vals = sph.bin_data_as_healpix(
    ...     catalog, data=flux, nside=64, statistic='median')
    """
    _require_healpy('bin_data_as_healpix')
    # frame_name=None: HEALPix maps carry no frame, so a SkyCoord
    # keeps its own (matching the bare vertex builders).
    lons, lats = _coords_or_arrays_deg(lons, lats, None, 'bin_data_as_healpix')
    if data is None or nside is None:
        raise TypeError('bin_data_as_healpix: data and nside are required.')
    data = np.asarray(data, dtype=float)
    npix = hp.nside2npix(nside)
    coords_hp = hp.ang2pix(nside, lons, lats, lonlat=True)

    valid = np.isfinite(data)
    clean_data = data[valid]
    clean_pix = coords_hp[valid]

    hpxmap = np.full(npix, blank_value)

    if statistic == 'sum':
        sums = np.bincount(clean_pix, weights=clean_data, minlength=npix)
        occupied = np.bincount(clean_pix, minlength=npix) > 0
        hpxmap[occupied] = sums[occupied]
    elif statistic == 'mean':
        sums = np.bincount(clean_pix, weights=clean_data, minlength=npix)
        counts = np.bincount(clean_pix, minlength=npix)
        occupied = counts > 0
        hpxmap[occupied] = sums[occupied] / counts[occupied]
    elif statistic == 'count':
        counts = np.bincount(clean_pix, minlength=npix)
        occupied = counts > 0
        hpxmap[occupied] = counts[occupied]
    elif statistic in ('median', 'min', 'max', 'std') or callable(statistic):
        # Group the per-cell values and aggregate. Named stats use pandas when
        # available (fast); a user CALLABLE always goes through the clean-data
        # loop so it receives only finite values per cell (matching
        # bin_data_sparse), rather than pandas' NaN-laden full groups.
        func: Any = None
        if callable(statistic):
            func = statistic
        else:
            try:
                import pandas as pd
                df = pd.DataFrame({'pix': coords_hp, 'val': data})
                grouped = df.groupby('pix')['val'].agg(statistic)
                hpxmap[grouped.index.values] = grouped.values
            except ImportError:
                func = {'median': np.nanmedian, 'min': np.nanmin,
                        'max': np.nanmax, 'std': np.nanstd}[statistic]
        if func is not None:
            sort_idx = np.argsort(clean_pix)
            sorted_pix = clean_pix[sort_idx]
            sorted_data = clean_data[sort_idx]
            split_pts = np.searchsorted(sorted_pix, np.arange(npix), side='left')
            split_pts = np.append(split_pts, len(sorted_pix))
            for i in range(npix):
                if split_pts[i + 1] > split_pts[i]:
                    hpxmap[i] = func(sorted_data[split_pts[i]:split_pts[i + 1]])
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    plonc, platc, pvals = healpix_to_celestial(
        hpxmap, lonlatlims, center_deg, xyres_pix, blank_value)
    return hpxmap, plonc, platc, pvals


def healpix_circle_query(centerlon_deg: float, centerlat_deg: float,
                         radius_deg: float, nside: int,
                         inclusive: bool = False, fact: int = 4,
                         nest: bool = False) -> np.ndarray:
    """Query HEALPix pixel indices within a disk."""
    _require_healpy('healpix_circle_query')
    vec = hp.ang2vec(centerlon_deg, centerlat_deg, lonlat=True)
    return hp.query_disc(nside, vec, np.radians(radius_deg),
                         inclusive=inclusive, fact=fact, nest=nest)


def healpix_polygon_query(vertices_deg: npt.ArrayLike, nside: int,
                          inclusive: bool = False, fact: int = 4,
                          nest: bool = False,
                          dither_lon_deg: float = 1e-6) -> np.ndarray:
    """
    Query HEALPix pixel indices within a convex polygon.

    Parameters
    ----------
    vertices_deg : array-like
        Polygon vertices as [[lon1, lat1], [lon2, lat2], ...] in degrees
    nside : int
    inclusive : bool
        If True, include pixels that overlap the polygon edge as well.
    fact : int
        Sub-pixel factor for inclusive mode (passed to ``hp.query_polygon``).
    nest : bool
        If True, return NESTED-ordered indices.
    dither_lon_deg : float
        Magnitude of the lon perturbation used to work around healpy's
        ``query_polygon`` precision edge case at HEALPix base-pixel
        boundaries (default 1e-6° = 3.6 milli-arcsec). The wrapper does
        two queries — one with the user's vertices, one with vertices
        shifted by this amount in lon — and unions the results. The
        perturbation must be smaller than the HEALPix tile size at the
        target nside so it doesn't shift the selection. Pass 0 to
        disable the workaround entirely (sub-microarcsec / VLBI-scale
        polygon queries should pass 0 or a much smaller value).

    Notes
    -----
    Healpy's ``hp.query_polygon`` silently excludes individual interior
    pixels whose centers fall on a HEALPix base-pixel boundary (e.g.
    lon=270° at high nside). The visual symptom is a faint diagonal
    line of "missing" pixels inside an otherwise-filled query region;
    ``dither_lon_deg`` works around it by nudging the query vertices.
    """
    _require_healpy('healpix_polygon_query')
    verts = np.atleast_2d(vertices_deg).astype(float)
    xyz = hp.ang2vec(verts[:, 0], verts[:, 1], lonlat=True)
    res = hp.query_polygon(nside, xyz, inclusive=inclusive, fact=fact, nest=nest)
    if dither_lon_deg and dither_lon_deg > 0:
        verts_p = verts.copy()
        verts_p[:, 0] += dither_lon_deg
        xyz_p = hp.ang2vec(verts_p[:, 0], verts_p[:, 1], lonlat=True)
        res_p = hp.query_polygon(nside, xyz_p, inclusive=inclusive,
                                 fact=fact, nest=nest)
        res = np.union1d(res, res_p)
    return res


###############################################################################
#                                                                             #
#              HEALPIX SPARSE ZOOM                               #
#                                                                             #
###############################################################################


def auto_nside(resolution_deg: float | None = None,
               resolution_arcmin: float | None = None,
               resolution_arcsec: float | None = None,
               resolution_mas: float | None = None) -> tuple[int, float]:
    """
    Select the HEALPix nside that gives pixels at or finer than the
    requested angular resolution.

    The pixel scale for a given nside is approximately:
        pixel_size ≈ sqrt(4π / (12 * nside²))  [in radians]

    Parameters
    ----------
    resolution_deg : float, optional
    resolution_arcmin : float, optional
    resolution_arcsec : float, optional
    resolution_mas : float, optional
        Desired pixel resolution. Exactly one must be specified.

    Returns
    -------
    nside : int
        Power-of-2 nside value
    actual_resolution_arcsec : float
        Actual pixel resolution in arcseconds for the returned nside

    Examples
    --------
    >>> auto_nside(resolution_arcmin=3.0)
    (2048, 103.06...)
    >>> auto_nside(resolution_arcsec=1.0)
    (262144, 0.805...)
    >>> auto_nside(resolution_mas=10.0)
    (33554432, 0.00629...)
    """
    _require_healpy('auto_nside')

    # Convert to degrees
    specs = [(resolution_deg, 1.0), (resolution_arcmin, 1/60.),
             (resolution_arcsec, 1/3600.), (resolution_mas, 1/3.6e6)]
    given = [(v * scale) for v, scale in specs if v is not None]
    if len(given) != 1:
        raise ValueError("Specify exactly one of resolution_deg, "
                         "resolution_arcmin, resolution_arcsec, resolution_mas")
    target_deg = given[0]

    # nside must be power of 2; find smallest where pixel size <= target
    # hp.nside2resol returns pixel resolution in radians
    nside = 1
    while nside <= 2**29:  # healpy max
        resol_deg = np.degrees(hp.nside2resol(nside))
        if resol_deg <= target_deg:
            resol_arcsec = resol_deg * 3600.
            return nside, resol_arcsec
        nside *= 2

    resol_arcsec = np.degrees(hp.nside2resol(nside)) * 3600.
    return nside, resol_arcsec


def healpix_pixel_corners(pixel_indices: npt.ArrayLike, nside: int,
                          step: int = 1, nest: bool = False,
                          ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Get the boundary vertices (corners) for HEALPix pixels.

    Parameters
    ----------
    pixel_indices : array-like
        Pixel indices to get boundaries for
    nside : int
    step : int
        Number of points per edge (1 = corners only, higher = smoother)
    nest : bool
        If True, assume NESTED pixel ordering

    Returns
    -------
    lon_boundaries : list of arrays
        Longitude vertices (degrees) for each pixel
    lat_boundaries : list of arrays
        Latitude vertices (degrees) for each pixel
    """
    _require_healpy('healpix_pixel_corners')
    pixel_indices = np.atleast_1d(pixel_indices)
    if pixel_indices.size == 0:
        return [], []
    # hp.boundaries + hp.vec2ang are both vectorized: one call for ALL pixels
    # (an array of thousands of pixels was previously a per-pixel Python loop
    # of thousands of healpy calls). Array input -> (npix, 3, 4*step).
    xyz = hp.boundaries(nside, pixel_indices, step=step, nest=nest)
    if xyz.ndim == 2:                       # single pixel -> (3, nvert)
        xyz = xyz[np.newaxis]
    npix, _, nvert = xyz.shape
    vecs = np.moveaxis(xyz, 1, 2).reshape(-1, 3)     # (npix*nvert, 3)
    lon, lat = hp.vec2ang(vecs, lonlat=True)
    lon = lon.reshape(npix, nvert)
    lat = lat.reshape(npix, nvert)
    return list(lon), list(lat)


def plot_healpix_sparse(pixel_indices: npt.ArrayLike,
                        values: npt.ArrayLike | None, nside: int,
                        ax: Any = None, nest: bool = False,
                        step: int = 1, show_boundaries: bool = False,
                        boundary_color: str = '0.5',
                        boundary_lw: float = 0.5, set_extent: bool = True,
                        padding_factor: float = 1.5,
                        cmap: Any = 'viridis',
                        vmin: float | None = None, vmax: float | None = None,
                        backend: str = 'patches',
                        sampling: str = 'canvas',
                        interp: bool = False,
                        xyres_pix: tuple[int, int] = (2000, 1000),
                        blank_value: float = np.nan,
                        **patch_kwargs: Any) -> Any:
    """
    Render a sparse subset of HEALPix pixels without materializing a full-sky
    array. This enables visualization at arbitrarily fine angular resolution
    (sub-arcsecond, mas, µas) by plotting only the queried pixel subset.

    Three rendering backends are available via ``backend``:

    * ``'patch'`` (default; ``'patches'`` accepted) — each tile drawn as
      an individual Polygon patch using its true spherical boundary,
      projected through the geometry pipeline. Produces sharp,
      pixel-accurate edges and handles antimeridian / pole / frame-edge
      crossings cleanly. This is the natural choice for sparse data and
      stays correct under user-driven zoom.
    * ``'imshow'`` — sample the sparse map at every canvas pixel and
      render via ``ax.imshow``. Returns an ``AxesImage``. Internally
      builds a full-size ``12*nside²`` array (not suited to extreme
      nside) and samples through ``healpix_to_canvas`` — crispest
      output for medium-nside maps where canvas pixels are larger
      than HEALPix tiles.
    * ``'pcolormesh'`` — same canvas-pixel sampling as ``imshow`` by
      default, rendered as a ``QuadMesh``. Pass ``sampling='lonlat'``
      to recover the legacy fuzzy lon/lat-grid behavior.

    Parameters
    ----------
    pixel_indices : array-like
        HEALPix pixel indices to render
    values : array-like
        Data value for each pixel (same length as pixel_indices). Used for
        colormapping. Pass None to render all pixels with a single color
        (set via ``facecolor`` in patch_kwargs).
    nside : int
        HEALPix nside parameter
    ax : WCSAxes, optional
        Axes to plot on. If None, uses current axes.
    nest : bool
        If True, pixel_indices are in NESTED ordering
    step : int
        (patches only) Points per pixel edge for boundary smoothness
        (1 = corners only). Higher values (4–8) help for large pixels at
        low nside.
    show_boundaries : bool
        (patches only) If True, draw pixel boundary lines
    boundary_color : str
        (patches only) Color for pixel boundaries
    boundary_lw : float
        (patches only) Line width for pixel boundaries
    set_extent : bool
        (patches only) If True, auto-set axis limits to show the
        rendered region (skipped when the data already fills most of
        the axes).
    padding_factor : float
        (patches only) Padding around the region extent, in units of
        pixel radius.
    cmap : str or Colormap
        Colormap for values
    vmin, vmax : float, optional
        Color scale limits. Defaults to data min/max.
    backend : {'patch', 'imshow', 'pcolormesh'}
        Rendering backend (see top of docstring). Default ``'patch'``
        (sparse maps benefit from per-tile control). Plural /
        singular forms accepted interchangeably (``'patches'`` →
        ``'patch'``).
    sampling : {'canvas', 'lonlat'}
        (``imshow`` / ``pcolormesh`` only) Sampling-grid mode. See
        ``plot_healpix_allsky`` for the full discussion;
        ``'canvas'`` is the crisper default,
        ``'lonlat'`` is the legacy fallback.
    interp : bool
        (canvas-sampling only) Bilinear interpolation in the
        HEALPix lookup. Default ``False`` (nearest-pixel).
    xyres_pix : tuple
        (``sampling='lonlat'`` only) Meshgrid resolution.
    blank_value : float
        Value used for non-given HEALPix pixels in the
        intermediate full-size array. Default ``np.nan``.
    **patch_kwargs
        Additional kwargs passed to the rendered artist (PatchCollection
        for ``backend='patch'``, QuadMesh for ``backend='pcolormesh'``).

    Returns
    -------
    artist : PatchCollection or QuadMesh
        The rendered artist (a ScalarMappable in either case, suitable
        for ``fig.colorbar``).

        For ``backend='patch'`` the returned ``PatchCollection`` carries
        an extra ``patch_pixel_index`` attribute: an integer array of
        length ``len(pc.get_paths())`` whose entry ``k`` is the source
        HEALPix pixel id of patch ``k``. The patches backend preserves
        the input ordering of ``pixel_indices``, but a tile that
        splits across the antimeridian or visible frame edge produces
        multiple patches that all carry the same source pixel id —
        so the mapping is many-to-one in general. Use it to recover
        which patch(es) belong to a given pixel for highlighting,
        per-patch alpha bumps, etc.

    Examples
    --------
    >>> # Zoom to a 2° disk around the Galactic center at 1' resolution
    >>> nside, _ = sph.auto_nside(resolution_arcmin=1.0)
    >>> pix = sph.healpix_circle_query(0, 0, 2.0, nside)
    >>> vals = np.random.rand(len(pix))  # or real data
    >>> ax = sph.make_wcs_frame(111, 'TAN', center=(0, 0))
    >>> sph.plot_healpix_sparse(pix, vals, nside, ax=ax)

    >>> # Render at mas scale around a VLBI source (no full-sky array!)
    >>> nside, _ = sph.auto_nside(resolution_mas=5.0)
    >>> pix = sph.healpix_circle_query(83.633, 22.015, 0.001, nside)
    >>> sph.plot_healpix_sparse(pix, data[pix], nside, ax=ax)

    >>> # Highlight the patch(es) for a chosen pixel using the
    >>> # patch_pixel_index attribute on the returned collection.
    >>> pc = sph.plot_healpix_sparse(pix, vals, nside, ax=ax)
    >>> target = pix[0]
    >>> mask = pc.patch_pixel_index == target
    >>> # Outline the matching patch(es) on top:
    >>> for path in np.asarray(pc.get_paths(), dtype=object)[mask]:
    ...     ax.plot(path.vertices[:, 0], path.vertices[:, 1],
    ...             color='red', lw=2.0)

    >>> # Use pcolormesh backend instead (smoother but full-array-bound)
    >>> sph.plot_healpix_sparse(pix, vals, nside, ax=ax, backend='pcolormesh')
    """
    _require_healpy('plot_healpix_sparse')
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    pixel_indices = np.atleast_1d(pixel_indices)
    if values is not None:
        values = np.atleast_1d(values)
        if len(values) != len(pixel_indices):
            raise ValueError(
                f"values ({len(values)}) must match pixel_indices ({len(pixel_indices)})")

    if ax is None:
        ax = plt.gca()

    # Normalize plural / singular backend forms.
    from .geometry._api import _resolve_backend
    backend = _resolve_backend(backend, helper_name='plot_healpix_sparse',
                                valid=('patch', 'pcolormesh', 'imshow'))

    if sampling not in ('canvas', 'lonlat'):
        raise ValueError(
            f"sampling must be 'canvas' or 'lonlat', got {sampling!r}")

    # ---------------- canvas-sampled backends (imshow / pcolormesh+canvas) ----------------
    if backend == 'imshow' or (backend == 'pcolormesh' and sampling == 'canvas'):
        # Both backends share the same canvas-pixel sampling — the
        # only difference is the final artist type. Materialize a
        # full-size HEALPix array filled with blank_value, set the
        # user's sparse pixels, and sample via healpix_to_canvas.
        from .projections.canvas import healpix_to_canvas
        npix = hp.nside2npix(nside)
        full = np.full(npix, blank_value, dtype=float)
        if values is not None:
            full[pixel_indices] = values
        else:
            full[pixel_indices] = 1.0  # uniform fill
        # Frame from axes WCS (default ICRS).
        from .wcs_frame import _get_wcs_frame_name
        ax_frame = _get_wcs_frame_name(ax)
        if ax.figure is not None:
            ax.figure.canvas.draw()
        arr, ext = healpix_to_canvas(full, ax,
                                      frame=ax_frame, nest=nest,
                                      interp=interp,
                                      blank_value=blank_value)
        if backend == 'imshow':
            return ax.imshow(arr, extent=ext, origin='lower',
                              cmap=cmap, vmin=vmin, vmax=vmax,
                              interpolation='nearest', **patch_kwargs)
        # pcolormesh + canvas
        xmin, xmax, ymin, ymax = ext
        ny, nx = arr.shape
        xe = np.linspace(xmin, xmax, nx + 1)
        ye = np.linspace(ymin, ymax, ny + 1)
        return ax.pcolormesh(xe, ye, arr, cmap=cmap,
                              vmin=vmin, vmax=vmax, **patch_kwargs)

    # ---------------- legacy pcolormesh+lonlat ----------------
    if backend == 'pcolormesh':  # sampling == 'lonlat' here
        # Legacy fallback: build a full-size HEALPix array, then go
        # through ``healpix_to_celestial`` (lon/lat meshgrid) +
        # pcolormesh with ``transform='world'``. Visibly fuzzy at
        # high latitudes — kept as a backwards-compat / debug path.
        npix = hp.nside2npix(nside)
        full = np.full(npix, blank_value, dtype=float)
        if values is not None:
            full[pixel_indices] = values
        else:
            full[pixel_indices] = 1.0
        try:
            center_deg = float(ax.wcs.wcs.crval[0])
        except Exception:
            center_deg = 180.0
        plonc, platc, pvals = healpix_to_celestial(
            full, 'allsky', center_deg, xyres_pix, blank_value)
        mesh = ax.pcolormesh(
            plonc, platc, pvals,
            transform=ax.get_transform('world'),
            cmap=cmap, vmin=vmin, vmax=vmax,
            **patch_kwargs,
        )
        return mesh

    # ---------------- patches backend (default) ----------------

    # Get pixel boundaries (world coords, one (lon, lat) array per
    # pixel). For large tiles (low nside), bump ``step`` automatically
    # so the boundary is densely sampled along the actual HEALPix
    # great-circle edges. ``hp.boundaries`` returns vertices along the
    # true geodesic edges of each tile, with no wrap ambiguity — this
    # avoids the antimeridian artifacts that ``_densify_polygon_edges``
    # produces when a tile sits entirely on one branch of the lon=±180
    # boundary but the densifier picks the wrong geodesic branch.
    max_pixrad_deg = np.degrees(hp.max_pixrad(nside))
    if max_pixrad_deg > 5.0 and step < 4:
        # Effective step: enough vertices to follow projection
        # curvature at low nside without overwhelming the renderer.
        # 8 points per edge gives a smooth diamond on MOL/AIT for
        # nside ≤ 8.
        step = max(step, 8)
    lon_bounds, lat_bounds = healpix_pixel_corners(
        pixel_indices, nside, step=step, nest=nest)

    # Determine if we have a WCS transform
    has_wcs = hasattr(ax, 'get_transform') and hasattr(ax, 'wcs')

    # Project each pixel's spherical polygon through the geometry
    # pipeline. This delegates antimeridian handling, frame-boundary
    # closure, and projection-curvature edge tracing to the same
    # ``_project_shape`` machinery that ``add_spherical_polygon`` uses,
    # so HEALPix tiles render correctly across all WCSAxes projections
    # and survive user-driven zoom / xlim / ylim changes (the polygons
    # are stored in pixel space and use ``ax.transData`` directly).
    #
    # A pixel that straddles the projection's antimeridian or visible
    # frame edge yields multiple paths — they all inherit the same
    # data value from the colormap and the same uniform color in the
    # no-values branch.
    if has_wcs:
        # Each HEALPix tile is a small spherical polygon (a 4-vertex
        # diamond). Rendering goes through the D3-style Sutherland-
        # Hodgman pipeline (``_antimeridian_clip`` + ``_stitch_and_project``)
        # — the same path ``add_spherical_polygon(clip='d3')`` uses.
        # This pre-clips each tile against the projection's antimeridian
        # as a great-circle line *before* WCS-projecting, so high-lat
        # tiles whose corners straddle the lon=center±180° wrap (e.g.
        # nside=32 RING pix 23 on AIT center=180) split cleanly into
        # the two hemispheres rather than tripping ``_project_shape``'s
        # frame-boundary closure logic. The default ``_project_shape``
        # path treats the WCS singularity at center±180° at high lat
        # as a "1 jump non-pole" case and walks the elliptical limb
        # from one segment endpoint to the other — gobbling up the
        # whole top of the frame.
        from .geometry._antimeridian import (
            _antimeridian_clip,
            _stitch_and_project,
        )
        from .geometry._densify import _densify_polygon_edges
        from .geometry._frame_geom import _get_frame_polygon

        try:
            lon_center = float(ax.wcs.wcs.crval[0])
        except Exception:
            lon_center = 0.0

        # Frame polygon is fixed for the axes — compute once.
        frame_poly = _get_frame_polygon(ax)

        # Densification resolution per edge. ``add_spherical_polygon``
        # defaults to 100 — overkill for small HEALPix tiles, so we
        # scale down for high nside to keep renders fast.
        max_pixrad_deg = np.degrees(hp.max_pixrad(nside))
        if max_pixrad_deg >= 10.0:
            edge_resolution = 100  # nside ≤ 4
        elif max_pixrad_deg >= 2.0:
            edge_resolution = 30   # nside ≤ 32
        else:
            edge_resolution = 10   # nside ≥ 64

        all_patches = []
        patch_values = []  # one entry per *patch* (not per pixel)
        patch_pixel_index = []  # source pixel_indices[i] per patch
        for i, (lons, lats) in enumerate(zip(lon_bounds, lat_bounds)):
            lons_arr = np.asarray(lons, dtype=float)
            lats_arr = np.asarray(lats, dtype=float)
            # Close the polygon for the densifier.
            lons_c = np.append(lons_arr, lons_arr[0])
            lats_c = np.append(lats_arr, lats_arr[0])
            # Detect antimeridian crossing: |dlon| > 180 between any two
            # consecutive vertices (after centering on lon_center) is the
            # signature of an edge that wraps the seam. Tiles that don't
            # cross can skip _densify_polygon_edges entirely — that
            # densifier's slerp introduces tiny floating-point variations
            # depending on edge orientation, which produces sub-pixel
            # gaps between the projected vertices of adjacent tiles.
            # The raw `hp.boundaries` vertices are deterministic per
            # pixel, so two adjacent tiles' shared edge endpoints are
            # byte-identical and project to the exact same pixel coords,
            # producing a watertight tessellation.
            lon_norm = ((lons_c - lon_center + 180.0) % 360.0) - 180.0
            crosses_antimeridian = bool(
                np.any(np.abs(np.diff(lon_norm)) > 180.0))
            if crosses_antimeridian:
                lons_d, lats_d = _densify_polygon_edges(
                    lons_c, lats_c,
                    resolution=edge_resolution,
                    geodesic='auto', geodesic_threshold=10.0,
                )
            else:
                lons_d, lats_d = lons_c, lats_c
            # D3-style: clip against antimeridian first, then project.
            # ``_stitch_and_project`` returns matplotlib Paths in pixel
            # coords clipped to the frame polygon.
            segments = _antimeridian_clip(lons_d, lats_d, lon_center)
            if not segments:
                continue
            # HEALPix tiles are small — no need for the expected-area
            # complement-detection heuristic. Pass est_frac=None so
            # ``_stitch_and_project`` skips the complement check. Also
            # set ``min_piece_area`` to effectively zero (the default
            # 5 px² is set for full-sized regions like survey footprints,
            # which create zero-area sliver artifacts that should be
            # filtered). HEALPix tiles are individual primitives; even
            # the antimeridian-split halves of pole tiles can be << 1 px²
            # but are exactly the pieces needed to close the apex gap.
            paths = _stitch_and_project(segments, ax, frame_poly,
                                        expected_frac=None,
                                        min_piece_area=0.0)
            if not paths:
                continue
            for path in paths:
                all_patches.append(MplPolygon(path.vertices, closed=True))
                patch_pixel_index.append(int(pixel_indices[i]))
                if values is not None:
                    patch_values.append(values[i])

        transform = ax.transData
    else:
        # Plain matplotlib axes — no WCS. Render polygons in
        # world-coord space directly (best effort).
        all_patches = []
        patch_values = []
        patch_pixel_index = []
        for i, (lons, lats) in enumerate(zip(lon_bounds, lat_bounds)):
            verts = np.column_stack([np.asarray(lons, float),
                                     np.asarray(lats, float)])
            verts = np.vstack([verts, verts[0]])
            all_patches.append(MplPolygon(verts, closed=True))
            patch_pixel_index.append(int(pixel_indices[i]))
            if values is not None:
                patch_values.append(values[i])
        transform = ax.transData

    # Set up edge styling
    edge_kw: dict[str, Any] = {}
    if show_boundaries:
        edge_kw['edgecolor'] = boundary_color
        edge_kw['linewidth'] = boundary_lw
    else:
        edge_kw['edgecolor'] = 'none'
        edge_kw['linewidth'] = 0

    if values is not None:
        values_arr = np.asarray(patch_values)
        if vmin is None:
            vmin = np.nanmin(values_arr) if len(values_arr) else 0.0
        if vmax is None:
            vmax = np.nanmax(values_arr) if len(values_arr) else 1.0

        collection = PatchCollection(all_patches, transform=transform,
                                     **edge_kw, **patch_kwargs)
        collection.set_array(values_arr)
        collection.set_cmap(cmap)
        collection.set_clim(vmin, vmax)
    else:
        # Uniform color
        fc = patch_kwargs.pop('facecolor', patch_kwargs.pop('color', 'C0'))
        collection = PatchCollection(all_patches, transform=transform,
                                     facecolor=fc, **edge_kw, **patch_kwargs)

    # Per-patch source-pixel mapping. The patches backend preserves the
    # input ordering of ``pixel_indices``: patch ``k`` corresponds to
    # ``pixel_indices[patch_pixel_index[k]]``-equivalent pixel id stored
    # directly in ``patch_pixel_index[k]``. A tile that splits across
    # the antimeridian or visible frame edge produces multiple patches
    # that all carry the same source pixel id, so the mapping is
    # many-to-one. Common workflows: ``mask = pc.patch_pixel_index ==
    # target_pix`` to find the patch(es) for a given pixel; ``np.unique
    # (pc.patch_pixel_index)`` to recover the rendered pixel set.
    # Dynamic attribute carried on the returned artist for per-patch
    # pixel lookup; matplotlib collections permit ad-hoc attributes.
    collection.patch_pixel_index = np.asarray(  # type: ignore[attr-defined]
        patch_pixel_index, dtype=int)

    ax.add_collection(collection)

    # Auto-set axis extent — but only when there's something to zoom
    # in to. The bounding box is computed from the projected polygon
    # vertices (already in pixel space) padded by ``max_pixrad``
    # scaled by ``padding_factor``. We then skip the zoom in two
    # cases where applying it would be wrong:
    #
    # (a) The data already spans most of the axes' natural extent in
    #     either dimension. Cropping a horizontal ring on an all-sky
    #     frame would zoom one dimension while leaving the other
    #     full-width — and on EllipticalFrame / SinusoidalFrame /
    #     other custom frames the visual frame spine is drawn in
    #     axes-fraction coords, so it doesn't track ``set_xlim`` /
    #     ``set_ylim`` and ends up offset from the gridlines and
    #     data after the zoom.
    #
    # (b) The data fills most of one dimension but only a thin slice
    #     of the other (e.g. 30 pixels along a single declination,
    #     spanning all RA). Same problem as (a): cropping the thin
    #     dimension produces a misaligned frame.
    #
    # The threshold below — zoom only when the data fills less than
    # 80% of the natural extent in *both* dimensions — keeps the
    # function's typical use case (a small cluster within a much
    # larger axes) working while skipping zoom when the user already
    # has the right view.
    if set_extent and has_wcs and all_patches:
        all_pix = np.vstack(
            [poly.get_xy() for poly in all_patches])
        cdelt = np.array(ax.wcs.wcs.cdelt[:2], dtype=float)
        scale = np.mean(np.abs(cdelt)) if np.any(cdelt) else 1.0
        eps_pix = (np.degrees(hp.max_pixrad(nside)) / scale) * padding_factor
        xmin = float(all_pix[:, 0].min()) - eps_pix
        xmax = float(all_pix[:, 0].max()) + eps_pix
        ymin = float(all_pix[:, 1].min()) - eps_pix
        ymax = float(all_pix[:, 1].max()) + eps_pix

        orig_xlim = ax.get_xlim()
        orig_ylim = ax.get_ylim()
        orig_xspan = float(orig_xlim[1] - orig_xlim[0])
        orig_yspan = float(orig_ylim[1] - orig_ylim[0])
        cur_xspan = xmax - xmin
        cur_yspan = ymax - ymin

        # Only zoom if data fills < 80% of natural extent in BOTH dims.
        x_frac = abs(cur_xspan) / abs(orig_xspan) if orig_xspan else 1.0
        y_frac = abs(cur_yspan) / abs(orig_yspan) if orig_yspan else 1.0
        if x_frac < 0.80 and y_frac < 0.80:
            # Match the axes' existing aspect ratio so a small cluster
            # stays diamond-shaped rather than getting stretched.
            target_aspect = (
                abs(orig_xspan) / abs(orig_yspan) if orig_yspan else 1.0
            )
            cur_aspect = cur_xspan / cur_yspan if cur_yspan else 1.0
            if cur_aspect > target_aspect:
                new_yspan = cur_xspan / target_aspect
                ycen = 0.5 * (ymin + ymax)
                ymin = ycen - 0.5 * new_yspan
                ymax = ycen + 0.5 * new_yspan
            elif cur_aspect < target_aspect:
                new_xspan = cur_yspan * target_aspect
                xcen = 0.5 * (xmin + xmax)
                xmin = xcen - 0.5 * new_xspan
                xmax = xcen + 0.5 * new_xspan

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    return collection


def bin_data_sparse(lons: SkyCoord | npt.ArrayLike, lats: npt.ArrayLike | None = None,
                    data: npt.ArrayLike | None = None,
                    nside: int | None = None,
                    statistic: str | Callable[[np.ndarray], float] = 'mean',
                    nest: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin data into HEALPix pixels and return only the occupied pixel indices
    and aggregated values — no full-sky array allocation.

    This is the sparse counterpart to ``bin_data_as_healpix()``: instead of
    creating a full ``12 * nside²`` array, it returns only the pixels that
    contain data, making it usable at arbitrarily high nside.

    Parameters
    ----------
    lons, lats : array-like, or SkyCoord in ``lons``
        Source coordinates in degrees. A ``SkyCoord`` array may be passed as
        ``lons`` instead, replacing both; following arguments must then be
        keywords (``nside=64``, not a bare ``64``). A HEALPix map carries no
        frame of its own, so a SkyCoord KEEPS its frame rather than being
        converted.
    data : array-like
        Data values per source
    nside : int
    statistic : str or callable
        ``'mean'``, ``'median'``, ``'sum'``, ``'min'``, ``'max'``, ``'std'``,
        ``'count'``, or a callable that takes the 1-D array of (finite) values
        in a cell and returns a scalar — e.g.
        ``lambda v: np.percentile(v, 90)``.
    nest : bool
        If True, use NESTED pixel ordering

    Returns
    -------
    pixel_indices : ndarray
        Occupied pixel indices (sorted)
    values : ndarray
        Aggregated value per occupied pixel

    Examples
    --------
    >>> pix, vals = sph.bin_data_sparse(ra, dec, flux, nside=2**17)
    >>> sph.plot_healpix_sparse(pix, vals, nside=2**17, ax=ax)
    """
    _require_healpy('bin_data_sparse')
    # frame_name=None: HEALPix maps carry no frame, so a SkyCoord
    # keeps its own (matching the bare vertex builders).
    lons, lats = _coords_or_arrays_deg(lons, lats, None, 'bin_data_sparse')
    if data is None or nside is None:
        raise TypeError('bin_data_sparse: data and nside are required.')
    lons_arr = np.asarray(lons, dtype=float)
    lats_arr = np.asarray(lats, dtype=float)
    data_arr = np.asarray(data, dtype=float)

    valid = np.isfinite(data_arr)
    lons_arr, lats_arr, data_arr = lons_arr[valid], lats_arr[valid], data_arr[valid]

    pix = hp.ang2pix(nside, lons_arr, lats_arr, lonlat=True, nest=nest)

    if statistic == 'count':
        unique_pix, counts = np.unique(pix, return_counts=True)
        return unique_pix, counts.astype(float)

    # Group by pixel
    sort_idx = np.argsort(pix)
    sorted_pix = pix[sort_idx]
    sorted_data = data_arr[sort_idx]

    # Find boundaries between groups
    unique_pix, first_idx = np.unique(sorted_pix, return_index=True)
    splits = np.append(first_idx, len(sorted_pix))

    func_map = {
        'mean': np.nanmean, 'median': np.nanmedian, 'sum': np.nansum,
        'min': np.nanmin, 'max': np.nanmax, 'std': np.nanstd,
    }
    func: Any
    if callable(statistic):
        func = statistic
    elif statistic in func_map:
        func = func_map[statistic]
    else:
        raise ValueError(f"Unknown statistic '{statistic}'. Choose from: "
                         f"{', '.join(func_map)}, count, or pass a callable")

    agg_values = np.array([
        func(sorted_data[splits[i]:splits[i+1]])
        for i in range(len(unique_pix))
    ])

    return unique_pix, agg_values


class HealpixBins(NamedTuple):
    """Sparse HEALPix binning result from :func:`image_to_healpix`.

    Carries the occupied ``pixels`` and their ``values``, plus the ``nside``
    (which sparse pixel indices do not themselves encode — so it must travel
    with them) and an optional ``counts`` coverage array (``None`` unless
    ``return_counts=True``).

    Supports attribute access AND tuple unpacking. The intended use is
    attribute access, which is robust to the optional ``counts`` field::

        r = sph.image_to_healpix(field, hdr, sparse=True)   # auto nside
        sph.plot_healpix_sparse(r.pixels, r.values, r.nside, ax=ax)
    """
    pixels:  np.ndarray
    values:  np.ndarray
    nside:   int
    counts:  np.ndarray | None = None


def _resolve_image_input(image: Any, hdr_or_wcs: Any) -> tuple[Any, Any]:
    """Normalize the flexible first argument of :func:`image_to_healpix` to a
    ``(data_array, header_or_wcs)`` pair.

    Accepts a bare array (then ``hdr_or_wcs`` is required), a ``(data, header)``
    / ``(data, wcs)`` tuple, an astropy HDU, an HDUList, or a path to a FITS
    file. For an HDU / HDUList / path, an explicitly-passed ``hdr_or_wcs`` (e.g.
    a corrected WCS) still wins over the file's own header.
    """
    import os

    from astropy.io import fits

    def _first_image_hdu(hdul: Any) -> Any:
        for hdu in hdul:
            if getattr(hdu, 'data', None) is not None \
                    and np.ndim(hdu.data) >= 2:
                return hdu
        raise ValueError("no image HDU (>=2-D data) found in the FITS input")

    if isinstance(image, (tuple, list)) and len(image) == 2:
        image, paired = image
        if hdr_or_wcs is None:
            hdr_or_wcs = paired
    elif isinstance(image, (str, os.PathLike)):
        with fits.open(image) as hdul:
            hdu = _first_image_hdu(hdul)
            data, header = np.asarray(hdu.data), hdu.header
        return data, (hdr_or_wcs if hdr_or_wcs is not None else header)
    elif isinstance(image, fits.HDUList):
        hdu = _first_image_hdu(image)
        return (np.asarray(hdu.data),
                hdr_or_wcs if hdr_or_wcs is not None else hdu.header)
    elif (hasattr(image, 'data') and hasattr(image, 'header')
            and not isinstance(image, np.ndarray)):
        # A single HDU (Primary/Image/Compressed).
        return (np.asarray(image.data),
                hdr_or_wcs if hdr_or_wcs is not None else image.header)

    if hdr_or_wcs is None:
        raise ValueError(
            "hdr_or_wcs is required when the image is a bare array (pass a "
            "header/WCS, or give an HDU, HDUList, or FITS path instead)")
    return image, hdr_or_wcs


def nside_from_array(healpix_array: npt.ArrayLike) -> int:
    """Infer the HEALPix ``nside`` from a full-sky map's length.

    A full-sky HEALPix map always has ``npix = 12 * nside**2`` elements, so the
    nside is recoverable from a dense array (this is what the dense plotters do
    internally). Raises ``ValueError`` for a length that is not a valid full-sky
    map — note a *sparse* index array does NOT carry nside (its indices only
    bound it from below), so this cannot be used on sparse output.
    """
    npix = int(np.asarray(healpix_array).size)
    if npix <= 0 or npix % 12 != 0:
        raise ValueError(
            f"length {npix} is not a valid full-sky HEALPix map "
            f"(needs npix = 12 * nside**2)")
    nside = int(round((npix / 12) ** 0.5))
    if 12 * nside * nside != npix:
        raise ValueError(
            f"length {npix} does not correspond to an integer nside")
    return nside


def _resolve_nside(nside: int | str | Any, wcs2d: Any,
                   oversample: float) -> int:
    """Resolve the ``nside`` argument of :func:`image_to_healpix` to an int.

    Accepts an explicit int (used as-is; ``oversample`` ignored), ``'auto'``
    (matched to the WCS pixel scale), or a target angular resolution (an
    astropy ``Quantity`` or a string like ``'5arcmin'`` / ``'30arcsec'``).
    ``oversample`` scales a resolution-derived nside finer (``>1``) or coarser
    (``<1``).
    """
    if isinstance(nside, (int, np.integer)) and not isinstance(nside, bool):
        return int(nside)
    if isinstance(nside, str) and nside == 'auto':
        from astropy.wcs.utils import proj_plane_pixel_scales
        base_res_deg = float(np.mean(np.abs(proj_plane_pixel_scales(wcs2d))))
    else:
        from astropy.coordinates import Angle
        try:
            base_res_deg = float(Angle(nside).to_value('deg'))
        except Exception as exc:
            raise ValueError(
                f"nside must be an int, 'auto', or an angular resolution "
                f"(e.g. '5arcmin' or 5*u.arcmin), got {nside!r}") from exc
    n, _ = auto_nside(resolution_deg=base_res_deg / float(oversample))
    return n


def image_to_healpix(data: npt.ArrayLike, hdr_or_wcs: Any = None,
                     nside: int | str | Any = 'auto',
                     statistic: str | Callable[[np.ndarray], float] = 'mean', *,
                     frame: str | None = None, sparse: bool | str = 'auto',
                     nest: bool = False, oversample: float = 1.0,
                     return_counts: bool = False) -> Any:
    """Bin a 2-D FITS image onto a HEALPix grid by sky position.

    Reprojects an image to HEALPix the simple, dependency-free way: take each
    image pixel's world coordinate from the WCS and aggregate the pixel values
    that land in each HEALPix cell (via :func:`bin_data_sparse`). This is a
    *binning* reprojection — no flux interpolation — appropriate when the
    HEALPix resolution is comparable to or coarser than the image's, the usual
    all-sky case. For a flux-conserving / interpolating reproject of a fine
    image onto a coarse grid (or vice versa), use the external ``reproject``
    package (``reproject.reproject_to_healpix``).

    Parameters
    ----------
    data : array, HDU, HDUList, FITS path, or (data, header/wcs) tuple
        The image. Most directly a 2-D array (with ``hdr_or_wcs`` given), but
        for convenience also an astropy ``ImageHDU`` / ``PrimaryHDU``, an
        ``HDUList`` (first image HDU used), a path to a FITS file, or a
        ``(data, header)`` / ``(data, wcs)`` tuple — in which cases
        ``hdr_or_wcs`` is optional. Degenerate leading axes (e.g. FREQ /
        STOKES) are squeezed; the result must be 2-D.
    hdr_or_wcs : astropy.io.fits.Header or astropy.wcs.WCS, optional
        WCS for the image (reduced to its 2-D celestial part if higher-D).
        Required only for a bare-array ``data``; if given alongside an HDU /
        path it overrides the file's own header.
    nside : int, 'auto', or angular resolution
        HEALPix nside. ``'auto'`` (default) picks the nside whose pixels are at
        or finer than the image's pixel scale (via :func:`auto_nside` on the
        WCS pixel scale), so HEALPix cells roughly match image pixels. May also
        be a target angular resolution — an astropy ``Quantity`` or a string
        like ``'5arcmin'`` / ``'30arcsec'`` — resolved through :func:`auto_nside`.
    statistic : str or callable
        Per-cell aggregation: ``'mean'`` (default), ``'sum'``, ``'median'``,
        ``'min'``, ``'max'``, ``'std'``, ``'count'``, or a callable taking the
        1-D array of (finite) values in a cell and returning a scalar — e.g.
        ``lambda v: np.percentile(v, 90)`` for a 90th-percentile map.
    frame : str, optional
        Target HEALPix frame (``'galactic'``, ``'icrs'``, ``'ecliptic'``, …).
        If ``None`` (default), bin in the image's native WCS frame. Otherwise
        the image pixel coordinates are transformed into ``frame`` before
        binning (e.g. drop an equatorial image onto a galactic HEALPix map).
    sparse : bool or 'auto'
        Output form. ``'auto'`` (default) returns the dense full-sky array when
        the image covers most of the sky and the sparse form when it is a small
        field (occupies under half the HEALPix cells) — which also avoids
        allocating a giant mostly-empty array when ``'auto'`` nside picks a high
        value for a fine field image. ``False`` forces dense; ``True`` forces
        sparse. Note the sparse form includes ``nside`` (sparse indices do not
        encode it; see :func:`nside_from_array` for the dense case).
    nest : bool
        Pixel ordering of the output (``False`` = RING, the convention the
        dense-map plotters expect; ``True`` = NESTED).
    oversample : float
        Scale a resolution-derived nside (``'auto'`` or a target resolution)
        finer (``>1``) or coarser (``<1``) than the raw pixel scale. Ignored
        for an explicit integer ``nside``.
    return_counts : bool
        If True, also return the per-cell count of image pixels binned into
        each cell (a *coverage* map) — useful to weight, or to mask thin /
        under-sampled cells. Aligned with the value output.

    Returns
    -------
    dense form (``sparse`` resolves to False)
        ``hpxmap`` — a full-sky ``12 * nside**2`` array (empty cells ``NaN``),
        ready for :func:`plot_healpix_map` (which infers nside from its length;
        see :func:`nside_from_array`). With ``return_counts`` it is instead a
        ``(hpxmap, counts_map)`` tuple (``counts_map`` empty cells ``0``).
    sparse form (``sparse`` resolves to True)
        a :class:`HealpixBins` NamedTuple ``(pixels, values, nside, counts)``
        — sparse indices do not encode nside, so it travels in the result.
        Use attribute access: ``plot_healpix_sparse(r.pixels, r.values,
        r.nside, ax=ax)``. ``counts`` is ``None`` unless ``return_counts``.

    Examples
    --------
    >>> hpx = sph.image_to_healpix(data, header)                # auto nside/form
    >>> sph.plot_healpix_map(hpx, ax=ax)                        # nside inferred
    >>> r = sph.image_to_healpix(field, hdr, sparse=True)       # auto nside
    >>> sph.plot_healpix_sparse(r.pixels, r.values, r.nside, ax=ax)
    """
    _require_healpy('image_to_healpix')
    from .core.fits_utils import header_coord_grids

    data, hdr_or_wcs = _resolve_image_input(data, hdr_or_wcs)
    arr = np.squeeze(np.asarray(data, dtype=float))
    if arr.ndim != 2:
        raise ValueError(
            f"data must be 2-D after squeezing degenerate axes, got "
            f"shape {arr.shape}")

    wcs = hdr_or_wcs if isinstance(hdr_or_wcs, WCS) else WCS(hdr_or_wcs)
    wcs2d = wcs.celestial if getattr(wcs, 'naxis', 2) > 2 else wcs
    nside = _resolve_nside(nside, wcs2d, oversample)

    lon, lat = header_coord_grids(
        wcs2d, shape=(int(arr.shape[0]), int(arr.shape[1])))
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    vals = arr.ravel()

    # Drop blank pixels and any off-projection (NaN) world coordinates before
    # binning — both bin paths skip non-finite data, but filtering here keeps
    # the optional frame transform cheap and SkyCoord-clean.
    good = np.isfinite(vals) & np.isfinite(lon) & np.isfinite(lat)
    lon, lat, vals = lon[good], lat[good], vals[good]

    if frame is not None:
        from astropy.wcs.utils import wcs_to_celestial_frame

        from .coord_overlay import _resolve_frame
        native = wcs_to_celestial_frame(wcs2d)
        sky = SkyCoord(lon, lat, unit='deg', frame=native)
        sky = sky.transform_to(_resolve_frame(frame))
        # .spherical is frame-agnostic (handles ra/dec, l/b, ecliptic lon/lat).
        lon = sky.spherical.lon.deg
        lat = sky.spherical.lat.deg

    pix, agg = bin_data_sparse(lon, lat, vals, nside,
                               statistic=statistic, nest=nest)
    counts = None
    if return_counts:
        # Same (sorted, unique) pixel set as the value binning, so counts
        # aligns element-wise with ``agg`` / scatters to the same cells.
        _, counts = bin_data_sparse(lon, lat, vals, nside,
                                    statistic='count', nest=nest)

    # Resolve 'auto' output form: a small field fills few cells → sparse (and
    # never allocate the giant array); an all-sky image fills most → dense.
    npix = hp.nside2npix(nside)
    if isinstance(sparse, str):
        if sparse != 'auto':
            raise ValueError(
                f"sparse must be a bool or 'auto', got {sparse!r}")
        sparse = len(pix) < 0.5 * npix

    if sparse:
        return HealpixBins(pix, agg, nside, counts)
    hpxmap = np.full(npix, np.nan)
    hpxmap[pix] = agg
    if return_counts:
        counts_map = np.zeros(npix)
        counts_map[pix] = counts
        return hpxmap, counts_map
    return hpxmap


###############################################################################
#                                                                             #
#              HEALPIX CONVENIENCE & MAP OPERATIONS              #
#                                                                             #
###############################################################################


class HealpixResult(NamedTuple):
    """Return type for :func:`plot_healpix_allsky` and
    :func:`healpix_allsky_figure`.

    Every artist created by the plot is reachable on the result so
    users can adjust them after the fact (relabel the colorbar,
    restyle the mappable, etc.). Supports tuple unpacking
    (``fig, ax, im, cbar = plot_healpix_allsky(...)``) and
    attribute access (``result.colorbar.set_label(...)``).
    """
    fig:      object  # matplotlib.figure.Figure
    ax:       object  # WCSAxes
    mappable: object  # AxesImage / QuadMesh / PatchCollection
    colorbar: object  # matplotlib.colorbar.Colorbar | None


def _check_image_kwargs_no_overlap(image_kwargs: dict[str, Any],
                                   **direct: Any) -> None:
    """Raise ``TypeError`` if a key is supplied both as a direct
    named kwarg and inside ``image_kwargs``. The image_kwargs
    pass-through is for matplotlib options the function doesn't
    expose directly; collisions almost always mean a user typo or
    copy-paste error and silent overrides are confusing."""
    if not image_kwargs:
        return
    overlap = [k for k in direct if k in image_kwargs]
    if overlap:
        raise TypeError(
            f"keyword(s) {overlap!r} supplied both as a named "
            "argument and inside image_kwargs. Pass each option "
            "in exactly one place."
        )


def plot_healpix_allsky(healpix_array: npt.ArrayLike, ax: Any = None, *,
                        cmap: Any = 'viridis',
                        vmin: float | None = None, vmax: float | None = None,
                        blank_value: float = np.nan,
                        xyres_pix: tuple[int, int] = (2000, 1000),
                        title: str | None = None, colorbar: bool = True,
                        cbar_label: str | None = None,
                        planes: Sequence[str] | None = None,
                        backend: str = 'imshow',
                        sampling: str = 'canvas',
                        nest: bool = False, interp: bool = False,
                        show_boundaries: bool = False,
                        boundary_color: str = '0.5',
                        boundary_lw: float = 0.3,
                        image_kwargs: dict[str, Any] | None = None,
                        ) -> HealpixResult:
    """Plot a HEALPix all-sky map onto an existing WCSAxes.

    Pure axis-plotter — does not create a figure. Use
    :func:`healpix_allsky_figure` for a one-line figure-builder
    convenience that owns ``projection`` / ``center`` / ``frame`` /
    ``figsize`` etc. and delegates here under the hood.

    Three rendering backends are available via ``backend``:

    * ``'imshow'`` (default) — sample the HEALPix map at every
      canvas pixel of the WCSAxes (the same approach
      ``hp.mollview`` uses) and render via ``ax.imshow``.
      Produces the crispest output — every output pixel is square
      in canvas space, so there's no high-latitude banding.
      Returns an ``AxesImage``.
    * ``'pcolormesh'`` — same canvas-pixel sampling as ``imshow``
      by default, rendered via ``ax.pcolormesh`` for a
      ``QuadMesh`` return type (per-cell event picking,
      cell-aligned outputs). Pass ``sampling='lonlat'`` to recover
      the legacy lon/lat-grid sampling — a fallback / debug
      path that produces visibly fuzzy edges at high latitudes.
    * ``'patches'`` — every non-blank HEALPix tile drawn as an
      individual Polygon via the geometry pipeline. Sharp tile
      structure, ideal for visualizing tile-resolution differences
      between maps; impractical at very high nside.

    Parameters
    ----------
    healpix_array : array-like
        Full-sky HEALPix map (RING ordering, npix = 12*nside²;
        pass ``nest=True`` for NESTED ordering).
    ax : WCSAxes, optional
        Target axes. If ``None``, builds a sensible default
        AIT all-sky frame for one-off interactive use; for
        composability into multi-panel figures, pre-create the
        axes via ``make_wcs_frame`` and pass it here.
    cmap : str or Colormap
    vmin, vmax : float, optional
    blank_value : float
        Value treated as blank/masked (default NaN).
    xyres_pix : tuple
        (``sampling='lonlat'`` only) Meshgrid resolution for the
        legacy lon/lat sampling path.
    title : str, optional
    colorbar : bool
        Show colorbar.
    cbar_label : str, optional
    planes : list of str, optional
        Plane overlays to add (e.g. ['galactic', 'ecliptic']).
    backend : {'imshow', 'pcolormesh', 'patch'}
        Rendering backend (see top of docstring). Default ``'imshow'``.
    sampling : {'canvas', 'lonlat'}
        (``imshow`` / ``pcolormesh`` only) Where to sample the
        HEALPix data. ``'canvas'`` (default) samples at every
        canvas-pixel center — crispest. ``'lonlat'`` builds a
        regular lon/lat grid and lets matplotlib reproject —
        kept as a fallback / debug option. ``imshow`` ignores
        ``sampling='lonlat'`` (it cannot reproject) and silently
        falls back to canvas sampling.

        .. warning::
           ``sampling='lonlat'`` goes through
           ``ax.pcolormesh(..., transform=ax.get_transform('world'))``
           which is fragile at certain lon/lat-grid sizes —
           matplotlib's quad rasterizer mishandles the
           lon=0/360 wraparound at specific resolutions (e.g.
           ``xyres_pix=(1600, 800)`` on AIT center=180 produces
           a giant mis-projected red strip across the lower
           half). Stick to common defaults like ``(2000, 1000)``
           if you use this path. ``sampling='canvas'`` is
           immune — every canvas pixel is sampled
           independently, no antimeridian seam in the
           rasterization step.
    nest : bool
        If True, treat ``healpix_array`` as NESTED-ordered; default
        ``False`` (RING).
    interp : bool
        (canvas-sampling only) If True, use bilinear interpolation
        in the HEALPix lookup (smoother edges); default ``False``
        (nearest-pixel lookup, sharp tile boundaries).
    show_boundaries : bool
        (patches only) Draw the boundary line of each tile.
    boundary_color : str
        (patches only) Color for the per-tile boundary line.
    boundary_lw : float
        (patches only) Line width for the per-tile boundary line.
    image_kwargs : dict, optional
        Extra kwargs forwarded to the rendered artist
        (``imshow`` / ``pcolormesh`` / ``PatchCollection``). Any
        key already exposed as a named parameter (``cmap``,
        ``vmin``, ``vmax`` etc.) raises ``TypeError`` if also
        present here — pass each option in exactly one place.

    Returns
    -------
    HealpixResult
        NamedTuple ``(fig, ax, mappable, colorbar)``. ``mappable``
        is an ``AxesImage`` for ``'imshow'``, a ``QuadMesh`` for
        ``'pcolormesh'``, a ``PatchCollection`` for ``'patches'``.
        ``colorbar`` is ``None`` if ``colorbar=False``.

    Examples
    --------
    >>> # Pre-create the axes for composability
    >>> fig = plt.figure(figsize=(14, 7))
    >>> ax = sph.make_wcs_frame(111, 'AIT', center=180, fig=fig)
    >>> result = sph.plot_healpix_allsky(hpxmap, ax=ax,
    ...     cmap='inferno', cbar_label='counts/pixel')
    >>> result.colorbar.set_label('counts (log scale)')

    >>> # One-line figure-builder convenience (separate function)
    >>> result = sph.healpix_allsky_figure(hpxmap, projection='MOL',
    ...     center=0, figsize=(11, 5))

    >>> # Show the underlying HEALPix tile structure
    >>> fig, ax, im, cbar = sph.plot_healpix_allsky(hpxmap, ax=ax,
    ...     backend='patch', show_boundaries=True, boundary_lw=0.2)
    """
    _require_healpy('plot_healpix_allsky')

    image_kwargs = dict(image_kwargs) if image_kwargs else {}
    _check_image_kwargs_no_overlap(image_kwargs, cmap=cmap,
                                   vmin=vmin, vmax=vmax)

    if ax is None:
        # Sensible-default figure for one-off use. Matches the
        # previous default but is now opt-in by leaving ax=None.
        fig = plt.figure(figsize=(14, 7))
        ax = make_wcs_frame(111, 'AIT', center=180, frame='ICRS', fig=fig)
        format_ticklabels(ax, style='publication')
    else:
        fig = ax.figure

    # Normalize plural/singular backend forms.
    from .geometry._api import _resolve_backend
    backend = _resolve_backend(backend, helper_name='plot_healpix_allsky',
                                valid=('patch', 'pcolormesh', 'imshow'))

    if sampling not in ('canvas', 'lonlat'):
        raise ValueError(
            f"sampling must be 'canvas' or 'lonlat', got {sampling!r}")

    healpix_array = np.asarray(healpix_array, dtype=float)

    # Resolve the axes' frame (for HEALPix → axes-frame conversion).
    ax_frame = _get_wcs_frame_name(ax) if hasattr(ax, 'wcs') else 'icrs'

    if backend == 'imshow':
        # Canvas-pixel sampling — crispest path. Returns AxesImage.
        from .projections.canvas import healpix_to_canvas
        fig.canvas.draw()  # WCSAxes needs a draw before pixel_to_world
        arr, ext = healpix_to_canvas(
            healpix_array, ax,
            frame=ax_frame, nest=nest, interp=interp,
            blank_value=blank_value)
        im = ax.imshow(arr, extent=ext, origin='lower',
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', **image_kwargs)
    elif backend == 'pcolormesh':
        if sampling == 'canvas':
            # Canvas-pixel sampling rendered as QuadMesh.
            from .projections.canvas import healpix_to_canvas
            fig.canvas.draw()
            arr, ext = healpix_to_canvas(
                healpix_array, ax,
                frame=ax_frame, nest=nest, interp=interp,
                blank_value=blank_value)
            xmin, xmax, ymin, ymax = ext
            ny, nx = arr.shape
            x_edges = np.linspace(xmin, xmax, nx + 1)
            y_edges = np.linspace(ymin, ymax, ny + 1)
            im = ax.pcolormesh(x_edges, y_edges, arr,
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               **image_kwargs)
        else:
            # Legacy fallback: lon/lat grid + transform='world'.
            try:
                center_deg = float(ax.wcs.wcs.crval[0])
            except Exception:
                center_deg = 180.0
            plonc, platc, pvals = healpix_to_celestial(
                healpix_array, 'allsky', center_deg, xyres_pix, blank_value)
            im = ax.pcolormesh(plonc, platc, pvals,
                               transform=ax.get_transform('world'),
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               **image_kwargs)
    else:  # backend == 'patch'
        # Identify non-blank pixels and delegate to the patches path.
        if np.isnan(blank_value):
            keep = np.isfinite(healpix_array)
        else:
            keep = healpix_array != blank_value
        pix_idx = np.where(keep)[0]
        pix_vals = healpix_array[keep]
        nside = hp.npix2nside(len(healpix_array))
        im = plot_healpix_sparse(
            pix_idx, pix_vals, nside=nside, ax=ax, nest=nest,
            cmap=cmap, vmin=vmin, vmax=vmax,
            show_boundaries=show_boundaries,
            boundary_color=boundary_color, boundary_lw=boundary_lw,
            set_extent=False,  # leave the axes' all-sky view alone
            backend='patch',
            **image_kwargs,
        )

    cbar = None
    if colorbar:
        cbar = add_colorbar(im, ax=ax, label=cbar_label)

    if planes:
        for p in planes:
            add_plane_overlay(ax, p, lw=1, ls='--', alpha=0.6)

    if title:
        ax.set_title(title)

    return HealpixResult(fig=fig, ax=ax, mappable=im, colorbar=cbar)


def healpix_allsky_figure(healpix_array: npt.ArrayLike,
                          projection: str = 'AIT', center: float = 180,
                          frame: str = 'ICRS',
                          figsize: tuple[float, float] | None = None,
                          dpi: int = 100,
                          style: str = 'publication',
                          savepath: str | None = None,
                          **plot_kwargs: Any) -> HealpixResult:
    """One-line HEALPix all-sky heatmap — like ``hp.mollview()`` but
    with full WCSAxes projections, formatting, and overlay support.

    Convenience wrapper that owns figure creation. Builds the
    figure + axes via ``make_wcs_frame`` and delegates to
    :func:`plot_healpix_allsky` for the actual data rendering;
    use that directly to plot into an existing axes.

    Parameters
    ----------
    healpix_array : array-like
        See :func:`plot_healpix_allsky`.
    projection : str
        Any supported projection name (e.g. 'AIT', 'MOL', 'SFL', 'CAR').
    center : float
        Center longitude in degrees.
    frame : str
        Coordinate frame ('ICRS', 'Galactic', etc.).
    figsize : tuple, optional
        Figure size. Default ``(14, 7)``.
    dpi : int
        Figure DPI.
    style : str
        Tick label style preset.
    savepath : str, optional
        If given, save figure to this path.
    **plot_kwargs
        All remaining kwargs forwarded to :func:`plot_healpix_allsky`
        (``cmap``, ``vmin``, ``vmax``, ``backend``, ``sampling``,
        ``image_kwargs``, …).

    Returns
    -------
    HealpixResult
        NamedTuple ``(fig, ax, mappable, colorbar)``.

    Examples
    --------
    >>> result = sph.healpix_allsky_figure(hpxmap, projection='MOL',
    ...     center=0, cmap='inferno', cbar_label='counts/pixel',
    ...     planes=['galactic', 'ecliptic'])
    >>> result.colorbar.set_label('counts (log)')
    """
    _require_healpy('healpix_allsky_figure')

    if figsize is None:
        figsize = (14, 7)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = make_wcs_frame(111, projection, center=center, frame=frame, fig=fig)
    format_ticklabels(ax, style=style)

    result = plot_healpix_allsky(healpix_array, ax=ax, **plot_kwargs)

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')

    return result


def healpix_smooth(healpix_array: npt.ArrayLike,
                   sigma_deg: float | None = None,
                   sigma_arcmin: float | None = None,
                   sigma_arcsec: float | None = None,
                   beam_fwhm_arcmin: float | None = None) -> np.ndarray:
    """
    Smooth a HEALPix map with a Gaussian beam.

    Convenience wrapper around ``hp.smoothing()`` with intuitive
    angular units.

    Parameters
    ----------
    healpix_array : array-like
        Full-sky HEALPix map
    sigma_deg : float, optional
    sigma_arcmin : float, optional
    sigma_arcsec : float, optional
        Gaussian sigma in the specified unit. Exactly one must be given,
        OR use beam_fwhm_arcmin.
    beam_fwhm_arcmin : float, optional
        FWHM of Gaussian beam in arcminutes (converted to sigma internally).

    Returns
    -------
    smoothed : ndarray
    """
    _require_healpy('healpix_smooth')

    if beam_fwhm_arcmin is not None:
        sigma_rad = np.radians(beam_fwhm_arcmin / 60.) / (2 * np.sqrt(2 * np.log(2)))
    else:
        specs = [(sigma_deg, 1.0), (sigma_arcmin, 1/60.),
                 (sigma_arcsec, 1/3600.)]
        given = [(v * scale) for v, scale in specs if v is not None]
        if len(given) != 1:
            raise ValueError("Specify exactly one of sigma_deg, sigma_arcmin, "
                             "sigma_arcsec, or beam_fwhm_arcmin")
        sigma_rad = np.radians(given[0])

    return hp.smoothing(np.asarray(healpix_array, dtype=float), sigma=sigma_rad)


def healpix_upgrade(healpix_array: npt.ArrayLike, nside_out: int) -> np.ndarray:
    """
    Upgrade a HEALPix map to a higher nside (finer resolution).

    Each pixel in the input map is split into sub-pixels that all inherit
    the parent pixel's value.

    Parameters
    ----------
    healpix_array : array-like
    nside_out : int
        Target nside (must be >= input nside, must be power of 2)

    Returns
    -------
    upgraded : ndarray
    """
    _require_healpy('healpix_upgrade')
    arr = np.asarray(healpix_array, dtype=float)
    nside_in = hp.npix2nside(len(arr))
    if nside_out <= nside_in:
        raise ValueError(f"nside_out ({nside_out}) must be > input nside ({nside_in})")
    return hp.ud_grade(arr, nside_out)


def healpix_downgrade(healpix_array: npt.ArrayLike, nside_out: int,
                      method: str = 'mean') -> np.ndarray:
    """
    Downgrade a HEALPix map to a lower nside (coarser resolution).

    Parameters
    ----------
    healpix_array : array-like
    nside_out : int
        Target nside (must be <= input nside, must be power of 2)
    method : str
        'mean' (default) averages sub-pixels; 'sum' sums them.

    Returns
    -------
    downgraded : ndarray
    """
    _require_healpy('healpix_downgrade')
    arr = np.asarray(healpix_array, dtype=float)
    nside_in = hp.npix2nside(len(arr))
    if nside_out >= nside_in:
        raise ValueError(f"nside_out ({nside_out}) must be < input nside ({nside_in})")

    if method == 'mean':
        return hp.ud_grade(arr, nside_out)
    elif method == 'sum':
        ratio = (nside_in // nside_out) ** 2
        return hp.ud_grade(arr, nside_out) * ratio
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mean' or 'sum'.")


def healpix_combine(map1: npt.ArrayLike, map2: npt.ArrayLike,
                    operation: str = 'add') -> tuple[np.ndarray, int]:
    """
    Combine two HEALPix maps with automatic nside matching.

    If the maps have different nsides, the lower-resolution one is
    upgraded to match.

    Parameters
    ----------
    map1, map2 : array-like
        HEALPix maps (RING ordering)
    operation : str
        'add', 'subtract', 'multiply', 'divide', 'max', 'min'

    Returns
    -------
    result : ndarray
    nside : int
        The nside of the result
    """
    _require_healpy('healpix_combine')
    m1 = np.asarray(map1, dtype=float)
    m2 = np.asarray(map2, dtype=float)
    nside1 = hp.npix2nside(len(m1))
    nside2 = hp.npix2nside(len(m2))

    nside_out = max(nside1, nside2)
    if nside1 < nside_out:
        m1 = hp.ud_grade(m1, nside_out)
    if nside2 < nside_out:
        m2 = hp.ud_grade(m2, nside_out)

    ops: dict[str, Any] = {
        'add': np.add, 'subtract': np.subtract,
        'multiply': np.multiply, 'divide': np.divide,
        'max': np.maximum, 'min': np.minimum,
    }
    if operation not in ops:
        raise ValueError(f"Unknown operation '{operation}'. "
                         f"Choose from: {', '.join(ops)}")

    return ops[operation](m1, m2), nside_out

