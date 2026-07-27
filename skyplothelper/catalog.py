"""Spatial filtering and matching of source catalogs.

Plot-independent utilities that filter a user's *own* catalog
(a pandas DataFrame, an astropy Table, a SkyCoord, or raw ``(ra, dec)``
arrays) by a region of interest. These are distinct from
:func:`skyplothelper.search_vizier`, which fetches a *new* catalog from
a remote service.

Three entry points:

* :func:`region_search` — the general engine. Filters a catalog by any
  region object exposing ``contains_points(ra, dec) -> bool mask``. A
  :class:`~skyplothelper.CompoundRegion` (arbitrary set-algebra ROI)
  plugs straight in, so an in/out test on a plotted region becomes a
  one-liner catalog filter.
* :func:`cone_search` — the common case: sources within an angular
  radius of a center. A light wrapper over :func:`region_search` using
  an *analytic* circle (true angular separation), so it is exact, works
  all-sky, and needs no plot or projection.
* :func:`crossmatch` — nearest-neighbor positional match between two
  catalogs within a tolerance (counterpart finding).

All inputs are assumed to be in the same coordinate frame, ICRS degrees
by default. The return is type-preserving: a DataFrame in yields a
filtered DataFrame out, a Table yields a Table, a SkyCoord yields a
filtered SkyCoord, and raw ``(ra, dec)`` arrays yield a boolean mask.
Only numpy/astropy are required (no pandas dependency — DataFrame
support is duck-typed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .core.coords import angulardistance
from .geometry._parsing import _coords_to_frame_deg, _parse_coords

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

# Common column-name spellings for sky coordinates, lowercased. Checked in
# order, so the most specific/standard names win. Catalogs from VizieR, Gaia,
# SIMBAD, and typical user CSVs are covered.
_RA_NAMES = (
    'ra', 'raj2000', '_raj2000', 'ra_icrs', 'radeg', 'ra_deg', 'raj2000_deg',
    'alpha', 'alpha_j2000', 'alphawin_j2000', 's_ra', 'right_ascension',
)
_DEC_NAMES = (
    'dec', 'de', 'dej2000', '_dej2000', 'de_icrs', 'dec_icrs', 'decdeg',
    'dec_deg', 'dej2000_deg', 'delta', 'delta_j2000', 'deltawin_j2000',
    's_dec', 'declination', 'decl', 'decj2000',
)

# Angular-unit conversions to degrees for the ``unit=`` convenience knob.
_UNIT_TO_DEG = {
    'deg': 1.0, 'degree': 1.0, 'degrees': 1.0,
    'arcmin': 1.0 / 60.0, 'arcminute': 1.0 / 60.0, 'arcminutes': 1.0 / 60.0,
    'arcsec': 1.0 / 3600.0, 'arcsecond': 1.0 / 3600.0, 'arcseconds': 1.0 / 3600.0,
    'rad': 180.0 / np.pi, 'radian': 180.0 / np.pi, 'radians': 180.0 / np.pi,
    'mas': 1.0 / 3.6e6, 'uas': 1.0 / 3.6e9,
}


# --- input parsing -------------------------------------------------------

def _to_deg(value: Any, unit: str = 'deg') -> float:
    """Resolve an angle to degrees.

    An astropy Quantity is converted directly (``unit`` ignored); a bare
    float is interpreted in ``unit`` (default degrees).
    """
    if hasattr(value, 'to'):  # Quantity duck-type
        return float(value.to('deg').value)
    try:
        factor = _UNIT_TO_DEG[unit.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown unit {unit!r}; use one of "
            f"{sorted(set(_UNIT_TO_DEG)) }, or pass an astropy Quantity.")
    return float(value) * factor


def _resolve_center(center: SkyCoord | tuple[float, float] | str, frame: str = 'icrs') -> tuple[float, float]:
    """Resolve a cone center to ``(lon_deg, lat_deg)`` in *frame*.

    Accepts a name string (resolved via :func:`~skyplothelper.resolve_name`,
    needs astroquery), a scalar SkyCoord, or a ``(lon, lat)`` pair in degrees.

    *frame* defaults to ICRS — the frame this module works in and the one
    catalog columns almost always use. Set it when the catalog's coordinate
    columns are in another frame (e.g. galactic ``l``/``b``), so the center and
    the catalog are compared in the same system. A bare ``(lon, lat)`` pair is
    assumed to already be in *frame* and is passed through unconverted.
    """
    if isinstance(center, str):
        from .queries import resolve_name
        center = resolve_name(center)
    if hasattr(center, 'transform_to'):  # SkyCoord duck-type
        lon, lat = _coords_to_frame_deg(center, frame)
        return float(lon), float(lat)
    lon, lat = center  # (lon, lat) pair, already in *frame*
    return float(lon), float(lat)


def _find_col(colnames: list[str], candidates: tuple[str, ...]) -> str | None:
    """Find the first column whose lowercased name matches a candidate."""
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _resolve_catalog_coords(
    catalog: Any, ra_col: str | None, dec_col: str | None,
    frame: str = 'icrs',
) -> tuple[np.ndarray, np.ndarray, Callable[..., Any]]:
    """Extract ``(lon, lat)`` arrays and a type-preserving repack function.

    The repack closure takes a boolean mask and rebuilds a filtered output
    of the same kind as ``catalog`` (DataFrame/Table/SkyCoord), or returns
    the mask itself for raw coordinate-array input.

    *frame* only affects a **SkyCoord** catalog, which carries its own frame
    and is converted into this one. Column- and array-based input is assumed to
    already be in *frame* (there is nothing to convert from).
    """
    # astropy Table (has .colnames). Checked before the DataFrame branch:
    # a Table also has .columns, but only the Table exposes .colnames, and
    # touching a DataFrame-style attribute (e.g. .iloc) on a Table raises.
    if hasattr(catalog, 'colnames'):
        cols = list(catalog.colnames)
        rc = ra_col or _find_col(cols, _RA_NAMES)
        dc = dec_col or _find_col(cols, _DEC_NAMES)
        if rc is None or dc is None:
            raise ValueError(
                "Could not auto-detect RA/Dec columns in Table "
                f"(columns: {cols}). Pass ra_col= and dec_col= explicitly.")
        ra = np.asarray(catalog[rc], dtype=float)
        dec = np.asarray(catalog[dc], dtype=float)

        def repack_tbl(mask: np.ndarray, *, extra_cols: dict[str, Any] | None = None,
                       sort_key: np.ndarray | None = None) -> Any:
            m = np.asarray(mask, dtype=bool)
            out = catalog[m]
            if extra_cols:
                for k, v in extra_cols.items():
                    out[k] = np.asarray(v)[m]
            if sort_key is not None:
                order = np.argsort(np.asarray(sort_key)[m], kind='stable')
                out = out[order]
            return out

        return ra, dec, repack_tbl

    # pandas DataFrame (duck-typed: has .columns but not .colnames).
    if hasattr(catalog, 'columns'):
        cols = list(catalog.columns)
        rc = ra_col or _find_col(cols, _RA_NAMES)
        dc = dec_col or _find_col(cols, _DEC_NAMES)
        if rc is None or dc is None:
            raise ValueError(
                "Could not auto-detect RA/Dec columns in DataFrame "
                f"(columns: {cols}). Pass ra_col= and dec_col= explicitly.")
        ra = np.asarray(catalog[rc], dtype=float)
        dec = np.asarray(catalog[dc], dtype=float)

        def repack_df(mask: np.ndarray, *, extra_cols: dict[str, Any] | None = None,
                      sort_key: np.ndarray | None = None) -> Any:
            m = np.asarray(mask, dtype=bool)
            out = catalog[m].copy()
            if extra_cols:
                for k, v in extra_cols.items():
                    out[k] = np.asarray(v)[m]
            if sort_key is not None:
                order = np.argsort(np.asarray(sort_key)[m], kind='stable')
                out = out.iloc[order]
            return out

        return ra, dec, repack_df

    # SkyCoord (array) — filter the SkyCoord itself.
    if hasattr(catalog, 'transform_to'):
        ra, dec = _coords_to_frame_deg(catalog, frame)

        def repack_coord(mask: np.ndarray, *, extra_cols: dict[str, Any] | None = None,
                         sort_key: np.ndarray | None = None) -> Any:
            m = np.asarray(mask, dtype=bool)
            out = catalog[m]
            if sort_key is not None:
                order = np.argsort(np.asarray(sort_key)[m], kind='stable')
                out = out[order]
            return out  # extra_cols can't attach to a SkyCoord; ignored

        return np.asarray(ra, float), np.asarray(dec, float), repack_coord

    # Raw (ra, dec) pair of array-likes — return the mask (nothing to subset).
    try:
        ra_in, dec_in = catalog
    except (ValueError, TypeError):
        raise TypeError(
            "catalog must be a pandas DataFrame, astropy Table, SkyCoord, "
            "or an (ra, dec) pair of arrays.")
    ra = np.asarray(ra_in, dtype=float)
    dec = np.asarray(dec_in, dtype=float)

    def repack_arrays(mask: np.ndarray, *, extra_cols: dict[str, Any] | None = None,
                      sort_key: np.ndarray | None = None) -> Any:
        return np.asarray(mask, dtype=bool)

    return ra, dec, repack_arrays


def _separations(ra: np.ndarray, dec: np.ndarray,
                 center_ra: float, center_dec: float) -> np.ndarray:
    """Angular separation (deg) of each ``(ra, dec)`` from a center."""
    pts = np.column_stack([np.ravel(ra), np.ravel(dec)])
    ctr = np.broadcast_to(np.array([center_ra, center_dec], float), pts.shape)
    sep = np.asarray(angulardistance(pts, ctr), dtype=float)
    return sep.reshape(np.shape(ra))


# --- analytic circle region ----------------------------------------------

class _SkyCircle:
    """Analytic small-circle region (center + angular radius).

    Exposes the same ``contains_points`` protocol as
    :class:`~skyplothelper.CompoundRegion`, but tests membership with true
    angular separation rather than a projected pixel-space shape — so it is
    exact, works all-sky, and needs no axes/projection.
    """

    def __init__(self, center_ra: float, center_dec: float, radius_deg: float):
        self.center_ra = center_ra
        self.center_dec = center_dec
        self.radius_deg = radius_deg

    def contains_points(self, ra: Any, dec: Any = None) -> np.ndarray:
        ra_arr, dec_arr = _parse_coords(ra, dec)
        sep = _separations(np.asarray(ra_arr, float), np.asarray(dec_arr, float),
                           self.center_ra, self.center_dec)
        return sep <= self.radius_deg


def _region_center(region: Any, center: SkyCoord | tuple[float, float] | str, frame: str = 'icrs',
                   ) -> tuple[float | None, float | None]:
    """Resolve a center for separation/sort: explicit ``center`` wins, else
    fall back to an analytic region's own center if it exposes one."""
    if center is not None:
        return _resolve_center(center, frame)
    if hasattr(region, 'center_ra') and hasattr(region, 'center_dec'):
        return region.center_ra, region.center_dec
    return None, None


# --- public API ----------------------------------------------------------

def region_search(catalog: Any, region: Any, *,
                  ra_col: str | None = None, dec_col: str | None = None,
                  center: SkyCoord | tuple[float, float] | str = None, frame: str = 'icrs',
                  return_mask: bool = False,
                  add_separation: bool = False, sep_col: str = 'separation',
                  sort: bool = False) -> Any:
    """Filter a catalog to the sources inside a region.

    The general spatial-filter engine: ``region`` can be any object with a
    ``contains_points(ra, dec) -> bool array`` method. A
    :class:`~skyplothelper.CompoundRegion` (arbitrary set-algebra ROI)
    satisfies this protocol directly, so a plotted region's in/out test
    becomes a catalog filter. :func:`cone_search` is a thin wrapper over
    this function.

    Parameters
    ----------
    catalog : DataFrame, Table, SkyCoord, or (ra, dec) arrays
        The sources to filter, in ICRS degrees (RA/Dec auto-detected for
        table-like input; override with ``ra_col``/``dec_col``).
    region : object
        Anything exposing ``contains_points(ra, dec)``, e.g. a
        :class:`~skyplothelper.CompoundRegion`.
    ra_col, dec_col : str, optional
        Column names for table-like input. Auto-detected if omitted.
    center : str, SkyCoord, or (ra, dec), optional
        Reference point for ``add_separation`` / ``sort``. Required for
        those options unless ``region`` exposes its own center (as the
        analytic circle from :func:`cone_search` does).
    return_mask : bool
        Return a boolean mask instead of a filtered subset.
    add_separation : bool
        Append a separation column (degrees from ``center``) to table-like
        output. Requires a center.
    sep_col : str
        Name of the separation column (default ``'separation'``).
    sort : bool
        Sort the output by ascending separation. Requires a center.

    Returns
    -------
    Filtered catalog of the same type as ``catalog`` (DataFrame → DataFrame,
    Table → Table, SkyCoord → SkyCoord), or a boolean ndarray for raw
    ``(ra, dec)`` input or when ``return_mask=True``.

    Examples
    --------
    >>> # sources inside a plotted set-algebra region
    >>> inside = region_search(df, my_compound_region)
    >>> # with a reference center for separations
    >>> inside = region_search(df, region, center='M31', add_separation=True)
    """
    ra, dec, repack = _resolve_catalog_coords(catalog, ra_col, dec_col, frame)
    mask = np.asarray(region.contains_points(ra, dec), dtype=bool)
    mask = mask.reshape(np.shape(ra))

    if return_mask:
        return mask

    sep = None
    if add_separation or sort:
        cra, cdec = _region_center(region, center, frame)
        if cra is None or cdec is None:
            raise ValueError(
                "add_separation/sort require a center; pass center=(ra, dec), "
                "a name, or a SkyCoord (cone_search supplies one automatically).")
        sep = _separations(ra, dec, cra, cdec)

    extra = {sep_col: sep} if (add_separation and sep is not None) else None
    return repack(mask, extra_cols=extra, sort_key=sep if sort else None)


def cone_search(catalog: Any, center: SkyCoord | tuple[float, float] | str, radius: Any, *, unit: str = 'deg',
                ra_col: str | None = None, dec_col: str | None = None,
                frame: str = 'icrs',
                return_mask: bool = False, add_separation: bool = False,
                sep_col: str = 'separation', sort: bool = False) -> Any:
    """Filter a catalog to sources within an angular radius of a center.

    A light wrapper over :func:`region_search` using an *analytic* circle:
    membership is the true angular separation ``sep <= radius``, computed
    with the Vincenty formula. Exact, all-sky, and independent of any plot
    or projection.

    Parameters
    ----------
    catalog : DataFrame, Table, SkyCoord, or (ra, dec) arrays
        The sources to filter, in ICRS degrees.
    center : str, SkyCoord, or (lon, lat)
        Cone center. A name string is resolved via
        :func:`~skyplothelper.resolve_name` (needs astroquery).
    frame : str
        Coordinate frame the search is carried out in. Default ``'icrs'``.
        A SkyCoord center (and a SkyCoord catalog) is converted into this
        frame; bare ``(lon, lat)`` pairs and catalog columns are assumed to
        already be in it. Set it when your catalog columns are galactic
        ``l``/``b`` or ecliptic, so center and catalog are compared in the
        same system.
    radius : float or Quantity
        Cone radius. A bare float is interpreted in ``unit`` (default
        degrees); an astropy Quantity overrides ``unit``.
    unit : str
        Unit for a bare-float ``radius``: ``'deg'`` (default), ``'arcmin'``,
        ``'arcsec'``, ``'rad'``, ``'mas'``, ``'uas'``.
    ra_col, dec_col : str, optional
        Column names for table-like input. Auto-detected if omitted.
    return_mask, add_separation, sep_col, sort
        As in :func:`region_search`. The center is supplied automatically,
        so ``add_separation`` and ``sort`` work without a separate
        ``center`` argument.

    Returns
    -------
    Filtered catalog of the same type as ``catalog`` (or a boolean mask for
    raw arrays / ``return_mask=True``).

    Examples
    --------
    >>> # 2MASS sources within 10 arcmin of M31, nearest first
    >>> near = cone_search(df, 'M31', 10, unit='arcmin', sort=True)
    >>> # within 0.5 deg of an explicit position, with separations
    >>> near = cone_search(tbl, (180.0, 45.0), 0.5, add_separation=True)
    """
    cra, cdec = _resolve_center(center, frame)
    radius_deg = _to_deg(radius, unit)
    circle = _SkyCircle(cra, cdec, radius_deg)
    return region_search(catalog, circle, ra_col=ra_col, dec_col=dec_col,
                         frame=frame,
                         return_mask=return_mask, add_separation=add_separation,
                         sep_col=sep_col, sort=sort)


def crossmatch(catalog: Any, reference: Any, max_sep: Any, *, unit: str = 'deg',
               ra_col: str | None = None, dec_col: str | None = None,
               ref_ra_col: str | None = None, ref_dec_col: str | None = None,
               sep_col: str = 'match_sep', idx_col: str = 'match_idx',
               sort: bool = False, return_indices: bool = False) -> Any:
    """Positionally match a catalog against a reference catalog.

    For each source in ``catalog``, finds its nearest neighbor in
    ``reference`` (via astropy's ``match_to_catalog_sky``) and keeps only
    those matched within ``max_sep``. Counterpart finding / cross-ID.

    Parameters
    ----------
    catalog, reference : DataFrame, Table, SkyCoord, or (ra, dec) arrays
        The catalog to match and the catalog to match against, ICRS degrees.
    max_sep : float or Quantity
        Maximum match separation. A bare float is interpreted in ``unit``
        (default degrees); a Quantity overrides ``unit``.
    unit : str
        Unit for a bare-float ``max_sep`` (see :func:`cone_search`).
    ra_col, dec_col : str, optional
        Coordinate columns for ``catalog`` (auto-detected if omitted).
    ref_ra_col, ref_dec_col : str, optional
        Coordinate columns for ``reference`` (auto-detected if omitted).
    sep_col : str
        Name of the match-separation column added to table-like output
        (degrees). Default ``'match_sep'``.
    idx_col : str
        Name of the column holding the matched row index into ``reference``.
        Default ``'match_idx'``.
    sort : bool
        Sort the output by ascending match separation.
    return_indices : bool
        Instead of a filtered catalog, return ``(idx, sep_deg, mask)`` raw
        arrays (full length of ``catalog``): nearest-reference index,
        separation in degrees, and the within-tolerance boolean mask.

    Returns
    -------
    Filtered subset of ``catalog`` (same type) holding only matched sources,
    with ``idx_col`` and ``sep_col`` added for table-like input — or the raw
    ``(idx, sep_deg, mask)`` arrays when ``return_indices=True``.

    Examples
    --------
    >>> # rows of `obs` with a Gaia source within 1 arcsec
    >>> matched = crossmatch(obs, gaia, 1.0, unit='arcsec')
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    ra, dec, repack = _resolve_catalog_coords(catalog, ra_col, dec_col)
    rra, rdec, _ = _resolve_catalog_coords(reference, ref_ra_col, ref_dec_col)

    cat_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    ref_coord = SkyCoord(ra=rra * u.deg, dec=rdec * u.deg)
    idx, sep2d, _ = cat_coord.match_to_catalog_sky(ref_coord)
    sep_deg = np.asarray(sep2d.deg, dtype=float)

    max_deg = _to_deg(max_sep, unit)
    mask = sep_deg <= max_deg

    if return_indices:
        return np.asarray(idx), sep_deg, mask

    extra = {idx_col: np.asarray(idx), sep_col: sep_deg}
    return repack(mask, extra_cols=extra, sort_key=sep_deg if sort else None)
