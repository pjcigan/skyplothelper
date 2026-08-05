"""Mutual sky-visibility (co-visibility) regions for a set of ground stations.

A *simple, geometric* calculation: at an instant, the sky a station can see
above its elevation limit ``el_min`` is exactly a spherical cap centered on the
point overhead — center ``Dec = φ`` (station latitude), ``RA = GAST + λ``
(Greenwich apparent sidereal time + east longitude), angular radius
``90° − el_min``. The mutually-visible sky is the intersection of the per-station
caps, which drops straight into :class:`~skyplothelper.CompoundRegion`.

This is the geometric horizon only — it ignores refraction (lifts the apparent
horizon ~34′ right at the horizon, negligible above ~15°), station altitude, and
mount keyholes / slew limits. For observatory-accurate visibility (named sites,
real horizon models, scheduling) use the ``obsplanning`` package; this module is
the lightweight standalone case so ``obsplanning`` isn't a dependency for a quick
co-visibility footprint.

Three entry points:

* :func:`covisibility_circles` — the pure geometry+time bridge: stations + time
  → list of caps ``(name, center SkyCoord, radius_deg)``. No plotting backend.
* :func:`covisibility_region` — builds a :class:`CompoundRegion` of the
  instantaneously co-visible sky (full intersection, or "≥ k of N" coverage),
  honoring per-station azimuth horizon masks.
* :func:`covisibility_duration_band` — the time-integrated companion: the
  declination band(s) co-visible for at least ``min_hours`` over a sidereal day
  (mutual-visibility duration depends only on declination).
"""

from __future__ import annotations

import itertools
import warnings
from collections import namedtuple
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from ._timeinput import to_time


@contextmanager
def _relaxed_iers() -> Iterator[None]:
    """Disable astropy's IERS staleness guard for the enclosed block.

    This is a *geometric* co-visibility calculation: the only thing the IERS
    table buys here is the UT1−UTC offset (< 0.9 s ≈ 0.004° of Earth rotation),
    which is far below the precision of a 75°-radius visibility cap. Relaxing
    the guard lets the calc run offline and for future dates without forcing
    the user to download tables or edit astropy config."""
    from astropy.utils import iers
    with iers.conf.set_temp("auto_max_age", None):
        yield

__all__ = [
    "covisibility_circles",
    "covisibility_region",
    "covisibility_coverage",
    "covisibility_duration_band",
]


# --- input normalization ----------------------------------------------------

def _to_deg(value: Any, default: float | None = None) -> float | None:
    """Coerce an angle (astropy Quantity or number-in-degrees) to a float in
    degrees. ``None`` returns ``default``."""
    if value is None:
        return default
    if hasattr(value, "unit"):
        return float(value.to_value(u.deg))
    return float(value)


def _parse_mask(mask: npt.ArrayLike | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Normalize an azimuth horizon mask to ``(az_deg, el_deg)`` arrays sorted
    by azimuth, or ``None``. Accepts ``[[az...], [el...]]`` or an ``(N, 2)``
    array of ``(az, el)`` rows."""
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == 2:          # [[az...], [el...]]
        az, el = arr[0], arr[1]
    elif arr.ndim == 2 and arr.shape[1] == 2:        # [(az, el), ...]
        az, el = arr[:, 0], arr[:, 1]
    else:
        raise ValueError(
            "hor_mask must be [[azimuths], [elevations]] or an (N, 2) array "
            f"of (az, el) rows; got shape {arr.shape}")
    order = np.argsort(az)
    return az[order], el[order]


def _parse_stations(stations: Any, el_min: Any) -> list[dict[str, Any]]:
    """Normalize ``stations`` to a list of dicts with keys ``name``, ``lat``,
    ``lon`` (degrees), ``min_el`` (degrees), ``hor_mask`` (``(az, el)`` or None).

    Accepts:

    * ``{name: {'lat':, 'lon':, 'min_el'?:, 'hor_mask'?:}}`` — the primary form.
    * ``{name: EarthLocation | SkyCoord}`` — with the global ``el_min``.
    * a list of any of: the per-station dict (with a ``'name'`` key), an
      ``EarthLocation`` / ``SkyCoord``, or ``(name, EarthLocation|SkyCoord)``.
    """
    default_el = _to_deg(el_min, 15.0)

    def _from_point(obj: Any) -> tuple[float, float]:
        if isinstance(obj, EarthLocation):
            return float(obj.lon.to_value(u.deg)), float(obj.lat.to_value(u.deg))
        if isinstance(obj, SkyCoord):
            sph = obj.spherical
            return (float(sph.lon.to_value(u.deg)),
                    float(sph.lat.to_value(u.deg)))
        raise TypeError(
            "station point must be an astropy EarthLocation or SkyCoord, or a "
            "dict with 'lat'/'lon' keys; got " + type(obj).__name__)

    def _from_dict(name: Any, d: dict[str, Any]) -> dict[str, Any]:
        return dict(
            name=str(name),
            lat=_to_deg(d["lat"]),
            lon=_to_deg(d["lon"]),
            min_el=_to_deg(d.get("min_el"), default_el),
            hor_mask=_parse_mask(d.get("hor_mask")),
        )

    out: list[dict[str, Any]] = []
    if isinstance(stations, dict):
        for name, val in stations.items():
            if isinstance(val, dict):
                out.append(_from_dict(name, val))
            else:
                lon, lat = _from_point(val)
                out.append(dict(name=str(name), lat=lat, lon=lon,
                                min_el=default_el, hor_mask=None))
    else:
        for i, entry in enumerate(stations):
            if isinstance(entry, dict):
                out.append(_from_dict(entry.get("name", f"S{i}"), entry))
            elif isinstance(entry, tuple) and len(entry) == 2:
                name, pt = entry
                lon, lat = _from_point(pt)
                out.append(dict(name=str(name), lat=lat, lon=lon,
                                min_el=default_el, hor_mask=None))
            else:
                lon, lat = _from_point(entry)
                out.append(dict(name=f"S{i}", lat=lat, lon=lon,
                                min_el=default_el, hor_mask=None))
    if not out:
        raise ValueError("stations is empty")
    return out


def _gast_deg(time: Any) -> float:
    """Greenwich apparent sidereal time in degrees for ``time`` (anything
    :class:`astropy.time.Time` accepts: a ``Time``, ISO string, datetime, ...)."""
    t = to_time(time, _caller="covisibility")
    with _relaxed_iers():
        return float(t.sidereal_time("apparent", "greenwich").to_value(u.deg))


# --- per-station boundaries -------------------------------------------------

def _cap(station: dict[str, Any], gast_deg: float) -> tuple[float, float, float]:
    """The visibility cap of a station: ``(ra_deg, dec_deg, radius_deg)``."""
    ra = (gast_deg + station["lon"]) % 360.0
    dec = station["lat"]
    radius = 90.0 - station["min_el"]
    return ra, dec, radius


def _mask_boundary(station: dict[str, Any], time: Any,
                   n_az: int = 181) -> tuple[np.ndarray, np.ndarray]:
    """Sample a station's azimuth-masked horizon as an ICRS ``(lons, lats)``
    boundary curve, by transforming ``(az, el_mask(az))`` through AltAz at the
    station + time. The enclosed (zenith-side) region is what's visible."""
    az_in, el_in = station["hor_mask"]
    # Periodic interpolation of the mask onto a dense azimuth grid.
    az = np.linspace(0.0, 360.0, n_az)
    el = np.interp(az % 360.0, az_in % 360.0, el_in, period=360.0)
    loc = EarthLocation.from_geodetic(station["lon"] * u.deg,
                                      station["lat"] * u.deg)
    altaz = AltAz(obstime=to_time(time, _caller="covisibility"), location=loc)
    with _relaxed_iers():
        sky = SkyCoord(az=az * u.deg, alt=el * u.deg, frame=altaz).icrs
    return sky.ra.to_value(u.deg), sky.dec.to_value(u.deg)


def _add_station(region: Any, station: dict[str, Any], gast_deg: float,
                 time: Any, method: str) -> None:
    """Add one station's visible region to ``region`` via ``method`` in
    {'add', 'intersect'} — a polygon if it has a horizon mask, else a cap."""
    if station["hor_mask"] is not None:
        lons, lats = _mask_boundary(station, time)
        getattr(region, f"{method}_polygon")(lons, lats)
    else:
        ra, dec, radius = _cap(station, gast_deg)
        getattr(region, f"{method}_circle")(ra, dec, radius)


# --- public: pure geometry bridge -------------------------------------------

def covisibility_circles(stations: Any, time: Any = None, *,
                         el_min: Any = 15 * u.deg) -> list[dict[str, Any]]:
    """Per-station visibility caps for ``stations`` at ``time`` (pure geometry).

    The backend-free bridge: returns a list of caps that any code (including
    ``obsplanning``) can build a region from. Horizon masks are ignored here —
    each station yields its ``el_min`` circle; use :func:`covisibility_region`
    for mask-aware polygons.

    Parameters
    ----------
    stations : dict or list
        Station definitions — see :func:`covisibility_region`.
    time : astropy.time.Time or str or datetime, optional
        Observation instant (UTC assumed for naive inputs). ``None`` (the
        default) uses the **current time** (:meth:`astropy.time.Time.now`) — a
        convenience for a quick look at the co-visible sky (or its overall sky
        fraction) when the exact instant doesn't matter.
    el_min : Quantity or float
        Default minimum elevation (degrees). Per-station ``'min_el'`` overrides.

    Returns
    -------
    list of dict
        One per station: ``{'name', 'center': SkyCoord (ICRS), 'radius_deg',
        'min_el_deg'}``. ``center`` is the overhead point; ``radius_deg`` is
        ``90 − min_el``.
    """
    if time is None:
        time = Time.now()
    gast = _gast_deg(time)
    caps = []
    for st in _parse_stations(stations, el_min):
        ra, dec, radius = _cap(st, gast)
        caps.append(dict(
            name=st["name"],
            center=SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
            radius_deg=radius,
            min_el_deg=st["min_el"],
        ))
    return caps


# --- backend routing --------------------------------------------------------

def _compound_region_for(target: Any) -> Any:
    """An empty :class:`CompoundRegion` bound to ``target`` — an mpl WCSAxes, a
    :class:`Projector`, or a plotly Figure (all-sky or FITS)."""
    from .geometry import CompoundRegion
    from .geometry._projector import Projector
    if isinstance(target, Projector):
        return CompoundRegion(target)
    # plotly Figure (duck-typed to avoid importing plotly here)
    if hasattr(target, "layout") and hasattr(target, "add_trace"):
        meta = getattr(target.layout, "meta", None) or {}
        if isinstance(meta, dict) and meta.get("sph_wcs_header"):
            from .plotly.fits import make_fits_compound_region
            return make_fits_compound_region(target)
        from .plotly.core import make_compound_region
        return make_compound_region(target)
    return CompoundRegion(target)   # assume mpl WCSAxes


# --- public: instantaneous co-visibility region -----------------------------

def covisibility_region(target: Any, stations: Any, time: Any = None, *,
                        el_min: Any = 15 * u.deg,
                        min_stations: int | None = None) -> Any:
    """The instantaneously co-visible sky as a :class:`CompoundRegion`.

    Parameters
    ----------
    target : WCSAxes, Projector, or plotly Figure
        Where the region will be drawn — supplies the projection. Render the
        result with :meth:`CompoundRegion.render` (matplotlib) or
        :func:`skyplothelper.plotly.add_compound_region` (plotly).
    stations : dict or list
        Station definitions. The primary form is a dict::

            {'Mk': {'lat': 19.8, 'lon': -155.5, 'min_el': 15,
                    'hor_mask': [[az...], [el...]]}, ...}

        ``lat`` / ``lon`` are geodetic degrees (or Quantity); ``min_el`` and
        ``hor_mask`` are optional (defaulting to ``el_min`` / no mask). Values
        may also be astropy ``EarthLocation`` / ``SkyCoord`` (with the global
        ``el_min``), which lets ITRF *xyz* coordinates flow in via astropy.
    time : astropy.time.Time or str or datetime, optional
        Observation instant. ``None`` (the default) uses the **current time**
        (:meth:`astropy.time.Time.now`) — handy for seeing the co-visible sky
        (or its overall fraction) at *some* time when the exact instant doesn't
        matter.
    el_min : Quantity or float
        Default minimum elevation (degrees). Default ``15``.
    min_stations : int or None
        ``None`` (default) → sky visible to **all** stations (full
        intersection). An integer ``k`` → sky visible to **at least k** of N
        stations (the union of all k-station intersections — e.g. ``k=2`` for
        "any baseline" in VLBI). The number of combinations is ``C(N, k)``; a
        warning is emitted if that is large.

    Returns
    -------
    CompoundRegion
        The co-visible region (possibly empty — check
        :attr:`CompoundRegion.is_empty`).
    """
    if time is None:
        time = Time.now()
    region = _compound_region_for(target)
    sts = _parse_stations(stations, el_min)
    gast = _gast_deg(time)
    n = len(sts)
    k = n if min_stations is None else int(min_stations)
    # A default name so region.annotate(ax) works out of the box (overridable).
    region.label = ("Co-visible" if min_stations is None
                    else f"Co-visible (≥{k} of {n})")
    if k < 1:
        raise ValueError(f"min_stations must be >= 1, got {min_stations!r}")
    if k > n:
        warnings.warn(
            f"min_stations={k} exceeds the {n} stations given; the region "
            "is empty.", stacklevel=2)
        region._geom = None
        return region

    if k == n:
        # Full intersection: chain add/intersect directly.
        for i, st in enumerate(sts):
            _add_station(region, st, gast, time, "add" if i == 0 else "intersect")
        return region

    # "≥ k of N": union over all k-subsets of their intersection. Build each
    # station's single-cap geometry once (same projector), then combine with
    # shapely so we don't re-project per subset.
    from shapely.ops import unary_union

    from .geometry import CompoundRegion
    n_comb = _n_choose_k(n, k)
    if n_comb > 200:
        warnings.warn(
            f"min_stations={k} of {n} → {n_comb} station combinations; this "
            "may be slow. Consider a higher k or fewer stations.", stacklevel=2)
    geoms: list[Any] = []
    for st in sts:
        single = CompoundRegion(region.projector)
        _add_station(single, st, gast, time, "add")
        g = single._geom
        # A projected horizon circle can come back with a self-touching sliver;
        # clean it so the k-of-N intersections don't propagate degeneracies.
        if g is not None and not g.is_empty and not g.is_valid:
            g = g.buffer(0)
        geoms.append(g)

    pieces: list[Any] = []
    # Nearly-tangent horizon circles intersect in near-zero-width slivers, over
    # which GEOS emits "RuntimeWarning: invalid value encountered in
    # intersection" and can leave thin sliver lobes in the union. Silence that
    # specific numeric noise and buffer(0) the union to drop the slivers.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=RuntimeWarning,
            message='invalid value encountered')
        for subset in itertools.combinations(geoms, k):
            inter = subset[0]
            for g in subset[1:]:
                if inter is None or inter.is_empty:
                    break
                inter = inter.intersection(g)
            if inter is not None and not inter.is_empty:
                pieces.append(inter)
        region._geom = unary_union(pieces).buffer(0) if pieces else None
    return region


CoverageLayer = namedtuple("CoverageLayer", ["k", "region", "color", "artists"])


def covisibility_coverage(target: Any, stations: Any, time: Any = None, *,
                          mode: str = "exactly", el_min: Any = 15 * u.deg,
                          min_k: int = 1, cmap: Any = "viridis",
                          alpha: float = 0.55, label: bool = True,
                          render: bool = True, **kwargs: Any,
                          ) -> list["CoverageLayer"]:
    """Co-visibility coverage as colored layers: the sky seen by ``k`` of the
    ``N`` stations, for ``k = min_k … N``.

    A convenience over :func:`covisibility_region` that builds and (by default)
    draws one layer per coverage count, colored along *cmap*. Two layerings:

    - ``mode='exactly'`` (default) — **disjoint** bands: the sky visible to
      *exactly* k stations (``≥k`` minus ``≥k+1``). Reads as a coverage-count
      choropleth ("seen by exactly 5, 4, 3, …").
    - ``mode='atleast'`` — **nested** shells: the sky visible to *at least* k
      stations (each a superset of the next). Useful for planning ("anywhere
      ≥2 stations see is a valid baseline").

    Parameters
    ----------
    target : WCSAxes, Projector, or plotly Figure
        Where the layers are drawn / projected (as in :func:`covisibility_region`).
    stations : dict or list
        Station definitions — see :func:`covisibility_region`.
    time : astropy.time.Time or str or datetime, optional
        Observation instant; ``None`` (default) → current time.
    mode : {'exactly', 'atleast'}
        Disjoint exactly-k bands (default) or nested ≥k shells.
    el_min : Quantity or float
        Default minimum elevation (degrees).
    min_k : int
        Lowest coverage count to draw (default 1 → include single-station sky).
    cmap : str or Colormap
        Matplotlib colormap sampled across ``k = min_k … N`` (low k → low end).
    alpha : float
        Fill alpha for each layer (default 0.55).
    label : bool
        Annotate each layer with its ``k`` (``'≥k'`` in ``atleast`` mode) at a
        point inside it. Skipped on non-FITS frames (no label anchor there).
    render : bool
        Draw the layers on *target* (matplotlib only). ``False`` returns the
        regions without drawing.
    **kwargs
        Forwarded to :meth:`CompoundRegion.render` (e.g. ``edgecolor``).

    Returns
    -------
    list of CoverageLayer
        One ``CoverageLayer(k, region, color, artists)`` per coverage count
        (``region.label`` is set; empty layers are included with no artists),
        ordered by ascending ``k``.

    Examples
    --------
    >>> layers = sph.covisibility_coverage(ax, stations)          # now, exactly-k
    >>> layers = sph.covisibility_coverage(ax, stations, mode='atleast', min_k=2)
    >>> {lay.k: round(lay.region.area_frac, 3) for lay in layers}  # coverage table
    """
    if mode not in ("exactly", "atleast"):
        raise ValueError(f"mode must be 'exactly' or 'atleast', got {mode!r}")
    if time is None:
        time = Time.now()
    n = len(_parse_stations(stations, el_min))
    if n == 0:
        return []
    min_k = max(1, int(min_k))
    ks = list(range(min_k, n + 1))

    # "≥k" regions for every k we need (plus k=N+... implicitly empty). Built via
    # covisibility_region so masks / min_stations semantics stay identical.
    atleast = {k: covisibility_region(target, stations, time, el_min=el_min,
                                      min_stations=k)
               for k in range(min_k, n + 1)}

    import matplotlib as mpl
    cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    denom = max(len(ks) - 1, 1)

    ax = getattr(atleast[min_k], "ax", None)
    layers: list[CoverageLayer] = []
    for i, k in enumerate(ks):
        if mode == "atleast":
            reg = atleast[k]
            reg.label = f"≥{k}"
        else:                                   # exactly k = ≥k minus ≥(k+1)
            reg = atleast[k].difference(atleast[k + 1]) if k < n else atleast[k]
            reg.label = f"{k}"
        color = cmap_obj(i / denom)
        artists: list[Any] = []
        if render and ax is not None and not reg.is_empty:
            artists = list(reg.render(facecolor=color, alpha=alpha,
                                      zorder=2 + i, **kwargs))
            if label and getattr(reg.projector, "wcs", None) is not None:
                t = reg.annotate(ax, zorder=20 + i)
                if t is not None:
                    artists.append(t)
        layers.append(CoverageLayer(k=k, region=reg, color=color, artists=artists))
    return layers


def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    for i in range(k):
        num = num * (n - i) // (i + 1)
    return num


# --- public: time-integrated duration band ----------------------------------

def _hour_angle_halfwidth_deg(lat_deg: float, dec_deg: float,
                              el_min_deg: float) -> float:
    """Half-width of the visibility hour-angle window (degrees) for a source at
    ``dec`` from a station at ``lat`` with limit ``el_min``. 0 → never up,
    180 → always up (circumpolar above the limit)."""
    phi = np.radians(lat_deg)
    dec = np.radians(dec_deg)
    el = np.radians(el_min_deg)
    denom = np.cos(phi) * np.cos(dec)
    num = np.sin(el) - np.sin(phi) * np.sin(dec)
    if abs(denom) < 1e-12:
        # Source or station at a pole: no HA dependence — up iff overhead point
        # clears the limit.
        return 180.0 if num <= 0 else 0.0
    cos_h = num / denom
    if cos_h <= -1.0:
        return 180.0
    if cos_h >= 1.0:
        return 0.0
    return float(np.degrees(np.arccos(cos_h)))


def _circular_coverage(centers: Sequence[float], halfwidths: Sequence[float],
                       min_count: int, period: float = 24.0) -> float:
    """Total measure (same units as ``period``) of the circle covered by at
    least ``min_count`` of the intervals ``[c - h, c + h]`` (mod period)."""
    n_full = sum(1 for h in halfwidths if 2.0 * h >= period)
    if n_full >= min_count:
        return period
    need = min_count - n_full
    # Partial intervals, split at the 0/period seam into [s, e) with s < e.
    segs: list[tuple[float, float]] = []
    for c, h in zip(centers, halfwidths):
        if h <= 0.0 or 2.0 * h >= period:
            continue
        s = (c - h) % period
        e = (c + h) % period
        if s < e:
            segs.append((s, e))
        else:
            segs.append((s, period))
            segs.append((0.0, e))
    if not segs:
        return 0.0
    events: list[tuple[float, int]] = []
    for s, e in segs:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()
    total = 0.0
    count = 0
    prev = 0.0
    for pos, delta in events:
        if count >= need:
            total += pos - prev
        count += delta
        prev = pos
    return total


def covisibility_duration_band(target: Any, stations: Any, min_hours: float, *,
                               el_min: Any = 15 * u.deg,
                               min_stations: int | None = None,
                               n_dec: int = 361) -> Any:
    """Declination band(s) co-visible for at least ``min_hours`` per sidereal day.

    The time-integrated companion to :func:`covisibility_region`. Mutual-
    visibility *duration* depends only on declination (shifting every station's
    hour-angle window by the source RA leaves their overlap unchanged), so the
    "co-visible ≥ N hours" locus is a declination band — returned as a
    :class:`CompoundRegion` of one or more :meth:`~CompoundRegion.add_latitude_band`
    strips.

    Parameters
    ----------
    target, stations, el_min, min_stations :
        As in :func:`covisibility_region`. ``min_stations`` counts how many
        stations must see the source simultaneously (default: all).
    min_hours : float
        Minimum mutual-visibility duration (hours of sidereal time).
    n_dec : int
        Declination samples across −90…+90 used to locate the band edges.

    Returns
    -------
    CompoundRegion
        Union of the qualifying declination band(s) (empty if none qualify).
    """
    sts = _parse_stations(stations, el_min)
    n = len(sts)
    k = n if min_stations is None else int(min_stations)
    centers = [(-st["lon"] / 15.0) % 24.0 for st in sts]   # hours

    decs = np.linspace(-90.0, 90.0, n_dec)
    dur = np.empty_like(decs)
    for j, d in enumerate(decs):
        halfwidths = [_hour_angle_halfwidth_deg(st["lat"], d, st["min_el"]) / 15.0
                      for st in sts]
        dur[j] = _circular_coverage(centers, halfwidths, k)

    region = _compound_region_for(target)
    qualifies = dur >= min_hours
    added = False
    for lo, hi in _contiguous_ranges(decs, qualifies):
        region.add_latitude_band(lo, hi)
        added = True
    if not added:
        region._geom = None
    return region


def _contiguous_ranges(x: npt.ArrayLike,
                       mask: npt.ArrayLike) -> Iterator[tuple[float, float]]:
    """Yield ``(x_start, x_end)`` for each run of True in ``mask``."""
    x = np.asarray(x)
    mask = np.asarray(mask)
    if not mask.any():
        return
    idx = np.flatnonzero(mask)
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[splits + 1]))
    ends = np.concatenate((idx[splits], [idx[-1]]))
    for s, e in zip(starts, ends):
        yield float(x[s]), float(x[e])
